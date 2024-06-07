import copy
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
from torch.ao.quantization.experimental.adaround_fake_quantize import (
    AdaroundFakeQuantizer,
)
from torch.ao.quantization.experimental.adaround_loss import AdaptiveRoundingLoss
from torch.ao.quantization.observer import MinMaxObserver
from torch.nn import functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, TensorDataset


class AdaptiveRoundingOptimizer:
    def __init__(
        self,
        model: Union[torch.nn.Module, torch.nn.DataParallel],
        callback: Callable[
            [
                Union[torch.nn.Module, torch.nn.DataParallel],
                Any,
                Optional[torch.nn.Module],
            ],
            None,
        ],
        forward_hook_wrapper: Callable[[List[torch.Tensor]], Callable],
        data: Any,
        observer: Type[torch.ao.quantization.observer.ObserverBase] = MinMaxObserver,
        max_iter=10000,
        dtype: torch.dtype = torch.qint8,
        quant_min=-128,
        quant_max=127,
        qscheme: torch.qscheme = torch.per_tensor_symmetric,
        batch_size: int = 256,
        feed_forward_wrapper: Optional[torch.nn.Module] = None,
    ):
        if torch.cuda.is_available():
            self.model = model.cuda()
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(model)
        else:
            self.model = model
        self.q_model = copy.deepcopy(self.model)
        self.device = torch.device("cuda") if torch.cuda.is_available() else None
        self.callback = callback
        self.forward_hook_wrapper = forward_hook_wrapper
        # TODO rather than having a data as list type or, we better pass *iterator* instead of list
        self.data = data
        self.batch_size = min(batch_size, len(data))
        self.max_iter = max_iter
        self.adaptive_round_loss_fn = AdaptiveRoundingLoss(
            max_iter=self.max_iter, warm_start=0.2
        )
        self.dtype = dtype
        self.observer = observer
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qscheme = qscheme
        self.feed_forward_wrapper = feed_forward_wrapper

    def run_adaround(self) -> torch.nn.Module:
        layer_list: List[Tuple[str, torch.nn.Module, torch.nn.Module]] = []
        for (name, module), q_module in zip(
            self.model.named_modules(), self.q_model.modules()
        ):
            if isinstance(module, torch.nn.ReLU):
                # Disable all inplace operations
                module.inplace = False
            if isinstance(q_module, torch.nn.ReLU):
                # Disable all inplace operations
                q_module.inplace = False
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                # Knowing activation ahead-of-time would be helpful for asymmetric formulation
                # But this is challenging in eager mode, but graph module.
                layer_list.append((name, module, q_module))
        print(f"Total number of layers : {len(layer_list)}")  # noqa: G004

        for name, module, q_module in layer_list:
            print(
                f"Kick start adaptive rounding on {name} module {module}"  # noqa: G004
            )
            self.optimize_adaptive_rounding(
                module,
                q_module,
                None,
            )

        return (
            self.q_model.module
            if isinstance(self.q_model, DataParallel)
            else self.q_model
        )

    def get_data_inp_out(
        self, module: torch.nn.Module, q_module: torch.nn.Module, data: List[Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        fp_out: List[torch.Tensor] = []
        q_input: List[torch.Tensor] = []
        fp_input: List[torch.Tensor] = []
        fp32_fetcher: List[torch.Tensor] = []
        quant_fetcher: List[torch.Tensor] = []
        handler1 = module.register_forward_hook(self.forward_hook_wrapper(fp32_fetcher))
        handler2 = q_module.register_forward_hook(
            self.forward_hook_wrapper(quant_fetcher)
        )
        if torch.cuda.is_available():
            # Somehow, we need to move the model continuously
            # Otherwise, the model will be lowered to CPU misteriously
            self.model = self.model.cuda()
            self.q_model = self.q_model.cuda()
        for data_ in data:
            with torch.no_grad():
                self.callback(self.model, data_, self.feed_forward_wrapper)
                self.callback(self.q_model, data_, self.feed_forward_wrapper)
            fp32_output = fp32_fetcher[1]
            quant_input = quant_fetcher[0]
            fp_out.append(fp32_output)
            q_input.append(quant_input)
            fp_input.append(fp32_fetcher[0])
        handler1.remove()
        handler2.remove()
        return q_input, fp_out, fp_input

    @torch.no_grad()
    def feed_forward(self, x, weight, module):
        if isinstance(module, torch.nn.Conv1d):
            out = torch.nn.functional.conv1d(
                x,
                weight,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        elif isinstance(module, torch.nn.Linear):
            out = torch.nn.functional.linear(
                x,
                weight,
                bias=module.bias,
            )
        else:
            raise NotImplementedError
        return out

    def _compute_and_display_local_losses(
        self,
        ada_quantizer: AdaroundFakeQuantizer,
        q_module: torch.nn.Module,
        q_inp: torch.Tensor,
        fp_out: torch.Tensor,
    ):
        with torch.no_grad():
            ada_quantizer.use_soft_rounding = False
            q_w_hard_round = ada_quantizer(q_module.weight)
            out_hard_quant = self.feed_forward(q_inp, q_w_hard_round, q_module)
            ada_quantizer.use_soft_rounding = True
            q_w_soft_round = ada_quantizer(q_module.weight)
            out_soft_quant = self.feed_forward(q_inp, q_w_soft_round, q_module)
            soft_quant_loss = F.mse_loss(out_soft_quant, fp_out)
            hard_quant_loss = F.mse_loss(out_hard_quant, fp_out)
            print(
                f"soft quant loss: {soft_quant_loss.item()} hard quant loss: {hard_quant_loss.item()}"  # noqa: G004
            )

    def optimize_adaptive_rounding(
        self,
        module: torch.nn.Module,
        q_module: torch.nn.Module,
        activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        ada_quantizer = AdaroundFakeQuantizer(
            dtype=self.dtype,
            observer=self.observer,
            qscheme=self.qscheme,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            reduce_range=False,
        )
        ada_quantizer.enable_observer()
        ada_quantizer(q_module.weight)
        ada_quantizer.disable_observer()
        ada_quantizer.enable_fake_quant()
        optimizer = torch.optim.Adam([ada_quantizer.V])
        inp, out, fp_in = self.get_data_inp_out(module, q_module, self.data)

        print("==================== Before adaround ====================")
        assert (
            torch.abs(out[0] - module(fp_in[0])).sum().item() == 0
        ), "In-placed activation is detected, please do not use activation in-placed"
        # Stack the tensors in each list into a single tensor
        # Assuming inp and out are your lists of tensors
        inp_tensor = torch.vstack(inp)
        out_tensor = torch.vstack(out)
        dataset = TensorDataset(inp_tensor, out_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._compute_and_display_local_losses(ada_quantizer, q_module, inp[0], out[0])
        global_idx = 0
        one_iter = len(out) // self.batch_size
        for iteration in range(self.max_iter // one_iter):
            reconstruction_loss = regularization_loss = torch.tensor(0)
            for q_inp, fp_out in dataloader:
                optimizer.zero_grad()
                q_weight = ada_quantizer(q_module.weight)
                if isinstance(module, torch.nn.Conv1d):
                    q_out = torch.nn.functional.conv1d(
                        q_inp,
                        q_weight,
                        bias=q_module.bias,
                        stride=q_module.stride,
                        padding=q_module.padding,
                        dilation=q_module.dilation,
                        groups=q_module.groups,
                    )
                elif isinstance(q_module, torch.nn.Linear):
                    q_out = torch.nn.functional.linear(
                        q_inp,
                        q_weight,
                        bias=q_module.bias,
                    )
                else:
                    raise NotImplementedError
                regularization_loss, reconstruction_loss = self.adaptive_round_loss_fn(
                    fp_out,
                    q_out,
                    ada_quantizer.V,
                    curr_iter=global_idx,
                )
                loss = regularization_loss + reconstruction_loss
                loss.backward()
                optimizer.step()
                global_idx += 1
                if global_idx >= self.max_iter:
                    break
            if global_idx >= self.max_iter:
                break
            if iteration % 30 == 0:
                print(
                    f"glob iter {global_idx} regularization_loss {regularization_loss.item()} "  # noqa: G004
                    f"reconstruction_loss {reconstruction_loss.item()}"  # noqa: G004
                )
        print("==================== After adaround ====================")
        self._compute_and_display_local_losses(ada_quantizer, q_module, inp[0], out[0])

        ada_quantizer.use_soft_rounding = True
        ada_quantizer.V.requires_grad = False
        ada_quantizer = ada_quantizer.eval()
        q_weight = ada_quantizer(q_module.weight)
        # At the end of optimization, we need to copy the adarounded weight back to the original module
        q_module.weight.data.copy_(q_weight)
        # Eager mode requires observer to be set as "weight_fake_quant" to be parsed
        q_module.weight_fake_quant = ada_quantizer.activation_post_process
