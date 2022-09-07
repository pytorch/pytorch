# Owner(s): ["oncall: distributed"]

import functools
import sys
from typing import Optional, Type

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestFSDPUseOrigParams(FSDPTest):
    def _get_optim(
        self,
        model: nn.Module,
        optim_class: Type[torch.optim.Optimizer],
        multi_tensor: bool,
    ) -> torch.optim.Optimizer:
        """
        Constructs an Adam optimizer with three parameter groups, one for
        weights, one for biases, and one for everything else, each with
        different weight decay and learning rates.
        """
        param_groups = [
            {"params": [], "weight_decay": 0.1, "lr": 1e-2},
            {"params": [], "weight_decay": 0.01, "lr": 1e-3},
            {"params": []}
        ]
        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                param_groups[0]["params"].append(param)
            elif "bias" in param_name:
                param_groups[1]["params"].append(param)
            else:
                param_groups[2]["params"].append(param)
        return optim_class(param_groups, lr=5e-3, foreach=multi_tensor)

    def _get_ddp_transformer(
        self,
        optim_class: Type[torch.optim.Optimizer],
        multi_tensor: bool,
        find_unused_params: bool,
    ):
        """
        Returns a transformer with shared parameters wrapped with DDP and a
        corresponding optimizer.
        """
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        ddp_model = DDP(
            model,
            device_ids=[self.rank],
            find_unused_parameters=find_unused_params,
        )
        ddp_optim = self._get_optim(ddp_model, optim_class, multi_tensor)
        return ddp_model, ddp_optim

    def _get_fsdp_transformer(
        self,
        init_optim_before_wrap: bool,
        optim_class: Type[torch.optim.Optimizer],
        multi_tensor: bool,
        backward_prefetch: Optional[BackwardPrefetch],
        cpu_offload: Optional[CPUOffload],
    ):
        """
        Returns a transformer with shared parameters wrapped with FSDP and a
        corresponding optimizer.
        """
        # Each transformer layer has multiple linear layers, so this policy, in
        # combination with the parameter group construction, ensures different
        # hyperparameter settings within one `FlatParameter`
        fsdp_kwargs = {
            "auto_wrap_policy": functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer},
            ),
            "use_orig_params": True,
            "backward_prefetch": backward_prefetch,
            "cpu_offload": cpu_offload,
        }
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        if init_optim_before_wrap:
            fsdp_optim = self._get_optim(model, optim_class, multi_tensor)
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
        else:
            fsdp_model = FSDP(model, self.process_group, **fsdp_kwargs)
            fsdp_optim = self._get_optim(fsdp_model, optim_class, multi_tensor)
        return fsdp_model, fsdp_optim

    def _check_train_parity(
        self,
        ddp_model: DDP,
        ddp_optim: torch.optim.Optimizer,
        fsdp_model: FSDP,
        fsdp_optim: torch.optim.Optimizer,
        num_iters: int = 10,
    ):
        """Checks training parity between DDP and FSDP."""
        device = torch.device("cuda")
        for _ in range(num_iters):
            iter_losses = []
            for model, optim in ((ddp_model, ddp_optim), (fsdp_model, fsdp_optim)):
                module = model.module
                optim.zero_grad()
                inp = module.get_input(device)
                output = model(*inp)
                loss = module.get_loss(inp, output).to(device)
                iter_losses.append(loss)
                module.run_backward(loss)
                # Perform the DDP optimizer step on CPU to match FSDP if needed
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(torch.device("cpu"))
                optim.step()
                if model is ddp_model and fsdp_model.cpu_offload.offload_params:
                    model.to(torch.device("cuda"))
            self.assertEqual(iter_losses[0].item(), iter_losses[1].item())
            iter_losses.clear()
        atol = 1e-5 if self.world_size <= 2 else 1e-4
        rtol = 1e-6 if self.world_size <= 2 else 1e-5
        with FSDP.summon_full_params(fsdp_model):
            for p1, p2 in zip(ddp_model.parameters(), fsdp_model.parameters()):
                torch.testing.assert_close(p1, p2, atol=atol, rtol=rtol)

    def _get_subtest_config(self):
        return {
            "init_optim_before_wrap": [False, True],
            "optim_class": [torch.optim.Adam, torch.optim.AdamW],
            "multi_tensor": [False, True],
            "backward_prefetch": [
                None,
                BackwardPrefetch.BACKWARD_PRE,
                BackwardPrefetch.BACKWARD_POST,
            ],
            "cpu_offload": [None, CPUOffload(True)]
        }

    def _test_diff_hyperparams(
        self,
        init_optim_before_wrap: bool,
        optim_class: Type[torch.optim.Optimizer],
        multi_tensor: bool,
        backward_prefetch: Optional[BackwardPrefetch],
        cpu_offload: Optional[CPUOffload],
    ):
        """
        Args:
            init_optim_before_wrap (bool): If ``True``, initializes the
                FSDP optimizer before wrapping the model with FSDP; otherwise,
                initializes the FSDP optimizer after wrapping the model with
                FSDP. We permit both forms of initialization to give users
                flexibility.
        """
        ddp_model, ddp_optim = self._get_ddp_transformer(
            optim_class=optim_class,
            multi_tensor=multi_tensor,
            find_unused_params=False,
        )
        fsdp_model, fsdp_optim = self._get_fsdp_transformer(
            init_optim_before_wrap,
            optim_class,
            multi_tensor,
            backward_prefetch,
            cpu_offload,
        )
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim)

    @skip_if_lt_x_gpu(2)
    def test_diff_hyperparams(self):
        """
        Tests FSDP parity with DDP when using multiple parameter groups with
        different hyperparameter settings.
        """
        self.run_subtests(
            self._get_subtest_config(),
            self._test_diff_hyperparams,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("multi_tensor", [False, True])
    def test_diff_trainability(self, multi_tensor: bool):
        """
        Tests FSDP parity with DDP when using multiple parameter groups and
        freezing the parameters in one parameter group.
        """
        optim_class = torch.optim.Adam
        ddp_model, ddp_optim = self._get_ddp_transformer(
            optim_class=optim_class,
            multi_tensor=multi_tensor,
            find_unused_params=True,
        )
        fsdp_model, fsdp_optim = self._get_fsdp_transformer(
            init_optim_before_wrap=False,
            optim_class=optim_class,
            multi_tensor=multi_tensor,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            cpu_offload=None,
        )
        # Freeze all biases (which happen to be in the same parameter group)
        for param_name, param in ddp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        for param_name, param in fsdp_model.named_parameters():
            if "bias" in param_name:
                param.requires_grad_(False)
        self._check_train_parity(ddp_model, ddp_optim, fsdp_model, fsdp_optim)


instantiate_parametrized_tests(TestFSDPUseOrigParams)

if __name__ == "__main__":
    run_tests()
