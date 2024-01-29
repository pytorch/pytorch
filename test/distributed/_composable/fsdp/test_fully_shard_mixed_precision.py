# Owner(s): ["oncall: distributed"]

import copy
import functools

from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_distributed import (
    requires_nccl_version,
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    check_1d_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    MLP,
    patch_reduce_scatter,
    reduce_scatter_with_assert,
)
from torch.testing._internal.common_utils import run_tests


class TestFullyShardMixedPrecisionTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    def _init_models_and_optims(
        self,
        reshard_after_forward: Union[bool, int],
        param_dtype: Optional[torch.dtype],
        reduce_dtype: Optional[torch.dtype],
    ):
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).cuda()
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        return ref_model, ref_optim, model, optim

    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_compute_dtype(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
            },
            self._test_compute_dtype,
        )

    def _test_compute_dtype(self, reshard_after_forward: Union[bool, int]):
        param_dtype = torch.bfloat16
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward, param_dtype=param_dtype, reduce_dtype=None
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, param_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device="cuda", dtype=param_dtype)
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model_bf16(inp.to(param_dtype)).sum()
            ref_loss.backward()
            for param in ref_model_bf16.parameters():
                dist.all_reduce(param.grad)  # bf16 reduction
                param.grad.div_(self.world_size)
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad.to(param_fp32.dtype)
                param_bf16.grad = None
            ref_optim.step()  # fp32 optimizer step
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)

            self.assertEqual(fsdp_loss, ref_loss)
            check_1d_sharded_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_reduce_dtype(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
            },
            self._test_reduce_dtype_fp32_reduce,
        )

    def _test_reduce_dtype_fp32_reduce(self, reshard_after_forward: Union[bool, int]):
        param_dtype, reduce_dtype = torch.bfloat16, torch.float32
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward, param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device="cuda", dtype=param_dtype)
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model_bf16(inp.to(param_dtype)).sum()
            ref_loss.backward()
            for param in ref_model_bf16.parameters():
                param.grad.data = param.grad.to(torch.float32)
                dist.all_reduce(param.grad)  # fp32 reduction
                param.grad.div_(self.world_size)
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_fp32.grad = param_bf16.grad
                param_bf16.grad = None
            ref_optim.step()  # fp32 optimizer step
            for param_fp32, param_bf16 in zip(
                ref_model.parameters(), ref_model_bf16.parameters()
            ):
                param_bf16.detach().copy_(param_fp32)

            self.assertEqual(fsdp_loss, ref_loss)
            check_1d_sharded_parity(self, ref_model, model)

    def _test_reduce_dtype_bf16_reduce(self, reshard_after_forward: Union[bool, int]):
        param_dtype, reduce_dtype = torch.float32, torch.bfloat16
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward, param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device="cuda", dtype=param_dtype)
        for iter_idx in range(10):
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            fsdp_loss = model(inp).sum()
            with patch_reduce_scatter(reduce_scatter):
                fsdp_loss.backward()
            optim.step()

            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            ref_loss = ref_model(inp).sum()
            ref_loss.backward()
            for param in ref_model.parameters():
                param.grad.data = param.grad.to(reduce_dtype)
                dist.all_reduce(param.grad)  # bf16 reduction
                param.grad.div_(self.world_size)
                param.grad = param.grad.to(param.dtype)  # upcast to fp32
            ref_optim.step()  # fp32 optimizer step

            self.assertEqual(fsdp_loss, ref_loss)
            check_1d_sharded_parity(self, ref_model, model)


class TestFullyShardMixedPrecisionCasts(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_float16_on_one_submodule(self):
        x = torch.zeros(2, 100, device="cuda")

        # Subtest 1: use fp16 on the second child submodule -- does not require
        # any additional casting logic
        forward_inputs: Dict[str, nn.Module] = {}
        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).cuda()
        fully_shard(model.c2, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        fully_shard(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float16)

        # Subtest 2: use fp16 on the second child module, where the user module
        # owns the cast
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=True
        ).cuda()
        fully_shard(
            model.c2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=False
            ),
        )
        fully_shard(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

        # Subtest 3: use fp16 on the first child module and specify its output
        # dtype so that the second child module does not need to cast
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        fully_shard(
            model.c1,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, output_dtype=torch.float32
            ),
        )
        fully_shard(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float16)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

    @skip_if_lt_x_gpu(2)
    def test_submodules_with_external_inputs(self):
        self.run_subtests(
            {"enable_submodule_cast": [False, True]},
            self._test_submodules_with_external_inputs,
        )

    def _test_submodules_with_external_inputs(self, enable_submodule_cast: bool):
        class ToyModule(nn.Module):
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                self.l = nn.Linear(100, 100)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                self.forward_inputs["l2_input_x"] = x
                self.forward_inputs["l2_input_y"] = y
                return self.l(x)

        class ToyModel(nn.Module):
            def __init__(self, forward_inputs: Dict[str, torch.Tensor]) -> None:
                super().__init__()
                self.l1 = nn.Linear(100, 100)
                self.l2 = ToyModule(forward_inputs)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.forward_inputs["model_input_x"] = x
                y = torch.ones(
                    2, 100, device="cuda", dtype=torch.float32
                )  # external input
                return self.l2(self.l1(x), y)

        forward_inputs: Dict[str, torch.Tensor] = {}
        model = ToyModel(forward_inputs).cuda()
        x = torch.zeros(2, 100, device="cuda", dtype=torch.float32)
        fully_shard(
            model.l2,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.float16, cast_forward_inputs=enable_submodule_cast
            ),
        )
        fully_shard(model, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        model(x).sum().backward()

        # If we enable `model.l2` to cast (as default), then `l2_input_y` gets
        # cast to fp16, and if we disable, then it says as fp32.
        self.assertEqual(forward_inputs["model_input_x"].dtype, torch.float16)
        self.assertEqual(forward_inputs["l2_input_x"].dtype, torch.float16)
        self.assertEqual(
            forward_inputs["l2_input_y"].dtype,
            torch.float16 if enable_submodule_cast else torch.float32,
        )


if __name__ == "__main__":
    run_tests()
