# Owner(s): ["oncall: distributed"]

import contextlib
import copy
import functools

from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
from _test_fully_shard_common import (
    check_sharded_grad_parity,
    MLP,
    patch_reduce_scatter,
    reduce_scatter_with_dtype_assert,
)

from torch.distributed._composable.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
    register_forward_cast_hooks,
)
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.testing._internal.common_distributed import (
    requires_nccl_version,
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShardMixedPrecision(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_compute_dtype(self):
        """
        Tests train parity against existing flat-parameter FSDP when using
        compute mixed precision.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "offload_policy": [OffloadPolicy(), OffloadPolicy("cpu")],
            },
            self._test_compute_dtype,
        )

    def _test_compute_dtype(
        self,
        reshard_after_forward: bool,
        offload_policy: OffloadPolicy,
    ):
        torch.manual_seed(42)
        param_dtype = torch.bfloat16
        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype)
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        ref_model = FSDP(
            copy.deepcopy(model).cuda(),
            mixed_precision=MixedPrecision(param_dtype=param_dtype),
            cpu_offload=CPUOffload(
                offload_params=(offload_policy.offload_type == "cpu")
            ),
        )
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        device = torch.device("cuda")
        for iter_idx in range(10):
            inp = torch.randn((4, 16), device=device, dtype=param_dtype)
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                self.assertEqual(losses[-1].dtype, param_dtype)
                losses[-1].backward()
                _optim.step()
            self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_reduce_dtype(self):
        """
        Tests train parity against existing flat-parameter FSDP when using
        reduce mixed precision.
        """
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "offload_policy": [OffloadPolicy(), OffloadPolicy("cpu")],
                "use_fp32_reduce": [False, True],
            },
            self._test_reduce_dtype,
        )

    def _test_reduce_dtype(
        self,
        reshard_after_forward: bool,
        offload_policy: OffloadPolicy,
        use_fp32_reduce: bool,
    ):
        torch.manual_seed(42)
        if use_fp32_reduce:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
        else:
            param_dtype = torch.float32
            reduce_dtype = torch.bfloat16
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        ref_model = FSDP(
            copy.deepcopy(model).cuda(),
            mixed_precision=MixedPrecision(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype
            ),
            cpu_offload=CPUOffload(
                offload_params=(offload_policy.offload_type == "cpu")
            ),
        )
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        orig_reduce_scatter = dist.reduce_scatter_tensor
        reduce_scatter = functools.partial(
            reduce_scatter_with_dtype_assert, self, orig_reduce_scatter, reduce_dtype
        )

        torch.manual_seed(42 + self.rank + 1)
        device = torch.device("cuda")
        # Train on the same input to avoid loss explosion
        inp = torch.randn((4, 16), device=device, dtype=param_dtype)
        for iter_idx in range(10):
            losses: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
                losses.append(_model(inp).sum())
                self.assertEqual(losses[-1].dtype, param_dtype)
                ctx = (
                    patch_reduce_scatter(reduce_scatter)
                    if _model is model
                    else contextlib.nullcontext()
                )
                with ctx:
                    losses[-1].backward()
                if _model is model:
                    for param in model.parameters():
                        self.assertEqual(param.grad.dtype, torch.float32)
                _optim.step()
            if reduce_dtype == torch.bfloat16:
                # Reducing in bf16 requires more precision tolerance by the 3rd
                # iteration. Relaxing atol/rtol allows all iterations to pass.
                # Otherwise, we check for expected behavior with dtype checks.
                torch.testing.assert_close(losses[0], losses[1], atol=2e-5, rtol=5e-5)
            else:
                self.assertEqual(losses[0], losses[1])

    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_grad_acc_with_reduce_dtype(self):
        """
        Tests that gradient accumulation without reduce-scatter when using
        bf16 compute and fp32 reduction accumulates the unsharded gradients in
        fp32.
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False]},
            self._test_grad_acc_with_reduce_dtype,
        )

    def _test_grad_acc_with_reduce_dtype(self, reshard_after_forward: bool):
        torch.manual_seed(42)
        param_dtype, reduce_dtype = (torch.bfloat16, torch.float32)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        # To emulate the mixed precision implementation where forward/backward
        # compute use bf16 and optimizer uses fp32, we maintain both an fp32
        # and a bf16 copy of the reference model
        ref_model = copy.deepcopy(model).cuda()
        ref_model_compute = copy.deepcopy(ref_model).to(param_dtype)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        for mlp in model:
            fully_shard(
                mlp, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy
            )
        fully_shard(
            model, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(42 + self.rank + 1)
        device = torch.device("cuda")
        # Train on the same input to avoid loss explosion
        num_microbatches = 4
        inp = torch.randn((2 * num_microbatches, 16), device=device, dtype=param_dtype)
        for iter_idx in range(10):
            microbatch_inps = torch.chunk(inp, 4)
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                model.set_requires_gradient_sync(is_last_microbatch)
                losses: List[torch.Tensor] = []
                for _model in (ref_model_compute, model):
                    losses.append(
                        _model(microbatch_inps[microbatch_idx].detach()).sum()
                    )
                    self.assertEqual(losses[-1].dtype, param_dtype)
                    losses[-1].backward()
                self.assertEqual(losses[0], losses[1])
                # Manually accumulate gradients into the base reference model
                # from the compute reference model in fp32
                for ref_param, ref_param_compute in zip(
                    ref_model.parameters(), ref_model_compute.parameters()
                ):
                    self.assertTrue(ref_param_compute.grad is not None)
                    self.assertEqual(ref_param.dtype, torch.float32)
                    if ref_param.grad is not None:
                        ref_param.grad += ref_param_compute.grad
                    else:
                        ref_param.grad = ref_param_compute.grad.to(ref_param.dtype)
                    ref_param_compute.grad = None
                # Manually reduce gradients for the reference model on the last
                # microbatch to implement data parallelism
                if is_last_microbatch:
                    for ref_param in ref_model.parameters():
                        self.assertTrue(ref_param.grad is not None)
                        dist.all_reduce(ref_param.grad)
                        ref_param.grad /= self.world_size
            check_sharded_grad_parity(self, ref_model, model)
            ref_optim.step()
            optim.step()
            ref_optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            optim.zero_grad(set_to_none=(iter_idx % 2 == 0))
            # Manually copy parameters from the base reference model to the
            # compute reference model to run the optimizer step for the latter
            for ref_param, ref_param_compute in zip(
                ref_model.parameters(), ref_model_compute.parameters()
            ):
                ref_param_compute.detach().copy_(ref_param)

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

        # Subtest 3: use fp16 on the first child module, where the user must
        # additionally register hooks on the second child module
        forward_inputs: Dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).cuda()
        fully_shard(model.c1, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        fully_shard(model)
        register_forward_cast_hooks(
            model.c2, input_dtype=torch.float32, output_dtype=torch.float32
        )  # explicitly register cast for second child module
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float16)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float32)

        # Subtest 4: use fp16 on the first child module and specify its output
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
