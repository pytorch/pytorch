# Owner(s): ["oncall: distributed"]

import copy
import dataclasses
import functools
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _get_gradient_divide_factors,
)
from torch.distributed.tensor import Shard
from torch.testing._internal.common_distributed import (
    requires_nccl_version,
    SaveForwardInputsModel,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    FSDPTestMultiThread,
    get_devtype,
    MLP,
    patch_reduce_scatter,
    reduce_scatter_with_assert,
)
from torch.testing._internal.common_utils import run_tests, skipIfRocm, TEST_HPU


device_type = torch.device(get_devtype())


class TestFullyShardMixedPrecisionTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.get_device_module(device_type).device_count())

    def _init_models_and_optims(
        self,
        reshard_after_forward: Union[bool, int],
        param_dtype: Optional[torch.dtype],
        reduce_dtype: Optional[torch.dtype],
        use_shard_placement_fn,
    ):
        torch.manual_seed(42)
        model = nn.Sequential(*[MLP(16, torch.device("cpu")) for _ in range(3)])
        ref_model = copy.deepcopy(model).to(device_type)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)

        def _shard_placement_fn(param: nn.Parameter) -> Optional[Shard]:
            largest_dim = -1
            largest_dim_size = -1
            for dim, dim_size in enumerate(param.shape):
                if dim_size > largest_dim_size:
                    largest_dim = dim
                    largest_dim_size = dim_size
            assert largest_dim >= 0, f"{param.shape}"
            return Shard(largest_dim)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        shard_placement_fn = _shard_placement_fn if use_shard_placement_fn else None
        fully_shard_fn = functools.partial(
            fully_shard,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            shard_placement_fn=shard_placement_fn,
        )
        for mlp in model:
            fully_shard_fn(mlp)
        fully_shard_fn(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
        return ref_model, ref_optim, model, optim

    def _get_use_shard_placement_fn_vals_for_bf16_reduce(self):
        use_shard_placement_fn_vals = [False]
        if self.world_size == 2:
            # For world size >2, gradient elements get reduced in different
            # orders for the baseline vs. dim-1 sharding, leading to numeric
            # differences for bf16 reduction, so only test world size 2.
            use_shard_placement_fn_vals.append(True)
        return use_shard_placement_fn_vals

    @skipIfRocm  # regressed in ROCm 6.4, but ROCm 6.5 fixes it
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_compute_dtype(self):
        use_shard_placement_fn_vals = (
            self._get_use_shard_placement_fn_vals_for_bf16_reduce()
        )
        self.run_subtests(
            {
                "param_dtype": [torch.bfloat16, torch.float16],
                "reshard_after_forward": [False, True, 2],
                "use_shard_placement_fn": use_shard_placement_fn_vals,
            },
            self._test_compute_dtype,
        )

    def _test_compute_dtype(
        self,
        param_dtype: torch.dtype,
        reshard_after_forward: Union[bool, int],
        use_shard_placement_fn: bool,
    ):
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward,
            param_dtype=param_dtype,
            reduce_dtype=None,
            use_shard_placement_fn=use_shard_placement_fn,
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, param_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        predivide_factor, postdivide_factor, _, _ = _get_gradient_divide_factors(
            self.process_group, all_reduce_group=None, reduce_dtype=param_dtype
        )

        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device=device_type.type, dtype=param_dtype)
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
                # Use reduce-scatter -> all-gather as all-reduce because for
                # world size >=4, NCCL all-reduce shows numeric differences
                # compared with NCCL reduce-scatter
                if predivide_factor is not None and predivide_factor > 1:
                    param.grad.div_(predivide_factor)
                elif predivide_factor is None:
                    param.grad.div_(self.world_size)
                output = torch.zeros_like(torch.chunk(param.grad, self.world_size)[0])
                dist.reduce_scatter_tensor(output, param.grad)
                dist.all_gather_into_tensor(param.grad, output)
                if postdivide_factor is not None and postdivide_factor > 1:
                    param.grad.div_(postdivide_factor)
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
            check_sharded_parity(self, ref_model, model)

    @skipIfRocm  # regressed in ROCm 6.4, but ROCm 6.5 fixes it
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_reduce_dtype(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "use_shard_placement_fn": [False, True],
            },
            self._test_reduce_dtype_fp32_reduce,
        )
        use_shard_placement_fn_vals = (
            self._get_use_shard_placement_fn_vals_for_bf16_reduce()
        )
        self.run_subtests(
            {
                "reshard_after_forward": [False, True, 2],
                "use_shard_placement_fn": use_shard_placement_fn_vals,
            },
            self._test_reduce_dtype_bf16_reduce,
        )

    def _test_reduce_dtype_fp32_reduce(
        self, reshard_after_forward: Union[bool, int], use_shard_placement_fn: bool
    ):
        if (
            self.world_size > 2
            and isinstance(reshard_after_forward, int)
            and use_shard_placement_fn
        ):
            return
        param_dtype, reduce_dtype = torch.bfloat16, torch.float32
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            use_shard_placement_fn=use_shard_placement_fn,
        )
        ref_model_bf16 = copy.deepcopy(ref_model).to(param_dtype)
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device=device_type.type, dtype=param_dtype)
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
            check_sharded_parity(self, ref_model, model)

    def _test_reduce_dtype_bf16_reduce(
        self, reshard_after_forward: Union[bool, int], use_shard_placement_fn: bool
    ):
        param_dtype, reduce_dtype = torch.float32, torch.bfloat16
        ref_model, ref_optim, model, optim = self._init_models_and_optims(
            reshard_after_forward,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            use_shard_placement_fn=use_shard_placement_fn,
        )
        group = dist.distributed_c10d._get_default_group()
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        inp = torch.randn((4, 16), device=device_type.type, dtype=param_dtype)
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
                param_grad = param.grad.to(reduce_dtype)
                # Use reduce-scatter -> all-gather to implement all-reduce
                # since for world size >2, bf16 all-reduce and reduce-scatter
                # have numeric differences
                sharded_grad = funcol.reduce_scatter_tensor(
                    param_grad, scatter_dim=0, reduceOp="avg", group=group
                )  # bf16 reduction
                param.grad = funcol.all_gather_tensor(
                    sharded_grad, gather_dim=0, group=group
                ).to(param.dtype)  # upcast to fp32
            ref_optim.step()  # fp32 optimizer step

            self.assertEqual(fsdp_loss, ref_loss)
            check_sharded_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(2)
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
        ref_model = copy.deepcopy(model).to(device_type)
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
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, reduce_dtype)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        torch.manual_seed(42 + self.rank + 1)
        device = device_type
        # Train on the same input to avoid loss explosion
        num_microbatches = 4
        inp = torch.randn((2 * num_microbatches, 16), device=device, dtype=param_dtype)
        for iter_idx in range(10):
            microbatch_inps = torch.chunk(inp, 4)
            for microbatch_idx in range(num_microbatches):
                is_last_microbatch = microbatch_idx == num_microbatches - 1
                model.set_requires_gradient_sync(is_last_microbatch)
                model.set_reshard_after_backward(
                    is_last_microbatch or reshard_after_forward
                )
                losses: list[torch.Tensor] = []
                for _model in (ref_model_compute, model):
                    losses.append(
                        _model(microbatch_inps[microbatch_idx].detach()).sum()
                    )
                    self.assertEqual(losses[-1].dtype, param_dtype)
                    with patch_reduce_scatter(reduce_scatter):
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
            check_sharded_parity(self, ref_model, model)
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


class TestFullyShardMixedPrecisionCasts(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(1)
    def test_float16_on_one_submodule(self):
        x = torch.zeros(2, 100, device=device_type)

        # Subtest 1: use fp16 on the second child submodule -- does not require
        # any additional casting logic
        forward_inputs: dict[str, nn.Module] = {}
        model = SaveForwardInputsModel(
            forward_inputs,
            cast_forward_inputs=False,
        ).to(device_type)
        fully_shard(model.c2, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float16))
        fully_shard(model)
        model(x).sum().backward()
        self.assertEqual(forward_inputs[model].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c1].dtype, torch.float32)
        self.assertEqual(forward_inputs[model.c2].dtype, torch.float16)

        # Subtest 2: use fp16 on the second child module, where the user module
        # owns the cast
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=True
        ).to(device_type)
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
        forward_inputs: dict[nn.Module, torch.Tensor] = {}
        model = SaveForwardInputsModel(
            forward_inputs=forward_inputs, cast_forward_inputs=False
        ).to(device_type)
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

    @skip_if_lt_x_gpu(1)
    def test_submodules_with_external_inputs(self):
        self.run_subtests(
            {"enable_submodule_cast": [False, True]},
            self._test_submodules_with_external_inputs,
        )

    def _test_submodules_with_external_inputs(self, enable_submodule_cast: bool):
        class ToyModule(nn.Module):
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                super().__init__()
                self.l = nn.Linear(100, 100)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                self.forward_inputs["l2_input_x"] = x
                self.forward_inputs["l2_input_y"] = y
                return self.l(x)

        class ToyModel(nn.Module):
            def __init__(self, forward_inputs: dict[str, torch.Tensor]) -> None:
                super().__init__()
                self.l1 = nn.Linear(100, 100)
                self.l2 = ToyModule(forward_inputs)
                self.forward_inputs = forward_inputs

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.forward_inputs["model_input_x"] = x
                y = torch.ones(
                    2, 100, device=device_type.type, dtype=torch.float32
                )  # external input
                return self.l2(self.l1(x), y)

        forward_inputs: dict[str, torch.Tensor] = {}
        model = ToyModel(forward_inputs).to(device_type)
        x = torch.zeros(2, 100, device=device_type.type, dtype=torch.float32)
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

    @skip_if_lt_x_gpu(1)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_norm_modules_bf16(self):
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        self._test_norm_modules(mp_policy)

    @skip_if_lt_x_gpu(1)
    def test_norm_modules_fp16(self):
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.float16)
        self._test_norm_modules(mp_policy)

    def _test_norm_modules(self, mp_policy: MixedPrecisionPolicy):
        def inner(model: nn.Module, x: torch.Tensor):
            # Run forward and backward to check for no type mismatch errors
            z = model(x)
            self.assertEqual(z.dtype, mp_policy.param_dtype)
            z.sum().backward()

        # Layer norm
        model = nn.Sequential(nn.Linear(32, 32), nn.LayerNorm(32), nn.Linear(32, 32))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        inner(model, torch.randn((4, 32)))

        # Batch norm 1D
        model = nn.Sequential(nn.Linear(32, 32), nn.BatchNorm1d(32), nn.Linear(32, 32))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        inner(model, torch.randn((4, 32)))

        # Batch norm 2D: error in backward from buffer dtype mismatch
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        if TEST_HPU:
            inner(model, torch.randn((3, 1, 9, 9)))
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected running_mean to have type",  # Error not seen on HPUs and hence it can be skipped
            ):
                # Errors in batch norm 2D backward
                inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: cast buffers down to lower precision
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        for module in (model[0], model[1], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        # Casting batch norm buffers to the lower precision allows backward
        model[1].running_mean = model[1].running_mean.to(mp_policy.param_dtype)
        model[1].running_var = model[1].running_var.to(mp_policy.param_dtype)
        inner(model, torch.randn((3, 1, 9, 9)))

        # Batch norm 2D: use special mixed precision policy
        model = nn.Sequential(nn.Conv2d(1, 5, 3), nn.BatchNorm2d(5), nn.Conv2d(5, 4, 3))
        bn_mp_policy = MixedPrecisionPolicy(output_dtype=mp_policy.param_dtype)
        fully_shard(model[1], mp_policy=bn_mp_policy)
        for module in (model[0], model[2], model):
            fully_shard(module, mp_policy=mp_policy)
        inner(model, torch.randn((3, 1, 9, 9)))

    @skip_if_lt_x_gpu(1)
    def test_clamp_reduce_dtype(self):
        # Initialize the model directly in bf16
        init_dtype = torch.bfloat16
        model = nn.Sequential(
            nn.Linear(32, 32, dtype=init_dtype),
            nn.Linear(32, 32, dtype=init_dtype),
        ).to(device_type.type)
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
        )
        # Check that we did not clamp the reduce dtype
        self.assertEqual(mp_policy.reduce_dtype, torch.bfloat16)
        for module in model:
            fully_shard((module), mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)

        # Check that the reduce-scatter runs in bf16 even after we change the
        # model from bf16 to fp32
        model.to(torch.float32)
        orig_reduce_scatter = dist.reduce_scatter_tensor

        def assert_fn(output: torch.Tensor):
            self.assertEqual(output.dtype, torch.bfloat16)

        reduce_scatter = functools.partial(
            reduce_scatter_with_assert, self, orig_reduce_scatter, assert_fn
        )
        with patch_reduce_scatter(reduce_scatter):
            inp = torch.randn((4, 32), device=device_type.type)
            loss = model(inp).sum()
            loss.backward()

    @skip_if_lt_x_gpu(1)
    def test_dataclass_input(self):
        @dataclasses.dataclass
        class Input:
            x: torch.Tensor

        class Model(nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self._layer = nn.Linear(10, 10)

            def forward(self, input: Input):
                return self._layer(input.x)

        mp_policy = MixedPrecisionPolicy(
            torch.bfloat16, torch.bfloat16, torch.bfloat16, True
        )
        model = Model()
        inp = Input(torch.randn(2, 10).cuda())

        fully_shard(model, mp_policy=mp_policy)
        loss = model(inp).sum()
        loss.backward()


if __name__ == "__main__":
    run_tests()
