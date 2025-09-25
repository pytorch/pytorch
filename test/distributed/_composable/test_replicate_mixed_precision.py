# Owner(s): ["oncall: distributed"]

import copy
import functools
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    _get_gradient_divide_factors,
)
from torch.distributed.tensor import Shard
from torch.testing._internal.common_distributed import (
    requires_nccl_version,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    FSDPTest,
    get_devtype,
    MLP,
    patch_reduce_scatter,
    reduce_scatter_with_assert,
)
from torch.testing._internal.common_utils import run_tests, skipIfRocmVersionLessThan


device_type = torch.device(get_devtype())


class TestReplicateMixedPrecisionTraining(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(2, torch.get_device_module(device_type).device_count())

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
        replicate_fn = functools.partial(
            replicate,
            reshard_after_forward=reshard_after_forward,
            mp_policy=mp_policy,
            shard_placement_fn=shard_placement_fn,
        )
        for mlp in model:
            replicate_fn(mlp)
        replicate_fn(model)
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

    @skipIfRocmVersionLessThan((7, 0))
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_compute_dtype(self):
        use_shard_placement_fn_vals = (
            self._get_use_shard_placement_fn_vals_for_bf16_reduce()
        )
        self.run_subtests(
            {
                "param_dtype": [torch.bfloat16, torch.float16],
                "reshard_after_forward": [False, True],
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

    @skipIfRocmVersionLessThan((7, 0))
    @skip_if_lt_x_gpu(2)
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for bf16 collectives")
    def test_reduce_dtype(self):
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
                "use_shard_placement_fn": [False, True],
            },
            self._test_reduce_dtype_fp32_reduce,
        )
        use_shard_placement_fn_vals = (
            self._get_use_shard_placement_fn_vals_for_bf16_reduce()
        )
        self.run_subtests(
            {
                "reshard_after_forward": [False, True],
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
            replicate(
                mlp, reshard_after_forward=reshard_after_forward, mp_policy=mp_policy
            )
        replicate(
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


if __name__ == "__main__":
    run_tests()
