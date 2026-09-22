# Owner(s): ["oncall: distributed"]

import copy
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    DataParallelMeshDims,
    fully_shard,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp._fully_shard._fsdp_collectives import DefaultReduceScatter
from torch.distributed.fsdp._fully_shard._fsdp_grad import FSDPGrad
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import parametrize, run_tests
from torch.utils._python_dispatch import TorchDispatchMode


if dist._is_spmd_types_available():
    import spmd_types as spmd
    from spmd_types.checker import typecheck


class TrackPendingGradCopies(TorchDispatchMode):
    def __init__(self, parameters):
        super().__init__()
        self.grads = []
        for param in parameters:
            grad = param.grad
            if isinstance(grad, FSDPGrad):
                components = (grad.reduced, grad.unreduced, grad.partial)
            else:
                components = (grad,)
            for component in components:
                if component is not None:
                    self.grads.append(
                        component._local_tensor
                        if isinstance(component, DTensor)
                        else component
                    )
        self.storages = {g.untyped_storage().data_ptr() for g in self.grads}
        self.h2d = 0
        self.d2h = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func == torch.ops.aten._to_copy.default:
            tensor = args[0]._local_tensor if isinstance(args[0], DTensor) else args[0]
            device = kwargs.get("device")
            if device is not None:
                target = torch.device(device).type
                if tensor.device.type == "cpu" and target != "cpu":
                    if tensor.untyped_storage().data_ptr() in self.storages:
                        self.h2d += 1
                elif tensor.device.type != "cpu" and target == "cpu":
                    self.d2h += 1
        return func(*args, **kwargs)


class OffloadPendingGradModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.first = nn.Linear(4, 2, bias=False, device=device)
        self.second = nn.Linear(4, 2, bias=False, device=device)

    def forward(self, inp, use_second=True):
        output = self.first(inp)
        if use_second:
            output = output + self.second(inp)
        return output


class TestFullyShardPendingGrad(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @parametrize("cpu_offload", [False, True])
    @parametrize("read_pending", [False, True])
    def test_reduced_and_unreduced_keep_native_dtypes(
        self, device, cpu_offload, read_pending
    ):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model,
            mesh=init_device_mesh(torch.device(device).type, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        for sync, value in ((True, 256), (False, 1), (True, -256)):
            model.set_requires_gradient_sync(sync)
            model(
                torch.full((1, 1), value, device=device, dtype=torch.bfloat16)
            ).sum().backward()
            if not sync:
                grad = model.weight.grad
                self.assertIsInstance(grad, FSDPGrad)
                self.assertEqual(grad.dtype, torch.float32)
                self.assertEqual(grad.reduced.dtype, torch.float32)
                self.assertEqual(grad.unreduced.dtype, torch.bfloat16)
                self.assertEqual(
                    grad.reduced.to_local(),
                    torch.full_like(grad.reduced.to_local(), 256),
                )
                self.assertEqual(
                    grad.unreduced.to_local(),
                    torch.ones_like(grad.unreduced.to_local()),
                )
                if read_pending:
                    for _ in range(2):
                        snapshot = grad.to(device).full_tensor()
                        self.assertEqual(snapshot, torch.full_like(snapshot, 257))
                    self.assertEqual(grad.unreduced.dtype, torch.bfloat16)
        actual = model.weight.grad.to(device).full_tensor()
        self.assertEqual(actual, torch.ones_like(actual))

        model.zero_grad(set_to_none=True)
        for sync, value in ((True, 3), (False, 256), (False, 1), (True, -256)):
            model.set_requires_gradient_sync(sync)
            model(
                torch.full((1, 1), value, device=device, dtype=torch.bfloat16)
            ).sum().backward()
            if not sync:
                self.assertEqual(model.weight.grad.unreduced.dtype, torch.bfloat16)
                if read_pending:
                    model.weight.grad.to(device).full_tensor()
        # The pending 256 + 1 must round in BF16 before the final -256.
        actual = model.weight.grad.to(device).full_tensor()
        self.assertEqual(actual, torch.full_like(actual, 3))

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "consumer", ["zero", "zero_none", "scale", "clip", "clip_foreach", "unscale"]
    )
    def test_mixed_component_consumers(self, device, consumer):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model,
            mesh=init_device_mesh(torch.device(device).type, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(torch.device(device).type, init_scale=8.0)
        for sync, value in ((True, 256), (False, 1)):
            model.set_requires_gradient_sync(sync)
            loss = model(
                torch.full((1, 1), value, device=device, dtype=torch.bfloat16)
            ).sum()
            if consumer == "unscale":
                loss = scaler.scale(loss)
            loss.backward()
        reduced = torch.tensor(
            2048 if consumer == "unscale" else 256, device=device, dtype=torch.float32
        )
        unreduced = torch.tensor(
            8 if consumer == "unscale" else 1, device=device, dtype=torch.bfloat16
        )
        if consumer in ("zero", "zero_none"):
            model.zero_grad(set_to_none=consumer == "zero_none")
            reduced.zero_()
            unreduced.zero_()
        elif consumer == "scale":
            torch._foreach_mul_([model.weight.grad], 0.5)
            reduced.mul_(0.5)
            unreduced.mul_(0.5)
        elif consumer.startswith("clip"):
            expected_norm = torch.full((2, 1), 257.0, device=device).norm()
            actual_norm = nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=consumer == "clip_foreach"
            )
            if isinstance(actual_norm, DTensor):
                actual_norm = actual_norm.full_tensor()
            self.assertEqual(actual_norm, expected_norm)
            factor = (1.0 / (expected_norm + 1e-6)).clamp(max=1.0)
            reduced.mul_(factor)
            unreduced.mul_(factor)
        else:
            scaler.unscale_(optimizer)
            reduced.div_(8)
            unreduced.div_(8)
            for found_inf in scaler._found_inf_per_device(optimizer).values():
                self.assertEqual(found_inf.item(), 0)
        if consumer == "zero_none":
            self.assertIsNone(model.weight.grad)
        else:
            grad = model.weight.grad
            self.assertIsInstance(grad, FSDPGrad)
            self.assertEqual(grad.reduced.dtype, torch.float32)
            self.assertEqual(grad.unreduced.dtype, torch.bfloat16)
            self.assertEqual(
                grad.reduced.to_local(), reduced.expand_as(grad.reduced.to_local())
            )
            self.assertEqual(
                grad.unreduced.to_local(),
                unreduced.expand_as(grad.unreduced.to_local()),
            )
        model.synchronize_gradients()
        if consumer == "zero_none":
            self.assertIsNone(model.weight.grad)
        else:
            self.assertIsInstance(model.weight.grad, DTensor)
            actual = model.weight.grad.full_tensor()
            self.assertEqual(actual, (reduced + unreduced.float()).expand_as(actual))

    @skip_if_lt_x_gpu(2)
    def test_custom_divide_factor_complete_value(self, device):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model,
            mesh=init_device_mesh(torch.device(device).type, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        model.set_gradient_divide_factor(3)
        for sync in (True, False):
            model.set_requires_gradient_sync(sync)
            model(
                torch.full((1, 1), 3.0, device=device, dtype=torch.bfloat16)
            ).sum().backward()
        self.assertIsInstance(model.weight.grad, FSDPGrad)
        before = model.weight.grad.full_tensor()
        self.assertEqual(before, torch.full_like(before, 4))
        model.synchronize_gradients()
        self.assertIsInstance(model.weight.grad, DTensor)
        self.assertEqual(model.weight.grad.full_tensor(), before)
        model.synchronize_gradients()
        self.assertEqual(model.weight.grad.full_tensor(), before)

    @skip_if_lt_x_gpu(2)
    @parametrize("existing_reduced", [False, True])
    def test_none_grad_dtype_uses_actual_reduction_dtype(
        self, device, existing_reduced
    ):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        model.weight.grad_dtype = None
        fully_shard(
            model,
            mesh=init_device_mesh(device, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        if existing_reduced:
            model.weight.grad = torch.ones_like(model.weight)
        model.set_requires_gradient_sync(False)
        model(
            torch.full((1, 1), 2.0, device=device, dtype=torch.bfloat16)
        ).sum().backward()
        grad = model.weight.grad
        self.assertIsInstance(grad, FSDPGrad)
        expected_dtype = torch.float32 if existing_reduced else torch.bfloat16
        self.assertEqual(grad.dtype, expected_dtype)
        self.assertEqual(grad.unreduced.dtype, torch.bfloat16)
        before = grad.full_tensor()
        self.assertEqual(before.dtype, expected_dtype)
        self.assertEqual(before, torch.full_like(before, 3 if existing_reduced else 2))
        model.synchronize_gradients()
        self.assertIsNone(model.weight.grad_dtype)
        self.assertIsInstance(model.weight.grad, DTensor)
        self.assertEqual(model.weight.grad.dtype, expected_dtype)
        self.assertEqual(model.weight.grad.full_tensor(), before)

    @skip_if_lt_x_gpu(2)
    @parametrize("boundary", ["backward", "unshard", "synchronize"])
    def test_grad_dtype_edit_after_forward_rejected(self, device, boundary):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model,
            mesh=init_device_mesh(device, (self.world_size,)),
            reshard_after_forward=False,
        )
        model.set_reshard_after_backward(False)
        model.set_requires_gradient_sync(False)
        output = model(torch.ones(1, 1, device=device))
        if boundary != "backward":
            output.sum().backward()
            self.assertIsInstance(model.weight.grad, FSDPGrad)
        param = model.weight
        grad = param.grad
        param.grad_dtype = None
        with self.assertRaisesRegex(RuntimeError, "grad_dtype.*fully_shard"):
            if boundary == "backward":
                output.sum().backward()
            elif boundary == "unshard":
                model.unshard()
            else:
                model.synchronize_gradients()
        self.assertIs(param.grad, grad)
        self.assertIsNone(param.grad_dtype)

    @skip_if_lt_x_gpu(2)
    def test_none_grad_dtype_preserved_with_pending_conversion(self, device):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        model.weight.grad_dtype = None
        fully_shard(model, mesh=init_device_mesh(device, (self.world_size,)))
        model.set_requires_gradient_sync(False)
        inp = torch.ones(1, 1, device=device)
        model(inp).sum().backward()
        self.assertIsInstance(model.weight.grad, FSDPGrad)
        model.to(device=device)
        self.assertIsNone(model.weight.grad_dtype)
        model.set_requires_gradient_sync(True)
        model(inp).sum().backward()
        self.assertIsNone(model.weight.grad_dtype)
        self.assertEqual(
            model.weight.grad.full_tensor(), torch.full((2, 1), 2.0, device=device)
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("callback", ["hook", "reduce_scatter", "packing"])
    def test_custom_hook_runs_only_on_synchronization(self, device, callback):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model, mesh=init_device_mesh(torch.device(device).type, (self.world_size,))
        )
        hook_calls = []

        def hook(output):
            hook_calls.append(1)
            output.mul_(2)

        class CountingReduceScatter(DefaultReduceScatter):
            def __call__(self, *args, **kwargs):
                hook_calls.append(1)
                return super().__call__(*args, **kwargs)

        if callback == "hook":
            model.set_all_reduce_hook(hook)
        elif callback == "reduce_scatter":
            model.set_custom_reduce_scatter(CountingReduceScatter())
        else:
            group = model._get_fsdp_state()._fsdp_param_groups[0]
            prepare_inputs = group._prepare_reduce_scatter_inputs

            def counting_prepare_inputs(*args, **kwargs):
                hook_calls.append(1)
                return prepare_inputs(*args, **kwargs)

            group._prepare_reduce_scatter_inputs = counting_prepare_inputs
        model.set_requires_gradient_sync(False)
        model(torch.ones(1, 1, device=device)).sum().backward()
        self.assertEqual(hook_calls, [])
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "synchronize_gradients"):
                model.weight.grad.full_tensor()
        self.assertEqual(hook_calls, [])
        model.synchronize_gradients()
        self.assertEqual(hook_calls, [1])
        actual = model.weight.grad.full_tensor()
        self.assertEqual(
            actual, torch.full_like(actual, 2 if callback == "hook" else 1)
        )
        model.synchronize_gradients()
        self.assertEqual(hook_calls, [1])

    @skip_if_lt_x_gpu(2)
    @parametrize("policy", ["factor", "sum", "hook", "reduce_scatter"])
    def test_pending_reduction_policy_is_frozen(self, device, policy):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model, mesh=init_device_mesh(torch.device(device).type, (self.world_size,))
        )
        model.set_gradient_divide_factor(3)
        model.set_requires_gradient_sync(False)
        model(torch.ones(1, 1, device=device)).sum().backward()
        group = model._get_fsdp_state()._fsdp_param_groups[0]
        original_comm = group._reduce_scatter_comm
        model.set_gradient_divide_factor(3)
        model.set_force_sum_reduction_for_comms(False)
        model.set_custom_reduce_scatter(original_comm)

        def change():
            if policy == "factor":
                model.set_gradient_divide_factor(2)
            elif policy == "sum":
                model.set_force_sum_reduction_for_comms(True)
            elif policy == "hook":
                model.set_all_reduce_hook(lambda output: output.mul_(2))
            else:
                model.set_custom_reduce_scatter(DefaultReduceScatter())

        with self.assertRaisesRegex(RuntimeError, "pending|synchronize_gradients"):
            change()
        self.assertEqual(group.gradient_divide_factor, 3)
        self.assertFalse(group.force_sum_reduction_for_comms)
        self.assertIsNone(group._all_reduce_hook)
        self.assertIs(group._reduce_scatter_comm, original_comm)
        model.synchronize_gradients()
        change()

    @skip_if_lt_x_gpu(2)
    @parametrize("convert", ["device", "dtype"])
    def test_pending_conversion_is_atomic(self, device, convert):
        device = torch.device(device).type
        model = nn.Linear(1, 2, bias=False, device=device)
        fully_shard(
            model,
            mesh=init_device_mesh(torch.device(device).type, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        model.set_requires_gradient_sync(False)
        model(torch.ones(1, 1, device=device, dtype=torch.bfloat16)).sum().backward()
        param = model.weight
        grad = param.grad
        unreduced = grad.unreduced
        with self.assertRaisesRegex(RuntimeError, "zero_grad|synchronize_gradients"):
            model.to(torch.device("cpu") if convert == "device" else torch.float32)
        self.assertIs(model.weight, param)
        self.assertIs(model.weight.grad, grad)
        self.assertIs(grad.unreduced, unreduced)
        self.assertEqual(param.device.type, device)
        self.assertEqual(param.dtype, torch.float32)
        self.assertEqual(unreduced.dtype, torch.bfloat16)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "mutation",
        ["none", "scale", "zero", "zero_none", "clone", "shard", "replicate", "unused"],
    )
    def test_cpu_offload_pending_grad(self, device, mutation):
        self.run_subtests(
            {"pin_memory": [False, True]},
            self._test_cpu_offload_pending_grad,
            torch.device(device).type,
            mutation,
        )

    @skip_if_lt_x_gpu(2)
    def test_cpu_offload_pending_grad_late_sync(self, device):
        self.run_subtests(
            {"pin_memory": [False, True]},
            self._test_cpu_offload_pending_grad,
            torch.device(device).type,
            "none",
            late_sync=True,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("set_to_none", [False, True])
    def test_cpu_offload_pending_grad_partial_group(self, device, set_to_none):
        self._test_cpu_offload_pending_grad(
            torch.device(device).type,
            "zero_none" if set_to_none else "zero",
            pin_memory=False,
            partial_group=True,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("set_to_none", [False, True])
    def test_partial_group_clear_after_forward(self, device, set_to_none):
        device = torch.device(device).type
        model = OffloadPendingGradModel(device)
        fully_shard(
            [model.first, model.second],
            mesh=init_device_mesh(device, (self.world_size,)),
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
        )
        fully_shard(model)
        model.set_reshard_after_backward(False)
        for sync, value in ((True, 4), (False, 2)):
            model.set_requires_gradient_sync(sync)
            model(
                torch.full((1, 4), value, device=device, dtype=torch.bfloat16)
            ).sum().backward()

        output = model.first(
            torch.ones(1, 4, device=device, dtype=torch.bfloat16, requires_grad=True)
        )
        with self.assertRaisesRegex(RuntimeError, "outside forward and backward"):
            model.synchronize_gradients()
        for param in model.parameters():
            self.assertIsInstance(param.grad, FSDPGrad)
            self.assertIsNotNone(param.grad.reduced)
            self.assertIsNotNone(param.grad.unreduced)
        model.zero_grad(set_to_none=set_to_none)
        output.sum().backward()
        model.synchronize_gradients()
        actual = model.first.weight.grad.full_tensor()
        self.assertEqual(actual, torch.ones_like(actual))
        if set_to_none:
            self.assertIsNone(model.second.weight.grad)
        else:
            actual = model.second.weight.grad.full_tensor()
            self.assertEqual(actual, torch.zeros_like(actual))

    def _test_cpu_offload_pending_grad(
        self, device, mutation, pin_memory, late_sync=False, partial_group=False
    ):
        model = OffloadPendingGradModel(device)
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.25)
        ref_model = copy.deepcopy(model).to(torch.bfloat16)
        for param in ref_model.parameters():
            param.grad_dtype = torch.float32
        mesh = init_device_mesh(device, (self.world_size,))
        fully_shard(
            [model.first, model.second] if partial_group else model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=pin_memory),
        )
        if partial_group:
            fully_shard(model)
        model.set_reshard_after_backward(pin_memory)
        state_module = model.first if partial_group else model
        fsdp_params = [
            param
            for group in state_module._get_fsdp_state()._fsdp_param_groups
            for param in group.fsdp_params
        ]
        for step in range(4):
            sync = step == 3
            before_forward_sync = not sync if late_sync and step >= 2 else sync
            model.set_requires_gradient_sync(before_forward_sync)
            inputs = [
                torch.arange(8, device=device, dtype=torch.bfloat16).view(2, 4) / 8
                + (rank + 1) / 2
                + step / 4
                for rank in range(self.world_size)
            ]
            use_second = mutation != "unused" or not sync
            if partial_group and step == 1:
                use_second = False
            ref_loss = sum(ref_model(inp, use_second).float().sum() for inp in inputs)
            ref_loss = ref_loss / self.world_size
            with TrackPendingGradCopies(model.parameters()) as copies:
                if partial_group and step == 1:
                    inp = inputs[self.rank].requires_grad_()
                    loss = model.first(inp).float().sum()
                else:
                    loss = model(inputs[self.rank], use_second).float().sum()
                model.set_requires_gradient_sync(sync)
                if step == 1 and mutation in ("zero", "zero_none"):
                    for param in model.parameters():
                        self.assertEqual(param.device.type, "cpu")
                        self.assertEqual(param.grad.device.type, "cpu")
                    for module in (model, ref_model):
                        module.zero_grad(set_to_none=mutation == "zero_none")
                ref_loss.backward()
                loss.backward()
            if step > 0 and not sync:
                if mutation not in ("shard", "replicate") or step > 1:
                    self.assertEqual(copies.h2d, 0)
            if sync:
                self.assertGreater(copies.h2d, 0)
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                self.assertEqual(param.device.type, "cpu")
                if ref_param.grad is None:
                    self.assertIsNone(param.grad)
                    continue
                self.assertEqual(param.grad.device.type, "cpu")
                self.assertEqual(param.grad.dtype, torch.float32)
                self.assertEqual(ref_param.grad, param.grad.to(device).full_tensor())
            if not sync:
                for param in fsdp_params:
                    self.assertIsNone(param._unsharded_param.grad)
            if step == 0:
                for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                    if mutation == "scale":
                        ref_param.grad.mul_(0.5)
                        param.grad.mul_(0.5)
                    elif mutation == "clone":
                        ref_param.grad = ref_param.grad.clone().mul_(2)
                        param.grad = param.grad.clone().mul_(2)
                    elif mutation in ("shard", "replicate"):
                        ref_param.grad = ref_param.grad.clone().mul_(2)
                        replacement = ref_param.grad.cpu()
                        placement = Replicate()
                        if mutation == "shard":
                            replacement = replacement.chunk(self.world_size)[
                                self.rank
                            ].clone()
                            placement = Shard(0)
                        param.grad = DTensor.from_local(
                            replacement, mesh, (placement,)
                        ).cpu()

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "mutation",
        ["none", "zero", "zero_none", "clone", "noncontiguous", "shard", "replicate"],
    )
    def test_unsharded_accumulated_grad(self, device, mutation):
        self.run_subtests(
            {"cpu_offload": [False, True]},
            self._test_unsharded_accumulated_grad,
            torch.device(device).type,
            mutation,
        )

    def _test_unsharded_accumulated_grad(self, device, mutation, cpu_offload):
        model = nn.Linear(4, 2, bias=False, device=device)
        ref_model = copy.deepcopy(model).to(torch.bfloat16)
        local_model = copy.deepcopy(ref_model)
        for module in (ref_model, local_model):
            module.weight.grad_dtype = torch.float32
        mesh = init_device_mesh(device, (self.world_size,))
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        model.set_reshard_after_backward(False)
        fsdp_param = model._get_fsdp_state()._fsdp_param_groups[0].fsdp_params[0]

        def check_getters(expected):
            leaf = getattr(fsdp_param, "_unsharded_param", None)
            pending = fsdp_param.sharded_param.grad
            owners = (
                model.weight,
                fsdp_param.sharded_param.grad,
                None if leaf is None else leaf.grad,
            )
            components = (
                (pending.reduced, pending.unreduced, pending.partial)
                if isinstance(pending, FSDPGrad)
                else ()
            )
            with (
                CommDebugMode() as comm,
                TrackPendingGradCopies(model.parameters()) as copies,
            ):
                grad = fsdp_param.unsharded_accumulated_grad
            self.assertEqual(comm.get_total_counts(), 0)
            self.assertEqual(copies.h2d, 0)
            self.assertEqual(copies.d2h, 0)
            for _ in range(2):
                with (
                    CommDebugMode() as comm,
                    TrackPendingGradCopies(model.parameters()) as copies,
                ):
                    if expected is None:
                        with self.assertRaisesRegex(
                            AssertionError,
                            "Expects unsharded_accumulated_grad to not be None",
                        ):
                            fsdp_param.unsharded_accumulated_grad_data
                    else:
                        data = fsdp_param.unsharded_accumulated_grad_data
                self.assertEqual(comm.get_total_counts(), 0)
                self.assertEqual(copies.h2d, int(cpu_offload and expected is not None))
                self.assertEqual(copies.d2h, 0)
            if isinstance(pending, FSDPGrad):
                for before, after in zip(
                    components, (pending.reduced, pending.unreduced, pending.partial)
                ):
                    self.assertIs(before, after)
            current = (
                model.weight,
                fsdp_param.sharded_param.grad,
                None if leaf is None else leaf.grad,
            )
            for before, after in zip(owners, current):
                self.assertIs(before, after)
            if expected is None:
                self.assertIsNone(grad)
                return
            self.assertNotIsInstance(data, DTensor)
            expected_device = "cpu" if cpu_offload else device
            self.assertEqual(data.device, fsdp_param.device)
            self.assertEqual(data, expected)
            self.assertNotIsInstance(grad, DTensor)
            self.assertEqual(grad.device.type, expected_device)
            self.assertEqual(grad.dtype, torch.float32)
            self.assertEqual(grad, expected.to(expected_device))
            owner = (
                pending.unreduced._local_tensor
                if isinstance(pending, FSDPGrad) and pending.unreduced is not None
                else leaf.grad
            )
            self.assertEqual(grad.data_ptr(), owner.data_ptr())
            if not cpu_offload:
                self.assertEqual(data.data_ptr(), owner.data_ptr())

        check_getters(None)
        for name in ("unsharded_accumulated_grad", "unsharded_accumulated_grad_data"):
            with self.assertRaises(AttributeError):
                setattr(fsdp_param, name, None)

        for step in range(3):
            model.set_requires_gradient_sync(step == 2)
            inputs = [
                torch.arange(8, device=device, dtype=torch.bfloat16).view(2, 4) / 8
                + (rank + 1) / 2
                + step / 4
                for rank in range(self.world_size)
            ]
            output = model(inputs[self.rank])
            check_getters(local_model.weight.grad)
            ref_loss = sum(ref_model(inp).float().sum() for inp in inputs)
            (ref_loss / self.world_size).backward()
            local_model(inputs[self.rank]).float().sum().backward()
            output.float().sum().backward()
            check_getters(None if step == 2 else local_model.weight.grad)
            self.assertEqual(
                ref_model.weight.grad, model.weight.grad.to(device).full_tensor()
            )
            if step != 0:
                continue
            if mutation in ("zero", "zero_none"):
                for module in (model, ref_model, local_model):
                    module.zero_grad(set_to_none=mutation == "zero_none")
            elif mutation in ("clone", "noncontiguous"):
                if mutation == "noncontiguous":
                    with self.assertRaisesRegex(RuntimeError, "synchronize_gradients"):
                        model.weight.grad.t()
                    model.synchronize_gradients()
                for module in (model, ref_model, local_model):
                    grad = module.weight.grad
                    grad = (
                        grad.clone()
                        if mutation == "clone"
                        else grad.t().contiguous().t()
                    )
                    module.weight.grad = grad.mul_(2)
                    if mutation == "noncontiguous":
                        self.assertFalse(module.weight.grad.is_contiguous())
                if mutation == "noncontiguous":
                    local_model.weight.grad = None
            elif mutation in ("shard", "replicate"):
                ref_model.weight.grad.mul_(2)
                local_model.weight.grad = None
                grad = ref_model.weight.grad.clone()
                placement = Replicate()
                if mutation == "shard":
                    grad = grad.chunk(self.world_size)[self.rank].clone()
                    placement = Shard(0)
                model.weight.grad = DTensor.from_local(grad, mesh, (placement,)).to(
                    "cpu" if cpu_offload else device
                )
            check_getters(local_model.weight.grad)

    @skip_if_lt_x_gpu(2)
    @parametrize("cpu_offload", [False, True])
    def test_unsharded_accumulated_grad_dtensor(self, device, cpu_offload):
        device = torch.device(device).type
        mesh = init_device_mesh(
            device, (1, self.world_size), mesh_dim_names=("dp", "tp")
        )
        ref_model = nn.Linear(4, 4, bias=False, device=device)
        model = parallelize_module(
            copy.deepcopy(ref_model), mesh["tp"], ColwiseParallel()
        )
        fully_shard(
            model,
            mesh=mesh["dp"],
            reshard_after_forward=False,
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        model.set_reshard_after_backward(False)
        model.set_requires_gradient_sync(False)
        fsdp_param = model._get_fsdp_state()._fsdp_param_groups[0].fsdp_params[0]
        for step in range(2):
            inp = torch.ones(2, 4, device=device) * (step + 1)
            output = model(inp)
            if step:
                self.assertIsNone(fsdp_param.unsharded_param.grad)
                accumulated = fsdp_param.unsharded_accumulated_grad
                self.assertIsInstance(accumulated, DTensor)
                self.assertIsInstance(model.weight.grad, FSDPGrad)
                self.assertEqual(
                    accumulated._local_tensor.data_ptr(),
                    model.weight.grad.unreduced._local_tensor.data_ptr(),
                )
            ref_model(inp).sum().backward()
            output.sum().backward()
            pending = model.weight.grad
            self.assertIsInstance(pending, FSDPGrad)
            unreduced = pending.unreduced
            self.assertIsNotNone(unreduced)
            with (
                CommDebugMode() as comm,
                TrackPendingGradCopies(model.parameters()) as copies,
            ):
                grad = fsdp_param.unsharded_accumulated_grad
            self.assertEqual(comm.get_total_counts(), 0)
            self.assertEqual(copies.h2d, 0)
            self.assertEqual(copies.d2h, 0)
            for _ in range(2):
                with (
                    CommDebugMode() as comm,
                    TrackPendingGradCopies(model.parameters()) as copies,
                ):
                    data = fsdp_param.unsharded_accumulated_grad_data
                self.assertEqual(comm.get_total_counts(), 0)
                self.assertEqual(copies.h2d, int(cpu_offload))
                self.assertEqual(copies.d2h, 0)
            self.assertIsInstance(grad, DTensor)
            self.assertEqual(grad.device_mesh, mesh["tp"])
            self.assertEqual(grad.placements, (Shard(0),))
            self.assertEqual(grad.device.type, "cpu" if cpu_offload else device)
            self.assertEqual(data.device, fsdp_param.device)
            self.assertEqual(grad.to(device).full_tensor(), ref_model.weight.grad)
            self.assertEqual(
                data, ref_model.weight.grad.chunk(self.world_size)[self.rank]
            )
            self.assertIs(model.weight.grad, pending)
            self.assertIs(pending.unreduced, unreduced)
            self.assertIsNone(fsdp_param.unsharded_param.grad)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "consumer",
        [
            "clip",
            "clip_foreach",
            "unscale",
            "unscale_inf",
            "mul",
            "zero",
            "zero_none",
        ],
    )
    def test_pending_grad_consumers(self, device, consumer):
        self.run_subtests(
            {
                "mesh_size": [1, self.world_size],
                "reduce_op": ["avg", "sum"],
                "reshard_after_backward": [False, True],
            },
            self._test_pending_grad_consumers,
            torch.device(device).type,
            consumer,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("replacement", ["clone", "shard", "replicate", "none"])
    def test_pending_grad_replacement(self, device, replacement):
        self.run_subtests(
            {
                "mesh_size": [1, self.world_size],
                "reshard_after_backward": [False, True],
                "reduce_op": ["avg", "sum"],
            },
            self._test_pending_grad_consumers,
            torch.device(device).type,
            f"replace_{replacement}",
        )

    def _test_pending_grad_consumers(
        self, device, consumer, mesh_size, reduce_op, reshard_after_backward
    ):
        mesh = init_device_mesh(
            device,
            (self.world_size // mesh_size, mesh_size),
            mesh_dim_names=("replicate", "shard"),
        )["shard"]
        ranks = mesh.mesh.tolist()
        ref_model = nn.Linear(4, 2, device=device)
        with torch.no_grad():
            ref_model.weight.fill_(0.25)
            ref_model.bias.fill_(0.5)
        model = copy.deepcopy(ref_model)
        fully_shard(model, mesh=mesh)
        model.set_reshard_after_backward(reshard_after_backward)
        model.set_requires_gradient_sync(False)
        if reduce_op == "sum":
            model.set_gradient_divide_factor(1)
        models = (ref_model, model)
        optims = [torch.optim.SGD(m.parameters(), lr=0.1) for m in models]
        scalers = [torch.amp.GradScaler(device, init_scale=8.0) for _ in models]
        use_scaler = consumer.startswith("unscale")

        def loss(module, rank, step):
            inp = torch.arange(8, device=device).view(2, 4) / 8
            inp = inp + (rank + 1) / 2 + step / 4
            output = module(inp).sum()
            if consumer == "unscale_inf" and step == 0 and rank == ranks[-1]:
                output = output * float("inf")
            return output

        def backward(step):
            ref_loss = sum(loss(ref_model, rank, step) for rank in ranks)
            if reduce_op == "avg":
                ref_loss = ref_loss / mesh_size
            losses = (ref_loss, loss(model, self.rank, step))
            for value, scaler in zip(losses, scalers):
                if use_scaler and step == 0:
                    value = scaler.scale(value)
                value.backward()

        def check_grads(placement):
            for ref_param, param in zip(ref_model.parameters(), model.parameters()):
                if ref_param.grad is None:
                    self.assertIsNone(param.grad)
                    continue
                if isinstance(placement, Partial):
                    self.assertIsInstance(param.grad, FSDPGrad)
                    self.assertEqual(param.grad.dtype, param.grad_dtype)
                    self.assertIsNotNone(param.grad.unreduced)
                    self.assertEqual(param.grad.unreduced.placements, (placement,))
                else:
                    self.assertIsInstance(param.grad, DTensor)
                    self.assertEqual(param.grad.placements, (placement,))
                self.assertEqual(ref_param.grad, param.grad.full_tensor())

        backward(0)
        check_grads(Partial(reduce_op))
        norms = []
        found_inf = []
        replacement_refs = []
        for module, optim, scaler in zip(models, optims, scalers):
            if consumer.startswith("clip"):
                norm = nn.utils.clip_grad_norm_(
                    module.parameters(), 1.0, foreach=consumer == "clip_foreach"
                )
                norms.append(norm.full_tensor() if isinstance(norm, DTensor) else norm)
            elif use_scaler:
                scaler.unscale_(optim)
                found_inf.append(scaler._found_inf_per_device(optim))
            elif consumer.startswith("zero"):
                module.zero_grad(set_to_none=consumer == "zero_none")
            elif consumer == "mul":
                for param in module.parameters():
                    param.grad.mul_(2)
            else:
                for param in module.parameters():
                    grad = param.grad
                    if consumer == "replace_none":
                        param.grad = None
                        continue
                    if (
                        isinstance(grad, (DTensor, FSDPGrad))
                        and consumer != "replace_clone"
                    ):
                        placement = (
                            Shard(0) if consumer == "replace_shard" else Replicate()
                        )
                        grad = grad.redistribute(placements=(placement,))
                    param.grad = grad.clone().mul_(2)
        if consumer == "replace_replicate":
            replacement_refs = [
                (p.grad, p.grad.full_tensor().clone()) for p in model.parameters()
            ]
        if norms:
            self.assertEqual(norms[0], norms[1])
        if found_inf:
            self.assertEqual(found_inf[0], found_inf[1])
            for value in found_inf[1].values():
                self.assertEqual(value.item(), float(consumer == "unscale_inf"))
        replacement_placement = {
            "replace_shard": Shard(0),
            "replace_replicate": Replicate(),
        }.get(consumer, Partial(reduce_op))
        check_grads(replacement_placement)

        final_step = 1
        if consumer.startswith("replace_"):
            with CommDebugMode() as comm_mode:
                backward(1)
            if consumer == "replace_clone" and not reshard_after_backward:
                self.assertEqual(comm_mode.get_total_counts(), 0)
            check_grads(Partial(reduce_op))
            final_step = 2

        # The unscale cases only check mutation preservation here. Normal
        # GradScaler use unscales after all gradient accumulation is complete.
        model.set_requires_gradient_sync(True)
        backward(final_step)
        check_grads(Shard(0))
        for grad, expected in replacement_refs:
            self.assertEqual(grad.full_tensor(), expected)

    @skip_if_lt_x_gpu(2)
    def test_pending_grad_replacement_different_mesh(self, device):
        device = torch.device(device).type
        mesh = init_device_mesh(device, (self.world_size,))
        other_mesh = init_device_mesh(device, (1, self.world_size))
        model = nn.Linear(4, 2, bias=False, device=device)
        fully_shard(model, mesh=mesh)
        model.set_requires_gradient_sync(False)
        inp = torch.ones(2, 4, device=device)
        model(inp).sum().backward()
        model.weight.grad = DTensor.from_local(
            model.weight.grad.full_tensor(), other_mesh, (Replicate(), Replicate())
        )
        with self.assertRaisesRegex(
            ValueError,
            "Replacing a pending gradient with a DTensor on a different mesh is not supported",
        ):
            model(inp)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "optimizer_cls,dtype",
        [
            (torch.optim.SGD, torch.float32),
            (torch.optim.AdamW, torch.float32),
            (torch.optim.SGD, torch.bfloat16),
        ],
    )
    @parametrize("foreach", [False, True])
    def test_optimizer_step_with_pending_grad(
        self, device, optimizer_cls, dtype, foreach
    ):
        self.run_subtests(
            {
                "mesh_size": [1, self.world_size],
                "reshard_after_backward": [False, True],
            },
            self._test_optimizer_step_with_pending_grad,
            torch.device(device).type,
            optimizer_cls,
            dtype,
            foreach,
        )

    def _test_optimizer_step_with_pending_grad(
        self, device, optimizer_cls, dtype, foreach, mesh_size, reshard_after_backward
    ):
        mesh = init_device_mesh(
            device,
            (self.world_size // mesh_size, mesh_size),
            mesh_dim_names=("replicate", "shard"),
        )["shard"]
        ranks = mesh.mesh.tolist()
        ref_model = nn.Linear(4, 2, device=device, dtype=dtype)
        with torch.no_grad():
            ref_model.weight.fill_(0.25)
            ref_model.bias.fill_(0.5)
        model = copy.deepcopy(ref_model)
        if dtype == torch.bfloat16:
            for module in (ref_model, model):
                for param in module.parameters():
                    param.grad_dtype = torch.float32
        fully_shard(
            model,
            mesh=mesh,
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
        )
        model.set_reshard_after_backward(reshard_after_backward)
        ref_optim, optim = [
            optimizer_cls(module.parameters(), lr=0.125, foreach=foreach, fused=False)
            for module in (ref_model, model)
        ]
        params = list(model.parameters())
        for step in range(4):
            sync = step >= 2
            model.set_requires_gradient_sync(sync)
            inputs = {
                rank: torch.arange(8, device=device, dtype=dtype).view(2, 4) / 8
                + (rank + 1) / 2
                + step / 4
                for rank in ranks
            }
            ref_outputs = {rank: ref_model(inp) for rank, inp in inputs.items()}
            with CommDebugMode() as comm_mode:
                output = model(inputs[self.rank])
            self.assertEqual(output, ref_outputs[self.rank])
            if step == 1 and not reshard_after_backward:
                self.assertEqual(comm_mode.get_total_counts(), 0)
            elif step >= 2 and mesh_size > 1:
                self.assertGreater(comm_mode.get_total_counts(), 0)
            ref_loss = sum(
                value.float().square().mean() for value in ref_outputs.values()
            )
            (ref_loss / mesh_size).backward()
            output.float().square().mean().backward()
            for ref_param, param, original in zip(
                ref_model.parameters(), model.parameters(), params
            ):
                self.assertIs(param, original)
                self.assertIsInstance(param.grad, DTensor if sync else FSDPGrad)
                if sync:
                    self.assertEqual(param.grad.placements, (Shard(0),))
                else:
                    self.assertEqual(param.grad.unreduced.placements, (Partial("avg"),))
                self.assertEqual(ref_param.grad, param.grad.full_tensor())
            if step == 0 and not reshard_after_backward and self.rank == 0:
                # A rank-local no-op must not trigger an implicit all-gather.
                with torch.no_grad():
                    params[0].add_(0)
            if step in (1, 2):
                model.synchronize_gradients()
                model.reshard()
                ref_optim.step()
                optim.step()
                for ref_param, param in zip(ref_model.parameters(), params):
                    self.assertEqual(ref_param, param.full_tensor())
                    self.assertEqual(
                        set(ref_optim.state[ref_param]), set(optim.state[param])
                    )
                    for key, value in optim.state[param].items():
                        if isinstance(value, DTensor):
                            value = value.full_tensor()
                        self.assertEqual(ref_optim.state[ref_param][key], value)

    @skip_if_lt_x_gpu(2)
    def test_pinned_cpu_offload_explicit_sync(self, device):
        device = torch.device(device).type
        model = nn.Linear(4, 4, bias=False, device=device)
        with torch.no_grad():
            model.weight.fill_(1)
        fully_shard(
            model,
            mesh=init_device_mesh(device, (self.world_size,)),
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            offload_policy=CPUOffloadPolicy(pin_memory=True),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        model.set_requires_gradient_sync(False)
        model(
            torch.full((2, 4), self.rank + 1, device=device, dtype=torch.bfloat16)
        ).sum().backward()
        self.assertIsInstance(model.weight.grad, FSDPGrad)
        model.synchronize_gradients()
        group = model._get_fsdp_state()._fsdp_param_groups[0]
        for fsdp_param in group.fsdp_params:
            self.assertIsNone(fsdp_param.grad_offload_event)
        self.assertEqual(model.weight.grad.device.type, "cpu")
        self.assertEqual(
            model.weight.grad.to_local(),
            torch.full_like(model.weight.to_local(), 3),
        )
        optimizer.step()
        self.assertEqual(
            model.weight.to_local(), torch.full_like(model.weight.to_local(), 0.25)
        )


class TestFullyShardPendingGradHSDP(FSDPTest):
    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    @parametrize("mutation", ["none", "scale", "zero", "zero_none", "reset"])
    @parametrize("cpu_offload", [False, True])
    def test_reduced_partial_and_unreduced_components(
        self, device, mutation, cpu_offload
    ):
        device = torch.device(device).type
        mesh = init_device_mesh(
            torch.device(device).type,
            (2, 2),
            mesh_dim_names=("replicate", "shard"),
        )
        model = nn.Linear(1, 4, bias=False, device=device)
        fully_shard(
            model,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        model(
            torch.full((1, 1), 4.0, device=device, dtype=torch.bfloat16)
        ).sum().backward()
        model.set_requires_all_reduce(False)
        model(
            torch.full((1, 1), 2.0, device=device, dtype=torch.bfloat16)
        ).sum().backward()
        grad = model.weight.grad
        self.assertIsInstance(grad, FSDPGrad)
        self.assertIsNotNone(grad.reduced)
        self.assertIsNotNone(grad.partial)
        self.assertIsNone(grad.unreduced)
        self.assertEqual(grad.reduced.dtype, torch.float32)
        self.assertEqual(grad.partial.dtype, torch.bfloat16)
        actual = grad.to(device).full_tensor()
        self.assertEqual(actual, torch.full_like(actual, 6))

        model.set_requires_gradient_sync(False)
        model(torch.ones(1, 1, device=device, dtype=torch.bfloat16)).sum().backward()
        grad = model.weight.grad
        self.assertIsInstance(grad, FSDPGrad)
        self.assertEqual(grad.reduced.dtype, torch.float32)
        self.assertEqual(grad.partial.dtype, torch.bfloat16)
        self.assertEqual(grad.unreduced.dtype, torch.bfloat16)
        actual = grad.to(device).full_tensor()
        self.assertEqual(actual, torch.full_like(actual, 7))
        expected = 7
        if mutation == "scale":
            grad.mul_(0.5)
            expected = 3.5
        elif mutation in ("zero", "zero_none"):
            model.zero_grad(set_to_none=mutation == "zero_none")
            expected = 0
            if mutation == "zero_none":
                # Discarding the public value also discards its pending HSDP
                # stage before changing the reduction recipe.
                model.set_gradient_divide_factor(self.world_size)
        elif mutation == "reset":
            original_forward = model.forward

            def fail_forward(*args, **kwargs):
                original_forward(*args, **kwargs)
                raise RuntimeError("expected forward failure")

            model.forward = fail_forward
            with self.assertRaisesRegex(RuntimeError, "expected forward failure"):
                model(torch.ones(1, 1, device=device, dtype=torch.bfloat16))
            model.forward = original_forward
            model.reset_iter_state()
            expected = 0
        model.synchronize_gradients()
        if mutation in ("zero_none", "reset"):
            self.assertIsNone(model.weight.grad)
        else:
            self.assertIsInstance(model.weight.grad, DTensor)
            actual = model.weight.grad.to(device).full_tensor()
            self.assertEqual(actual, torch.full_like(actual, expected))
        # A later backward must not resurrect the consumed HSDP stage.
        model.set_requires_gradient_sync(True)
        model(torch.ones(1, 1, device=device, dtype=torch.bfloat16)).sum().backward()
        actual = model.weight.grad.to(device).full_tensor()
        self.assertEqual(actual, torch.full_like(actual, expected + 1))

    @skip_if_lt_x_gpu(4)
    @parametrize("action", ["clear", "replace", "policy", "offload"])
    def test_saved_partial_alias_after_replacement(self, device, action):
        device = torch.device(device).type
        mesh = init_device_mesh(device, (2, 2), mesh_dim_names=("replicate", "shard"))
        model = nn.Linear(1, 4, bias=False, device=device)
        fully_shard(
            model,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16),
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if action == "offload"
            else OffloadPolicy(),
        )
        model.set_requires_all_reduce(False)
        model(
            torch.full((1, 1), self.rank + 1, device=device, dtype=torch.bfloat16)
        ).sum().backward()
        saved = model.weight.grad
        self.assertIsInstance(saved, FSDPGrad)
        expected_partial = saved.partial.to_local().clone()
        model.zero_grad(set_to_none=True)
        if action == "replace":
            model.weight.grad = torch.full_like(model.weight, 7)
        elif action == "policy":
            model.set_gradient_divide_factor(self.world_size)
        elif action == "offload":
            # Publishing the next offloaded contribution must also
            # release the old owner without overwriting saved aliases.
            model.set_requires_gradient_sync(False)
            model(
                torch.zeros((1, 1), device=device, dtype=torch.bfloat16)
            ).sum().backward()
        model.synchronize_gradients()
        self.assertEqual(saved.partial.to_local(), expected_partial)
        if action in ("clear", "policy"):
            self.assertIsNone(model.weight.grad)
        else:
            self.assertEqual(
                model.weight.grad.to_local(),
                torch.full_like(
                    model.weight.to_local(), 7 if action == "replace" else 0
                ),
            )


instantiate_device_type_tests(
    TestFullyShardPendingGrad, globals(), only_for=(get_devtype().type,)
)
instantiate_device_type_tests(
    TestFullyShardPendingGradHSDP, globals(), only_for=(get_devtype().type,)
)


class SpmdPendingGradModel(nn.Module):
    def __init__(self, ref_model, mesh):
        super().__init__()
        self.tp_pg = mesh.get_group("tp")
        self.replicated_weight = nn.Parameter(ref_model[0].weight.detach().clone())
        self.sharded_weight = nn.Parameter(
            ref_model[1].weight.detach().chunk(2)[mesh["tp"].get_local_rank()].clone()
        )
        dp_axis = spmd.MeshAxis.of(mesh.get_group("dp"))
        tp_axis = spmd.MeshAxis.of(self.tp_pg)
        spmd.assert_type(self.replicated_weight, {dp_axis: spmd.R, tp_axis: spmd.R})
        spmd.assert_type(self.sharded_weight, {dp_axis: spmd.R, tp_axis: spmd.S(0)})
        for param in self.parameters():
            param.grad_dtype = torch.float32

    def forward(self, inp):
        output = inp @ self.replicated_weight.t()
        output = spmd.redistribute(
            output,
            self.tp_pg,
            src=spmd.S(0),
            dst=spmd.R,
            backward_options={"op_dtype": torch.float32},
        )
        output = output @ self.sharded_weight.t()
        output = spmd.redistribute(
            output,
            self.tp_pg,
            src=spmd.S(1),
            dst=spmd.I,
            backward_options={"op_dtype": torch.float32},
        )
        return output.sum()


@unittest.skipUnless(dist._is_spmd_types_available(), "requires spmd_types")
class TestFullyShardSpmdPendingGrad(FSDPTest):
    @property
    def world_size(self):
        return 4

    @skip_if_lt_x_gpu(4)
    def test_cpu_offload_spmd_pending_grad(self, device):
        device = torch.device(device).type
        mesh = init_device_mesh(device, (2, 2), mesh_dim_names=("dp", "tp"))
        self._test_reduced_to_pending_spmd_grad(
            device,
            torch.bfloat16,
            mesh,
            "avg",
            "mul",
            False,
            cpu_offload=True,
        )

    @skip_if_lt_x_gpu(4)
    def test_pending_grad_replacement_spmd(self, device):
        device = torch.device(device).type
        mesh = init_device_mesh(device, (2, 2), mesh_dim_names=("dp", "tp"))
        self.run_subtests(
            {"reshard_after_backward": [False, True]},
            self._test_reduced_to_pending_spmd_grad,
            device,
            torch.float32,
            mesh=mesh,
            reduce_op="avg",
            mutation="replicate",
        )

    @skip_if_lt_x_gpu(4)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_reduced_to_pending_spmd_grad(self, device, dtype):
        device = torch.device(device).type
        meshes = [
            init_device_mesh(
                device,
                (self.world_size // (dp_size * 2), dp_size, 2),
                mesh_dim_names=("replicate", "dp", "tp"),
            )["dp", "tp"]
            for dp_size in (1, 2)
        ]
        self.run_subtests(
            {
                "mesh": meshes,
                "reduce_op": ["avg", "sum"],
                "mutation": ["mul", "zero", "zero_none"],
                "reshard_after_backward": [False, True],
            },
            self._test_reduced_to_pending_spmd_grad,
            device,
            dtype,
        )

    def _test_reduced_to_pending_spmd_grad(
        self,
        device,
        dtype,
        mesh,
        reduce_op,
        mutation,
        reshard_after_backward,
        cpu_offload=False,
    ):
        ref_model = nn.Sequential(
            nn.Linear(4, 4, bias=False, device=device, dtype=dtype),
            nn.Linear(4, 8, bias=False, device=device, dtype=dtype),
        )
        for param in ref_model.parameters():
            with torch.no_grad():
                param.fill_(0.25)
            param.grad_dtype = torch.float32
        model = SpmdPendingGradModel(ref_model, mesh)
        fully_shard(
            model,
            mesh=mesh,
            dp_mesh_dims=DataParallelMeshDims(shard="dp"),
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
            reshard_after_forward=False,
            offload_policy=CPUOffloadPolicy(pin_memory=False)
            if cpu_offload
            else OffloadPolicy(),
        )
        model.set_reshard_after_backward(reshard_after_backward)
        if reduce_op == "sum":
            model.set_gradient_divide_factor(1)
        dp_axis = spmd.MeshAxis.of(mesh.get_group("dp"))
        tp_axis = spmd.MeshAxis.of(mesh.get_group("tp"))
        ranks_by_dp = mesh.mesh.tolist()

        def make_input(rank, step):
            return (
                torch.arange(8, device=device, dtype=dtype).view(2, 4) / 8
                + (rank + 1) / 2
                + step / 4
            )

        def check_grads(pending, replicated=False):
            for index, (ref_param, param) in enumerate(
                zip(ref_model.parameters(), model.parameters())
            ):
                if ref_param.grad is None:
                    self.assertIsNone(param.grad)
                    continue
                self.assertIsInstance(
                    param.grad, FSDPGrad if pending and not replicated else DTensor
                )
                visible_grad = (
                    param.grad.unreduced
                    if isinstance(param.grad, FSDPGrad)
                    else param.grad
                )
                self.assertIsNotNone(visible_grad)
                if replicated:
                    self.assertEqual(
                        visible_grad.placements, (Replicate(), Replicate())
                    )
                else:
                    tp_placement = Partial() if pending and index == 0 else Replicate()
                    if index == 1:
                        tp_placement = Shard(0)
                    self.assertEqual(visible_grad.placements[1], tp_placement)
                    if pending:
                        self.assertEqual(visible_grad.placements[0], Partial(reduce_op))
                    else:
                        self.assertEqual(
                            visible_grad.placements[0], param.placements[0]
                        )
                if cpu_offload:
                    self.assertEqual(param.grad.device.type, "cpu")
                self.assertEqual(ref_param.grad, param.grad.to(device).full_tensor())

        syncs = (True, False, True)
        if mutation == "replicate":
            syncs = (True, False, False, True)
        if cpu_offload:
            syncs = (False, False, False, True)
        replacement_refs = []
        for step, sync in enumerate(syncs):
            model.set_requires_gradient_sync(sync)
            ref_loss = sum(
                ref_model(torch.cat([make_input(rank, step) for rank in ranks])).sum()
                for ranks in ranks_by_dp
            )
            if reduce_op == "avg":
                ref_loss = ref_loss / len(ranks_by_dp)
            ref_loss.backward()
            inp = make_input(self.rank, step)
            with TrackPendingGradCopies(model.parameters()) as copies:
                with (
                    spmd.set_current_mesh(mesh),
                    typecheck(strict_mode="strict", local=False),
                ):
                    spmd.assert_type(
                        inp,
                        {dp_axis: spmd.V, tp_axis: spmd.V},
                        partition_spec=spmd.PartitionSpec((dp_axis, tp_axis), None),
                    )
                    model(inp).backward()
            if cpu_offload and step > 0 and not sync:
                self.assertEqual(copies.h2d, 0)
            check_grads(pending=not sync)
            for grad, expected in replacement_refs:
                self.assertEqual(grad.full_tensor(), expected)
            if step == 1:
                for module in (ref_model, model):
                    if mutation == "mul":
                        for param in module.parameters():
                            param.grad.mul_(2)
                    elif mutation == "replicate":
                        for param in module.parameters():
                            grad = param.grad
                            if isinstance(grad, (DTensor, FSDPGrad)):
                                grad = grad.redistribute(
                                    placements=(Replicate(), Replicate())
                                )
                            param.grad = grad.clone().mul_(2)
                    else:
                        module.zero_grad(set_to_none=mutation == "zero_none")
                if mutation == "replicate":
                    replacement_refs = [
                        (p.grad, p.grad.full_tensor().clone())
                        for p in model.parameters()
                    ]
                check_grads(pending=True, replicated=mutation == "replicate")


instantiate_device_type_tests(
    TestFullyShardSpmdPendingGrad, globals(), only_for=(get_devtype().type,)
)
if __name__ == "__main__":
    run_tests()
