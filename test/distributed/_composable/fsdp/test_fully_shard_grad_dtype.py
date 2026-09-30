# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.utils.checkpoint import checkpoint


class _FP32WGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        ctx.save_for_backward(inp, weight)
        return inp @ weight.T

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        grad_weight = torch.mm(grad_output.T.contiguous(), inp, out_dtype=torch.float32)
        return grad_output @ weight, grad_weight


class _Linear(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.full((32, 32), 0.125, device=device, dtype=dtype))

    def forward(self, inp):
        return _FP32WGrad.apply(inp, self.weight)


@instantiate_parametrized_tests
class TestGradientDtypePolicy(TestCase):
    @parametrize("dtype", [torch.int64, torch.bool, torch.complex64, "float32"])
    def test_invalid_dtype(self, dtype):
        with self.assertRaisesRegex(ValueError, "floating-point dtype"):
            MixedPrecisionPolicy(grad_dtype=dtype)

    def test_positional_compatibility(self):
        policy = MixedPrecisionPolicy(torch.bfloat16, torch.float32, None, False)
        self.assertIsNone(policy.grad_dtype)
        self.assertFalse(policy.cast_forward_inputs)


class TestFullyShardGradientDtype(FSDPTest):
    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    @skip_if_lt_x_gpu(2)
    @skipIfTorchDynamo("grad_dtype is not supported in compile")
    @parametrize("storage_dtype", [torch.bfloat16, torch.float32])
    @parametrize("reshard_after_forward", [False, True])
    @parametrize("schedule", ["sync", "no_sync", "checkpoint"])
    @parametrize("hsdp", [False, True])
    def test_fp32_gradient(
        self, device, storage_dtype, reshard_after_forward, schedule, hsdp
    ):
        if hsdp and self.world_size < 4:
            self.skipTest("HSDP coverage requires four GPUs")
        device = torch.device(device).type
        mesh_shape = (2, self.world_size // 2) if hsdp else (self.world_size,)
        mesh_dim_names = ("replicate", "shard") if hsdp else ("shard",)
        mesh = init_device_mesh(device, mesh_shape, mesh_dim_names=mesh_dim_names)
        projection = _Linear(torch.device(device, self.rank), storage_dtype)
        model = nn.Sequential(projection)
        # An omitted reduce_dtype must not restore the BF16 compute dtype.
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, grad_dtype=torch.float32
        )
        fully_shard(
            projection,
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=reshard_after_forward,
        )
        fully_shard(model, mesh=mesh)
        projection.set_gradient_divide_factor(1.0)
        projection.set_force_sum_reduction_for_comms(True)
        param_group = fully_shard.state(projection)._fsdp_param_groups[0]
        fsdp_param = param_group.fsdp_params[0]
        shard_size = mesh_shape[-1]
        shard_rank = mesh.get_local_rank(mesh.ndim - 1)

        for step in range(2):
            model.zero_grad(set_to_none=step == 0)
            expected = torch.zeros((32, 32), device=self.rank, dtype=torch.float32)
            for microbatch in range(2):
                projection.set_requires_gradient_sync(
                    schedule != "no_sync" or microbatch == 1
                )
                inp = torch.full(
                    (2 + microbatch, 32),
                    1.0078125,
                    device=self.rank,
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                output = (
                    checkpoint(model, inp, use_reentrant=False)
                    if schedule == "checkpoint"
                    else model(inp)
                )
                self.assertEqual(fsdp_param.unsharded_param.dtype, torch.bfloat16)
                grad_output = torch.full_like(output, 1.0078125 * (self.rank + 1))
                # FP32 arithmetic on the same BF16 operands is an independent
                # reference for the mixed-input GEMM's FP32 output.
                local_grad = grad_output.float().T @ inp.detach().float()
                self.assertFalse(
                    torch.equal(local_grad, local_grad.bfloat16().float())
                )
                received = []
                handle = fsdp_param.unsharded_param.register_hook(
                    lambda grad: received.append(grad.detach().clone())
                )
                output.backward(grad_output)
                handle.remove()
                self.assertEqual(len(received), 1)
                self.assertEqual(received[0].dtype, torch.float32)
                self.assertEqual(received[0], local_grad, atol=0, rtol=0)
                dist.all_reduce(local_grad, op=dist.ReduceOp.SUM)
                expected.add_(local_grad)
                if schedule == "no_sync" and microbatch == 0:
                    continue
                grad = projection.weight.grad
                self.assertEqual(projection.weight.dtype, storage_dtype)
                self.assertEqual(grad.dtype, torch.float32)
                self.assertEqual(grad._spec.tensor_meta.dtype, torch.float32)
                self.assertEqual(
                    grad.to_local(),
                    expected.chunk(shard_size)[shard_rank],
                    atol=0,
                    rtol=0,
                )
            with torch.no_grad():
                projection.weight.add_(
                    projection.weight.grad.to(storage_dtype), alpha=-0.001
                )

    @skip_if_lt_x_gpu(2)
    @skipIfTorchDynamo("grad_dtype is not supported in compile")
    @parametrize("storage_dtype", [torch.bfloat16, torch.float32])
    def test_default_gradient_dtype(self, device, storage_dtype):
        projection = _Linear(
            torch.device(torch.device(device).type, self.rank), storage_dtype
        )
        fully_shard(
            projection,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        inp = torch.full((2, 32), 1.0078125, device=self.rank, dtype=torch.bfloat16)
        output = projection(inp)
        param_group = fully_shard.state(projection)._fsdp_param_groups[0]
        fsdp_param = param_group.fsdp_params[0]
        received = []
        handle = fsdp_param.unsharded_param.register_hook(
            lambda grad: received.append(grad.dtype)
        )
        output.backward(torch.full_like(output, 1.0078125))
        handle.remove()
        self.assertEqual(received, [torch.bfloat16])
        self.assertEqual(projection.weight.grad.dtype, storage_dtype)

    @skip_if_lt_x_gpu(2)
    @skipIfTorchDynamo("grad_dtype is not supported in compile")
    @parametrize("reduce_dtype", [torch.bfloat16, torch.float32])
    def test_explicit_reduction_dtype_and_unused_parameter(self, device, reduce_dtype):
        projection = _Linear(
            torch.device(torch.device(device).type, self.rank), torch.bfloat16
        )
        projection.register_parameter(
            "unused", nn.Parameter(torch.zeros_like(projection.weight))
        )
        fully_shard(
            projection,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_dtype,
                grad_dtype=torch.float32,
            ),
        )
        projection.set_reduce_scatter_unused_params(True)
        projection.set_gradient_divide_factor(1.0)
        projection.set_force_sum_reduction_for_comms(True)
        inp = torch.full((2, 32), 1.0078125, device=self.rank, dtype=torch.bfloat16)
        output = projection(inp)
        grad_output = torch.full_like(output, 1.0078125)
        expected = (grad_output.float().T @ inp.float()).to(reduce_dtype)
        dist.all_reduce(expected, op=dist.ReduceOp.SUM)
        output.backward(grad_output)
        grad = projection.weight.grad
        self.assertEqual(grad.dtype, torch.float32)
        self.assertEqual(
            grad.to_local(),
            expected.float().chunk(self.world_size)[self.rank],
            atol=0,
            rtol=0,
        )
        self.assertEqual(projection.unused.grad.dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(projection.unused.grad.to_local()), 0)

    @skip_if_lt_x_gpu(2)
    @skipIfTorchDynamo("Parameter conversion is tested in eager mode")
    @parametrize("storage_dtype", [torch.bfloat16, torch.float32])
    @parametrize("grad_dtype", [None, torch.float32])
    def test_parameter_conversion_with_gradient(self, device, storage_dtype, grad_dtype):
        device = torch.device(torch.device(device).type, self.rank)
        model = nn.Sequential(_Linear(device, storage_dtype))
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, grad_dtype=grad_dtype
            ),
        )
        inp = torch.full((3, 32), 1.0078125, device=device, dtype=torch.bfloat16)
        model(inp).backward(torch.full_like(inp, 1.0078125))
        param = model[0].weight
        expected_param = param.to_local().detach().clone()
        expected_grad = param.grad.to_local().clone()
        expected_dtype = grad_dtype or storage_dtype

        def fail_on_parameter(tensor):
            if tensor.requires_grad:
                raise RuntimeError("parameter conversion failed")
            return tensor.clone()

        original_grad = param.grad
        with self.assertRaisesRegex(RuntimeError, "parameter conversion failed"):
            model._apply(fail_on_parameter)
        self.assertIs(param.grad, original_grad)
        del original_grad

        model.cpu()
        self.assertIs(model[0].weight, param)
        self.assertEqual(param.device.type, "cpu")
        self.assertEqual(param.grad.device.type, "cpu")
        self.assertEqual(param.grad_dtype, expected_dtype)
        self.assertEqual(param.grad.dtype, expected_dtype)
        self.assertEqual(param.grad._spec.tensor_meta.dtype, expected_dtype)
        self.assertEqual(param.to_local(), expected_param.cpu(), atol=0, rtol=0)
        self.assertEqual(param.grad.to_local(), expected_grad.cpu(), atol=0, rtol=0)

        model.to(device=device)
        self.assertEqual(param.grad.dtype, expected_dtype)
        self.assertEqual(param.grad.to_local(), expected_grad, atol=0, rtol=0)
        model(inp).backward(torch.full_like(inp, 1.0078125))
        self.assertEqual(param.grad.to_local(), expected_grad * 2, atol=0, rtol=0)

    @skip_if_lt_x_gpu(2)
    @skipIfTorchDynamo("Parameter conversion is tested in eager mode")
    @parametrize("storage_dtype", [torch.bfloat16, torch.float32])
    def test_partial_parameter_conversion_failure(self, device, storage_dtype):
        device = torch.device(torch.device(device).type, self.rank)
        model = nn.Sequential(
            _Linear(device, storage_dtype), _Linear(device, storage_dtype)
        )
        model[0].register_buffer("conversion_sentinel", torch.ones(1, device=device))
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, grad_dtype=torch.float32
            ),
        )
        inp = torch.full((3, 32), 1.0078125, device=device, dtype=torch.bfloat16)
        model(inp).backward(torch.full_like(inp, 1.0078125))
        params = list(model.parameters())
        expected_grads = [param.grad.to_local().clone() for param in params]

        def fail_on_buffer(tensor):
            if tensor.numel() == 1:
                raise RuntimeError("buffer conversion failed")
            return tensor.cpu()

        with self.assertRaisesRegex(RuntimeError, "buffer conversion failed"):
            model._apply(fail_on_buffer)
        self.assertEqual(params[0].device.type, "cpu")
        self.assertEqual(params[1].device, device)
        for param, expected in zip(params, expected_grads):
            self.assertIsNotNone(param.grad)
            self.assertEqual(param.grad.device, param.device)
            self.assertEqual(param.grad.dtype, torch.float32)
            self.assertEqual(
                param.grad.to_local(), expected.to(param.device), atol=0, rtol=0
            )

        model.to(device=device)
        model(inp).backward(torch.full_like(inp, 1.0078125))
        for param, expected in zip(params, expected_grads):
            self.assertEqual(param.grad.to_local(), expected * 2, atol=0, rtol=0)

    @skip_if_lt_x_gpu(2)
    @skipIfTorchDynamo("Parameter conversion is tested in eager mode")
    @parametrize("storage_dtype", [torch.bfloat16, torch.float32])
    @parametrize("conversion", ["bfloat16", "half", "to_bfloat16", "to_cpu_bfloat16"])
    def test_parameter_dtype_conversion_preserves_gradient(
        self, device, storage_dtype, conversion
    ):
        device = torch.device(torch.device(device).type, self.rank)
        model = _Linear(device, storage_dtype)
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, grad_dtype=torch.float32
            ),
        )
        inp = torch.full((3, 32), 1.0078125, device=device, dtype=torch.bfloat16)
        model(inp).backward(torch.full_like(inp, 1.0078125))
        expected = model.weight.grad.to_local().clone()
        self.assertNotEqual(expected, expected.bfloat16().float())
        self.assertNotEqual(expected, expected.half().float())

        target_dtype = torch.float16 if conversion == "half" else torch.bfloat16
        target_device = (
            torch.device("cpu") if conversion == "to_cpu_bfloat16" else device
        )
        if conversion in ("bfloat16", "half"):
            getattr(model, conversion)()
        else:
            model.to(device=target_device, dtype=target_dtype)
        self.assertEqual(model.weight.dtype, target_dtype)
        self.assertEqual(model.weight.grad.dtype, torch.float32)
        self.assertEqual(model.weight.grad.device, target_device)
        self.assertEqual(
            model.weight.grad.to_local(), expected.to(target_device), atol=0, rtol=0
        )

        model.to(device=device, dtype=storage_dtype)
        model(inp).backward(torch.full_like(inp, 1.0078125))
        self.assertEqual(model.weight.grad.to_local(), expected * 2, atol=0, rtol=0)


instantiate_device_type_tests(TestFullyShardGradientDtype, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
