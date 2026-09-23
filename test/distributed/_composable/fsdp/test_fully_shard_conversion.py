# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestFullyShardConversion(TestCase):
    def setUp(self):
        super().setUp()
        dist.init_process_group(
            "nccl" if self.device_type == "cuda" else "gloo",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        self.addCleanup(dist.destroy_process_group)
        self.mesh = init_device_mesh(self.device_type, (1,))

    def _assert_parity(self, model, reference, *, check_override=False):
        for actual, expected in zip(model.parameters(), reference.parameters()):
            self.assertIsInstance(actual, DTensor)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertEqual(actual.grad_dtype, expected.grad_dtype)
            if check_override:
                self.assertEqual(
                    actual._has_grad_dtype_override, expected._has_grad_dtype_override
                )
            self.assertEqual(actual.to_local(), expected)
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                self.assertIsInstance(actual.grad, DTensor)
                self.assertEqual(actual.grad.dtype, expected.grad.dtype)
                self.assertEqual(actual.grad.to_local(), expected.grad)

    @parametrize("grad_dtype", ["default", torch.float32, None])
    @parametrize("existing_grad", [False, True])
    def test_device_conversion_matches_plain_module(
        self, device, grad_dtype, existing_grad
    ):
        reference = nn.Linear(4, 4, device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            reference.weight.fill_(0.25)
            reference.bias.fill_(0.125)
        reference.bias.requires_grad_(False)
        model = copy.deepcopy(reference)
        for module in (model, reference):
            if grad_dtype != "default":
                for param in module.parameters():
                    param.grad_dtype = grad_dtype
        fully_shard(model, mesh=self.mesh)
        inp = torch.arange(8, device=device, dtype=torch.bfloat16).view(2, 4) / 8
        if existing_grad:
            for module in (model, reference):
                for param in module.parameters():
                    if param.requires_grad:
                        param.grad = torch.ones_like(
                            param, dtype=param.grad_dtype or torch.float32
                        )

        for module in (model, reference):
            module.cpu()
        self._assert_parity(model, reference, check_override=True)
        for module in (model, reference):
            module.to(device=device)
        self._assert_parity(model, reference, check_override=True)
        for group in model._get_fsdp_state()._fsdp_param_groups:
            for param in group.fsdp_params:
                self.assertEqual(
                    param._has_sharded_grad_dtype_override, grad_dtype != "default"
                )
        for iteration in range(2):
            if iteration:
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
            expected = reference(inp)
            actual = model(inp)
            self.assertEqual(actual, expected)
            expected.sum().backward()
            actual.sum().backward()
            del actual, expected
            self._assert_parity(model, reference)

            optimizer_types = [torch.optim.SGD]
            if all(
                p.grad is None or p.dtype == p.grad.dtype
                for p in reference.parameters()
            ):
                optimizer_types.extend((torch.optim.Adam, torch.optim.AdamW))
            for optimizer_type in optimizer_types:
                for module in (model, reference):
                    optimizer_type(module.parameters(), lr=0.01).step()
                self._assert_parity(model, reference)

    def test_device_conversion_after_backward(self, device):
        reference = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
        model = copy.deepcopy(reference)
        for module in (model, reference):
            module.weight.grad_dtype = torch.float32
        fully_shard(model, mesh=self.mesh)
        inp = torch.ones(2, 4, device=device, dtype=torch.bfloat16)
        for module in (model, reference):
            module(inp).sum().backward()
        for module in (model, reference):
            module.cpu()
        self._assert_parity(model, reference, check_override=True)
        for module in (model, reference):
            if self.device_type == "cuda":
                module.cuda(device=device)
            else:
                module.to(device=device)
        self._assert_parity(model, reference, check_override=True)
        for module in (model, reference):
            module(inp).sum().backward()
        self._assert_parity(model, reference, check_override=True)

    @parametrize("grouped", [False, True])
    def test_nested_device_conversion_with_grad(self, device, grouped):
        model = nn.Sequential(
            nn.Linear(4, 4, device=device, dtype=torch.bfloat16),
            nn.Linear(4, 4, device=device, dtype=torch.bfloat16),
        )
        reference = copy.deepcopy(model)
        for module in (model, reference):
            module[1].weight.grad_dtype = torch.float32
        mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
        if grouped:
            fully_shard(list(model), mesh=self.mesh, mp_policy=mp_policy)
        else:
            for layer in model:
                fully_shard(layer, mesh=self.mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=self.mesh, mp_policy=mp_policy)
        for module in (model, reference):
            for param in module.parameters():
                param.grad = torch.ones_like(param, dtype=param.grad_dtype)
        for target in ("cpu", device):
            for module in (model, reference):
                module.to(device=target)
            self._assert_parity(model, reference, check_override=True)
        model(torch.ones(2, 4, device=device, dtype=torch.bfloat16)).sum().backward()
        for param in model.parameters():
            self.assertEqual(param.dtype, torch.bfloat16)
            self.assertIsNotNone(param.grad)
            self.assertEqual(param.grad.dtype, param.grad_dtype)

    def test_shared_parameter_device_conversion_with_grad(self, device):
        first = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
        second = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
        second.weight = first.weight
        first.weight.grad_dtype = torch.float32
        fully_shard([first, second], mesh=self.mesh)
        first.weight.grad = torch.ones_like(first.weight, dtype=torch.float32)
        for target in ("cpu", device):
            second.to(device=target)
            self.assertIs(first.weight, second.weight)
            self.assertEqual(first.weight.dtype, torch.bfloat16)
            self.assertEqual(first.weight.grad_dtype, torch.float32)
            self.assertEqual(first.weight.grad.device, torch.device(target))
            self.assertEqual(first.weight.grad.dtype, torch.float32)
            self.assertEqual(first.weight.grad.to_local(), torch.ones(4, 4))

    @parametrize("grad_dtype", ["default", None])
    def test_device_conversion_leaves_pending_grad_on_compute_device(
        self, device, grad_dtype
    ):
        model = nn.Linear(4, 4, bias=False, device=device)
        if grad_dtype is None:
            model.weight.grad_dtype = None
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32
            ),
        )
        model.set_requires_gradient_sync(False)
        model.set_reshard_after_backward(False)
        inp = torch.ones(2, 4, device=device, dtype=torch.bfloat16)
        model(inp).sum().backward()
        group = model._get_fsdp_state()._fsdp_param_groups[0]
        param = group.fsdp_params[0]
        self.assertTrue(group.is_unsharded)
        pending = model.weight.grad
        self.assertIs(pending, param.unsharded_param.grad)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.dtype, torch.float32)
        self.assertNotIsInstance(pending, DTensor)
        pending_value = pending.clone()
        for target in ("cpu", device):
            model.to(device=target)
            self.assertTrue(group.is_sharded)
            self.assertEqual(model.weight.device, torch.device(target))
            self.assertIs(param.unsharded_accumulated_grad, pending)
            self.assertEqual(pending.device, torch.device(device))
            self.assertEqual(pending, pending_value)
        model.set_requires_gradient_sync(True)
        model(inp).sum().backward()
        model.reshard()
        self.assertEqual(model.weight.grad.to_local(), 2 * pending_value)

    def test_no_op_conversion_preserves_pending_grad(self, device):
        model = nn.Linear(4, 4, bias=False, device=device)
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
            ),
        )
        sharded_param = model.weight
        model.set_requires_gradient_sync(False)
        inp = torch.ones(2, 4, device=device, dtype=torch.bfloat16)
        model(inp).sum().backward()
        self.assertIs(model.weight, sharded_param)
        self.assertEqual(sharded_param.dtype, torch.float32)
        self.assertIsNone(sharded_param.grad)
        model.unshard()
        pending = model.weight.grad
        self.assertEqual(pending.dtype, torch.bfloat16)
        pending_value = pending.clone()
        model.reshard()

        model.to(device=device)
        self.assertIs(model.weight, sharded_param)
        model.unshard()
        self.assertIs(model.weight.grad, pending)
        self.assertEqual(pending, pending_value)
        model.reshard()

        model.set_requires_gradient_sync(True)
        model(inp).sum().backward()
        self.assertEqual(model.weight.grad.dtype, torch.float32)
        self.assertEqual(
            model.weight.grad.full_tensor(),
            torch.full((4, 4), 4.0, device=device),
        )

    @parametrize("grad_dtype", [torch.float32, None])
    def test_same_device_conversion_preserves_grad(self, device, grad_dtype):
        reference = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
        model = copy.deepcopy(reference)
        for module in (model, reference):
            module.weight.grad_dtype = grad_dtype
        fully_shard(model, mesh=self.mesh)
        for module in (model, reference):
            module.weight.grad = torch.ones_like(module.weight, dtype=torch.float32)
        grad = model.weight.grad
        for module in (model, reference):
            module.to(device=device)
        self.assertIs(model.weight.grad, grad)
        self._assert_parity(model, reference, check_override=True)
        inp = torch.ones(2, 4, device=device, dtype=torch.bfloat16)
        for module in (model, reference):
            module(inp).sum().backward()
        self._assert_parity(model, reference, check_override=True)

    @parametrize(
        "orig_dtype,param_dtype",
        [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)],
    )
    @parametrize("override_reduce_dtype", [False, True])
    def test_default_reduction_uses_original_param_dtype(
        self, device, orig_dtype, param_dtype, override_reduce_dtype
    ):
        model = nn.Linear(1, 2, bias=False, device=device, dtype=orig_dtype)
        reference = copy.deepcopy(model).to(param_dtype)
        reduce_dtype = param_dtype if override_reduce_dtype else None
        unsharded_grad_dtype = reduce_dtype or orig_dtype
        reference.weight.grad_dtype = unsharded_grad_dtype
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=param_dtype, reduce_dtype=reduce_dtype
            ),
        )
        model.set_reshard_after_backward(False)
        for step, value in enumerate((256, 1, -256)):
            sync = step == 2
            model.set_requires_gradient_sync(sync)
            model.set_is_last_backward(sync)
            inp = torch.full((1, 1), value, device=device, dtype=param_dtype)
            model(inp).sum().backward()
            reference(inp).sum().backward()
            self.assertEqual(model.weight.dtype, param_dtype)
            self.assertEqual(model.weight.grad_dtype, unsharded_grad_dtype)
            if not sync:
                self.assertEqual(model.weight.grad.dtype, unsharded_grad_dtype)
                self.assertEqual(model.weight.grad, reference.weight.grad)
        expected = 0 if unsharded_grad_dtype == torch.bfloat16 else 1
        self.assertEqual(
            reference.weight.grad, torch.full_like(reference.weight.grad, expected)
        )
        self.assertIsNone(model.weight.grad)
        model.reshard()
        self.assertEqual(model.weight.grad_dtype, orig_dtype)
        self.assertEqual(model.weight.grad.dtype, orig_dtype)
        self.assertEqual(
            model.weight.grad.full_tensor(), reference.weight.grad.to(orig_dtype)
        )

    @parametrize("param_dtype", [None, torch.bfloat16])
    def test_grad_dtype_default_reduction_preserves_accumulation(
        self, device, param_dtype
    ):
        model = nn.Linear(
            1,
            2,
            bias=False,
            device=device,
            dtype=torch.bfloat16 if param_dtype is None else torch.float32,
        )
        reference = copy.deepcopy(model).to(torch.bfloat16)
        for module in (model, reference):
            module.weight.grad_dtype = torch.float32
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype),
        )
        model.set_reshard_after_backward(False)
        for step, value in enumerate((256, 1, -256)):
            sync = step == 2
            model.set_requires_gradient_sync(sync)
            model.set_is_last_backward(sync)
            inp = torch.full((1, 1), value, device=device, dtype=torch.bfloat16)
            model(inp).sum().backward()
            reference(inp).sum().backward()
            if not sync:
                self.assertEqual(model.weight.grad, reference.weight.grad)
        self.assertEqual(reference.weight.grad, torch.ones_like(reference.weight.grad))
        self.assertEqual(model.weight.grad_dtype, torch.float32)
        self.assertIsNone(model.weight.grad)
        model.reshard()
        self.assertEqual(model.weight.grad.full_tensor(), reference.weight.grad)

    @parametrize("reduce_dtype", [None, torch.bfloat16])
    def test_grad_dtype_none_preserves_incoming_dtype(self, device, reduce_dtype):
        class FloatGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight):
                return weight.clone()

            @staticmethod
            def backward(ctx, grad):
                return grad.float()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.ones(2, device=device, dtype=torch.bfloat16)
                )

            def forward(self, inp):
                return FloatGrad.apply(self.weight) * inp

        model, reference = Model(), Model()
        model.weight.grad_dtype = None
        reference.weight.grad_dtype = reduce_dtype
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=reduce_dtype),
        )
        model.set_reshard_after_backward(False)
        for step, value in enumerate((256, 1, -256)):
            sync = step == 2
            model.set_requires_gradient_sync(sync)
            model.set_is_last_backward(sync)
            inp = torch.full((2,), value, device=device, dtype=torch.bfloat16)
            model(inp).sum().backward()
            reference(inp).sum().backward()
            if not sync:
                self.assertEqual(model.weight.grad.dtype, reference.weight.grad.dtype)
                self.assertEqual(model.weight.grad, reference.weight.grad)
        self.assertEqual(model.weight.grad_dtype, reduce_dtype)
        self.assertIsNone(model.weight.grad)
        model.reshard()
        self.assertIsNone(model.weight.grad_dtype)
        self.assertEqual(model.weight.grad.full_tensor(), reference.weight.grad)

    @parametrize("first_dtype", [torch.bfloat16, torch.float32])
    def test_grad_dtype_none_partial_dtype_changes(self, device, first_dtype):
        class InputGrad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, weight, inp):
                ctx.save_for_backward(inp)
                return weight * inp

            @staticmethod
            def backward(ctx, grad):
                (inp,) = ctx.saved_tensors
                return grad.to(inp.dtype) * inp, None

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.ones(2, device=device, dtype=torch.bfloat16)
                )
                self.weight.grad_dtype = None

            def forward(self, inp):
                return InputGrad.apply(self.weight, inp)

        model = Model()
        mesh = init_device_mesh(
            torch.device(device).type,
            (1, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        fully_shard(model, mesh=mesh)
        second_dtype = (
            torch.float32 if first_dtype == torch.bfloat16 else torch.bfloat16
        )
        for step, dtype in enumerate((first_dtype, second_dtype)):
            model.set_requires_all_reduce(step == 1)
            value = 257 if dtype == torch.float32 else -256
            model(torch.full((2,), value, device=device, dtype=dtype)).sum().backward()
        self.assertIsNone(model.weight.grad_dtype)
        self.assertEqual(model.weight.grad.dtype, torch.float32)
        self.assertEqual(model.weight.grad.full_tensor(), torch.ones(2, device=device))

    @parametrize("reduce_dtype", [None, torch.float32])
    def test_grad_dtype_none_unused_params_requires_dtype(self, device, reduce_dtype):
        model = nn.Linear(2, 2, device=device)
        model.weight.grad_dtype = None
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=reduce_dtype),
        )
        if reduce_dtype is None:
            with self.assertRaisesRegex(ValueError, "grad_dtype=None.*reduce_dtype"):
                model.set_reduce_scatter_unused_params(True)
        else:
            model.set_reduce_scatter_unused_params(True)

    @parametrize("grad_dtype", ["default", torch.float32, None])
    @parametrize("assign", [False, True])
    def test_grad_dtype_policy_preserved_after_device_conversion_and_load_state_dict(
        self, device, grad_dtype, assign
    ):
        reference = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
        model = copy.deepcopy(reference)
        if grad_dtype != "default":
            for module in (model, reference):
                module.weight.grad_dtype = grad_dtype
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(
                reduce_dtype=None if grad_dtype == "default" else grad_dtype
            ),
        )
        for module in (model, reference):
            module.to(device=device)
        self._assert_parity(model, reference, check_override=True)
        previous = model.weight
        state_dict = model.state_dict()
        replacement = nn.Parameter(state_dict["weight"])
        replacement.grad_dtype = torch.float64
        state_dict["weight"] = replacement
        model.weight.grad_dtype = torch.float64
        model.load_state_dict(state_dict, assign=assign)
        self.assertIs(model.weight, replacement if assign else previous)
        self._assert_parity(model, reference, check_override=grad_dtype != "default")
        for module in (model, reference):
            module.to(device=device)
        self._assert_parity(model, reference, check_override=grad_dtype != "default")
        dtype = reference.weight.dtype
        inp = torch.arange(8, device=device, dtype=dtype).view(2, 4) / 8
        for _ in range(2):
            for module in (model, reference):
                module.zero_grad(set_to_none=True)
                module(inp).sum().backward()
            self._assert_parity(
                model, reference, check_override=grad_dtype != "default"
            )
            for group in model._get_fsdp_state()._fsdp_param_groups:
                for param in group.fsdp_params:
                    self.assertEqual(
                        param._has_sharded_grad_dtype_override, grad_dtype != "default"
                    )


instantiate_device_type_tests(
    TestFullyShardConversion, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
