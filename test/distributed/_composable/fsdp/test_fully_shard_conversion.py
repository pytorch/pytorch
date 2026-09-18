# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Partial
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
                    actual._is_grad_dtype_explicit, expected._is_grad_dtype_explicit
                )
            self.assertEqual(actual.full_tensor(), expected)
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                self.assertIsInstance(actual.grad, DTensor)
                self.assertEqual(actual.grad.dtype, expected.grad.dtype)
                self.assertEqual(actual.grad.full_tensor(), expected.grad)

    def _assert_conversion_requires_clear(self, model, target):
        saved = [
            (
                param,
                param.detach().clone(),
                param.grad,
                None if param.grad is None else param.grad.clone(),
                param.grad_dtype,
                param._is_grad_dtype_explicit,
            )
            for param in model.parameters()
        ]
        with self.assertRaisesRegex(RuntimeError, r"zero_grad\(set_to_none=True\)"):
            model.to(target)
        for actual, (param, value, grad, grad_value, grad_dtype, override) in zip(
            model.parameters(), saved
        ):
            self.assertIs(actual, param)
            self.assertEqual(actual, value)
            self.assertEqual(actual.dtype, value.dtype)
            self.assertIs(actual.grad, grad)
            if grad is not None:
                self.assertEqual(grad, grad_value)
                self.assertEqual(grad.dtype, grad_value.dtype)
            self.assertEqual(actual.grad_dtype, grad_dtype)
            self.assertEqual(actual._is_grad_dtype_explicit, override)

    @parametrize("grad_dtype", ["default", torch.float32, None])
    @parametrize("existing_grad", [False, True])
    def test_to_matches_plain_module(self, device, grad_dtype, existing_grad):
        reference = nn.Linear(4, 4, device=device)
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
        inp = torch.arange(8, device=device, dtype=torch.float32).view(2, 4) / 8
        if existing_grad:
            for module in (model, reference):
                for param in module.parameters():
                    if param.requires_grad:
                        param.grad = torch.ones_like(param)

        if existing_grad and grad_dtype == torch.float32:
            self._assert_conversion_requires_clear(model, torch.bfloat16)
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)

        for module in (model, reference):
            module.to(torch.bfloat16)
        self._assert_parity(model, reference, check_override=True)
        for group in model._get_fsdp_state()._fsdp_param_groups:
            for param in group.fsdp_params:
                self.assertEqual(
                    param._has_sharded_grad_dtype_override, grad_dtype != "default"
                )
        inp = inp.to(torch.bfloat16)
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

    def test_same_dtype_to_after_backward(self, device):
        reference = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
        model = copy.deepcopy(reference)
        for module in (model, reference):
            module.weight.grad_dtype = torch.float32
        fully_shard(model, mesh=self.mesh)
        inp = torch.ones(2, 4, device=device, dtype=torch.bfloat16)
        for module in (model, reference):
            module(inp).sum().backward()
        self._assert_conversion_requires_clear(model, torch.bfloat16)
        model.zero_grad(set_to_none=False)
        self._assert_conversion_requires_clear(model, torch.bfloat16)
        for module in (model, reference):
            module.zero_grad(set_to_none=True)
            module.to(torch.bfloat16)
        self._assert_parity(model, reference, check_override=True)
        for module in (model, reference):
            module(inp).sum().backward()
        self._assert_parity(model, reference, check_override=True)

    @parametrize("grouped", [False, True])
    def test_nested_conversion_requires_clear(self, device, grouped):
        model = nn.Sequential(
            nn.Linear(4, 4, device=device), nn.Linear(4, 4, device=device)
        )
        model[1].weight.grad_dtype = torch.float32
        if grouped:
            fully_shard(list(model), mesh=self.mesh)
        else:
            for layer in model:
                fully_shard(layer, mesh=self.mesh)
        fully_shard(model, mesh=self.mesh)
        model[1].weight.grad = torch.ones_like(model[1].weight)
        self._assert_conversion_requires_clear(model, torch.bfloat16)

        model.zero_grad(set_to_none=True)
        model.to(torch.bfloat16)
        model(torch.ones(2, 4, device=device, dtype=torch.bfloat16)).sum().backward()
        for param in model.parameters():
            self.assertEqual(param.dtype, torch.bfloat16)
            self.assertIsNotNone(param.grad)
            self.assertEqual(param.grad.dtype, param.grad_dtype)

    def test_shared_parameter_conversion_requires_clear(self, device):
        first = nn.Linear(4, 4, bias=False, device=device)
        second = nn.Linear(4, 4, bias=False, device=device)
        second.weight = first.weight
        first.weight.grad_dtype = torch.float32
        fully_shard([first, second], mesh=self.mesh)
        first.weight.grad = torch.ones_like(first.weight)
        self._assert_conversion_requires_clear(second, torch.bfloat16)
        self.assertIs(first.weight, second.weight)
        self.assertEqual(first.weight.dtype, torch.float32)

    @parametrize("grad_dtype", ["default", None])
    def test_pending_grad_conversion_requires_clear(self, device, grad_dtype):
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
        self.assertEqual(model.weight.grad.placements, (Partial("avg"),))
        storage_size = param.unsharded_param.untyped_storage().nbytes()
        pending_spec = param._pending_grad_spec
        self._assert_conversion_requires_clear(model, torch.bfloat16)
        self.assertTrue(group.is_unsharded)
        self.assertEqual(param.unsharded_param.untyped_storage().nbytes(), storage_size)
        self.assertIs(param._pending_grad_spec, pending_spec)

        model.zero_grad(set_to_none=True)
        model.to(torch.bfloat16)
        model.set_requires_gradient_sync(True)
        model(inp).sum().backward()
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertIsNotNone(model.weight.grad)

    @parametrize("grad_dtype", [torch.float32, None])
    def test_device_only_conversion_requires_clear(self, device, grad_dtype):
        reference = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
        model = copy.deepcopy(reference)
        for module in (model, reference):
            module.weight.grad_dtype = grad_dtype
        fully_shard(model, mesh=self.mesh)
        for module in (model, reference):
            module.weight.grad = torch.ones_like(module.weight, dtype=torch.float32)
        self._assert_conversion_requires_clear(model, torch.device(device))
        for module in (model, reference):
            module.zero_grad(set_to_none=True)
            module.to(device=device)
        self._assert_parity(model, reference, check_override=True)
        inp = torch.ones(2, 4, device=device, dtype=torch.bfloat16)
        for module in (model, reference):
            module(inp).sum().backward()
        self._assert_parity(model, reference, check_override=True)

    @parametrize("grad_dtype", ["default", torch.float32, torch.bfloat16, None])
    @parametrize("convert", [False, True])
    def test_grad_dtype_change_before_first_forward(self, device, grad_dtype, convert):
        reference = nn.Linear(4, 4, bias=False, device=device)
        model = copy.deepcopy(reference)
        if grad_dtype is None:
            for module in (model, reference):
                module.weight.grad_dtype = torch.float32
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(
                reduce_dtype=None if grad_dtype == "default" else grad_dtype
            ),
        )
        for module in (model, reference):
            if grad_dtype != "default":
                module.weight.grad_dtype = grad_dtype
            if convert:
                module.to(torch.bfloat16)
        self._assert_parity(model, reference, check_override=True)
        dtype = reference.weight.dtype
        inp = torch.arange(8, device=device, dtype=dtype).view(2, 4) / 8
        for _ in range(2):
            for module in (model, reference):
                module.zero_grad(set_to_none=True)
                module(inp).sum().backward()
            self._assert_parity(model, reference, check_override=True)


instantiate_device_type_tests(
    TestFullyShardConversion, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
