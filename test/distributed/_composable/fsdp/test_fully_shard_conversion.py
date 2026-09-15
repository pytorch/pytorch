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
            self.assertEqual(actual.full_tensor(), expected)
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                self.assertIsInstance(actual.grad, DTensor)
                self.assertEqual(actual.grad.dtype, expected.grad.dtype)
                self.assertEqual(actual.grad.full_tensor(), expected.grad)

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
            module.to(torch.bfloat16)
        self._assert_parity(model, reference, check_override=True)
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
