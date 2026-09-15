# Owner(s): ["oncall: distributed"]

import copy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Partial
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
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
    @parametrize("initialized", [False, True])
    def test_to_matches_plain_module(
        self, device, grad_dtype, existing_grad, initialized
    ):
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
        if initialized:
            # Finish backward before conversion so no live graph uses the old leaf.
            for module in (model, reference):
                module(inp).sum().backward()
                module.zero_grad(set_to_none=True)
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

    @parametrize("compute_dtype", [torch.float32, torch.bfloat16])
    def test_initialized_conversion_preserves_mixed_precision(
        self, device, compute_dtype
    ):
        model = nn.Linear(4, 4, bias=False, device=device)
        with torch.no_grad():
            model.weight.fill_(0.25)
        reference = copy.deepcopy(model).to(compute_dtype)
        reference.weight.grad_dtype = torch.float32
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=compute_dtype, reduce_dtype=torch.float32
            ),
        )
        inp = torch.arange(8, device=device, dtype=torch.float32).view(2, 4) / 8
        model(inp).sum().backward()
        model.zero_grad(set_to_none=True)

        for storage_dtype in (torch.bfloat16, torch.float32):
            model.to(storage_dtype)
            self.assertEqual(model.weight.dtype, storage_dtype)
            self.assertEqual(model.weight.grad_dtype, storage_dtype)
            self.assertFalse(model.weight._has_grad_dtype_override)
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                actual = model(inp)
                expected = reference(inp.to(compute_dtype))
                self.assertEqual(actual.dtype, compute_dtype)
                self.assertEqual(actual, expected)
                actual.sum().backward()
                expected.sum().backward()
                del actual, expected
                self.assertEqual(model.weight.grad.dtype, storage_dtype)
                self.assertEqual(
                    model.weight.grad.full_tensor(),
                    reference.weight.grad.to(storage_dtype),
                )

    def test_initialized_grouped_conversion(self, device):
        reference = nn.Sequential(
            nn.Linear(4, 4, bias=False, device=device),
            nn.Linear(4, 4, bias=False, device=device),
        )
        with torch.no_grad():
            for param in reference.parameters():
                param.fill_(0.25)
        model = copy.deepcopy(reference)
        fully_shard(list(model), mesh=self.mesh)
        fully_shard(model, mesh=self.mesh)
        inp = torch.ones(2, 4, device=device)
        for module in (model, reference):
            module(inp).sum().backward()
            module.zero_grad(set_to_none=True)
            module.to(torch.bfloat16)
        self._assert_parity(model, reference, check_override=True)
        for module in (model, reference):
            module(inp.to(torch.bfloat16)).sum().backward()
        self._assert_parity(model, reference)

    def test_convert_pending_gradient(self, device):
        reference = nn.Linear(4, 4, bias=False, device=device)
        with torch.no_grad():
            reference.weight.fill_(0.25)
        model = copy.deepcopy(reference)
        reference.weight.grad_dtype = torch.float32
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
        )
        fsdp_param = model._get_fsdp_state()._fsdp_param_groups[0].fsdp_params[0]
        inp = torch.arange(8, device=device, dtype=torch.float32).view(2, 4) / 8
        model.set_requires_gradient_sync(False)
        for module in (model, reference):
            module(inp).sum().backward()
            module.to(torch.bfloat16)
        self.assertTrue(fsdp_param._grad_is_partial)
        self.assertFalse(fsdp_param._has_sharded_grad_dtype_override)
        self.assertEqual(model.weight.grad.placements, (Partial("avg"),))
        self.assertEqual(model.weight.grad.dtype, torch.bfloat16)
        self.assertEqual(model.weight.grad.full_tensor(), reference.weight.grad)

        model.set_requires_gradient_sync(True)
        for module in (model, reference):
            module(inp.to(torch.bfloat16)).sum().backward()
        self.assertFalse(fsdp_param._grad_is_partial)
        self.assertEqual(model.weight.grad.dtype, reference.weight.grad.dtype)
        self.assertEqual(model.weight.grad.full_tensor(), reference.weight.grad)
        for module in (model, reference):
            torch.optim.SGD(module.parameters(), lr=0.01).step()
        self.assertEqual(model.weight.full_tensor(), reference.weight)

    def test_convert_pending_gradient_unused_next_backward(self, device):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(4, 4, bias=False, device=device)
                self.second = nn.Linear(4, 4, bias=False, device=device)

            def forward(self, inp, use_first):
                return self.first(inp) if use_first else self.second(inp)

        reference = Model()
        with torch.no_grad():
            for param in reference.parameters():
                param.fill_(0.25)
        model = copy.deepcopy(reference)
        for module in (model.first, model.second, model):
            fully_shard(
                module,
                mesh=self.mesh,
                mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32),
            )
        inp = torch.ones(2, 4, device=device)
        model.set_requires_gradient_sync(False)
        for module in (model, reference):
            module(inp, use_first=True).sum().backward()
            module.to(torch.bfloat16)
        model.set_requires_gradient_sync(True)
        for module in (model, reference):
            module(inp.to(torch.bfloat16), use_first=False).sum().backward()
        self._assert_parity(model, reference)
        first_param = model.first._get_fsdp_state()._fsdp_param_groups[0].fsdp_params[0]
        self.assertFalse(first_param._grad_is_partial)

    @parametrize("explicit_reduce_dtype", [False, True])
    def test_convert_pending_gradient_with_fresh_gradient(
        self, device, explicit_reduce_dtype
    ):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Parameter(torch.full((4,), 0.25, device=device))
                self.second = nn.Parameter(torch.full((4,), 0.25, device=device))

            def forward(self, inp, use_first):
                return ((self.first if use_first else self.second) * inp).sum()

        model = Model()
        reference = copy.deepcopy(model)
        for param in reference.parameters():
            param.grad_dtype = torch.float32
        fully_shard(
            model,
            mesh=self.mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=None if explicit_reduce_dtype else torch.float32,
                reduce_dtype=torch.float32 if explicit_reduce_dtype else None,
            ),
        )
        inp = torch.arange(4, device=device, dtype=torch.float32) / 8
        model.set_requires_gradient_sync(False)
        for module in (model, reference):
            module(inp, use_first=True).backward()
            module.to(torch.bfloat16)
        self.assertEqual(model.first.grad.dtype, torch.bfloat16)
        self.assertIsNone(model.second.grad)

        compute_dtype = torch.bfloat16 if explicit_reduce_dtype else torch.float32
        # Match the FSDP compute policy after converting the existing gradient.
        reference.to(compute_dtype)
        inp = inp.to(compute_dtype)
        model.set_requires_gradient_sync(True)
        actual = model(inp, use_first=False)
        expected = reference(inp, use_first=False)
        self.assertEqual(actual.dtype, compute_dtype)
        self.assertEqual(actual, expected)
        actual.backward()
        expected.backward()
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            self.assertEqual(param.dtype, torch.bfloat16)
            self.assertEqual(param.grad.dtype, torch.bfloat16)
            self.assertEqual(param.grad.full_tensor(), ref_param.grad.to(param.dtype))
        for param in model._get_fsdp_state()._fsdp_param_groups[0].fsdp_params:
            self.assertFalse(param._grad_is_partial)


class TestFullyShardDistributedConversion(FSDPTest):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_conversion_with_pending_all_reduce(self):
        mesh = init_device_mesh(
            get_devtype().type, (2, 1), mesh_dim_names=("replicate", "shard")
        )
        self.run_subtests(
            {
                "config": [
                    (torch.float32, torch.float32, "root"),
                    (torch.float32, torch.float32, "nested"),
                    (torch.float32, torch.float32, "grouped"),
                    (torch.bfloat16, torch.float32, "root"),
                    (torch.float32, torch.bfloat16, "root"),
                ]
            },
            self._test_conversion_with_pending_all_reduce,
            mesh,
        )

    def _test_conversion_with_pending_all_reduce(self, mesh, config):
        param_dtype, carry_dtype, topology = config
        device = get_devtype()
        model = nn.Sequential(
            nn.Linear(4, 4, bias=False, device=device, dtype=param_dtype),
            nn.Linear(4, 4, bias=False, device=device, dtype=param_dtype),
        )
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.25)
        reference = copy.deepcopy(model).to(torch.bfloat16)
        for param in reference.parameters():
            param.grad_dtype = carry_dtype
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=carry_dtype
        )
        if topology == "nested":
            fully_shard(model[1], mesh=mesh, mp_policy=policy)
        elif topology == "grouped":
            fully_shard(list(model), mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        groups = list(
            {
                id(group): group
                for module in model.modules()
                if hasattr(module, "_get_fsdp_state")
                for group in module._get_fsdp_state()._fsdp_param_groups
            }.values()
        )

        def backward(step):
            inputs = [
                torch.arange(8, device=device, dtype=torch.bfloat16).view(2, 4) / 8
                + rank / 2
                + step / 4
                for rank in range(self.world_size)
            ]
            ref_loss = sum(reference(inp).float().sum() for inp in inputs)
            (ref_loss / self.world_size).backward()
            model(inputs[self.rank]).float().sum().backward()

        def snapshot():
            state = []
            for group in groups:
                carry = group._partial_reduce_output
                state.append(
                    (id(carry), carry.dtype, carry.clone())
                    if carry is not None
                    else None
                )
                for param in group.fsdp_params:
                    public = param.sharded_param
                    grad = public.grad
                    state.append(
                        (
                            id(public),
                            public.dtype,
                            public.grad_dtype,
                            public._has_grad_dtype_override,
                            public._local_tensor.clone(),
                            (id(grad), grad.dtype, grad._local_tensor.clone())
                            if grad is not None
                            else None,
                            id(public._spec),
                            id(param._pending_grad_spec),
                            id(param._unsharded_param),
                            tuple(id(buffer) for buffer in param.all_gather_outputs),
                        )
                    )
            return state

        backward(0)
        # Keep existing public gradients as well as a private all-reduce carry.
        partial_module = model[1] if topology == "nested" else model
        partial_module.set_requires_all_reduce(False)
        backward(1)
        torch.get_device_module(device).synchronize()
        carries = [g._partial_reduce_output for g in groups]
        self.assertTrue(any(carry is not None for carry in carries))
        for carry in carries:
            if carry is not None:
                self.assertEqual(carry.dtype, carry_dtype)
        # No parameters are owned directly by the Sequential, even when its
        # FSDP group manages the parameters of ordinary unwrapped children.
        model._apply(lambda tensor: tensor.to(torch.bfloat16), recurse=False)
        self.assertTrue(all(p.dtype == param_dtype for p in model.parameters()))
        if carry_dtype != torch.bfloat16:
            before = snapshot()
            with self.assertRaisesRegex(
                RuntimeError,
                "Cannot change the dtype of gradients awaiting all-reduce",
            ):
                model.to(torch.bfloat16)
            self.assertEqual(snapshot(), before)
        else:
            model.to(torch.bfloat16)
            self.assertTrue(all(p.dtype == torch.bfloat16 for p in model.parameters()))

        # Preserving the carry dtype is supported, including parameter no-ops.
        model.to(carry_dtype)
        for group, carry in zip(groups, carries):
            self.assertIs(group._partial_reduce_output, carry)
        model.set_requires_all_reduce(True)
        backward(2)
        for group in groups:
            self.assertIsNone(group._partial_reduce_output)
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            self.assertEqual(param.grad.full_tensor(), ref_param.grad.to(param.dtype))

        for module in (model, reference):
            module.to(torch.bfloat16)
        backward(3)
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            self.assertEqual(param.grad.full_tensor(), ref_param.grad)
        for module in (model, reference):
            torch.optim.SGD(module.parameters(), lr=0.01).step()
        for param, ref_param in zip(model.parameters(), reference.parameters()):
            self.assertEqual(param.full_tensor(), ref_param)

    @skip_if_lt_x_gpu(2)
    def test_initialized_conversion_with_existing_gradient(self):
        device = get_devtype()
        reference = nn.Linear(4, 4, bias=False, device=device)
        with torch.no_grad():
            reference.weight.fill_(0.25)
        model = copy.deepcopy(reference)
        for module in (model, reference):
            module.weight.grad_dtype = torch.float32
        fully_shard(model, mesh=init_device_mesh(device.type, (self.world_size,)))
        inputs = [
            torch.arange(8, device=device, dtype=torch.float32).view(2, 4) / 8
            + rank / 2
            for rank in range(self.world_size)
        ]
        reference_loss = sum(reference(inp).sum() for inp in inputs) / self.world_size
        reference_loss.backward()
        model(inputs[self.rank]).sum().backward()
        del reference_loss
        self.assertEqual(model.weight.grad.full_tensor(), reference.weight.grad)

        for module in (model, reference):
            module.to(torch.bfloat16)
        self.assertEqual(model.weight.grad_dtype, torch.float32)
        self.assertTrue(model.weight._has_grad_dtype_override)
        self.assertEqual(model.weight.grad.dtype, torch.bfloat16)
        self.assertEqual(model.weight.grad.full_tensor(), reference.weight.grad)
        inputs = [inp.to(torch.bfloat16) for inp in inputs]
        reference_loss = sum(reference(inp).sum() for inp in inputs) / self.world_size
        reference_loss.backward()
        actual = model(inputs[self.rank])
        self.assertEqual(actual, reference(inputs[self.rank]))
        actual.sum().backward()
        self.assertEqual(model.weight.grad.dtype, reference.weight.grad.dtype)
        self.assertEqual(model.weight.grad.full_tensor(), reference.weight.grad)
        for module in (model, reference):
            torch.optim.SGD(module.parameters(), lr=0.01).step()
        self.assertEqual(model.weight.full_tensor(), reference.weight)


instantiate_device_type_tests(
    TestFullyShardConversion, globals(), only_for=("cpu", "cuda")
)

if __name__ == "__main__":
    run_tests()
