# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard._fsdp_grad import FSDPGrad
from torch.distributed.tensor import DTensor, Partial, Shard
from torch.distributed.tensor.debug import CommDebugMode
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import parametrize, run_tests


class TestFSDPGrad(MultiProcContinuousTest):
    world_size = 2

    @classmethod
    def backend_str(cls):
        return "gloo"

    def _make_grad(self, device):
        mesh = init_device_mesh(torch.device(device).type, (self.world_size,))
        reduced = DTensor.from_local(
            torch.full((2, 4), 256.0, device=device), mesh, (Shard(0),)
        )
        unreduced = DTensor.from_local(
            torch.full((4, 4), self.rank + 1, dtype=torch.bfloat16, device=device),
            mesh,
            (Partial("sum"),),
        )
        return FSDPGrad(reduced, unreduced, None, reduced._spec, self._materialize)

    def _materialize(self, grad):
        result = grad.reduced.clone()
        for component in (grad.unreduced, grad.partial):
            if component is not None:
                contribution = component.redistribute(
                    placements=grad.sharded_spec.placements
                )
                result.add_(contribution.to(dtype=grad.dtype))
        return result

    @parametrize("foreach", [False, True])
    def test_component_mutations(self, device, foreach):
        grad = self._make_grad(device)
        param = torch.nn.Parameter(torch.zeros_like(grad.reduced))
        param.grad = grad
        self.assertIs(param.grad, grad)
        self.assertEqual(grad.dtype, torch.float32)
        self.assertEqual(grad.unreduced.dtype, torch.bfloat16)
        with CommDebugMode() as comm_mode:
            local = grad.to_local()
            self.assertEqual(local.shape, (2, 4))
            local.view(-1).mul_(2)
            if foreach:
                torch._foreach_div_([grad], 2)
                torch._foreach_mul_([grad], torch.tensor(0.5, device=device))
            else:
                grad.div_(2).mul_(0.5)
            grad.detach().to_local().t().mul_(2)
            grad.data.to_local().data.mul_(2).div_(2)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        self.assertEqual(
            grad.reduced.to_local(), torch.full((2, 4), 256.0, device=device)
        )
        self.assertEqual(
            grad.unreduced.to_local(),
            torch.full((4, 4), self.rank + 1, dtype=torch.bfloat16, device=device),
        )
        with CommDebugMode() as comm_mode:
            optimizer = torch.optim.SGD([param], lr=0.1, foreach=foreach)
            optimizer.zero_grad(set_to_none=False)
        self.assertEqual(comm_mode.get_total_counts(), 0)
        for component in grad.local_components().values():
            self.assertEqual(component.count_nonzero(), 0)

    def test_clone_and_snapshot_independence(self, device):
        grad = self._make_grad(device)
        cloned = grad.clone()
        self.assertIsInstance(cloned, FSDPGrad)
        cloned.mul_(2)
        expected = torch.full((2, 4), 259.0, device=device)
        self.assertEqual(grad.materialize().to_local(), expected)
        self.assertEqual(cloned.materialize().to_local(), expected * 2)
        for snapshot in (grad.materialize(), grad.redistribute(placements=(Shard(0),))):
            self.assertIs(type(snapshot), DTensor)
            snapshot.zero_()
        grad.full_tensor().zero_()
        grad.to_local().clone().zero_()
        self.assertEqual(grad.materialize().to_local(), expected)

    def test_mixed_dtensor_dispatch(self, device):
        grad = self._make_grad(device)
        expected = torch.full((2, 4), 259.0, device=device)
        param = torch.zeros_like(grad.reduced)
        for operation in (
            lambda: param.add_(grad),
            lambda: torch._foreach_add_([param], [grad]),
            lambda: param.lerp_(grad, 0.5),
            lambda: param.addcmul_(grad, grad),
        ):
            with self.assertRaisesRegex(RuntimeError, "synchronize_gradients"):
                operation()
        snapshot = grad.materialize()
        param.add_(snapshot)
        self.assertEqual(param.to_local(), expected)
        torch._foreach_add_([param], [snapshot])
        self.assertEqual(param.to_local(), expected * 2)
        param.lerp_(snapshot, 0.5)
        self.assertEqual(param.to_local(), expected * 1.5)
        param.zero_().addcmul_(snapshot, snapshot)
        self.assertEqual(param.to_local(), expected.square())
        self.assertEqual(grad.square().to_local(), expected.square())

    @parametrize("foreach", [False, True])
    def test_clip(self, device, foreach):
        grad = self._make_grad(device)
        param = torch.nn.Parameter(torch.zeros_like(grad.reduced))
        param.grad = grad
        reference = torch.nn.Parameter(torch.zeros((4, 4), device=device))
        reference.grad = torch.full((4, 4), 259.0, device=device)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference, 1, foreach=foreach)
        norm = torch.nn.utils.clip_grad_norm_(param, 1, foreach=foreach)
        self.assertEqual(norm.full_tensor(), expected_norm)
        self.assertIs(param.grad, grad)
        self.assertEqual(grad.unreduced.dtype, torch.bfloat16)
        scale = (1 / (expected_norm + 1e-6)).clamp(max=1)
        self.assertEqual(
            grad.reduced.to_local(), torch.full((2, 4), 256.0, device=device) * scale
        )
        self.assertEqual(
            grad.unreduced.to_local(),
            torch.full((4, 4), self.rank + 1, dtype=torch.bfloat16, device=device).mul_(
                scale
            ),
        )

    @parametrize("nonfinite", [False, True])
    def test_amp_unscale(self, device, nonfinite):
        grad = self._make_grad(device)
        if nonfinite and self.rank == 1:
            grad.reduced.to_local()[0, 0] = float("inf")
        found_inf = torch.zeros((), device=device)
        inv_scale = torch.tensor(0.5, device=device)
        torch._amp_foreach_non_finite_check_and_unscale_([grad], found_inf, inv_scale)
        self.assertEqual(found_inf, float(nonfinite))
        self.assertEqual(
            grad.unreduced.to_local(),
            torch.full(
                (4, 4), (self.rank + 1) / 2, dtype=torch.bfloat16, device=device
            ),
        )
        self.assertEqual(grad.unreduced.dtype, torch.bfloat16)

    def test_rejected_writes_and_exports(self, device):
        grad = self._make_grad(device)
        with CommDebugMode() as comm_mode:
            for operation in (
                lambda: grad.clamp_(-1, 1),
                lambda: grad.copy_(grad.reduced),
                lambda: grad.to_local()[0].zero_(),
                lambda: grad.to_local().numpy(),
                lambda: grad.to_local().untyped_storage(),
                lambda: grad.untyped_storage(),
                lambda: grad.view(-1),
                lambda: grad.mul_(torch.ones((2, 4), device=device)),
            ):
                with self.assertRaisesRegex(RuntimeError, "synchronize_gradients"):
                    operation()
            with self.assertRaisesRegex(RuntimeError, "synchronize_gradients"):
                grad.data = grad.reduced
        self.assertEqual(comm_mode.get_total_counts(), 0)


instantiate_device_type_tests(TestFSDPGrad, globals(), only_for=("cpu",))

if __name__ == "__main__":
    run_tests()
