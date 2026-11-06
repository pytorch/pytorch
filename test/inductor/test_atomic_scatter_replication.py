# Owner(s): ["module: inductor"]

import re

import torch
import torch._inductor.config as inductor_config
from torch._inductor import utils
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import needs_fallback_due_to_atomic_add_limitations
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


REPLICA_ADD = ".view(-1).add_(workspace_"
RUNTIME_BRANCH = re.compile(r"^\s+if .* <= .*:$", re.M)


def scatter_add(dest_rows, idx, src):
    out = torch.zeros(dest_rows, src.shape[1], dtype=src.dtype, device=src.device)
    return out.index_put_((idx,), src * 2, accumulate=True)


@inductor_config.patch(
    {
        "triton.replicate_atomic_scatters": True,
        "partitioned_scatter_enabled": False,
    }
)
@instantiate_parametrized_tests
class AtomicScatterReplicationTest(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _inputs(self, dest_rows, rows, cols=64, dtype=torch.float32):
        idx = torch.randint(0, dest_rows, (rows,), device=GPU_TYPE)
        src = torch.randn(rows, cols, dtype=dtype, device=GPU_TYPE)
        return idx, src

    def _check(self, actual, idx, src, dest_rows, tol):
        expected = scatter_add(dest_rows, idx.cpu(), src.cpu().double())
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=tol, rtol=tol
        )

    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_contended_scatter_is_replicated(self, dtype):
        if needs_fallback_due_to_atomic_add_limitations(dtype):
            self.skipTest(f"no Triton atomic_add for {dtype} on this device")
        dest_rows = 50
        rows, tol = (1 << 16, 1e-3) if dtype == torch.float32 else (1 << 14, 2.0)
        idx, src = self._inputs(dest_rows, rows, dtype=dtype)

        actual, (code,) = utils.run_and_get_code(
            torch.compile(scatter_add), dest_rows, idx, src
        )

        # Static sizes that pass both limits need no runtime branch.
        self.assertIn(REPLICA_ADD, code)
        self.assertIn("tl.program_id(0) % 16", code)
        self.assertNotRegex(code, RUNTIME_BRANCH)
        self._check(actual, idx, src, dest_rows, tol)

    def test_uncontended_scatter_is_not_replicated(self):
        dest_rows = 1 << 14
        idx, src = self._inputs(dest_rows, 1 << 14)

        actual, (code,) = utils.run_and_get_code(
            torch.compile(scatter_add), dest_rows, idx, src
        )

        self.assertNotIn(REPLICA_ADD, code)
        self.assertNotIn("tl.program_id(0) % 16", code)
        self._check(actual, idx, src, dest_rows, 1e-3)

    @inductor_config.patch({"triton.atomic_scatter_replica_max_mib": 1})
    def test_replicas_over_budget_are_not_used(self):
        # 16 replicas of 8192 x 64 fp32 need 32 MiB.
        dest_rows = 8192
        idx, src = self._inputs(dest_rows, 1 << 20)

        actual, (code,) = utils.run_and_get_code(
            torch.compile(scatter_add), dest_rows, idx, src
        )

        self.assertNotIn(REPLICA_ADD, code)
        self._check(actual, idx, src, dest_rows, 1e-3)

    def test_dynamic_sizes_pick_the_kernel_at_runtime(self):
        dest_rows = 64
        compiled = torch.compile(scatter_add, dynamic=True)

        idx, src = self._inputs(dest_rows, 1 << 16)
        actual, (code,) = utils.run_and_get_code(compiled, dest_rows, idx, src)
        self.assertIn(REPLICA_ADD, code)
        self.assertRegex(code, RUNTIME_BRANCH)
        self._check(actual, idx, src, dest_rows, 1e-3)

        # The same graph on both sides of the contention limit.
        for rows in (64, 1 << 12, 1 << 17):
            idx, src = self._inputs(dest_rows, rows)
            self._check(compiled(dest_rows, idx, src), idx, src, dest_rows, 1e-3)

    def test_two_destinations(self):
        def f(idx, src):
            a = torch.zeros(32, 64, device=src.device)
            b = torch.zeros(48, 64, device=src.device)
            a.index_put_((idx % 32,), src, accumulate=True)
            b.index_put_((idx % 48,), src * 3, accumulate=True)
            return a, b

        idx, src = self._inputs(1 << 20, 1 << 16)
        actual, (code,) = utils.run_and_get_code(torch.compile(f), idx, src)

        self.assertEqual(code.count(REPLICA_ADD), 2)
        self.assertEqual(code.count("workspace_0 = empty_strided"), 1)
        idx_c, src_c = idx.cpu(), src.cpu().double()
        expected = (
            torch.zeros(32, 64, dtype=torch.double).index_put_(
                (idx_c % 32,), src_c, accumulate=True
            ),
            torch.zeros(48, 64, dtype=torch.double).index_put_(
                (idx_c % 48,), src_c * 3, accumulate=True
            ),
        )
        for act, exp in zip(actual, expected):
            torch.testing.assert_close(act.cpu().double(), exp, atol=1e-3, rtol=1e-4)

    @inductor_config.patch({"triton.replicate_atomic_scatters": False})
    def test_off_by_default(self):
        dest_rows = 50
        idx, src = self._inputs(dest_rows, 1 << 16)

        actual, (code,) = utils.run_and_get_code(
            torch.compile(scatter_add), dest_rows, idx, src
        )

        self.assertNotIn(REPLICA_ADD, code)
        self.assertNotIn("tl.program_id(0) % 16", code)
        self._check(actual, idx, src, dest_rows, 1e-3)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
