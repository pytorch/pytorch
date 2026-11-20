# Owner(s): ["module: inductor"]

import re
from unittest import mock

import torch
import torch._inductor.codegen.common as common
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor import metrics, utils
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.codegen.wrapper_fxir import WrapperFxCodegen
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import needs_fallback_due_to_atomic_add_limitations
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import HardwareClassification
from torch.testing._internal.inductor_utils import HAS_GPU


REPLICA_ADD = ".view(-1).add_(reinterpret_tensor(workspace_"
RUNTIME_BRANCH = re.compile(r"^\s+if .* <= .*:$", re.MULTILINE)
KERNEL_LAUNCH = re.compile(r"^\s+(\w+)\.run\(", re.MULTILINE)
SKIPPED = "atomic_scatter_replica_skipped_"


def scatter_add(dest_rows, idx, src):
    out = torch.zeros(dest_rows, src.shape[1], dtype=src.dtype, device=src.device)
    return out.index_put_((idx,), src * 2, accumulate=True)


def skips():
    """The atomic_scatter_replica_skipped_<reason> counters, by reason."""
    return {
        name.removeprefix(SKIPPED): count
        for name, count in counters["inductor"].items()
        if name.startswith(SKIPPED)
    }


def branch_kernels(code):
    """Names of the kernels launched in the `if` and the `else` branch."""
    branch = RUNTIME_BRANCH.search(code)
    replicated = KERNEL_LAUNCH.search(code, branch.end())
    plain = KERNEL_LAUNCH.search(code, code.index("else:", replicated.end()))
    return replicated.group(1), plain.group(1)


def launched_kernels(fn, *args):
    """The result of fn(*args) and the names of the kernels it launched."""
    with torch.profiler.profile() as prof:
        out = fn(*args)
    return out, {event.name for event in prof.events()}


class AtomicScatterReplicationTest(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        # A class decorator would hide the test methods from
        # instantiate_device_type_tests.
        patch = inductor_config.patch(
            {
                "triton.atomic_scatter_replication": True,
                "partitioned_scatter_enabled": False,
            }
        )
        patch.__enter__()
        self.addCleanup(patch.__exit__, None, None, None)

    def _inputs(self, device, dest_rows, rows, cols=64, dtype=torch.float32):
        idx = torch.randint(0, dest_rows, (rows,), device=device)
        src = torch.randn(rows, cols, dtype=dtype, device=device)
        return idx, src

    def _expected(self, idx, src, dest_rows):
        return scatter_add(dest_rows, idx.cpu(), src.cpu().double())

    def _check(self, actual, idx, src, dest_rows):
        expected = self._expected(idx, src, dest_rows)
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=1e-3, rtol=1e-4
        )

    def _compile(self, *args, fn=scatter_add, **kwargs):
        return utils.run_and_get_code(torch.compile(fn, **kwargs), *args)

    def test_contended_scatter_is_replicated(self, device):
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16)

        metrics.reset()
        actual, (code,) = self._compile(dest_rows, idx, src)
        kernel_count = metrics.generated_kernel_count
        with inductor_config.patch({"triton.atomic_scatter_replication": False}):
            torch._dynamo.reset()
            metrics.reset()
            self._compile(dest_rows, idx, src)

        # Static sizes that pass both limits need no runtime branch, and the
        # replicated kernel replaces the plain one.
        self.assertIn(REPLICA_ADD, code)
        self.assertIn("tl.program_id(0) % 16", code)
        self.assertNotRegex(code, RUNTIME_BRANCH)
        self.assertEqual(counters["inductor"]["atomic_scatter_replicated"], 1)
        self.assertEqual(
            counters["inductor"]["atomic_scatter_replica_runtime_branch"], 0
        )
        self.assertEqual(skips(), {})
        self.assertEqual(kernel_count, metrics.generated_kernel_count)
        self._check(actual, idx, src, dest_rows)

    @dtypes(torch.float16, torch.bfloat16)
    def test_contended_low_precision_scatter_is_more_accurate(self, device, dtype):
        if needs_fallback_due_to_atomic_add_limitations(dtype):
            self.skipTest(f"no Triton {dtype} atomic_add on this device")
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16, dtype=dtype)
        expected = self._expected(idx, src, dest_rows)

        actual, (code,) = self._compile(dest_rows, idx, src)
        with inductor_config.patch({"triton.atomic_scatter_replication": False}):
            torch._dynamo.reset()
            plain = torch.compile(scatter_add)(dest_rows, idx, src)

        # Each 16-bit atomic_add rounds; a replica receives 16x fewer of them.
        self.assertIn(REPLICA_ADD, code)
        self.assertIn(".sum(dim=0, dtype=torch.float32))", code)
        error = (actual.cpu().double() - expected).abs().max().item()
        plain_error = (plain.cpu().double() - expected).abs().max().item()
        self.assertLess(error, plain_error / 2)
        self.assertLess(error, 0.05 * expected.abs().max().item())

    @dtypes(torch.float16, torch.bfloat16)
    @inductor_config.patch({"triton.atomic_scatter_replica_fp32_accumulation": True})
    def test_low_precision_replicas_accumulate_in_fp32(self, device, dtype):
        if needs_fallback_due_to_atomic_add_limitations(dtype):
            self.skipTest(f"no Triton {dtype} atomic_add on this device")
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16, dtype=dtype)
        expected = self._expected(idx, src, dest_rows)

        actual, (code,) = self._compile(dest_rows, idx, src)

        self.assertIn(REPLICA_ADD, code)
        self.assertRegex(code, r"workspace_0 = empty_strided_\w+\(.*torch.float32\)")
        # Only the final add to the 16-bit destination rounds.
        error = (actual.cpu().double() - expected).abs().max().item()
        self.assertLess(error, 0.01 * expected.abs().max().item())

    def test_replicas_add_to_a_nonzero_destination(self, device):
        # 50 x 60 elements is not a multiple of 16, so the replica stride is
        # padded and the add-back must cut the padding off.
        def f(idx, src):
            out = torch.full((50, 60), 1.5, device=src.device)
            return out.index_put_((idx,), src, accumulate=True)

        idx, src = self._inputs(device, 50, 1 << 16, cols=60)
        actual, (code,) = self._compile(idx, src, fn=f)

        self.assertIn(REPLICA_ADD, code)
        expected = torch.full((50, 60), 1.5, dtype=torch.double).index_put_(
            (idx.cpu(),), src.cpu().double(), accumulate=True
        )
        torch.testing.assert_close(
            actual.cpu().double(), expected, atol=1e-3, rtol=1e-4
        )

    def test_uncontended_scatter_is_not_replicated(self, device):
        dest_rows = 1 << 14
        idx, src = self._inputs(device, dest_rows, 1 << 14)

        actual, (code,) = self._compile(dest_rows, idx, src)

        self.assertNotIn(REPLICA_ADD, code)
        self.assertNotIn("tl.program_id(0) % 16", code)
        self.assertEqual(skips(), {"low_contention": 1})
        self._check(actual, idx, src, dest_rows)

    @inductor_config.patch({"triton.atomic_scatter_replica_max_mib": 1})
    def test_replicas_over_budget_are_not_used(self, device):
        # 16 replicas of 512 x 64 fp32 need 2 MiB. With 256 stores per element,
        # the contention limit would pass.
        dest_rows = 512
        idx, src = self._inputs(device, dest_rows, 1 << 17)

        actual, (code,) = self._compile(dest_rows, idx, src)

        self.assertNotIn(REPLICA_ADD, code)
        self.assertEqual(skips(), {"memory_budget": 1})
        self._check(actual, idx, src, dest_rows)

    @inductor_config.patch({"triton.atomic_scatter_replicas": 1})
    def test_one_replica_disables_replication(self, device):
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16)

        actual, (code,) = self._compile(dest_rows, idx, src)

        self.assertNotIn(REPLICA_ADD, code)
        self.assertNotIn("tl.program_id(0) %", code)
        self.assertEqual(skips(), {"one_replica": 1})
        self._check(actual, idx, src, dest_rows)

    @inductor_config.patch({"partitioned_scatter_enabled": True})
    def test_partitioned_scatter_takes_precedence(self, device):
        # Contended enough that the partitioned copies would pass both limits.
        dest_rows = 4
        idx, src = self._inputs(device, dest_rows, 1 << 16)

        actual, (code,) = self._compile(dest_rows, idx, src)

        self.assertEqual(counters["inductor"]["partitioned_scatter_applied"], 1)
        self.assertNotIn(REPLICA_ADD, code)
        self.assertNotIn("tl.program_id(0) % 16", code)
        self.assertEqual(skips(), {"partitioned_scatter_enabled": 1})
        self._check(actual, idx, src, dest_rows)

    @inductor_config.patch({"triton.prefer_nd_tiling": True})
    def test_2d_tiled_kernel_is_not_replicated(self, device):
        # With a column-major source, the scatter kernel is tiled in 2D, where
        # the programs along y would share replica program_id(0) % 16.
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16)
        src = src.t().contiguous().t()

        actual, (code,) = self._compile(dest_rows, idx, src)

        self.assertIn("ynumel", code)
        self.assertNotIn(REPLICA_ADD, code)
        self.assertEqual(skips(), {"tiled": 1})
        self._check(actual, idx, src, dest_rows)

    def test_dynamic_sizes_pick_the_kernel_at_runtime(self, device):
        dest_rows = 64
        idx, src = self._inputs(device, dest_rows, 1 << 16)
        compiled = torch.compile(scatter_add, dynamic=True)

        actual, (code,) = utils.run_and_get_code(compiled, dest_rows, idx, src)

        self.assertIn(REPLICA_ADD, code)
        self.assertRegex(code, RUNTIME_BRANCH)
        self.assertEqual(
            counters["inductor"]["atomic_scatter_replica_runtime_branch"], 1
        )
        replicated, plain = branch_kernels(code)
        self._check(actual, idx, src, dest_rows)

        # One graph on both sides of the limit of 128 * 64 * 64 stores.
        with torch.compiler.set_stance("fail_on_recompile"):
            for rows, used, unused in (
                (64, plain, replicated),
                (1 << 12, plain, replicated),
                (1 << 17, replicated, plain),
            ):
                idx, src = self._inputs(device, dest_rows, rows)
                actual, launched = launched_kernels(compiled, dest_rows, idx, src)
                self.assertIn(used, launched)
                self.assertNotIn(unused, launched)
                self._check(actual, idx, src, dest_rows)

    def test_dynamic_destination_size_that_is_a_sum(self, device):
        # The replica strides print as sums: 64 + 64*ks0 for n + 1 rows of 64,
        # and 16 + 16*(ks0 // 16) for n + 1 elements.
        def rows(n, idx, src):
            out = torch.zeros(n + 1, 64, device=src.device)
            return out.index_put_((idx,), src, accumulate=True)

        def elements(n, idx, src):
            out = torch.zeros(n + 1, device=src.device)
            return out.index_add_(0, idx, src)

        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16)
        vals = src[:, 0].contiguous()
        for fn, source in ((rows, src), (elements, vals)):
            expected = torch.zeros(
                dest_rows, *source.shape[1:], dtype=torch.double
            ).index_put_((idx.cpu(),), source.cpu().double(), accumulate=True)
            torch._dynamo.reset()
            compiled = torch.compile(fn, dynamic=True)
            _, (code,) = utils.run_and_get_code(compiled, dest_rows - 1, idx, source)
            replicated, _ = branch_kernels(code)

            actual, launched = launched_kernels(compiled, dest_rows - 1, idx, source)

            self.assertIn(replicated, launched)
            torch.testing.assert_close(
                actual.cpu().double(), expected, atol=1e-3, rtol=1e-4
            )

    @inductor_config.patch({"memory_planning": True})
    def test_memory_planning_gets_no_runtime_branch(self, device):
        # MemoryPlanner does not see the branch, so a pool created or freed
        # inside it would be unbound on the other side.
        def f(idx, src, w):
            out = torch.zeros(64, 64, device=src.device)
            out.index_put_((idx,), src, accumulate=True)
            return out, out.sin() @ w

        w = torch.randn(64, 64, device=device)
        compiled = torch.compile(f, dynamic=True)
        with torch.no_grad():
            idx, src = self._inputs(device, 64, 1 << 16)
            _, (code,) = utils.run_and_get_code(compiled, idx, src, w)
            self.assertNotRegex(code, RUNTIME_BRANCH)
            self.assertNotIn(REPLICA_ADD, code)
            self.assertEqual(skips(), {"memory_planning": 1})
            for rows in (1 << 10, 1 << 16):
                idx, src = self._inputs(device, 64, rows)
                out, product = compiled(idx, src, w)
                expected = torch.zeros(64, 64, dtype=torch.double).index_put_(
                    (idx.cpu(),), src.cpu().double(), accumulate=True
                )
                torch.testing.assert_close(
                    out.cpu().double(), expected, atol=1e-3, rtol=1e-4
                )
                torch.testing.assert_close(
                    product.cpu().double(),
                    expected.sin() @ w.cpu().double(),
                    atol=1e-2,
                    rtol=1e-3,
                )

            # Without memory_planning, the same graph gets the branch.
            with inductor_config.patch({"memory_planning": False}):
                torch._dynamo.reset()
                _, (control,) = utils.run_and_get_code(
                    torch.compile(f, dynamic=True), idx, src, w
                )
            self.assertRegex(control, RUNTIME_BRANCH)

    def test_two_destinations_share_one_workspace(self, device):
        def f(idx, src):
            a = torch.zeros(32, 64, device=src.device)
            b = torch.zeros(48, 64, device=src.device)
            a.index_put_((idx % 32,), src, accumulate=True)
            b.index_put_((idx % 48,), src * 3, accumulate=True)
            return a, b

        idx, src = self._inputs(device, 1 << 20, 1 << 16)
        actual, (code,) = self._compile(idx, src, fn=f)

        self.assertEqual(code.count(REPLICA_ADD), 2)
        self.assertIn("workspace_0", code)
        self.assertNotIn("workspace_1", code)
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

    def test_destinations_of_two_dtypes(self, device):
        if needs_fallback_due_to_atomic_add_limitations(torch.float16):
            self.skipTest("no Triton float16 atomic_add on this device")

        def f(idx, src):
            a = torch.zeros(32, 64, device=src.device)
            b = torch.zeros(48, 64, dtype=torch.float16, device=src.device)
            a.index_put_((idx % 32,), src, accumulate=True)
            b.index_put_((idx % 48,), (src * 3).half(), accumulate=True)
            return a, b

        idx, src = self._inputs(device, 1 << 20, 1 << 14)
        idx_c = idx.cpu()
        expected = (
            torch.zeros(32, 64, dtype=torch.double).index_put_(
                (idx_c % 32,), src.cpu().double(), accumulate=True
            ),
            torch.zeros(48, 64, dtype=torch.double).index_put_(
                (idx_c % 48,), (src * 3).half().cpu().double(), accumulate=True
            ),
        )
        # The workspace has one dtype, so only one destination gets replicas,
        # unless the fp16 one is kept in fp32 too.
        for fp32, add_backs, skipped in (
            (False, 1, {"mixed_dtypes": 1}),
            (True, 2, {}),
        ):
            with (
                self.subTest(fp32_accumulation=fp32),
                inductor_config.patch(
                    {"triton.atomic_scatter_replica_fp32_accumulation": fp32}
                ),
            ):
                counters.clear()
                torch._dynamo.reset()
                actual, (code,) = self._compile(idx, src, fn=f)

                self.assertEqual(code.count(REPLICA_ADD), add_backs)
                self.assertNotIn("workspace_1", code)
                self.assertEqual(skips(), skipped)
                torch.testing.assert_close(
                    actual[0].cpu().double(), expected[0], atol=1e-3, rtol=1e-4
                )
                # fp16 atomics round at every add.
                torch.testing.assert_close(
                    actual[1].cpu().double(),
                    expected[1],
                    atol=0.05 * expected[1].abs().max().item(),
                    rtol=0,
                )

    @inductor_config.patch({"nan_asserts": True})
    def test_nan_asserts_check_the_destination(self, device):
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16)

        actual, (code,) = self._compile(dest_rows, idx, src)

        # The replicated kernel stores to the workspace, so its arguments do not
        # include the destination, which is checked after the add-back.
        add_back = re.search(r"(\w+)" + re.escape(REPLICA_ADD), code)
        dest = add_back.group(1)
        self.assertIn(f"assert not {dest}.isnan().any().item()", code[add_back.end() :])
        self._check(actual, idx, src, dest_rows)

    def test_fx_wrapper_backend_gets_no_replicas(self, device):
        # A device can use WrapperFxCodegen without config.fx_wrapper. Its FX
        # conversion rejects the add-back, which is a plain Python line.
        common.init_backend_registration()
        for registry in (
            common.device_codegens,
            common.custom_backend_passes,
            common.custom_backend_codegen_configs,
        ):
            restore = mock.patch.dict(registry)
            restore.start()
            self.addCleanup(restore.stop)
        common.register_backend_for_device(
            self.device_type, TritonScheduling, WrapperFxCodegen
        )
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16)

        actual = torch.compile(scatter_add)(dest_rows, idx, src)

        self.assertEqual(counters["inductor"]["atomic_scatter_replicated"], 0)
        self.assertEqual(skips(), {"fx_wrapper": 1})
        self._check(actual, idx, src, dest_rows)

    @inductor_config.patch({"triton.atomic_scatter_replication": False})
    def test_knob_off_emits_no_replication(self, device):
        dest_rows = 50
        idx, src = self._inputs(device, dest_rows, 1 << 16)

        actual, (code,) = self._compile(dest_rows, idx, src)

        self.assertNotIn(REPLICA_ADD, code)
        self.assertNotIn("tl.program_id(0) % 16", code)
        self.assertEqual(skips(), {})
        self._check(actual, idx, src, dest_rows)


instantiate_device_type_tests(
    AtomicScatterReplicationTest, globals(), except_for="cpu", allow_xpu=True
)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
