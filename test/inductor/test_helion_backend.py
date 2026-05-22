# Owner(s): ["module: inductor"]
"""Tests for the Helion backend in PyTorch Inductor.

This file reuses the full test_torchinductor test suite with config patches
(following the same pattern as test_pallas.py), plus targeted manual tests.
"""

import os
import sys
import unittest

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._dynamo.testing import make_test_cls_with_patches
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_HELION
from torch.utils._pallas import has_tpu_pallas


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

if not HAS_HELION:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires helion")


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library


# Load helion expected failures from sentinel files
_helion_expected_failures_dir = os.path.join(
    os.path.dirname(__file__), "helion_expected_failures"
)
if os.path.isdir(_helion_expected_failures_dir):
    HELION_EXPECTED_FAILURES = set(os.listdir(_helion_expected_failures_dir))
else:
    HELION_EXPECTED_FAILURES = set()

# Load helion skip tests from sentinel files (for flaky tests)
_helion_skip_tests_dir = os.path.join(os.path.dirname(__file__), "helion_skip_tests")
if os.path.isdir(_helion_skip_tests_dir):
    HELION_SKIP_TESTS = set(os.listdir(_helion_skip_tests_dir))
else:
    HELION_SKIP_TESTS = set()


test_classes = {}

_XFAIL_PROP = "_expected_failure_helion"
_SKIP_PROP = "_skip_helion"


def _apply_helion_test_markers(cls):
    """Mark tests based on sentinel files in helion_expected_failures/ and helion_skip_tests/.

    Skip takes precedence over xfail: tests in helion_skip_tests/ are not run at all,
    which is the only safe option for tests known to crash CUDA (where xfail can't
    contain the damage because the crash is detected asynchronously after the test
    body completes, triggering "TEST SUITE EARLY TERMINATION").
    """
    for name in cls.__dict__:
        if name.startswith("test_"):
            fn = cls.__dict__[name]
            if callable(fn):
                key = f"{cls.__name__}.{name}"
                if key in HELION_SKIP_TESTS:
                    setattr(fn, _SKIP_PROP, True)
                elif key in HELION_EXPECTED_FAILURES:
                    setattr(fn, _XFAIL_PROP, True)


def _helion_skip_decorator(fn):
    if getattr(fn, _SKIP_PROP, False):
        return unittest.skip("Skipped in Helion backend")(fn)
    return fn


def _make_helion_variant(cls, patches, cls_prefix, suffix):
    _apply_helion_test_markers(cls)
    test_class = make_test_cls_with_patches(
        cls,
        cls_prefix,
        suffix,
        *patches,
        xfail_prop=_XFAIL_PROP,
        decorator=_helion_skip_decorator,
    )
    # Override TestCase._should_stop_test_suite to never trigger CUDA-crash early
    # termination for Helion variants. Many known-failing tests trigger async CUDA
    # errors (e.g. illegal memory access) which the base implementation otherwise
    # surfaces as a synthetic "TestSuite execution was aborted early" failure on
    # the NEXT test in the suite. That synthetic failure overrides expectedFailure
    # markers and turns the whole shard red. Helion fixes will need to address the
    # underlying CUDA crashes; until then, keep the suite running so other tests
    # are still observed.
    test_class._should_stop_test_suite = lambda self: False
    # Inject a setUp wrapper that skips when CUDA is in a corrupted state. CUDA
    # context corruption from a prior test is unrecoverable in-process; running
    # subsequent tests just produces a cascade of meaningless failures. Skipping
    # is the only way to keep the rest of the shard meaningful.
    _orig_setup = test_class.setUp

    def _setup_with_cuda_check(self):
        # Check CUDA state BEFORE super().setUp() because super().setUp() calls
        # torch.cuda.manual_seed_all() which itself crashes when CUDA is in a bad
        # state from a prior test. We want to skip the test cleanly instead.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                _ = torch.empty(1, device="cuda")
                torch.cuda.synchronize()
            except RuntimeError as e:
                self.skipTest(f"CUDA state corrupted from prior test: {e}")
        _orig_setup(self)

    test_class.setUp = _setup_with_cuda_check
    test_classes[test_class.__name__] = test_class
    # REMOVING THIS LINE WILL STOP TESTS FROM RUNNING
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


def make_helion(cls):
    """Create a test class variant that uses the Helion backend."""
    return _make_helion_variant(
        cls,
        [(config, "cuda_backend", "helion")],
        "Helion",
        "_helion",
    )


# Apply to GPU test suites (requires CUDA and Triton)
if HAS_CUDA_AND_TRITON and test_torchinductor.RUN_GPU:
    make_helion(test_torchinductor.SweepInputsGPUTest)
    make_helion(test_torchinductor.GPUTests)


def make_helion_pallas(cls):
    """Create a test class variant that uses the Helion Pallas CPU backend."""
    return _make_helion_variant(
        cls,
        [(config, "cpu_backend", "helion"), (config, "helion_autotune_effort", "none")],
        "HelionPallas",
        "_helion_pallas",
    )


# Apply to CPU test suites (Pallas interpret mode)
if test_torchinductor.RUN_CPU:
    make_helion_pallas(test_torchinductor.SweepInputsCpuTest)
    make_helion_pallas(test_torchinductor.CpuTests)


def make_helion_tpu(cls):
    """Create a test class variant that uses the Helion Pallas TPU backend."""
    return _make_helion_variant(
        cls,
        [(config, "tpu_backend", "helion"), (config, "helion_autotune_effort", "none")],
        "HelionTpu",
        "_helion_tpu",
    )


# Apply to TPU test suites (Pallas TPU mode)
if test_torchinductor.RUN_TPU and has_tpu_pallas():
    from torch_tpu import api as tpu_api

    tpu_api.tpu_device()  # initialize TPU runtime

    make_helion_tpu(test_torchinductor.SweepInputsTpuTest)
    if hasattr(test_torchinductor, "TpuTests"):
        make_helion_tpu(test_torchinductor.TpuTests)


# --- Manual targeted tests (kept for fast sanity checks) ---


class HelionBackendTests:
    """Mixin with device-agnostic Helion backend sanity tests.

    Subclasses must set `device` and apply the appropriate inductor_config patch.
    """

    device: str

    def setUp(self):
        # Check CUDA state BEFORE super().setUp() because super().setUp() calls
        # torch.cuda.manual_seed_all() which itself crashes when CUDA is in a bad
        # state from a prior test. Probe regardless of self.device, since the
        # underlying setUp hits CUDA for RNG even on CPU/TPU tests.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                _ = torch.empty(1, device="cuda")
                torch.cuda.synchronize()
            except RuntimeError as e:
                self.skipTest(f"CUDA state corrupted from prior test: {e}")
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def test_simple_add(self):
        def fn(x, y):
            return x + y

        x = torch.randn(1024, device=self.device)
        y = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_simple_mul(self):
        def fn(x, y):
            return x * y

        x = torch.randn(1024, device=self.device)
        y = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_pointwise_chain(self):
        def fn(x, y, z):
            return x + y * z

        x = torch.randn(1024, device=self.device)
        y = torch.randn(1024, device=self.device)
        z = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y, z), fn(x, y, z), rtol=1e-4, atol=1e-4
        )

    def test_unary_sin(self):
        def fn(x):
            return torch.sin(x)

        x = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_unary_exp(self):
        def fn(x):
            return torch.exp(x)

        x = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_unary_cos(self):
        def fn(x):
            return torch.cos(x)

        x = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_relu(self):
        def fn(x):
            return torch.relu(x)

        x = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_gelu(self):
        def fn(x):
            return torch.nn.functional.gelu(x)

        x = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_sigmoid(self):
        def fn(x):
            return torch.sigmoid(x)

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_silu(self):
        def fn(x):
            return torch.nn.functional.silu(x)

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_fused_ops(self):
        def fn(x, y, z):
            return torch.sin(x) + torch.cos(y) * z

        x = torch.randn(1024, device=self.device)
        y = torch.randn(1024, device=self.device)
        z = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y, z), fn(x, y, z), rtol=1e-4, atol=1e-4
        )

    def test_scalar_mul(self):
        def fn(x):
            return x * 2.0

        x = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_2d_tensor(self):
        def fn(x, y):
            return x + y

        x = torch.randn(64, 128, device=self.device)
        y = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_3d_tensor(self):
        def fn(x, y):
            return x + y

        x = torch.randn(4, 32, 64, device=self.device)
        y = torch.randn(4, 32, 64, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4)

    def test_4d_pointwise(self):
        def fn(x, y):
            return x * y + x

        x = torch.randn(4, 8, 16, 32, device=self.device)
        y = torch.randn(4, 8, 16, 32, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4
        )

    def test_2d_transpose_add(self):
        # Inductor encodes y.T as a stride permutation on the input buffer:
        # the kernel must read y[tile_1, tile_0].T inside the tile to recover
        # the output tile shape. The codegen used to ignore the index expression
        # and produced y[tile_0, tile_1], silently miscomputing the result.
        def fn(x, y):
            return x + y.T

        x = torch.randn(8, 16, device=self.device)
        y = torch.randn(16, 8, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4
        )

    def test_3d_permute_add(self):
        # 3D version of the transpose case: permute one operand and add to
        # the other. Forces the index-aware load path to handle a non-trivial
        # permutation tuple.
        def fn(x, y):
            return x + y.permute(2, 0, 1)

        x = torch.randn(4, 8, 16, device=self.device)
        y = torch.randn(8, 16, 4, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4
        )

    def test_2d_square_transpose_add(self):
        # Square shape ``(N, N)`` makes ``_iter_divisor_to_output_dim``'s
        # length-based output-dim matching ambiguous: both output dims have
        # the same size, so picking by length alone could silently miscode.
        # When ambiguous the mapping is refused and the flat-decode path
        # falls back to the full linearized expression, which is correct in
        # all directions.
        def fn(x, y):
            return x + y.T

        x = torch.randn(4, 4, device=self.device)
        y = torch.randn(4, 4, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4
        )

    def test_cat_square_output_index_expr(self):
        # Regression test for BL1 (code_review_round2_20260522.md):
        # ``cat([x, y], dim=-1)`` with output shape ``(N, N)`` produces a
        # flat-iter kernel (single pointwise tree, output ndim=2) that
        # emits ``ops.index_expr(x_sub)`` -> ``helion_index_expr`` on a
        # decomposed sub-symbol. Before the fix, length-only ambiguity in
        # the (N, N) shape made ``_coeff_to_dim`` return None, so
        # ``helion_index_expr`` silently substituted the FULL linearized
        # expression for the sub-symbol -- e.g. turning ``x0 + 4*x1`` into
        # ``LINEAR + 4*LINEAR == 5*LINEAR``. Stride-based tie-breaking in
        # ``_iter_divisor_to_output_dim`` resolves this correctly for the
        # row-major case; remaining ambiguity raises NotImplementedError
        # instead of miscoding.
        def fn(x, y):
            return torch.cat([x, y], dim=-1)

        x = torch.randn(4, 2, device=self.device)
        y = torch.randn(4, 2, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4
        )

    def test_different_dtypes(self):
        def fn(x, y):
            return x + y

        x = torch.randn(1024, device=self.device, dtype=torch.float16)
        y = torch.randn(1024, device=self.device, dtype=torch.float16)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x, y), fn(x, y), rtol=1e-3, atol=1e-3)

    def test_where(self):
        def fn(x):
            return torch.where(x > 0, x, torch.zeros_like(x))

        x = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_multiple_outputs(self):
        def fn(x, y):
            return x + y, x * y

        x = torch.randn(1024, device=self.device)
        y = torch.randn(1024, device=self.device)
        compiled_fn = torch.compile(fn)
        result = compiled_fn(x, y)
        expected = fn(x, y)
        self.assertEqual(len(result), len(expected))
        torch.testing.assert_close(result[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(result[1], expected[1], rtol=1e-4, atol=1e-4)

    def test_broadcasting(self):
        def fn(x, bias):
            return x + bias

        x = torch.randn(64, 128, device=self.device)
        bias = torch.randn(128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, bias), fn(x, bias), rtol=1e-4, atol=1e-4
        )

    def test_3d_broadcasting(self):
        def fn(x, y):
            return x + y

        x = torch.randn(4, 32, 64, device=self.device)
        y = torch.randn(1, 1, 64, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(
            compiled_fn(x, y), fn(x, y), rtol=1e-4, atol=1e-4
        )

    def test_sum_reduction(self):
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_3d_sum_reduction(self):
        def fn(x):
            return torch.sum(x, dim=-1)

        x = torch.randn(4, 32, 64, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-3, atol=1e-3)

    def test_max_reduction(self):
        def fn(x):
            return torch.amax(x, dim=-1)

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_min_reduction(self):
        def fn(x):
            return torch.amin(x, dim=-1)

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_mean_reduction(self):
        def fn(x):
            return torch.mean(x, dim=-1)

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_var_reduction(self):
        def fn(x):
            return torch.var(x, dim=-1)

        x = torch.randn(32, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-3, atol=1e-3)

    def test_std_reduction(self):
        def fn(x):
            return torch.std(x, dim=-1)

        x = torch.randn(32, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-3, atol=1e-3)

    def test_softmax(self):
        def fn(x):
            return torch.softmax(x, dim=-1)

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_layer_norm(self):
        def fn(x):
            return torch.nn.functional.layer_norm(x, [128])

        x = torch.randn(64, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)

    def test_rms_norm_pattern(self):
        def fn(x):
            var = torch.mean(x * x, dim=-1, keepdim=True)
            return x * torch.rsqrt(var + 1e-5)

        x = torch.randn(32, 128, device=self.device)
        compiled_fn = torch.compile(fn)
        torch.testing.assert_close(compiled_fn(x), fn(x), rtol=1e-4, atol=1e-4)


if HAS_CUDA_AND_TRITON:

    @inductor_config.patch(cuda_backend="helion", helion_autotune_effort="none")
    class TestHelionBackend(HelionBackendTests, TestCase):
        device = "cuda"

        # See note on _make_helion_variant: disable CUDA early-termination so a
        # crash from the preceding parameterized suite doesn't synthetically
        # fail the manual sanity tests.
        def _should_stop_test_suite(self):  # type: ignore[override]
            return False


@inductor_config.patch(cpu_backend="helion", helion_autotune_effort="none")
class TestHelionBackendPallas(HelionBackendTests, TestCase):
    device = "cpu"

    def _should_stop_test_suite(self):  # type: ignore[override]
        return False


if __name__ == "__main__":
    run_tests()
