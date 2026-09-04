# Owner(s): ["module: inductor"]
"""Tests for strict numerics mode."""

import os
import subprocess
import sys
import unittest


# Native reductions register during import, so enable the rollout first.
os.environ["PYTORCH_SUM_INNER_TREE"] = "1"

import torch
from torch._inductor import config, metrics
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._native.ops.reductions.inner_tree_plan import (
    compute_inner_tree_params,
    vec_size,
)
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TEST_CUTEDSL,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.testing._internal.opinfo.core import BinaryUfuncInfo, UnaryUfuncInfo
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._triton import has_triton_reduction_ordering


def _singleton_input(device):
    return torch.as_strided(torch.full((1,), -0.0, device=device), (1, 1), (1, 0))


SUM_CASES = (
    ("persistent_fp16", (64, 256), 1, torch.float16),
    ("looped_bf16", (8, 12000), 1, torch.bfloat16),
    ("split_fp32", (8, 65536), 1, torch.float32),
    ("persistent_fp64", (64, 256), 1, torch.float64),
    ("looped_fp64", (8, 12000), 1, torch.float64),
    ("split_fp64", (8, 65536), 1, torch.float64),
)
INNER_TREE_CALL = "reduction_ordering=tl.constexpr(tl.ReductionOrdering.INNER_TREE)"

SUM_VARIANTS = (
    ("autotune", (64, 256), 1, torch.float32, False, {"max_autotune": True}),
)

PROD_CASES = (
    ("persistent_fp16", (64, 256), 1, torch.float16),
    ("looped_fp32", (8, 12000), 1, torch.float32),
    ("split_fp32", (8, 65536), 1, torch.float32),
)

DYNAMIC_CASES = (("plan_change", (512, 65537), {}),)

OUT_OF_SCOPE_CASES = (
    ("multidim", lambda z: torch.sum(z, (0, 1))),
    ("dtype", lambda z: torch.sum(z, 1, dtype=torch.float64)),
)

LAYOUT_CASES = (
    (
        "outer_strided",
        lambda d: (torch.randn(128, 300, device=d)[::2], -1),
        True,
    ),
    ("singleton_dim1", lambda d: (_singleton_input(d), 1), True),
    (
        "noncollapsible",
        lambda d: (torch.randn(16, 8, 300, device=d)[::2], -1),
        False,
    ),
)

SIGNED_ZERO_CASES = (
    ("multirow_fp32", 4, 1, True, torch.float32),
    ("persistent_fp32", 64, 128, False, torch.float32),
    ("persistent_fp64", 64, 128, False, torch.float64),
)

FUSION_CASES = (
    "nested",
    "mix",
    "multi_kernel",
    "multi_output",
)

EFFECTIVE_NUMERICS = {
    "eager_numerics.division_rounding": config.use_eager_division_rounding,
    "eager_numerics.disable_ftz": config.should_disable_ftz,
    "emulate_precision_casts": config.should_emulate_precision_casts,
}


def _numerics_options(numerics, enabled):
    return {
        key: numerics if key == "numerics" else enabled
        for key in ("numerics", *EFFECTIVE_NUMERICS)
    }


def _effective_numerics():
    return {key: value() for key, value in EFFECTIVE_NUMERICS.items()}


class StrictNumericsConfigTest(TestCase):
    def test_config_patch_enables_eager_numerics(self):
        with config.patch(_numerics_options("strict", False)):
            self.assertEqual(
                _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, True)
            )
            with config.patch(numerics="default"):
                self.assertEqual(
                    _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, False)
                )
        with config.patch(_numerics_options("default", True)):
            self.assertEqual(
                _effective_numerics(), dict.fromkeys(EFFECTIVE_NUMERICS, True)
            )

    def test_strict_env_enables_eager_numerics(self):
        env = os.environ.copy()
        env["TORCHINDUCTOR_NUMERICS"] = "strict"
        env["TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING"] = "0"
        env["TORCHINDUCTOR_EMULATE_PRECISION_CASTS"] = "0"
        output = subprocess.check_output(
            [
                sys.executable,
                "-c",
                (
                    "from torch._inductor import config; "
                    "print(config.use_eager_division_rounding(), "
                    "config.should_disable_ftz(), "
                    "config.should_emulate_precision_casts())"
                ),
            ],
            env=env,
            text=True,
        )
        self.assertEqual(output.strip(), "True True True")


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and torch.version.hip is None,
    "requires NVIDIA CUDA and Triton",
)
class StrictNumericsCompileTest(TestCase):
    def test_compile_options_enable_eager_division(self, device):
        x = torch.full((1024,), 11.0, device=device)
        y = torch.full((1024,), 7.0, device=device)

        result, codes = run_and_get_code(
            torch.compile(
                lambda a, b: a / b,
                fullgraph=True,
                options={"numerics": "strict"},
            ),
            x,
            y,
        )

        self.assertEqual(result.view(torch.int32), (x / y).view(torch.int32))
        self.assertIn("div_rn", "\n".join(codes))


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON
    and torch.version.hip is None
    and has_triton_reduction_ordering(),
    "requires CUDA, tl.ReductionOrdering, and the eager inner-tree implementation",
)
class StrictNumericsTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def _run(self, fn, *args, **cfg):
        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        return result, "\n".join(codes)

    def _assert_bitwise_equal(self, eager, result):
        if not TEST_CUTEDSL:
            return
        self.assertEqual(
            eager.contiguous().reshape(-1).view(torch.uint8),
            result.contiguous().reshape(-1).view(torch.uint8),
        )

    def _check_sum(self, device, shape, dim, dtype, keepdim=False, **cfg):
        x = torch.randn(*shape, device=device, dtype=dtype)

        def fn(z):
            return torch.sum(z, dim, keepdim=keepdim)

        eager = fn(x)
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(eager, result)
        self.assertIn(INNER_TREE_CALL, code)
        return code

    @parametrize("case", SUM_CASES, name_fn=lambda c: c[0])
    def test_sum_bitwise(self, device, case):
        name, shape, dim, dtype = case
        code = self._check_sum(device, shape, dim, dtype)
        if name.startswith("persistent"):
            self.assertIn("@triton_heuristics.persistent_reduction(", code)
        elif name.startswith("looped"):
            self.assertIn("for r0_offset in", code)
            self.assertEqual(code.count(INNER_TREE_CALL), 1)
        else:
            self.assertEqual(code.count(INNER_TREE_CALL), 2)

    def _make_prod_input(self, shape, dtype, device):
        # Perturb each element by O(1)/n so the length-n product stays finite
        # (no over/underflow) while remaining order-sensitive; prod shares the
        # sum inner-tree order, only the combiner (*) and identity (1) differ.
        m, n = shape
        cols = torch.arange(n, device=device, dtype=torch.float32).reshape(1, n)
        pattern = (cols % 2) * 2 - 1
        if m > 1:
            rows = torch.arange(m, device=device, dtype=torch.float32).reshape(m, 1)
            pattern = pattern + ((rows % 5) - 2)
        return (1.0 + pattern / n).to(dtype)

    @parametrize("case", PROD_CASES, name_fn=lambda c: c[0])
    def test_prod_bitwise(self, device, case):
        name, shape, dim, dtype = case
        x = self._make_prod_input(shape, dtype, device)

        def fn(z):
            return torch.prod(z, dim)

        eager = fn(x)
        result, code = self._run(fn, x)
        self._assert_bitwise_equal(eager, result)
        self.assertIn(INNER_TREE_CALL, code)
        if name.startswith("split"):
            self.assertEqual(code.count(INNER_TREE_CALL), 2)

    def test_prod_out_of_scope_uses_default_order(self, device):
        # A dtype-casting prod is out of scope -> falls back to the default order.
        x = self._make_prod_input((64, 300), torch.float32, device)
        _, code = self._run(lambda z: torch.prod(z, 1, dtype=torch.float64), x)
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", SUM_VARIANTS, name_fn=lambda c: c[0])
    def test_sum_variants(self, device, case):
        _, shape, dim, dtype, keepdim, cfg = case
        self._check_sum(device, shape, dim, dtype, keepdim, **cfg)

    @parametrize(
        "input_dtype",
        (torch.float32, torch.float64),
        name_fn=lambda dtype: str(dtype).removeprefix("torch."),
    )
    def test_special_values_match_eager(self, device, input_dtype):
        x = torch.zeros(6, 300, device=device, dtype=input_dtype)
        x[0, 0] = torch.nan
        x[1, 0] = torch.inf
        x[2, 0] = -torch.inf
        x[3, :2] = torch.tensor(
            [torch.inf, -torch.inf], device=device, dtype=input_dtype
        )
        x[4] = -0.0
        x[5] = torch.tensor(
            [1e20, 1.0, -1e20, 1.0], device=device, dtype=input_dtype
        ).repeat(75)

        def fn(z):
            return z.sum(1)

        result, code = self._run(fn, x)
        self._assert_bitwise_equal(fn(x), result)
        self.assertIn(INNER_TREE_CALL, code)

    @skipIfNoCuteDSL
    @parametrize("case", DYNAMIC_CASES, name_fn=lambda c: c[0])
    def test_dynamic_sum(self, device, case):
        _, sizes, cfg = case

        def fn(z):
            return torch.sum(z, 1)

        with config.patch({"numerics": "strict", "force_disable_caches": True, **cfg}):
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=True)
            for n in sizes:
                x = torch.randn(8, n, device=device)
                self._assert_bitwise_equal(fn(x), compiled(x))

    @parametrize("case", OUT_OF_SCOPE_CASES, name_fn=lambda c: c[0])
    def test_out_of_scope_uses_default_order(self, device, case):
        _, fn = case
        _, code = self._run(fn, torch.randn(64, 300, device=device))
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", LAYOUT_CASES, name_fn=lambda c: c[0])
    def test_layout_eligibility(self, device, case):
        _, make_input, eligible = case
        x, dim = make_input(device)

        def fn(z):
            return torch.sum(z, dim)

        result, code = self._run(fn, x)
        self.assertEqual(INNER_TREE_CALL in code, eligible)
        if eligible:
            self._assert_bitwise_equal(fn(x), result)

    def test_unbacked_reduction_size_uses_default_order(self, device):
        def fn(z):
            return z[z > 0].sum(0)

        x = torch.randn(1024, device=device)
        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            result, code = self._run(fn, x)
        self.assertEqual(result, fn(x))
        self.assertNotIn(INNER_TREE_CALL, code)

    @parametrize("case", SIGNED_ZERO_CASES, name_fn=lambda c: c[0])
    def test_signed_zero(self, device, case):
        _, rows, n, multirow, dtype = case
        x = torch.full((rows, n), -0.0, device=device, dtype=dtype)

        def fn(z):
            return torch.sum(z, 1)

        cfg = {"triton.persistent_reductions": False} if multirow else {}
        result, code = self._run(fn, x, **cfg)
        self._assert_bitwise_equal(fn(x), result)
        vector_size = vec_size(x.element_size())
        if multirow:
            num_loads = (n + vector_size - 1) // vector_size
            rblock = (1 << (num_loads - 1).bit_length()) * vector_size
        else:
            rblock = compute_inner_tree_params(n, 1, vector_size).batch_total_elements
        self.assertIn(f"R0_BLOCK: tl.constexpr = {rblock}", code)

    def _make_fusion_case(self, kind, device):
        cfg = {}
        expected_metrics = {}
        kernel_count = None
        result_index = None

        if kind == "nested":
            batch, width, group = 8, 4096, 16
            args = (torch.randn(batch, width, device=device),)
            cfg = {"triton.nested_reduction": True}
            expected_metrics = {"codegen_nested_reduction": 0}

            def fn(x):
                outer = x.amax(-1, keepdim=True)
                y = torch.ops._inductor_test.realize(x + outer)
                return y.reshape(batch, width // group, group).sum(-1)

        elif kind == "mix":
            args = (torch.randn(8, 12000, device=device),)
            cfg = {
                "triton.mix_order_reduction": True,
                "triton.mix_order_reduction_non_strict_mode": True,
            }
            result_index = 0
            expected_metrics = {"codegen_mix_order_reduction": 0}

            def fn(x):
                return x.sum(-1), x.prod(0)

        elif kind == "multi_kernel":
            args = (torch.randn(8, 12000, device=device),)
            cfg = {"triton.multi_kernel": True}

            def fn(x):
                return x.sum(1)

        else:
            args = (torch.randn(8, 300, device=device),)
            cfg = {"online_softmax": True}
            result_index = 1
            kernel_count = 2

            def fn(x):
                return torch.softmax(x, -1), x.sum(-1)

        return fn, args, result_index, cfg, expected_metrics, kernel_count

    @parametrize("kind", FUSION_CASES)
    def test_fusion_preserves_strict_reduction(self, device, kind):
        fn, args, index, cfg, expected_metrics, kernel_count = self._make_fusion_case(
            kind, device
        )
        eager = fn(*args)
        metrics.reset()
        result, code = self._run(fn, *args, **cfg)
        expected = eager if index is None else eager[index]
        actual = result if index is None else result[index]
        self._assert_bitwise_equal(expected, actual)
        self.assertEqual(code.count(INNER_TREE_CALL), 1)
        for metric, expected_value in expected_metrics.items():
            self.assertEqual(getattr(metrics, metric), expected_value)
        if kernel_count is not None:
            self.assertEqual(metrics.generated_kernel_count, kernel_count)
        if kind == "multi_kernel":
            self.assertNotIn("async_compile.multi_kernel(", code)
            self.assertIn("for r0_offset in", code)
        elif kind == "multi_output":
            self.assertEqual(eager[0], result[0])

    def test_combo_kernel_preserves_strict_reduction_blocks(self, device):
        args = (
            torch.randn(8, 12000, device=device),
            torch.randn(8, 12000, device=device),
        )

        def fn(a, b):
            return a.sum(1), b.sum(1)

        eager = fn(*args)
        result, code = self._run(
            fn,
            *args,
            combo_kernels=True,
            combo_kernels_autotune=0,
            combo_kernel_peak_memory_pct_threshold=None,
        )
        for expected, actual in zip(eager, result, strict=True):
            self._assert_bitwise_equal(expected, actual)
        self.assertIn(INNER_TREE_CALL, code)
        self.assertNotIn("combo_grid_meta", code)

    @unittest.skipIf(not SM90OrLater, "requires TMA support")
    @parametrize("kind", ("multirow", "split"))
    def test_tma_preserves_strict_reduction(self, device, kind):
        if kind == "multirow":
            x = torch.randn(64, 5, device=device)
        else:
            x = torch.zeros(1, 65536, device=device)
            params = compute_inner_tree_params(
                x.shape[1], x.shape[0], vec_size(x.element_size())
            )
            for batch, value in enumerate((1e20, 1, -1e20, 1)):
                x[0, batch * params.batch_total_elements] = value

        def fn(z):
            return torch.sum(z, 1)

        eager = fn(x)
        if kind == "split":
            self._assert_bitwise_equal(eager, torch.ones_like(eager))
        result, code = self._run(
            fn,
            x,
            assume_aligned_inputs=True,
            **{"triton.use_tensor_descriptor": True},
        )
        self._assert_bitwise_equal(eager, result)
        if kind == "split":
            self.assertEqual(code.count(INNER_TREE_CALL), 2)
        self.assertIn("tensor_descriptor" if kind == "split" else "tl.store", code)


instantiate_device_type_tests(StrictNumericsCompileTest, globals(), only_for="cuda")
instantiate_device_type_tests(StrictNumericsTest, globals(), only_for="cuda")


# ---------------------------------------------------------------------------
# Pointwise strict numerics: eager vs torch.compile bitwise equivalence.
#
# Every pointwise OpInfo is run through eager and torch.compile under numerics="strict"
# and byte-compared; ops that do not yet match are listed in POINTWISE_XFAIL /
# BACKWARD_XFAIL / NONFLOAT_XFAIL. Ops in COMPILE_UNSUPPORTED are skipped outright, as
# are RNG ops and any call whose output is not the broadcast of its tensor inputs.
#
# Two input sources. Reference inputs supply the non-contiguous and arbitrarily strided
# layouts that drive codegen; a raw bit-pattern call supplies the subnormals, NaN
# encodings and exact special values that reference inputs never contain. Bit patterns
# are skipped for BITPATTERN_SLOW ops and for signatures that cannot take substituted
# data (bool masks, per-channel weights), which fall back to reference inputs alone.
# ---------------------------------------------------------------------------


# Non-ufunc pointwise ops (composite elementwise with multi-tensor / scalar / kwargs
# signatures). Unioned with the ufunc set so coverage spans all pointwise ops.
POINTWISE_EXTRA = frozenset(
    {
        # composite elementwise (multi-tensor / scalar / kwargs signatures)
        "addcmul",
        "addcdiv",
        "clamp",
        "lerp",
        "where",
        "masked_fill",
        "logaddexp2",
        "masked.logaddexp",
        "nn.functional.gelu",
        "nn.functional.hardswish",
        "nn.functional.leaky_relu",
        "native_dropout_backward",
        # Losses are deliberately absent: pointwise only at reduction="none", a corner
        # that decomposes into already-covered refs (sub/abs/where/mul/log).
        # Revisit once reductions are bitwise.
    }
)


def _op_id(op):
    name = f"{op.name}_{op.variant_test_name}" if op.variant_test_name else op.name
    return name.replace(".", "_")


def _pointwise_ops():
    seen = set()
    pointwise = []
    for op in op_db:
        if not (
            isinstance(op, (UnaryUfuncInfo, BinaryUfuncInfo))
            or op.name in POINTWISE_EXTRA
        ):
            continue
        oid = _op_id(op)
        if oid in seen:
            continue
        seen.add(oid)
        pointwise.append(op)
    return pointwise


POINTWISE_OPS = _pointwise_ops()
# Float dtypes swept by the bitwise test. bf16 and fp16 are where
# emulate_precision_casts and disable_ftz actually bite, so they are the key coverage.
POINTWISE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _dtype_label(dtype):
    return str(dtype).split(".")[-1]


# Bit-pattern coverage. reference_inputs come from randn/linspace and contain no
# subnormals -- exactly what disable_ftz / emulate_precision_casts govern -- so we add
# one call over raw bit patterns: exhaustive for 16-bit (all 65536 values), sampled for
# fp32.
NUM_BITPATTERN_SAMPLES = 65536


def _exhaustive_16bit(dtype, device):
    # All 65536 patterns; the narrowing int32->int16 cast wraps so 32768..65535 land on
    # negative int16 and reinterpret as the negative half of the float line.
    return (
        torch.arange(0, 65536, dtype=torch.int32, device=device)
        .to(torch.int16)
        .view(dtype)
    )


def _sampled_fp32(n, device, seed=0):
    # randint can't span 2**32 in int32, so draw 31 bits and OR the sign in. Random
    # sampling never lands on the exact special values, so they are appended.
    gen = torch.Generator(device=device).manual_seed(seed)
    bits = torch.randint(
        0, 2**31, (n,), dtype=torch.int32, device=device, generator=gen
    )
    signs = torch.randint(0, 2, (n,), dtype=torch.int32, device=device, generator=gen)
    fi = torch.finfo(torch.float32)
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            2.0,
            float("inf"),
            -float("inf"),
            float("nan"),
            fi.smallest_normal,
            -fi.smallest_normal,
            fi.max,
            fi.min,
            fi.eps,
            1.0 + fi.eps,
            1.0 - fi.eps / 2,
        ],
        device=device,
    )
    return torch.cat([(bits | (signs << 31)).view(torch.float32), specials])


_BIT_VIEW = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


def _diff_kind(a, b):
    """Coarse label for a known mismatch, so a regenerated list stays triageable.

    "nan-payload" when every differing element is NaN on both sides (eager quiets a
    signalling NaN where Triton passes it through), "signed-zero" when every differing
    element is zero on both sides, "value" for a real arithmetic difference. Computed
    only on failure.
    """
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return "shape"
        kinds = set()
        for x, y in zip(a, b):
            if x is None or y is None:
                if x is not y:
                    return "shape"
                continue
            if not _outputs_equal(x, y):
                kinds.add(_diff_kind(x, y))
        return "value" if "value" in kinds or not kinds else "+".join(sorted(kinds))
    if a.dtype != b.dtype or a.shape != b.shape:
        return "shape"
    if a.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b)
    int_dtype = _BIT_VIEW.get(a.dtype)
    if int_dtype is None:
        return "value"
    differ = a.contiguous().view(int_dtype) != b.contiguous().view(int_dtype)
    nan = differ & torch.isnan(a) & torch.isnan(b)
    zero = differ & (a == 0) & (b == 0)
    if not bool((differ == (nan | zero)).all()):
        return "value"
    parts = []
    if bool(nan.any()):
        parts.append("nan-payload")
    if bool(zero.any()):
        parts.append("signed-zero")
    return "+".join(parts) or "value"


def _outputs_equal(a, b):
    """True iff eager and compiled outputs are bit-for-bit identical.

    torch.equal on the raw bytes, not the values: comparing floats directly would
    call two identical NaNs unequal and +0.0/-0.0 equal. Complex is compared as
    (real, imag); tuple outputs (frexp, and gradient tuples whose entries may be
    None) element by element.
    """
    if isinstance(a, (tuple, list)):
        if not isinstance(b, (tuple, list)) or len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if x is None or y is None:
                if x is not y:
                    return False
                continue
            if not _outputs_equal(x, y):
                return False
        return True
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.numel() == 0:
        return True
    if a.is_complex():
        a, b = torch.view_as_real(a), torch.view_as_real(b)
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),
        b.contiguous().reshape(-1).view(torch.uint8),
    )


# Fallback input dtypes for ops without fp32 support (int first, then complex).
_NONFLOAT_DTYPES = (
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.complex64,
    torch.complex128,
    torch.complex32,
)


def _first_nonfloat_dtype(op):
    for d in _NONFLOAT_DTYPES:
        if d in op.supported_dtypes("cuda"):
            return d
    return None


# Ops with no fp32 support (gcd/lcm on ints, imag on complex).
NONFLOAT_INPUT_OPS = [
    op
    for op in POINTWISE_OPS
    if torch.float32 not in op.supported_dtypes("cuda")
    and _first_nonfloat_dtype(op) is not None
]

# Pointwise ops supporting autograd; their (elementwise) backward is bitwise-checked
# against eager in test_pointwise_backward.
BACKWARD_OPS = [op for op in POINTWISE_OPS if op.supports_autograd]

# (op_id, dtype_label) pairs not yet bitwise-identical to eager under strict numerics,
# generated from a full run on sm_100. Every entry is a real difference; the failure
# message tags each mismatching call as nan-payload, signed-zero or value.
POINTWISE_XFAIL = frozenset(
    {
        ("abs", "bfloat16"),
        ("abs", "float16"),
        ("abs", "float32"),
        ("addcdiv", "bfloat16"),
        ("addcdiv", "float16"),
        ("addcdiv", "float32"),
        ("angle", "bfloat16"),
        ("angle", "float16"),
        ("angle", "float32"),
        ("clamp", "bfloat16"),
        ("clamp", "float16"),
        ("copysign", "bfloat16"),
        ("copysign", "float16"),
        ("div_floor_rounding", "bfloat16"),
        ("div_floor_rounding", "float16"),
        ("div_floor_rounding", "float32"),
        ("double", "float16"),
        ("float_power", "bfloat16"),
        ("float_power", "float16"),
        ("float_power", "float32"),
        ("floor_divide", "bfloat16"),
        ("floor_divide", "float16"),
        ("floor_divide", "float32"),
        ("fmax", "bfloat16"),
        ("fmax", "float16"),
        ("fmax", "float32"),
        ("fmin", "bfloat16"),
        ("fmin", "float16"),
        ("fmin", "float32"),
        ("frexp", "bfloat16"),
        ("frexp", "float16"),
        ("frexp", "float32"),
        ("ldexp", "bfloat16"),
        ("ldexp", "float16"),
        ("logaddexp2", "float32"),
        ("mvlgamma_mvlgamma_p_1", "bfloat16"),
        ("mvlgamma_mvlgamma_p_1", "float16"),
        ("mvlgamma_mvlgamma_p_1", "float32"),
        ("mvlgamma_mvlgamma_p_3", "bfloat16"),
        ("mvlgamma_mvlgamma_p_3", "float16"),
        ("mvlgamma_mvlgamma_p_3", "float32"),
        ("mvlgamma_mvlgamma_p_5", "bfloat16"),
        ("mvlgamma_mvlgamma_p_5", "float16"),
        ("mvlgamma_mvlgamma_p_5", "float32"),
        ("nextafter", "bfloat16"),
        ("nextafter", "float16"),
        ("nn_functional_gelu", "float32"),
        ("nn_functional_hardtanh", "bfloat16"),
        ("nn_functional_hardtanh", "float16"),
        ("nn_functional_hardtanh", "float32"),
        ("nn_functional_relu6", "bfloat16"),
        ("nn_functional_relu6", "float16"),
        ("nn_functional_relu6", "float32"),
        ("nn_functional_relu", "bfloat16"),
        ("nn_functional_relu", "float16"),
        ("nn_functional_relu", "float32"),
        ("nn_functional_softplus", "float32"),
        ("nn_functional_softshrink", "bfloat16"),
        ("nn_functional_softshrink", "float16"),
        ("nn_functional_softshrink", "float32"),
        ("remainder", "bfloat16"),
        ("remainder", "float16"),
        ("remainder", "float32"),
        ("__rmod__", "bfloat16"),
        ("__rmod__", "float16"),
        ("__rmod__", "float32"),
        ("round_decimals_3", "bfloat16"),
        ("round_decimals_3", "float16"),
        ("round_decimals_3", "float32"),
        ("round_decimals_neg_3", "bfloat16"),
        ("round_decimals_neg_3", "float16"),
        ("round_decimals_neg_3", "float32"),
        ("rsub", "bfloat16"),
        ("rsub", "float16"),
        ("rsub", "float32"),
        ("special_bessel_j0", "float32"),
        ("special_bessel_j1", "float32"),
        ("special_bessel_y0", "float32"),
        ("special_bessel_y1", "float32"),
        ("special_entr", "bfloat16"),
        ("special_entr", "float16"),
        ("special_modified_bessel_i0", "float32"),
        ("special_modified_bessel_i1", "float32"),
        ("special_xlog1py", "bfloat16"),
        ("special_xlog1py", "float16"),
        ("sub", "bfloat16"),
        ("sub", "float16"),
        ("sub", "float32"),
        ("xlogy", "bfloat16"),
        ("xlogy", "float16"),
    }
)

BACKWARD_XFAIL = frozenset(
    {
        ("__rmod__", "bfloat16"),
        ("__rmod__", "float16"),
        ("__rmod__", "float32"),
        ("__rpow__", "bfloat16"),
        ("__rpow__", "float16"),
        ("__rpow__", "float32"),
        ("addcdiv", "bfloat16"),
        ("addcdiv", "float16"),
        ("addcdiv", "float32"),
        ("double", "bfloat16"),
        ("double", "float16"),
        ("float_power", "bfloat16"),
        ("float_power", "float16"),
        ("float_power", "float32"),
        ("ldexp", "bfloat16"),
        ("ldexp", "float16"),
        ("ldexp", "float32"),
        ("logaddexp2", "float32"),
        ("logit", "bfloat16"),
        ("logit", "float16"),
        ("logit", "float32"),
        ("mvlgamma_mvlgamma_p_1", "float32"),
        ("mvlgamma_mvlgamma_p_3", "float32"),
        ("mvlgamma_mvlgamma_p_5", "float32"),
        ("nn_functional_gelu", "bfloat16"),
        ("nn_functional_gelu", "float16"),
        ("nn_functional_gelu", "float32"),
        ("nn_functional_hardswish", "bfloat16"),
        ("nn_functional_hardswish", "float16"),
        ("nn_functional_hardswish", "float32"),
        ("nn_functional_mish", "bfloat16"),
        ("nn_functional_mish", "float16"),
        ("nn_functional_mish", "float32"),
        ("nn_functional_silu", "float16"),
        ("nn_functional_silu", "float32"),
        ("nn_functional_softshrink", "bfloat16"),
        ("nn_functional_softshrink", "float16"),
        ("nn_functional_softshrink", "float32"),
        ("nn_functional_tanhshrink", "bfloat16"),
        ("nn_functional_tanhshrink", "float16"),
        ("nn_functional_tanhshrink", "float32"),
        ("remainder", "bfloat16"),
        ("remainder", "float16"),
        ("remainder", "float32"),
        ("rsqrt", "bfloat16"),
        ("rsqrt", "float16"),
        ("sigmoid", "bfloat16"),
        ("sigmoid", "float16"),
        ("sigmoid", "float32"),
        ("special_bessel_j0", "float32"),
        ("special_bessel_j1", "float32"),
        ("special_bessel_y0", "float32"),
        ("special_bessel_y1", "float32"),
        ("special_modified_bessel_i0", "float32"),
        ("special_modified_bessel_i1", "float32"),
        ("special_xlog1py", "bfloat16"),
        ("special_xlog1py", "float16"),
        ("tanh", "bfloat16"),
        ("tanh", "float16"),
        ("tanh", "float32"),
        ("xlogy", "bfloat16"),
        ("xlogy", "float16"),
    }
)

NONFLOAT_XFAIL = frozenset(
    {
        ("nn_functional_silu_complex", "complex128"),
    }
)

POINTWISE_STRICT_CFG = {
    # The base inductor TestCase already gives every test a fresh, isolated (cold) cache
    # via fresh_cache() + fx_graph_cache=True, so force_disable_caches is unnecessary
    # (and would override fx_graph_cache, killing in-test reuse of repeated shapes).
    "numerics": "strict",
}


# Ops excluded from the bit-pattern sweep because they are value-dependent: their
# kernels iterate a series whose length depends on the argument, so a 65k vector of
# arbitrary bit patterns drives them for minutes. Measured on a full run: polygamma
# 64-363s per test, shifted_chebyshev >300s, everything else under 60s. They keep their
# reference_inputs coverage; only the value sweep is skipped (sample_inputs feed the bit-
# pattern call's signature, so for these ops those values never execute).
BITPATTERN_SLOW = frozenset(
    {
        "polygamma",
        "special.polygamma",
        "special.shifted_chebyshev_polynomial_t",
        "special.shifted_chebyshev_polynomial_u",
        "special.shifted_chebyshev_polynomial_v",
        "special.shifted_chebyshev_polynomial_w",
        # Not independently timed; excluded pending measurement.
        "special.laguerre_polynomial_l",
        "special.legendre_polynomial_p",
    }
)

# Ops that cannot compile under fullgraph (skipped, not xfail'd -- the exception happens
# before the comparison). jiterator kernels are runtime string-JIT'd CUDA.
COMPILE_UNSUPPORTED = frozenset(
    {
        "jiterator_unary",
        "jiterator_binary",
        "jiterator_binary_return_by_ref",
    }
)


class _RngOpDetector(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.has_rng = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if torch.Tag.nondeterministic_seeded in getattr(func, "tags", ()):
            self.has_rng = True
        return func(*args, **(kwargs or {}))


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and torch.version.hip is None,
    "requires CUDA and Triton",
)
class PointwiseStrictNumericsTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        # Reference inputs materialise millions of elements per op, so cached allocator
        # blocks accumulate across tests and OOM when several run concurrently.
        torch.cuda.empty_cache()

    def _is_tensor_output(self, out):
        return isinstance(out, torch.Tensor) or (
            isinstance(out, (tuple, list))
            and all(isinstance(t, torch.Tensor) for t in out)
        )

    def _is_pointwise_output(self, inp, args, kwargs, out):
        # A pointwise call preserves shape: output == broadcast of tensor inputs.
        shapes = [inp.shape] + [a.shape for a in args if isinstance(a, torch.Tensor)]
        shapes += [v.shape for v in kwargs.values() if isinstance(v, torch.Tensor)]
        try:
            bshape = torch.broadcast_shapes(*shapes)
        except Exception:
            return False
        outs = out if isinstance(out, (tuple, list)) else (out,)
        return all(isinstance(o, torch.Tensor) and o.shape == bshape for o in outs)

    def _build_calls(self, samples, dtype):
        """One call per usable sample at its natural shape: (input, args, kwargs).

        No flattening: shape drives Inductor codegen (tiling, masking, index math,
        strides), so each natural shape is compiled and checked as its own kernel.
        Only samples whose input is a tensor and that carry a `dtype` tensor are kept.
        The bit-pattern call (added separately) is a 1-D value sweep, not a shape sweep.
        """
        calls = []
        for sample in samples:
            if not isinstance(sample.input, torch.Tensor):
                continue
            tensors = [sample.input, *sample.args]
            if any(isinstance(t, torch.Tensor) and t.dtype == dtype for t in tensors):
                calls.append((sample.input, tuple(sample.args), dict(sample.kwargs)))
        return calls

    def _bitpattern_call(self, op, dtype, device):
        """Raw bit-pattern calls: one per non-tensor signature in sample_inputs.

        Exhaustive for 16-bit (all 65536 values), sampled for fp32. Tensor operands are
        replaced by the bit-pattern vector while scalars and kwargs are taken from the
        sample, so ops carrying a scalar parameter (polygamma n, mvlgamma p, threshold,
        fill value) are swept at each of their values rather than skipped. The trial
        call drops signatures that cannot take substituted data -- prelu's per-channel
        weight, and the bool masks of where / masked_fill / native_dropout_backward.
        """
        if op.name in BITPATTERN_SLOW:
            return []
        if dtype in (torch.float16, torch.bfloat16):
            x = _exhaustive_16bit(dtype, device)
            y = x.flip(0)
        elif dtype == torch.float32:
            x = _sampled_fp32(NUM_BITPATTERN_SAMPLES, device, seed=0)
            y = _sampled_fp32(NUM_BITPATTERN_SAMPLES, device, seed=1)
        else:
            return []
        try:
            samples = list(op.sample_inputs(device, dtype, requires_grad=False))
        except Exception:
            return []

        def scalar(v):
            return None if isinstance(v, torch.Tensor) else repr(v)

        calls = []
        seen = set()
        for sample in samples:
            if not isinstance(sample.input, torch.Tensor):
                continue
            args = tuple(y if isinstance(a, torch.Tensor) else a for a in sample.args)
            kwargs = {
                k: (y if isinstance(v, torch.Tensor) else v)
                for k, v in sample.kwargs.items()
            }
            key = (
                tuple(scalar(a) for a in args),
                tuple(sorted((k, scalar(v)) for k, v in kwargs.items())),
            )
            if key in seen:
                continue
            seen.add(key)
            try:
                op.op(x, *args, **kwargs)
            except Exception:
                continue  # e.g. prelu, whose per-channel weight is not elementwise
            calls.append((x, args, kwargs))
        return calls

    def _sweep(self, device, op, dtype, cfg):
        """Run reference inputs and bit patterns through eager and torch.compile.

        Returns mismatching (source, index, shape, kwargs, kind) records, where
        source is "ref" (reference inputs) or "bits" (raw bit patterns); both are
        held to the same bitwise standard. Skips known-uncompilable / RNG / no-sample
        ops; other compile failures are left to fail.
        """
        if op.name in COMPILE_UNSUPPORTED:
            self.skipTest("uncompilable op under fullgraph")

        def fn(inp, args, kwargs):
            return op.op(inp, *args, **kwargs)

        # Reference inputs are the only source of non-contiguous / arbitrarily strided
        # layouts; the bit-pattern call is the only source of subnormals and exhaustive
        # 16-bit values. Both always run.
        try:
            samples = list(op.reference_inputs(device, dtype, requires_grad=False))
        except Exception as e:
            self.skipTest(f"reference_inputs failed: {type(e).__name__}")
        calls = [("ref", *c) for c in self._build_calls(samples, dtype)]
        calls += [("bits", *c) for c in self._bitpattern_call(op, dtype, device)]
        if calls:
            # Probe every call: RNG use can depend on the scalar signature.
            detector = _RngOpDetector()
            with detector:
                for _, inp, args, kwargs in calls:
                    fn(inp, args, kwargs)
                    if detector.has_rng:
                        break
            if detector.has_rng:
                self.skipTest("RNG op excluded (RNG-source equivalence is separate)")
        tested = 0
        mismatches = []
        with (
            config.patch(cfg),
            torch._dynamo.config.patch(
                recompile_limit=sys.maxsize,
                accumulated_recompile_limit=sys.maxsize,
            ),
        ):
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            for idx, (tag, inp, args, kwargs) in enumerate(calls):
                eager = fn(inp, args, kwargs)
                if not self._is_tensor_output(eager):
                    continue
                if not self._is_pointwise_output(inp, args, kwargs, eager):
                    continue
                result = compiled(inp, args, kwargs)
                tested += 1
                if not _outputs_equal(eager, result):
                    kind = _diff_kind(eager, result)
                    mismatches.append((tag, idx, tuple(inp.shape), kwargs, kind))

        if tested == 0:
            self.skipTest("no usable sample")
        return mismatches

    @ops(POINTWISE_OPS, allowed_dtypes=POINTWISE_DTYPES)
    def test_pointwise_bitwise(self, device, dtype, op):
        # Every reference input must match eager bitwise under strict numerics. No-fp32
        # ops are covered by test_pointwise_nonfloat. @ops intersects allowed_dtypes
        # with each op's supported dtypes, so unsupported combos are never generated.
        mismatches = self._sweep(device, op, dtype, POINTWISE_STRICT_CFG)
        key = (_op_id(op), _dtype_label(dtype))
        all_match = not mismatches
        if key in POINTWISE_XFAIL:
            self.assertFalse(
                all_match,
                f"{key} now matches eager under strict numerics; "
                f"remove it from POINTWISE_XFAIL.",
            )
        else:
            self.assertTrue(
                all_match,
                f"{key} forward differs from eager under strict numerics "
                f"on (source, index, shape, kwargs, kind): {mismatches}.",
            )

    @ops(NONFLOAT_INPUT_OPS, allowed_dtypes=_NONFLOAT_DTYPES)
    def test_pointwise_nonfloat(self, device, dtype, op):
        # No-fp32 ops (bitwise/gcd/lcm on ints, imag on complex): no rounding, so eager
        # and compile must agree exactly. Swept over every supported int/complex dtype.
        mismatches = self._sweep(device, op, dtype, POINTWISE_STRICT_CFG)
        key = (_op_id(op), _dtype_label(dtype))
        all_match = not mismatches
        if key in NONFLOAT_XFAIL:
            self.assertFalse(
                all_match,
                f"{key} nonfloat now matches eager; remove it from NONFLOAT_XFAIL.",
            )
        else:
            self.assertTrue(
                all_match,
                f"{key} nonfloat differs from eager "
                f"on (source, index, shape, kwargs, kind): {mismatches}.",
            )

    def _input_grads(self, call_fn, inp, args, kwargs, grad_output):
        # Clone float tensor inputs as grad-tracking leaves, run call_fn, return the
        # output and the input gradients (grad_output is shared across eager/compiled).
        leaves = []

        def leafify(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                leaf = t.detach().clone().requires_grad_(True)
                leaves.append(leaf)
                return leaf
            return t

        inp2 = leafify(inp)
        args2 = tuple(leafify(a) for a in args)
        out = call_fn(inp2, args2, kwargs)
        if not leaves or not isinstance(out, torch.Tensor):
            return out, None
        if out.shape != grad_output.shape:
            # Compiled produced a different output shape than eager: report it
            # rather than let autograd.grad raise on the mismatched grad_outputs.
            return out, None
        grads = torch.autograd.grad(
            out, leaves, grad_outputs=grad_output, allow_unused=True
        )
        return out, grads

    def _sweep_backward(self, device, op, dtype, cfg):
        """Run each pointwise sample's backward through eager and torch.compile.

        Backward of a pointwise op is elementwise, so input gradients must match eager
        bitwise. Same two input sources as _sweep. Per-sample (backward needs leaf
        inputs); a fixed grad_output is shared so both paths see identical upstream
        gradients.
        """
        if op.name in COMPILE_UNSUPPORTED:
            self.skipTest("uncompilable op under fullgraph")

        def fn(inp, args, kwargs):
            return op.op(inp, *args, **kwargs)

        try:
            samples = list(op.reference_inputs(device, dtype, requires_grad=False))
        except Exception as e:
            self.skipTest(f"reference_inputs failed: {type(e).__name__}")
        calls = [("ref", *c) for c in self._build_calls(samples, dtype)]
        calls += [("bits", *c) for c in self._bitpattern_call(op, dtype, device)]
        if not calls:
            self.skipTest("no usable sample")

        # Probe every call: RNG use can depend on the scalar signature.
        detector = _RngOpDetector()
        with detector:
            for _, inp, args, kwargs in calls:
                fn(inp, args, kwargs)
                if detector.has_rng:
                    break
        if detector.has_rng:
            self.skipTest("RNG op excluded (RNG-source equivalence is separate)")

        tested = 0
        mismatches = []
        with (
            config.patch(cfg),
            torch._dynamo.config.patch(
                recompile_limit=sys.maxsize,
                accumulated_recompile_limit=sys.maxsize,
            ),
        ):
            torch._dynamo.reset()
            compiled = torch.compile(fn, fullgraph=True, dynamic=False)
            for idx, (tag, inp, args, kwargs) in enumerate(calls):
                with torch.no_grad():
                    probe = fn(inp, args, kwargs)
                if not isinstance(probe, torch.Tensor) or not probe.is_floating_point():
                    continue
                if not self._is_pointwise_output(inp, args, kwargs, probe):
                    continue
                # No reduction in backward: require every differentiable input to match
                # the output shape (broadcasting would sum-reduce the gradient).
                diff_ts = [
                    t
                    for t in (inp, *args)
                    if isinstance(t, torch.Tensor) and t.is_floating_point()
                ]
                if any(t.shape != probe.shape for t in diff_ts):
                    continue
                gen = torch.Generator(device=probe.device).manual_seed(0)
                grad_output = torch.randn(
                    probe.shape, generator=gen, device=probe.device, dtype=probe.dtype
                )
                try:
                    _, eager_grads = self._input_grads(
                        fn, inp, args, kwargs, grad_output
                    )
                except Exception:
                    continue
                if eager_grads is None:
                    continue
                _, comp_grads = self._input_grads(
                    compiled, inp, args, kwargs, grad_output
                )
                tested += 1
                if comp_grads is None:
                    # Output shape diverged, so there are no gradients to compare.
                    mismatches.append((tag, idx, tuple(inp.shape), kwargs, "shape"))
                    continue
                if not _outputs_equal(eager_grads, comp_grads):
                    kind = _diff_kind(eager_grads, comp_grads)
                    mismatches.append((tag, idx, tuple(inp.shape), kwargs, kind))

        if tested == 0:
            self.skipTest("no differentiable sample")
        return mismatches

    @ops(BACKWARD_OPS, allowed_dtypes=POINTWISE_DTYPES)
    def test_pointwise_backward(self, device, dtype, op):
        # Backward of a pointwise op is elementwise; grads must match eager bitwise.
        mismatches = self._sweep_backward(device, op, dtype, POINTWISE_STRICT_CFG)
        key = (_op_id(op), _dtype_label(dtype))
        all_match = not mismatches
        if key in BACKWARD_XFAIL:
            self.assertFalse(
                all_match,
                f"{key} backward now matches eager under strict numerics; "
                f"remove it from BACKWARD_XFAIL.",
            )
        else:
            self.assertTrue(
                all_match,
                f"{key} backward differs from eager under strict numerics "
                f"on (source, index, shape, kwargs, kind): {mismatches}.",
            )


instantiate_device_type_tests(PointwiseStrictNumericsTest, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
