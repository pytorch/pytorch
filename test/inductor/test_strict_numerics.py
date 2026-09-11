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
from torch.testing._internal.common_cuda import IS_SM100, IS_SM89, IS_SM90, SM90OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import (
    LazyVal,
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

EFFECTIVE_NUMERICS = (
    "eager_numerics.division_rounding",
    "eager_numerics.disable_ftz",
    "eager_numerics.use_pytorch_libdevice",
    "emulate_precision_casts",
)

# Two distinct NaN encodings per dtype: a canonical NaN on both sides cannot
# tell "first operand wins" apart from "second operand wins".
NAN_PAYLOADS = {
    torch.float16: (torch.int16, 0x7C11, 0x7E22),
    torch.bfloat16: (torch.int16, 0x7F81, 0x7FC2),
    torch.float32: (torch.int32, 0x7FAB2AC8, 0x7FC13579),
    torch.float64: (torch.int64, 0x7FFABCDEF0123456, 0x7FF8123456789ABC),
}


def _numerics_options(numerics, enabled):
    return {
        key: numerics if key == "numerics" else enabled
        for key in ("numerics", *EFFECTIVE_NUMERICS)
    }


def _effective_numerics():
    return {
        "eager_numerics.division_rounding": config.eager_numerics.division_rounding,
        "eager_numerics.disable_ftz": config.eager_numerics.disable_ftz,
        "eager_numerics.use_pytorch_libdevice": config.eager_numerics.use_pytorch_libdevice,
        "emulate_precision_casts": config.emulate_precision_casts,
    }


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
        with config.patch(
            {**_numerics_options("default", False), "emulate_precision_casts": True}
        ):
            self.assertFalse(config.eager_numerics.use_pytorch_libdevice)

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
                    "print(config.eager_numerics.division_rounding, "
                    "config.eager_numerics.disable_ftz, "
                    "config.eager_numerics.use_pytorch_libdevice, "
                    "config.emulate_precision_casts)"
                ),
            ],
            env=env,
            text=True,
        )
        self.assertEqual(output.strip(), "True True True True")


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

    @parametrize("case", ("tail", "mixed", "scalar", "broadcast", "nan"))
    def test_erfcx_branch_selection(self, device, case):
        dtype = torch.float32
        values = torch.tensor(
            [60.0, 80.0, 1e8, -8.0, -30.0, float("inf"), -float("inf"), float("nan")],
            dtype=dtype,
            device=device,
        )
        if case in ("mixed", "broadcast"):
            edges = torch.tensor(
                [-26.7, -6.1, -0.0, 0.0, 50.0, 5e7], dtype=dtype, device=device
            )
            values = torch.cat(
                (
                    values,
                    edges,
                    torch.nextafter(edges, torch.full_like(edges, -float("inf"))),
                    torch.nextafter(edges, torch.full_like(edges, float("inf"))),
                )
            )

        def fn(x, y=None):
            result = torch.special.erfcx(x)
            return result if y is None else result + y

        if case == "scalar":
            args = (values[0],)
        elif case == "broadcast":
            args = (values[:, None], torch.zeros((1, 17), dtype=dtype, device=device))
        else:
            if case == "nan":
                values.fill_(float("nan"))
            args = (values.repeat(129),)
        with config.patch(force_disable_caches=True):
            compiled = torch.compile(fn, fullgraph=True, options={"numerics": "strict"})
            result = compiled(*args)
        self.assertEqual(result.view(torch.int32), fn(*args).view(torch.int32))

    @parametrize("dtype", tuple(NAN_PAYLOADS), name_fn=lambda d: str(d).split(".")[-1])
    @parametrize("op", (torch.minimum, torch.maximum), name_fn=lambda f: f.__name__)
    def test_min_max_nan_payload(self, device, dtype, op):
        int_dtype, first, second = NAN_PAYLOADS[dtype]

        def nan(bits):
            return torch.tensor([bits], dtype=int_dtype, device=device).view(dtype)

        a, b = nan(first), nan(second)
        with config.patch(force_disable_caches=True):
            compiled = torch.compile(op, fullgraph=True, options={"numerics": "strict"})
            # Swapping operands must swap the preserved NaN payload.
            for x, y in ((a, b), (b, a)):
                result = compiled(x, y)
                self.assertEqual(result.view(int_dtype), op(x, y).view(int_dtype))


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


# Compare eager and compiled pointwise OpInfos on reference and raw-bit inputs.


# Pointwise ops not represented by UnaryUfuncInfo or BinaryUfuncInfo.
POINTWISE_EXTRA = frozenset(
    {
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
POINTWISE_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _dtype_label(dtype):
    return str(dtype).split(".")[-1]


# Exhaust all 16-bit encodings and sample float32 bit patterns.
NUM_BITPATTERN_SAMPLES = 65536

_FINFO32 = torch.finfo(torch.float32)
# Include exact values that random bit sampling is unlikely to hit.
_SPECIALS = torch.tensor(
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
        _FINFO32.smallest_normal,
        -_FINFO32.smallest_normal,
        _FINFO32.max,
        _FINFO32.min,
        _FINFO32.eps,
        1.0 + _FINFO32.eps,
        1.0 - _FINFO32.eps / 2,
    ]
)


def _exhaustive_16bit(dtype, device):
    # Narrowing wraps the upper half into negative int16 encodings.
    return (
        torch.arange(0, 65536, dtype=torch.int32, device=device)
        .to(torch.int16)
        .view(dtype)
    )


def _substitute(t, y, n, parity=0):
    r"""Substitute data vectors and resize masks, preserving scalar operands."""
    if not isinstance(t, torch.Tensor):
        return t
    if t.dtype == torch.bool:
        # Alternate parity so masks select every 16-bit encoding across two calls.
        mask = torch.zeros(n, dtype=torch.bool, device=y.device)
        mask[parity::2] = True
        return mask
    if t.is_floating_point() and t.numel() != 1:
        return y
    return t


def _sampled_fp32(n, device, seed=0):
    # Draw the sign separately because randint cannot span 2**32 in int32.
    gen = torch.Generator(device=device).manual_seed(seed)
    bits = torch.randint(
        0, 2**31, (n,), dtype=torch.int32, device=device, generator=gen
    )
    signs = torch.randint(0, 2, (n,), dtype=torch.int32, device=device, generator=gen)
    return torch.cat([(bits | (signs << 31)).view(torch.float32), _SPECIALS.to(device)])


_BIT_VIEW = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


def _diff_kind(a, b):
    r"""Classify mismatches as shape, value, NaN payload, or signed zero."""
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


def _diff_output(out):
    r"""Select the floating-point output, excluding frexp's integer exponent."""
    if isinstance(out, torch.Tensor):
        return out if out.is_floating_point() else None
    if isinstance(out, (tuple, list)):
        for t in out:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return t
    return None


def _outputs_equal(a, b):
    r"""Compare raw bytes, including NaN payloads and signed zeros."""
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


NONFLOAT_INPUT_OPS = [
    op
    for op in POINTWISE_OPS
    if torch.float32 not in op.supported_dtypes("cuda")
    and _first_nonfloat_dtype(op) is not None
]

BACKWARD_OPS = [op for op in POINTWISE_OPS if op.supports_autograd]

# (op_id, dtype_label) pairs that must still differ from eager.
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
        ("clamp", "float32"),
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
        ("logaddexp2", "bfloat16"),
        ("logaddexp2", "float16"),
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
        ("neg", "bfloat16"),
        ("neg", "float16"),
        ("neg", "float32"),
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
        ("special_log_ndtr", "float32"),
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
        ("remainder", "bfloat16"),
        ("remainder", "float16"),
        ("remainder", "float32"),
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
        ("nn_functional_silu", "bfloat16"),
        ("nn_functional_silu", "float16"),
        ("nn_functional_silu", "float32"),
        ("nn_functional_softshrink", "bfloat16"),
        ("nn_functional_softshrink", "float16"),
        ("nn_functional_softshrink", "float32"),
        ("nn_functional_tanhshrink", "bfloat16"),
        ("nn_functional_tanhshrink", "float16"),
        ("nn_functional_tanhshrink", "float32"),
        ("rsqrt", "bfloat16"),
        ("rsqrt", "float16"),
        ("sigmoid", "bfloat16"),
        ("sigmoid", "float16"),
        ("sigmoid", "float32"),
        ("special_bessel_j0", "float32"),
        ("special_bessel_j1", "float32"),
        ("special_bessel_y0", "float32"),
        ("special_bessel_y1", "float32"),
        ("special_log_ndtr", "float32"),
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
    "numerics": "strict",
}


# Skip raw-bit sweeps for prohibitively slow value-dependent series.
BITPATTERN_SLOW = frozenset(
    {
        "polygamma",
        "special.polygamma",
        "special.shifted_chebyshev_polynomial_t",
        "special.shifted_chebyshev_polynomial_u",
        "special.shifted_chebyshev_polynomial_v",
        "special.shifted_chebyshev_polynomial_w",
        "special.laguerre_polynomial_l",
        "special.legendre_polynomial_p",
    }
)

# Jiterator kernels cannot compile under fullgraph.
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


# The ledger was measured on sm_89 (CUDA 13.0/13.2), sm_90, and sm_100.
_LEDGER_ARCH = LazyVal(lambda: bool(IS_SM89 or IS_SM90 or IS_SM100))


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and torch.version.hip is None and _LEDGER_ARCH,
    "requires CUDA and Triton on sm_89, sm_90 or sm_100 (the ledger's measured arches)",
)
class PointwiseStrictNumericsTest(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)
        # Release large reference-input allocations between concurrent tests.
        torch.cuda.empty_cache()

    def _is_tensor_output(self, out):
        return isinstance(out, torch.Tensor) or (
            isinstance(out, (tuple, list))
            and all(isinstance(t, torch.Tensor) for t in out)
        )

    def _is_pointwise_output(self, inp, args, kwargs, out):
        shapes = [inp.shape] + [a.shape for a in args if isinstance(a, torch.Tensor)]
        shapes += [v.shape for v in kwargs.values() if isinstance(v, torch.Tensor)]
        try:
            bshape = torch.broadcast_shapes(*shapes)
        except Exception:
            return False
        outs = out if isinstance(out, (tuple, list)) else (out,)
        return all(isinstance(o, torch.Tensor) and o.shape == bshape for o in outs)

    def _build_calls(self, samples, dtype):
        r"""Build calls without flattening so reference layouts exercise codegen."""
        calls = []
        for sample in samples:
            if not isinstance(sample.input, torch.Tensor):
                continue
            tensors = [sample.input, *sample.args]
            if any(isinstance(t, torch.Tensor) and t.dtype == dtype for t in tensors):
                calls.append((sample.input, tuple(sample.args), dict(sample.kwargs)))
        return calls

    def _bitpattern_call(self, op, dtype, device):
        r"""Build valid raw-bit calls for each sampled signature and mask parity."""
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
        # Cross special values to cover missing pairs such as (+0, -0).
        sp = _SPECIALS.to(device=device, dtype=dtype)
        x = torch.cat([x, sp.repeat_interleave(sp.numel())])
        y = torch.cat([y, sp.repeat(sp.numel())])
        try:
            samples = list(op.sample_inputs(device, dtype, requires_grad=False))
        except Exception:
            return []

        def sig(v):
            # Keep scalar-tensor and vector-tensor signatures distinct.
            return tuple(v.shape) if isinstance(v, torch.Tensor) else repr(v)

        calls = []
        seen = set()
        for sample in samples:
            if not isinstance(sample.input, torch.Tensor):
                continue
            operands = (*sample.args, *sample.kwargs.values())
            masked = any(
                isinstance(t, torch.Tensor) and t.dtype == torch.bool for t in operands
            )
            for parity in (0, 1) if masked else (0,):
                args = tuple(_substitute(a, y, x.numel(), parity) for a in sample.args)
                kwargs = {
                    k: _substitute(v, y, x.numel(), parity)
                    for k, v in sample.kwargs.items()
                }
                key = (
                    parity,
                    tuple(sig(a) for a in args),
                    tuple(sorted((k, sig(v)) for k, v in kwargs.items())),
                )
                if key in seen:
                    continue
                seen.add(key)
                try:
                    op.op(x, *args, **kwargs)
                except Exception:
                    continue  # e.g. per-channel PReLU weights
                calls.append((x, args, kwargs))
        return calls

    def _collect_calls(self, device, op, dtype):
        r"""Return (source, input, args, kwargs) calls for both sweeps."""
        if op.name in COMPILE_UNSUPPORTED:
            self.skipTest("uncompilable op under fullgraph")
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
                op.op(inp, *args, **kwargs)
                if detector.has_rng:
                    break
        if detector.has_rng:
            self.skipTest("RNG op excluded (RNG-source equivalence is separate)")
        return calls

    def _require_kernel(self, tested, nothing_tested):
        # ATen fallbacks compare eager against itself, so do not count them as coverage.
        if tested == 0:
            self.skipTest(nothing_tested)
        if metrics.generated_kernel_count == 0:
            self.skipTest("no Triton kernel generated (op falls back to ATen)")

    def _sweep(self, device, op, dtype, cfg):
        r"""Return mismatching (source, index, shape, kwargs, kind) records."""
        calls = self._collect_calls(device, op, dtype)

        def fn(inp, args, kwargs):
            return op.op(inp, *args, **kwargs)

        tested = 0
        mismatches = []
        with (
            config.patch(cfg),
            torch._dynamo.config.patch(
                recompile_limit=sys.maxsize,
                accumulated_recompile_limit=sys.maxsize,
            ),
        ):
            metrics.reset()
            for idx, (tag, inp, args, kwargs) in enumerate(calls):
                eager = fn(inp, args, kwargs)
                if not self._is_tensor_output(eager):
                    continue
                if not self._is_pointwise_output(inp, args, kwargs, eager):
                    continue
                torch._dynamo.reset()
                result = torch.compile(fn, fullgraph=True)(inp, args, kwargs)
                tested += 1
                if not _outputs_equal(eager, result):
                    kind = _diff_kind(eager, result)
                    mismatches.append((tag, idx, tuple(inp.shape), kwargs, kind))

        if not mismatches:
            # A mismatch must not be turned into a no-kernel skip.
            self._require_kernel(tested, "no usable sample")
        return mismatches

    def _assert_ledger(self, mismatches, op, dtype, what, xfail, xfail_name):
        r"""Require listed pairs to differ and unlisted pairs to match eager."""
        key = (_op_id(op), _dtype_label(dtype))
        if key in xfail:
            self.assertTrue(
                mismatches,
                f"{key} {what} now matches eager under strict numerics; "
                f"remove it from {xfail_name}.",
            )
        else:
            self.assertFalse(
                mismatches,
                f"{key} {what} differs from eager under strict numerics "
                f"on (source, index, shape, kwargs, kind): {mismatches}.",
            )

    @ops(POINTWISE_OPS, allowed_dtypes=POINTWISE_DTYPES)
    def test_pointwise_bitwise(self, device, dtype, op):
        mismatches = self._sweep(device, op, dtype, POINTWISE_STRICT_CFG)
        self._assert_ledger(
            mismatches, op, dtype, "forward", POINTWISE_XFAIL, "POINTWISE_XFAIL"
        )

    @ops(NONFLOAT_INPUT_OPS, allowed_dtypes=_NONFLOAT_DTYPES)
    def test_pointwise_nonfloat(self, device, dtype, op):
        mismatches = self._sweep(device, op, dtype, POINTWISE_STRICT_CFG)
        self._assert_ledger(
            mismatches, op, dtype, "nonfloat", NONFLOAT_XFAIL, "NONFLOAT_XFAIL"
        )

    def _input_grads(self, call_fn, inp, args, kwargs, grad_output):
        leaves = []

        def leafify(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                # Preserve non-dense strides; clone expanded inputs to avoid overlap.
                src = t.detach()
                if src.is_contiguous() or 0 in src.stride():
                    leaf = src.clone().requires_grad_(True)
                else:
                    leaf = torch.empty_strided(
                        src.shape, src.stride(), dtype=src.dtype, device=src.device
                    )
                    leaf.copy_(src)
                    leaf.requires_grad_(True)
                leaves.append(leaf)
                return leaf
            return t

        inp2 = leafify(inp)
        args2 = tuple(leafify(a) for a in args)
        out = call_fn(inp2, args2, kwargs)
        target = _diff_output(out)
        if not leaves or target is None:
            return out, None
        if target.shape != grad_output.shape:
            return out, None
        grads = torch.autograd.grad(
            target, leaves, grad_outputs=grad_output, allow_unused=True
        )
        return out, grads

    def _sweep_backward(self, device, op, dtype, cfg):
        r"""Compare input gradients with identical upstream gradients."""
        calls = self._collect_calls(device, op, dtype)

        def fn(inp, args, kwargs):
            return op.op(inp, *args, **kwargs)

        tested = 0
        mismatches = []
        with (
            config.patch(cfg),
            torch._dynamo.config.patch(
                recompile_limit=sys.maxsize,
                accumulated_recompile_limit=sys.maxsize,
            ),
        ):
            metrics.reset()
            for idx, (tag, inp, args, kwargs) in enumerate(calls):
                with torch.no_grad():
                    probe = fn(inp, args, kwargs)
                probe_out = _diff_output(probe)
                if probe_out is None:
                    continue
                if not self._is_pointwise_output(inp, args, kwargs, probe):
                    continue
                # Exclude broadcasted inputs whose gradients require a reduction.
                diff_ts = [
                    t
                    for t in (inp, *args)
                    if isinstance(t, torch.Tensor) and t.is_floating_point()
                ]
                if any(t.shape != probe_out.shape for t in diff_ts):
                    continue
                gen = torch.Generator(device=probe_out.device).manual_seed(0)
                grad_output = torch.randn(
                    probe_out.shape,
                    generator=gen,
                    device=probe_out.device,
                    dtype=probe_out.dtype,
                )
                # Seed signed zeros, which randn does not reliably generate.
                flat = grad_output.reshape(-1)
                if flat.numel() >= 2:
                    flat[0] = 0.0
                    flat[1] = -0.0
                try:
                    _, eager_grads = self._input_grads(
                        fn, inp, args, kwargs, grad_output
                    )
                except Exception:
                    continue
                if eager_grads is None:
                    continue
                torch._dynamo.reset()
                compiled = torch.compile(fn, fullgraph=True)
                _, comp_grads = self._input_grads(
                    compiled, inp, args, kwargs, grad_output
                )
                tested += 1
                if comp_grads is None:
                    mismatches.append((tag, idx, tuple(inp.shape), kwargs, "shape"))
                    continue
                if not _outputs_equal(eager_grads, comp_grads):
                    kind = _diff_kind(eager_grads, comp_grads)
                    mismatches.append((tag, idx, tuple(inp.shape), kwargs, kind))

        if not mismatches:
            self._require_kernel(tested, "no differentiable sample")
        return mismatches

    @ops(BACKWARD_OPS, allowed_dtypes=POINTWISE_DTYPES)
    def test_pointwise_backward(self, device, dtype, op):
        mismatches = self._sweep_backward(device, op, dtype, POINTWISE_STRICT_CFG)
        self._assert_ledger(
            mismatches, op, dtype, "backward", BACKWARD_XFAIL, "BACKWARD_XFAIL"
        )


instantiate_device_type_tests(PointwiseStrictNumericsTest, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
