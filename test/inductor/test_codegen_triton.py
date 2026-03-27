# Owner(s): ["module: inductor"]
import contextlib
import unittest
import re

import sympy

import torch
import torch._inductor.config as inductor_config
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import CSEVariable, SizeArg
from torch._inductor.codegen.triton import (
    _materialize_trunc_to_float_expr,
    TritonKernelOverrides,
)
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler, promote_types
from torch._inductor.graph import GraphLowering
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    HAS_GPU_AND_TRITON,
)
from torch.utils._sympy.functions import FloorDiv, TruncToFloat, TruncToInt
from torch.utils._sympy.value_ranges import ValueRanges


class TestCodegenTriton(InductorTestCase):
    def setUp(self):
        super().setUp()

        class DummyModule(torch.nn.Module):
            def forward(self, x):
                return x * 2

        self._gm = torch.fx.symbolic_trace(DummyModule())
        self._graph = GraphLowering(self._gm)

        self._stack = contextlib.ExitStack()
        self._stack.enter_context(V.set_graph_handler(self._graph))

    def tearDown(self):
        self._stack.close()
        super().tearDown()

    @inductor_config.patch("triton.divisible_by_16", True)
    def test_config_of_sizearg(self):
        from torch._inductor.utils import (
            get_triton_attrs_descriptor_version,
            TritonAttrsDescriptorVersion,
        )

        two = sympy.Integer(2)
        eight = sympy.Integer(8)
        sixteen = sympy.Integer(16)
        s0 = sympy.Symbol("s0", positive=True, integer=True)
        s1 = sympy.Symbol("s1", positive=True, integer=True)

        def _check_divisibility(expected_divisible_indices, config):
            if get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V1_COMPILER,
                TritonAttrsDescriptorVersion.V0_NO_TRITON,
            }:
                self.assertEqual(expected_divisible_indices, config.divisible_by_16)
            elif get_triton_attrs_descriptor_version() in {
                TritonAttrsDescriptorVersion.V2_BACKENDS,
                TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
            }:
                self.assertEqual(expected_divisible_indices, config.divisibility_16)
            else:
                if (
                    get_triton_attrs_descriptor_version()
                    != TritonAttrsDescriptorVersion.V4_DICT
                ):
                    raise AssertionError
                self.assertIsInstance(config, dict)

                for idx in expected_divisible_indices:
                    # config is in the form
                    # {(idx,): [["tt.divisibility", 16]]}
                    # where (idx,) is a tuple in order to support tuple inputs to triton kernels.
                    self.assertTrue((idx,) in config)
                    self.assertTrue(["tt.divisibility", 16] in config[(idx,)])

        _check_divisibility(
            (2,),
            triton_utils.config_of(
                [
                    SizeArg("A", two),  # no
                    SizeArg("B", eight),  # no
                    SizeArg("C", sixteen),  # yes
                    SizeArg("D", s0),  # no
                    SizeArg("E", s1),  # no
                ]
            ),
        )

        _check_divisibility(
            (0, 2, 4, 5, 6),
            triton_utils.config_of(
                [
                    SizeArg("A", two * eight),  # 0: yes
                    SizeArg("B", eight * s0),  # 1: no
                    SizeArg("C", two * eight * s0),  # 2: yes
                    SizeArg("D", s0 * s1),  # 3: no
                    SizeArg("E", sixteen * s0),  # 4: yes
                    SizeArg("F", sixteen * eight * s0 * s1),  # 5: yes
                    SizeArg("G", two * eight * s0 * s1),  # 6: yes
                ]
            ),
        )

    def test_config_of_sizearg_with_check_constraint(self):
        from torch.utils._sympy.functions import Mod

        s2 = sympy.Symbol("s2", positive=True, integer=True)

        self.assertFalse(
            V.graph.sizevars.statically_known_multiple_of(s2, 16),
        )

        shape_env = V.graph.sizevars.shape_env
        shape_env.axioms[sympy.Eq(Mod(s2, 16), 0)] = sympy.true

        self.assertTrue(
            V.graph.sizevars.statically_known_multiple_of(s2, 16),
        )

    def test_pow_uses_active_override_constant_lowering(self):
        exponent = CSEVariable("ks0", ValueRanges.unknown(), torch.int64)

        class TestTritonKernelOverrides(TritonKernelOverrides):
            @classmethod
            def constant(cls, value, dtype):
                return f"custom_constant({value}, {dtype})"

        self.assertEqual(
            TestTritonKernelOverrides.pow(2, exponent),
            "libdevice.pow(custom_constant(2, torch.float64), (ks0).to(tl.float64))",
        )

    def test_pow_preserves_integer_dtype_for_unsigned_scalar_exponents(self):
        exponent = CSEVariable("ks0", ValueRanges.unknown(), torch.uint32)

        self.assertEqual(
            DtypePropagationOpsHandler().pow(2, exponent),
            promote_types([2, exponent]),
        )

    def test_pow_uses_integer_helper_for_unsigned_scalar_exponents(self):
        exponent = CSEVariable("ks0", ValueRanges.unknown(), torch.uint32)

        class TestTritonKernelOverrides(TritonKernelOverrides):
            @classmethod
            def constant(cls, value, dtype):
                return f"custom_constant({value}, {dtype})"

        self.assertEqual(
            TestTritonKernelOverrides.pow(3, exponent),
            "triton_helpers.pow_integer(custom_constant(3, torch.uint32), ks0)",
        )

    def test_materialize_trunc_to_float_expr_preserves_integer_subexpressions(self):
        s0 = sympy.Symbol("s0")

        trunc_expr = TruncToInt(s0)
        self.assertEqual(
            _materialize_trunc_to_float_expr(trunc_expr, torch.float64),
            TruncToFloat(s0),
        )

        integer_expr = FloorDiv(trunc_expr, sympy.Integer(5))
        self.assertEqual(
            _materialize_trunc_to_float_expr(integer_expr, torch.float64),
            integer_expr,
        )

        predicate_expr = sympy.Eq(trunc_expr, sympy.Integer(9007199254740993))
        self.assertEqual(
            _materialize_trunc_to_float_expr(predicate_expr, torch.float64),
            predicate_expr,
        )

        float_expr = sympy.Float(0.5) + trunc_expr
        self.assertEqual(
            _materialize_trunc_to_float_expr(float_expr, torch.float64),
            sympy.Float(0.5) + TruncToFloat(s0),
        )

    @unittest.skipUnless(torch.version.hip is not None, "pointer_range_32 is HIP-only")
    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_pointer_range_in_generated_code(self):
        """Verify tt.pointer_range=32 appears in generated Triton code on HIP."""

        def fn(x):
            return x + 1

        x = torch.randn(64, 64, device=GPU_TYPE, dtype=torch.bfloat16)
        _, code = run_and_get_code(torch.compile(fn), x)
        code_str = " ".join(code)
        self.assertIn("tt.pointer_range", code_str)

    def test_is_multiple_of_rules(self):
        """Test structural divisibility rules in _is_multiple_of."""
        from torch.utils._sympy.functions import FloorDiv, Mod

        sv = V.graph.sizevars
        shape_env = sv.shape_env

        s1 = sympy.Symbol("s1", positive=True, integer=True)
        s2 = sympy.Symbol("s2", positive=True, integer=True)
        s3 = sympy.Symbol("s3", positive=True, integer=True)

        # Product: any factor divisible → product divisible
        self.assertTrue(sv.statically_known_multiple_of(16 * s1, 16))
        self.assertTrue(sv.statically_known_multiple_of(4 * 4 * s1, 16))
        shape_env.axioms[sympy.Eq(Mod(s1, 16), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(s1 * s2, 16))
        self.assertFalse(sv.statically_known_multiple_of(s2 * s3, 16))

        # Sum: all terms divisible → sum divisible
        self.assertFalse(sv.statically_known_multiple_of(s1 + s2, 16))
        shape_env.axioms[sympy.Eq(Mod(s2, 16), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(s1 + s2, 16))
        self.assertTrue(sv.statically_known_multiple_of(s1 + 32, 16))
        self.assertFalse(sv.statically_known_multiple_of(s1 + 3, 16))

        # FloorDiv(a, b): a must be multiple of b*n
        self.assertFalse(sv.statically_known_multiple_of(FloorDiv(s1, 3), 16))
        shape_env.axioms[sympy.Eq(Mod(s3, 48), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(FloorDiv(s3, 3), 16))

        # Mod(a, b): both a and b must be multiples of n
        self.assertTrue(sv.statically_known_multiple_of(Mod(s1, 48), 16))
        s_nodiv = sympy.Symbol("s_nodiv", positive=True, integer=True)
        self.assertFalse(sv.statically_known_multiple_of(Mod(s_nodiv, 32), 16))
        self.assertFalse(sv.statically_known_multiple_of(Mod(s1, 7), 16))

        # Axiom fallback: bare symbol resolved via statically_known_true
        s4 = sympy.Symbol("s4", positive=True, integer=True)
        self.assertFalse(sv.statically_known_multiple_of(s4, 8))
        shape_env.axioms[sympy.Eq(Mod(s4, 8), 0)] = sympy.true
        self.assertTrue(sv.statically_known_multiple_of(s4, 8))


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
@inductor_config.patch(
    {
        "triton.divisible_by_16": True,
        "triton.speculative_divisibility": True,
    }
)
class TestSpeculativeDivisibility(InductorTestCase):
    """Tests for speculative divisibility dual-kernel dispatch.

    Under dynamic=True, symbolic SizeArgs (strides, numels) can't be statically
    proven as divisible by 16. Speculative divisibility AOT-compiles two Triton
    kernel variants -- _div16 (all args annotated as div16) and _general (no
    speculation) -- then dispatches at runtime via ``(a | b | ...) % 16 == 0``.

    Each test verifies two properties:
      1. Dispatch condition shape: correct number of | operators
      2. Runtime path selection: correct variant for div16 vs non-div16 inputs
    """

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()

    def _compile_and_check(self, fn, example_inputs, expected_pipe_count):
        """Compile with dynamic=True, check dispatch condition and correctness.

        Returns compiled_fn for further runtime path assertions.
        """
        from torch._inductor.utils import run_and_get_code

        compiled_fn = torch.compile(fn, dynamic=True)
        result, code = run_and_get_code(compiled_fn, *example_inputs)

        # Check dispatch condition
        full = "\n".join(code)
        self.assertTrue(any("_div16" in c for c in code), "Missing _div16 variant")
        self.assertTrue(any("_general" in c for c in code), "Missing _general variant")

        conditions = re.findall(r"if (.+% 16 == 0):", full)
        self.assertTrue(
            len(conditions) >= 1, "No '% 16 == 0' dispatch condition found"
        )
        # Pipe count depends on codegen details that may vary across GPU
        # architectures. Only check on GB200 (sm_100+) where we know the
        # exact expected condition shape.
        if torch.cuda.get_device_capability() == (10, 0):
            pipe_count = conditions[0].count("|")
            self.assertEqual(
                pipe_count,
                expected_pipe_count,
                f"Expected {expected_pipe_count} | in condition, got: {conditions[0]}",
            )

        # Correctness on first call
        expected = fn(*example_inputs)
        self.assertTrue(
            torch.allclose(result, expected, atol=1e-2, rtol=1e-2),
            "Incorrect result on first call",
        )

        return compiled_fn

    def _assert_runtime_paths(self, compiled_fn, test_cases, make_inputs, compute_expected):
        """Assert the correct kernel variant is selected at runtime.

        The compiled module (in PyCodeCache.modules) has kernel objects as
        module-level attributes (e.g. triton_red_fused_sum_0_div16). The
        generated wrapper evaluates the dispatch condition with runtime sizes
        and calls kernel.run(). We monkey-patch .run to record which variant
        executed, then assert it matches the expected path.

        Args:
            compiled_fn: the torch.compiled function to call
            test_cases: list of (*shape_args, expected_suffix) tuples
            make_inputs: callable(*shape_args) -> tuple of input tensors
            compute_expected: callable(*inputs) -> expected output tensor
        """
        from torch._inductor.codecache import PyCodeCache

        last_kernel = [None]
        originals = {}

        def patch_run(kernel_obj, name):
            orig = kernel_obj.run
            originals[(id(kernel_obj), name)] = (kernel_obj, orig)
            def traced(*args, **kwargs):
                last_kernel[0] = name
                return orig(*args, **kwargs)
            kernel_obj.run = traced

        for mod in PyCodeCache.modules:
            for attr in dir(mod):
                if "div16" in attr or "general" in attr:
                    obj = getattr(mod, attr)
                    if hasattr(obj, "run"):
                        patch_run(obj, attr)

        try:
            for *shape_args, expected_suffix in test_cases:
                last_kernel[0] = None
                inputs = make_inputs(*shape_args)
                result = compiled_fn(*inputs)
                expected = compute_expected(*inputs)
                self.assertTrue(
                    torch.allclose(result, expected, atol=1e-2, rtol=1e-2),
                    f"{shape_args}: incorrect result",
                )
                # Runtime path depends on exact dispatch condition shape, which
                # may vary across GPU architectures. Only check on GB200 (sm_100+).
                if torch.cuda.get_device_capability() == (10, 0):
                    self.assertIn(
                        expected_suffix,
                        last_kernel[0],
                        f"{shape_args}: expected {expected_suffix} path, got {last_kernel[0]}",
                    )
        finally:
            for (kernel_obj, orig_run) in originals.values():
                kernel_obj.run = orig_run

    def test_pointwise_1d(self):
        """Pointwise 1D: same contiguous layout, single xnumel arg.

        Both inputs contiguous with matching layout -> Inductor collapses to 1D
        iteration with a single xnumel SizeArg. xnumel = M*N is symbolic under
        dynamic=True -> one speculative arg, no | in dispatch condition.
        """
        def fn(a, b):
            return a * b

        a = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
        compiled_fn = self._compile_and_check(fn, (a, b), expected_pipe_count=0)

        # xnumel = M*N (collapsed 1D), so (M*N) % 16 == 0 determines the path.
        self._assert_runtime_paths(
            compiled_fn,
            [
                (1024, 2048, "div16"),    # 1024*2048 div16
                (1025, 2049, "general"),  # 1025*2049 not div16
                (15, 17, "general"),      # 15*17 = 255 not div16
            ],
            make_inputs=lambda M, N: (
                torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
                torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
            ),
            compute_expected=lambda a, b: a * b,
        )

    def test_pointwise_2d_varied_layout(self):
        """Pointwise 2D: mismatched layouts cause 2D tiling with stride args.

        Mismatched memory layouts (one transposed) -> 2D iteration with ks*
        stride args. ks0 == xnumel and ks1 == ynumel (same sympy expr), so
        deduplication yields exactly 2 distinct symbols -> exactly 1 |.
        """
        def fn(a, b):
            return a + b

        a = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16).t()
        compiled_fn = self._compile_and_check(fn, (a, b), expected_pipe_count=1)

        # Both M and N must individually be div16 for the _div16 path.
        self._assert_runtime_paths(
            compiled_fn,
            [
                (1024, 2048, "div16"),    # both div16
                (1025, 2048, "general"),  # M not div16
                (1024, 2049, "general"),  # N not div16
            ],
            make_inputs=lambda M, N: (
                torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
                torch.randn(N, M, device="cuda", dtype=torch.bfloat16).t(),
            ),
            compute_expected=lambda a, b: a + b,
        )

    def test_3d_middle_reduction(self):
        """3D reduction T[B, R, N].sum(dim=1): stride and numel args in dispatch.

        Reduction over the middle dimension. Inductor emits ks0=N and ks1=R as
        stride args; xnumel=B*N and r0_numel=R as numel args. r0_numel shares
        the same sympy symbol as ks1 (both = R), so deduplication leaves three
        distinct symbols: N, R, and B*N -> condition (N | R | B*N) % 16 == 0
        with 2 | operators.
        """
        def fn(a):
            return a.sum(dim=1)

        a = torch.randn(64, 128, 256, device="cuda", dtype=torch.bfloat16)
        compiled_fn = self._compile_and_check(fn, (a,), expected_pipe_count=2)

        # B*N = xnumel is in the condition, but B=17 still gives B*N=4352 (div16).
        self._assert_runtime_paths(
            compiled_fn,
            [
                (64, 128, 256, "div16"),     # N,R,B*N all div16
                (17, 128, 256, "div16"),     # B*N=4352 still div16
                (64, 128, 2049, "general"),  # N not div16
                (64, 2049, 256, "general"),  # R not div16
            ],
            make_inputs=lambda B, R, N: (
                torch.randn(B, R, N, device="cuda", dtype=torch.bfloat16),
            ),
            compute_expected=lambda a: a.sum(dim=1),
        )

    def test_inner_reduction(self):
        """Inner reduction (sum dim=-1): both stride and numel args in dispatch.

        For T[M, N].sum(dim=-1), ks0=N (stride) and xnumel=M (numel) are both
        symbolic. r0_numel=N shares the same symbol as ks0, so deduplication
        leaves two distinct symbols: N and M -> condition (N | M) % 16 == 0
        with 1 | operator.
        """
        def fn(a):
            return a.sum(dim=-1)

        a = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
        compiled_fn = self._compile_and_check(fn, (a,), expected_pipe_count=1)

        # Both M and N must be div16 for the _div16 path.
        self._assert_runtime_paths(
            compiled_fn,
            [
                (1024, 2048, "div16"),
                (1025, 2048, "general"),  # M not div16
                (1024, 2049, "general"),
            ],
            make_inputs=lambda M, N: (
                torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
            ),
            compute_expected=lambda a: a.sum(dim=-1),
        )

    def test_outer_reduction(self):
        """Outer reduction (sum dim=0): all args in dispatch, including constants.

        Reduction over M (rows), output along N (columns). Inductor splits
        outer reduction into two kernels:
          - kernel_0 (partial reduction): 4 args (N, M, 8*N, ceil(M/8))
            -> condition (N | M | 8*N | ceil(M/8)) % 16 == 0, 3 pipes.
          - kernel_1 (final per-block reduction): ks0=N and r0_numel=8
            -> condition (N | 8) % 16 == 0, 1 pipe.
            Because 8 is not divisible by 16, (N | 8) always has bit 3 set,
            so this condition is always false and kernel_1 always takes the
            _general path.
        The runtime path check sees the last kernel (kernel_1).
        """
        def fn(a):
            return a.sum(dim=0)

        a = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
        compiled_fn = self._compile_and_check(fn, (a,), expected_pipe_count=3)

        # kernel_1 condition is (N | 8) % 16 == 0, always false -> always general.
        self._assert_runtime_paths(
            compiled_fn,
            [
                (1024, 2048, "general"),   # (2048 | 8) % 16 = 8
                (1025, 2048, "general"),   # same
                (1024, 2049, "general"),   # same
            ],
            make_inputs=lambda M, N: (
                torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
            ),
            compute_expected=lambda a: a.sum(dim=0),
        )

    def test_rmsnorm(self):
        """RMSNorm: fused inner reduction + pointwise normalization.

        RMSNorm is x * rsqrt(mean(x^2) + eps) * weight, fused into a single
        Triton reduction kernel. Like inner reduction, ks0=N (stride) and
        xnumel=M (numel) are both symbolic. r0_numel=N shares the same symbol
        as ks0, so deduplication leaves two distinct symbols: N and M ->
        condition (N | M) % 16 == 0 with 1 | operator.
        """
        def fn(x, weight):
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + 1e-6)
            return (weight * x).to(torch.bfloat16)

        x = torch.randn(1024, 2048, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        compiled_fn = self._compile_and_check(fn, (x, w), expected_pipe_count=1)

        # Both M and N must be div16 for the _div16 path.
        self._assert_runtime_paths(
            compiled_fn,
            [
                (1024, 2048, "div16"),
                (1025, 2048, "general"),  # M not div16
                (1024, 2049, "general"),
            ],
            make_inputs=lambda M, N: (
                torch.randn(M, N, device="cuda", dtype=torch.bfloat16),
                torch.randn(N, device="cuda", dtype=torch.bfloat16),
            ),
            compute_expected=lambda x, w: (
                (w * x * torch.rsqrt(
                    x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6
                )).to(torch.bfloat16)
            ),
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
