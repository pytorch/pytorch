# Owner(s): ["module: inductor"]
import contextlib
import unittest

import sympy

import torch
import torch._inductor.config as inductor_config
from torch._inductor.codegen import triton_utils
from torch._inductor.codegen.common import (
    CSE,
    CSEProxy,
    CSEVariable,
    IndentedBuffer,
    SizeArg,
)
from torch._inductor.codegen.triton import (
    _materialize_trunc_to_float_expr,
    TritonKernel,
    TritonKernelOverrides,
)
from torch._inductor.codegen.triton_ir import StructuredTritonKernelIR
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

    def test_structured_ir_tracks_reduction_loop_scope(self):
        structured_ir = StructuredTritonKernelIR(
            kernel_name=None,
            kernel_kind="TritonKernel",
            numels={"x": sympy.Integer(16), "r0_": sympy.Integer(8)},
            range_trees=[
                {
                    "name": "xindex",
                    "prefix": "x",
                    "numel": sympy.Integer(16),
                    "is_loop": False,
                    "is_reduction": False,
                    "grid_dim": 0,
                    "tensor_dim": 0,
                },
                {
                    "name": "r0_index",
                    "prefix": "r0_",
                    "numel": sympy.Integer(8),
                    "is_loop": True,
                    "is_reduction": True,
                    "grid_dim": None,
                    "tensor_dim": 1,
                },
            ],
        )

        structured_ir.register_loop_carried(
            CSEVariable("accum0", ValueRanges.unknown(), torch.float32, ("XBLOCK",))
        )
        with structured_ir.scope(
            "masked", "masked", section="compute", attrs={"mask": "xmask"}
        ):
            structured_ir.record_node(
                kind="op",
                op="add",
                section="compute",
                inputs=("tmp0", "tmp1"),
                outputs=("tmp2",),
                attrs={"expr": "(tmp0 + tmp1)"},
                dedupe_outputs=True,
            )

        structured = structured_ir.to_dict()
        self.assertEqual(structured["section_scopes"]["compute"], 1)
        self.assertEqual(structured["scopes"][1]["kind"], "loop")
        self.assertEqual(structured["scopes"][1]["loop_carried"][0]["name"], "accum0")
        self.assertEqual(structured["scopes"][2]["parent"], 1)
        self.assertEqual(structured["nodes"][0]["scope"], 2)

    def test_cse_proxy_records_structured_ir_for_triton_ops(self):
        class FakeTritonKernel:
            def __init__(self):
                self.compute = IndentedBuffer()
                self.cse = CSE()
                self.current_node = None
                self.node_to_bounds = None
                self.structured_ir = StructuredTritonKernelIR(
                    kernel_name="fake_kernel",
                    kernel_kind="FakeTritonKernel",
                    numels={"x": sympy.Integer(8)},
                    range_trees=[
                        {
                            "name": "xindex",
                            "prefix": "x",
                            "numel": sympy.Integer(8),
                            "is_loop": False,
                            "is_reduction": False,
                            "grid_dim": 0,
                            "tensor_dim": 0,
                        }
                    ],
                )

            def create_cse_var(self, name, bounds, dtype, shape):
                return CSEVariable(name, bounds, dtype, shape)

            def record_codegen_operation(
                self, *, name, args, kwargs, raw_value, result, section
            ):
                self.structured_ir.record_node(
                    kind="op",
                    op=name,
                    section=section,
                    inputs=args,
                    outputs=result,
                    attrs={"expr": raw_value, "kwargs": kwargs},
                    dedupe_outputs=True,
                )

        kernel = FakeTritonKernel()
        proxy = CSEProxy(kernel, TritonKernelOverrides())
        lhs = CSEVariable("tmp_lhs", ValueRanges.unknown(), torch.float32, ("XBLOCK",))
        rhs = CSEVariable("tmp_rhs", ValueRanges.unknown(), torch.float32, ("XBLOCK",))

        with (
            unittest.mock.patch(
                "torch._inductor.codegen.common.get_current_backend",
                return_value="triton",
            ),
            V.set_kernel_handler(kernel),
            V.set_ops_handler(proxy),
        ):
            first = V.ops.add(lhs, rhs)
            second = V.ops.add(lhs, rhs)

        self.assertEqual(str(first), str(second))
        self.assertEqual(len(kernel.compute._lines), 1)
        structured = kernel.structured_ir.to_dict()
        self.assertEqual(len(structured["nodes"]), 1)
        self.assertEqual(structured["nodes"][0]["op"], "add")
        self.assertEqual(structured["nodes"][0]["outputs"][0]["name"], str(first))
        self.assertEqual(structured["nodes"][0]["inputs"][0]["value"], "tmp_lhs")
        self.assertEqual(structured["nodes"][0]["inputs"][1]["value"], "tmp_rhs")

    def test_cse_proxy_records_partial_accumulate_with_named_args(self):
        class FakeTritonKernel:
            def __init__(self):
                self.compute = IndentedBuffer()
                self.cse = CSE()
                self.current_node = None
                self.node_to_bounds = None
                self.saved_partial_accumulates = []
                self.structured_ir = StructuredTritonKernelIR(
                    kernel_name="fake_kernel",
                    kernel_kind="FakeTritonKernel",
                    numels={},
                    range_trees=[],
                )

            def partial_accumulate(self, name, reduction_type, value, extra_meta):
                self.saved_partial_accumulates.append(
                    (name, reduction_type, value, extra_meta)
                )

            def record_codegen_partial_accumulate(
                self, *, name, reduction_type, value, extra_meta
            ):
                self.structured_ir.record_node(
                    kind="reduction",
                    op="partial_accumulate",
                    section="compute",
                    inputs=(value,),
                    attrs={
                        "buffer": name,
                        "reduction_type": reduction_type,
                        "extra_meta": extra_meta,
                    },
                )

        kernel = FakeTritonKernel()
        proxy = CSEProxy(kernel, TritonKernelOverrides())
        value = CSEVariable(
            "tmp_val", ValueRanges.unknown(), torch.float32, ("XBLOCK",)
        )

        proxy.partial_accumulate(
            "acc",
            "sum",
            value,
            {"is_first_reduction": True},
        )

        self.assertEqual(
            kernel.saved_partial_accumulates,
            [("acc", "sum", value, {"is_first_reduction": True})],
        )
        structured = kernel.structured_ir.to_dict()
        self.assertEqual(len(structured["nodes"]), 1)
        self.assertEqual(structured["nodes"][0]["op"], "partial_accumulate")
        self.assertEqual(structured["nodes"][0]["attrs"]["buffer"], "acc")
        self.assertEqual(structured["nodes"][0]["attrs"]["reduction_type"], "sum")

    @unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_real_triton_codegen_exposes_structured_ir(self):
        captured_structured_irs = []
        original_codegen_kernel = TritonKernel.codegen_kernel

        def capture_codegen_kernel(kernel, name=None):
            code = original_codegen_kernel(kernel, name)
            captured_structured_irs.append(kernel.structured_ir_to_dict())
            return code

        def fn(x):
            return torch.argmax(x, dim=1)

        x = torch.randn(64, 32, device=GPU_TYPE)
        expected = fn(x)
        with unittest.mock.patch.object(
            TritonKernel,
            "codegen_kernel",
            new=capture_codegen_kernel,
        ):
            actual = torch.compile(fn)(x)

        self.assertEqual(actual, expected)
        self.assertTrue(captured_structured_irs)
        self.assertTrue(
            any(
                any(node["op"] == "argmax" for node in structured_ir["nodes"])
                for structured_ir in captured_structured_irs
            )
        )
        self.assertTrue(
            any(
                any(
                    value["name"].endswith("_index")
                    and value["dtype"] is not None
                    and value["shape"] is not None
                    for scope in structured_ir["scopes"]
                    for value in scope["loop_carried"]
                )
                for structured_ir in captured_structured_irs
            )
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
