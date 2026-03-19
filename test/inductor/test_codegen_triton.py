# Owner(s): ["module: inductor"]
import contextlib

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
from torch._inductor.virtualized import V
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU
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


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests("sympy")
