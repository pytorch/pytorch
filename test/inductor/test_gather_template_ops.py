# Owner(s): ["module: inductor"]

from unittest import mock

import sympy

import torch
import torch._inductor.utils
# Importing lowering populates op_dtype_propagation_rules.
import torch._inductor.lowering
from torch._inductor.codegen.triton import TritonKernelOverrides
from torch._inductor.dtype_propagation import DtypePropagationOpsHandler
from torch._inductor.shape_propagation import broadcast_shapes_for_args
from torch._inductor.utils import OpDtypeRule
from torch._prims_common import ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.testing._internal.common_utils import TestCase, run_tests


class TestGatherTemplateOps(TestCase):
    """Contract tests for out-of-tree gather_template op support.

    An out-of-tree backend can register a "gather_template" op rule whose
    lowering receives backend metadata (strings) and scalar index
    expressions (sympy) next to the value arguments. These tests pin the
    base passthrough lowering and the dtype/shape propagation tolerance;
    without a registered rule none of this code is reachable.
    """

    def test_triton_base_lowering_passthrough(self):
        # The base lowering returns the already-computed gathered value; the
        # remaining arguments are backend-specific metadata. self is unused.
        x_value = object()
        self.assertIs(
            TritonKernelOverrides.gather_template(
                None, "loader_name", x_value, "index", "gather_idx", "boundary"
            ),
            x_value,
        )

    def test_dtype_rule_skips_metadata_args(self):
        f32 = torch.empty((), dtype=torch.float32)
        i64 = torch.empty((), dtype=torch.int64)
        i0 = sympy.Symbol("i0", integer=True)
        kwargs = {"type_promotion_kind": ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT}
        # Only value arguments participate in promotion; metadata strings and
        # sympy index expressions are ignored.
        self.assertEqual(
            DtypePropagationOpsHandler.op_dtype_rule_skip_str(
                "loader_name", i64, i0, f32, **kwargs
            ),
            torch.float32,
        )
        # Numbers still participate.
        self.assertEqual(
            DtypePropagationOpsHandler.op_dtype_rule_skip_str(
                "loader_name", i64, 1, **kwargs
            ),
            torch.int64,
        )

    def test_base_dtype_rule_rejects_str_args(self):
        # The base rule asserts on str args; this is why the skip-str variant
        # is needed for gather_template-style ops.
        with self.assertRaises(AssertionError):
            DtypePropagationOpsHandler.op_dtype_rule(
                "loader_name",
                torch.empty((), dtype=torch.float32),
                type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            )

    def test_dtype_handler_registers_gather_template_rule(self):
        rules = dict(torch._inductor.utils.op_dtype_propagation_rules)
        rules["gather_template"] = OpDtypeRule(
            ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, None
        )
        saved_key = DtypePropagationOpsHandler._rules_key
        try:
            with mock.patch.object(
                torch._inductor.utils, "op_dtype_propagation_rules", rules
            ):
                # Bypass the singleton; __init__ meta-programs the rules onto
                # this throwaway instance.
                handler = object.__new__(DtypePropagationOpsHandler)
                DtypePropagationOpsHandler.__init__(handler)
                self.assertEqual(
                    handler.gather_template(
                        "loader_name", torch.empty((), dtype=torch.float32), "meta"
                    ),
                    torch.float32,
                )
        finally:
            DtypePropagationOpsHandler._rules_key = saved_key

    def test_broadcast_shapes_for_args_tolerates_metadata(self):
        i0 = sympy.Symbol("i0", integer=True)
        # A scalar index expression contributes a 0-d shape.
        self.assertEqual(broadcast_shapes_for_args((i0,)), ())
        self.assertEqual(broadcast_shapes_for_args(("loader_name", i0)), ())
        # A metadata string alone contributes no shape information.
        self.assertIsNone(broadcast_shapes_for_args(("loader_name",)))
        # Real shapes still broadcast.
        self.assertEqual(
            broadcast_shapes_for_args(("loader_name", torch.empty(2, 3))), (2, 3)
        )
        # Unknown types still raise: the tolerance must not swallow bugs.
        with self.assertRaises(TypeError):
            broadcast_shapes_for_args((object(),))


if __name__ == "__main__":
    run_tests()
