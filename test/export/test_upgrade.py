# Owner(s): ["oncall: export"]
import unittest
from unittest.mock import patch

import torch
import torch._dynamo as torchdynamo
from torch.export import export
from torch._export.serde.serialize import GraphModuleOpUpgrader
from torch._export.serde.upgrade import get_target_version, get_upgraders
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    IS_WINDOWS,
)

TEST_UPGRADERS = {
    "aten::div__Scalar_mode_0_3": (
        "div.Scalar_mode(Tensor self, Scalar other, *, str rounding_mode) -> Tensor",
        """
from typing import Any, Optional
def div__Scalar_mode_0_3(self: torch.Tensor, other: Any,  *, rounding_mode: Optional[str]=None) -> torch.Tensor:
    return self.divide(other, rounding_mode=rounding_mode)
        """,
    ),
    "aten::gelu_0_9": (
        "gelu(Tensor self) -> Tensor",
        """
def gelu_0_9(self: Tensor) -> Tensor:
  return torch.gelu(self, approximate='none')
        """,
    ),
}

TEST_UPGRADERS_ENTRY_MAP = {
    "div__Scalar_mode_0_3":
        """
from typing import Any, Optional
def div__Scalar_mode_0_3(self: torch.Tensor, other: Any,  *, rounding_mode: Optional[str]=None) -> torch.Tensor:
    return self.divide_(other, rounding_mode=rounding_mode)"""
}

TEST_OP_VERSION_MAP = {
    "aten::div_.Scalar_mode": [
        torch._C._UpgraderEntry(
            4,
            "div__Scalar_mode_0_3",
            "aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)"
        )
    ]
}


def count_op(graph, target_str):
    return len(
        [n for n in graph.nodes if isinstance(n.target, torch._ops.OpOverload) and n.target.name() == target_str])


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo doesn't support")
class TestUpgrade(TestCase):
    def test_get_upgraders(self):
        with patch.object(torch._C, "_get_upgraders_entry_map", return_value=TEST_UPGRADERS_ENTRY_MAP), \
                patch.object(torch._C, "_get_operator_version_map", return_value=TEST_OP_VERSION_MAP):
            op_upgraders = get_upgraders()
            self.assertEqual(op_upgraders, {
                "div__Scalar_mode_0_3": (
                    "aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)",
                    """
from typing import Any, Optional
def div__Scalar_mode_0_3(self: torch.Tensor, other: Any,  *, rounding_mode: Optional[str]=None) -> torch.Tensor:
    return self.divide_(other, rounding_mode=rounding_mode)""",
                )})

    def test_get_upgraders_missing_from_entry_map_raises(self):
        with patch.object(torch._C, "_get_upgraders_entry_map", return_value={}), \
                patch.object(torch._C, "_get_operator_version_map", return_value=TEST_OP_VERSION_MAP):
            with self.assertRaises(RuntimeError):
                get_upgraders()

    def test_upgrader_with_invalid_format_throws_exception(self):
        """Invalid upgrader function string should throw exception"""
        upgraders = [("div(Tensor a, Tensor b) -> Tensor", "TEST")]
        with self.assertRaises(RuntimeError):
            GraphModuleOpUpgrader._populate_passes(upgraders)

    def test_get_target_version_invalid_format_throws_exception(self):
        with self.assertRaises(RuntimeError):
            get_target_version("div_0")
        with self.assertRaises(RuntimeError):
            get_target_version("div_0_")
        with self.assertRaises(RuntimeError):
            get_target_version("div")

    def test_creates_upgrader_pass(self):
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)
        self.assertEqual(len(upgrader.upgrader_passes), 1)

    def test_div_upgrader_replaces_op_with_old_version(self):
        class Foo(torch.nn.Module):
            def forward(self, a: torch.Tensor, b):
                return torch.ops.aten.div.Scalar_mode(a, b, rounding_mode='trunc')

        fn = Foo()

        inputs = (torch.ones([2, 3]) * 4, 2.)
        ep = export(fn, inputs, [])
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)
        upgraded = ep._transform_do_not_use(*upgrader.upgrader_passes)
        upgraded.graph_module.print_readable()

        count = count_op(upgraded.graph, "aten::div.Scalar_mode")
        self.assertEqual(count, 0)
        custom_op_count = count_op(upgraded.graph, "aten::div__Scalar_mode_0_3")
        self.assertEqual(custom_op_count, 1)

    @unittest.skipIf(IS_WINDOWS, "Test case not supported on Windows")
    def test_div_upgrader_pass_return_new_op_after_retrace(self):
        class Foo(torch.nn.Module):
            def forward(self, a: torch.Tensor, b):
                return torch.ops.aten.div.Scalar_mode(a, b, rounding_mode='trunc')

        fn = Foo()

        inputs = (torch.ones([2, 3]) * 4, 2.)
        ep = export(fn, inputs)
        compiler_opset_version = {"aten": 4}
        model_opset_version = {"aten": 3}
        upgrader = GraphModuleOpUpgrader(compiler_opset_version, model_opset_version, TEST_UPGRADERS)

        count = count_op(ep.graph, "aten::div.Scalar_mode")
        self.assertEqual(count, 1)

        # upgrade: replace op (div.Scalar_mode -> div__Scalar_mode_0_3) then retrace
        upgraded_ep = upgrader.upgrade(ep)
        upgraded_ep.graph_module.print_readable()

        # no old version of op (div__Scalar_mode_0_3) anymore.
        custom_op_count = count_op(upgraded_ep.graph, "aten::div__Scalar_mode_0_3")
        self.assertEqual(custom_op_count, 0)

        # div__Scalar_mode_0_3 decomposes into div.Tensor.
        decomposed_op_count = count_op(upgraded_ep.graph, "aten::div.Tensor_mode")
        self.assertEqual(decomposed_op_count, 1)


if __name__ == '__main__':
    run_tests()
