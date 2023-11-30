# Owner(s): ["oncall: quantization"]

import unittest

from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import generate_numeric_debug_handle
from torch.fx import Node
from torch.testing._internal.common_quantization import TestHelperModules
from torch.testing._internal.common_utils import IS_WINDOWS, TestCase


@unittest.skipIf(IS_WINDOWS, "Windows not yet supported for torch.compile")
class TestGenerateNumericDebugHandle(TestCase):
    def test_simple(self):
        m = TestHelperModules.Conv2dThenConv1d()
        m = capture_pre_autograd_graph(m, m.example_inputs())
        generate_numeric_debug_handle(m)
        unique_ids = set()
        count = 0
        for n in m.graph.nodes:
            if "numeric_debug_handle" in n.meta:
                for arg in n.args:
                    if isinstance(arg, Node):
                        unique_ids.add(n.meta["numeric_debug_handle"][arg])
                        count += 1
                unique_ids.add(n.meta["numeric_debug_handle"]["output"])
                count += 1
        self.assertEqual(len(unique_ids), count)
