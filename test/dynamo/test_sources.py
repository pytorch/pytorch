# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_export_persist_assert)
"""

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.source import (
    AttrSource,
    GlobalSource,
    is_from_local_source,
    LocalSource,
)


class SourceTests(torch._dynamo.test_case.TestCase):
    def test_is_local(self):
        x_src = LocalSource("x")
        y_src = GlobalSource("y")

        attr_x_a = AttrSource(x_src, "a")
        attr_y_b = AttrSource(y_src, "b")

        self.assertTrue(is_from_local_source(attr_x_a))
        self.assertEqual(is_from_local_source(attr_y_b), False)
