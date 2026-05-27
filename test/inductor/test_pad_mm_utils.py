# Owner(s): ["module: inductor"]
import operator

import torch
from torch._inductor.fx_passes.pad_mm import get_non_view_def
from torch._inductor.test_case import run_tests, TestCase


class PadMMUtilsTest(TestCase):
    def test_get_non_view_def_traverses_getitem(self):
        graph = torch.fx.Graph()
        arg = graph.placeholder("arg")
        sort = graph.call_function(torch.ops.aten.sort.default, (arg,))
        getitem = graph.call_function(operator.getitem, (sort, 0))

        self.assertIs(get_non_view_def(getitem), sort)


if __name__ == "__main__":
    run_tests()
