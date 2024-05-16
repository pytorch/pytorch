# Owner(s): ["oncall: jit"]

import os
import sys

import torch
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestFunctionalBlocks(JitTestCase):
    def test_subgraph_creation(self):
        def fn(x, y, z):
            x = x + 1
            y = y + 1
            z = z + 1
            z.add_(2)
            z = z * z
            y = y * z
            if y < 2:
                y = y + 5
            return x + y + z

        graph = torch.jit.script(fn).graph
        self.run_pass("create_functional_graphs", graph)

        # all uses of x and y should be sunk
        FileCheck().check(r"%x").check_not(r"%x").check("FunctionalGraph").check(
            r"%x"
        ).run(graph)
        FileCheck().check(r"%y").check_not(r"%y").check("FunctionalGraph").check(
            r"%y"
        ).run(graph)

        # Don't allow any outputs which escape scope, so there is one final addition in the graph
        FileCheck().check("Tensor = prim::Functional").check_next("aten::add").run(
            graph
        )

        # z + 1, z.add_(2) considered non functional, z = z * z should be considered functional
        FileCheck().check("add").check("add_").check_not("mul").check(
            "FunctionalGraph"
        ).run(graph)
