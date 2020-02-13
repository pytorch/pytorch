import os
import sys

import torch
from torch.nn import functional as F
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, freeze_rng_state

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

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
        self.run_pass('create_functional_graphs', graph)

        # all uses of x and y should be sunk
        FileCheck().check(r"%x").check_not(r"%x").check("FunctionalGraph").check(r"%x").run(graph)
        FileCheck().check(r"%y").check_not(r"%y").check("FunctionalGraph").check(r"%y").run(graph)

        # Don't allow any outputs which escape scope, so there is one final addition in the graph
        FileCheck().check("Tensor = prim::Functional").check_next("aten::add").run(graph)

        # z + 1, z.add_(2) considered non functional, z = z * z should be considered functional
        FileCheck().check("add").check("add_").check_not("mul").check("FunctionalGraph").run(graph)

    def test_lower_linear(self):
        # linear is one of main use cases of removing mutation so add test so it doesnt regress
        @torch.jit.script
        def foo(x):
            return F.linear(x, torch.randn(20, 20), torch.randn(20))

        self.run_pass('inline', foo.graph)
        self.run_pass('peephole', foo.graph)
        self.run_pass('constant_propagation', foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)
        input = torch.randn(20, 20)
        with freeze_rng_state():
            out1 = foo(input)

        self.run_pass('remove_mutation', foo.graph)
        FileCheck().check_not("aten::add_").run(foo.graph)
        with freeze_rng_state():
            out2 = foo(input)
        self.assertEqual(out1, out2)

    def test_remove_mutation_aten_inplace(self):
        def test_not_new_alias(x):
            y = x[0]
            y.add_(2)
            return y

        fn = torch.jit.script(test_not_new_alias)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(torch.ones([2, 2])), test_not_new_alias(torch.ones([2, 2])))

        def test_no_lowering():
            x = torch.tensor([2, 2])
            x[0] = 3
            return x

        # there is no functional equivalent of x[0] = ...
        fn = torch.jit.script(test_no_lowering)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::copy_").run(graph)
        self.assertEqual(fn(), test_no_lowering())

        def test_move_before_not_valid():
            y = torch.tensor([2, 2])
            z = y + 2
            y.add_(2)
            return y, z

        fn = torch.jit.script(test_move_before_not_valid)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(), test_move_before_not_valid())

        def test_successful():
            x = torch.tensor([2, 2])
            x.add_(1)
            x.add_(3)
            y = x + 4
            return x, y

        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass('remove_mutation', graph)
        FileCheck().check_not("aten::add_").run(graph)
        self.assertEqual(test_successful(), fn())

        def test_intermediary_use():
            x = torch.tensor([2, 2])
            x.add_(1)
            y = x + 4
            x.add_(3)
            return x, y

        fn = torch.jit.script(test_intermediary_use)
        graph = fn.graph
        FileCheck().check_count("aten::add_", 2).run(graph)
        self.run_pass('remove_mutation', graph)
        # Unable to remove the second add_ because of the y = x + 4 use
        # In the future we could duplicating the value of x as a temporary and replacing
        # its intermediary use (so long as aliasing is safe)
        FileCheck().check_count("aten::add_", 1).run(graph)
        self.assertEqual(test_intermediary_use(), fn())

    def test_remove_mutation_lists_append(self):
        def successful_remove():
            return [i for i in range(5)]

        fn = torch.jit.script(successful_remove)
        graph = fn.graph
        self.run_pass('loop_unrolling', graph)
        self.run_pass('remove_mutation', graph)
        self.run_pass('constant_propagation', graph)
        FileCheck().check("graph").check_next("Constant").check_next("return").run(graph)
        self.assertEqual(successful_remove(), successful_remove())

        def intermediary_use():
            a = [1, 2]
            b = len(a)
            a.append(3)
            return a

        fn = torch.jit.script(intermediary_use)
        graph = fn.graph
        FileCheck().check("append").run(graph)
        self.run_pass('remove_mutation', graph)
        # it is possible to remove the append here but don't currently have the logic for it
        FileCheck().check_not("append").run(graph)
        self.assertEqual(intermediary_use(), fn())
