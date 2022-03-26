# Owner(s): ["oncall: jit"]

import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestIfHoisting(JitTestCase):
    def test_if_hoist_basic(self):
        def fn(x: bool, y: int):
            if x:
                z = y + 3
            else:
                z = y + 3
            return z


        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)
        FileCheck().check_count("prim::If", 0, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)
        self.assertEqual(fn(True, 1), fn_script(True, 1))

    def test_if_hoist_transposed_expr(self):
        """
        Making sure that we can properly eliminate
        an expression even if it is not at the start
        of a block
        """
        def fn(x: bool, y: int):
            if x:
                a = y + 3
                b = y * 2
            else:
                b = y * 2
                a = y + 3
            return a, b

        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 0, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1), fn_script(True, 1))
        self.assertEqual(fn(False, 5), fn_script(False, 5))

    def test_if_hoist_swapped_expr(self):
        """
        Making sure that the if statement
        doesn't get fully eliminated here
        """
        def fn(x: bool, y: int):
            if x:
                a = y + 3
                b = y * 2
            else:
                a = y * 2
                b = y + 3
            return a, b

        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1), fn_script(True, 1))
        self.assertEqual(fn(False, 5), fn_script(False, 5))

    def test_if_hoist_reused_var(self):
        """
        Making sure that cases where the python variable is reused
        is handled correctly
        """
        def fn(x: bool, y: int):
            b = 6
            if x:
                a = y + 3
                a = y * 2
            else:
                a = y * 2
                b = y + 3
            return a, b

        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::mul", 1, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1), fn_script(True, 1))
        self.assertEqual(fn(False, 5), fn_script(False, 5))

    def test_no_hoist(self):
        """
        Nothing should happen here, expressions are different
        """
        def fn(x: bool, y: int, z: int):
            if x:
                a = y + 3
            else:
                a = z + 3
            return a

        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 2, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1, 3), fn_script(True, 1, 3))
        self.assertEqual(fn(False, 5, 10), fn_script(False, 5, 10))

    def test_mutate_before(self):
        """
        Make sure that if there is a mutation before the common
        op, the hoist doesn't happen
        """
        def fn(x: bool, y: torch.Tensor):
            if x:
                y.add_(8)
                a = y + 3
            else:
                a = y + 3
            return a

        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add(", 2, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add_", 1, exactly=True).run(op_graph)

        t1 = torch.Tensor([1])
        t2 = torch.Tensor([5, 6])
        self.assertEqual(fn(True, t1.clone()), fn_script(True, t1.clone()))
        self.assertEqual(fn(False, t2.clone()), fn_script(False, t2.clone()))

    def test_mutate_after(self):
        """
        Check that the hoist can happen properly, and
        that the output is still correct.
        """
        def fn(x: bool, y: torch.Tensor):
            if x:
                b = 1
                a = y + 3
                y.add_(8)
            else:
                b = 2
                a = y + 3
            c = b + a
            return a

        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 2, exactly=True).run(op_graph)
        t1 = torch.Tensor([1])
        t2 = torch.Tensor([5, 6])
        self.assertEqual(fn(True, t1.clone()), fn_script(True, t1.clone()))
        self.assertEqual(fn(False, t2.clone()), fn_script(False, t2.clone()))

    def test_multiple_hoists(self):
        """
        test that hoists that depend on other hoists are done correctly
        """
        def fn(x: bool, y: torch.Tensor):
            if x:
                a = y + 3
                b = a + y
            else:
                a = y + 3
                b = a + y
            c = b * 2
            return c

        fn_script = torch.jit.script(fn)
        op_graph = fn_script.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 0, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 2, exactly=True).run(op_graph)

        t1 = torch.Tensor([1])
        t2 = torch.Tensor([5, 6])
        self.assertEqual(fn(True, t1), fn_script(True, t1))
        self.assertEqual(fn(False, t2), fn_script(False, t2))

    def test_hoist_mutation_2(self):
        def fn(cond: bool, x, y):
            if cond:
                x.sqrt_()
                x = x + y
                x.relu_()
            else:
                x.relu_()
                x = x + y
                x.sqrt_()
            z = x * 2
            return z

        fn_s = torch.jit.script(fn)
        op_graph = fn_s.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)
        FileCheck().check_count("aten::relu_", 2, exactly=True).run(op_graph)
        FileCheck().check_count("aten::sqrt_", 2, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 2, exactly=True).run(op_graph)
