import unittest
from typing import List

import torch
from torch import nn
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA, _inline_everything

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestIfHoisting(JitTestCase):
    def test_if_hoist_basic(self):
        @torch.jit.script
        def fn(x: bool, y: int):
            if x:
                z = y + 3
            else:
                z = y + 3
            return z


        op_graph = fn.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)
        FileCheck().check_count("prim::If", 0, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)
        self.assertEqual(fn(True, 1), 4)

    def test_if_hoist_transposed_expr(self):
        # Making sure that the if statement
        # doesn't get fully eliminated here
        @torch.jit.script
        def fn(x: bool, y: int):
            if x:
                a = y + 3
                b = y * 2
            else:
                b = y * 2
                a = y + 3
            return a, b

        op_graph = fn.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 0, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1), (4, 2))
        self.assertEqual(fn(False, 5), (8, 10))

    def test_if_hoist_swapped_expr(self):
        # Making sure that the if statement
        # doesn't get fully eliminated here
        @torch.jit.script
        def fn(x: bool, y: int):
            if x:
                a = y + 3
                b = y * 2
            else:
                a = y * 2
                b = y + 3
            return a, b

        op_graph = fn.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1), (4, 2))
        self.assertEqual(fn(False, 5), (10, 8))

    def test_if_hoist_shadowed_expr(self):
        # Making sure that multiple forms of shadowing are handled correctly
        @torch.jit.script
        def fn(x: bool, y: int):
            b = 6
            if x:
                a = y + 3
                a = y * 2
            else:
                a = y * 2
                b = y + 3
            return a, b

        op_graph = fn.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::mul", 1, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1), (2, 6))
        self.assertEqual(fn(False, 5), (10, 8))

    def test_no_hoist(self):
        # Nothing should happen here, expressions are different
        @torch.jit.script
        def fn(x: bool, y: int, z: int):
            if x:
                a = y + 3
            else:
                a = z + 3
            return a

        op_graph = fn.graph
        self.run_pass("common_expression_hoisting", op_graph)
        self.run_pass("dce", op_graph)

        FileCheck().check_count("prim::If", 1, exactly=True).run(op_graph)
        FileCheck().check_count("aten::add", 2, exactly=True).run(op_graph)

        self.assertEqual(fn(True, 1, 3), 4)
        self.assertEqual(fn(False, 5, 10), 13)
