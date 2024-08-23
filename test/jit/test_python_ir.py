# Owner(s): ["oncall: jit"]

import unittest

import numpy as np

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.jit_utils import JitTestCase


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestPythonIr(JitTestCase):
    def test_param_strides(self):
        def trace_me(arg):
            return arg

        t = torch.zeros(1, 3, 16, 16)
        traced = torch.jit.trace(trace_me, t)
        value = list(traced.graph.param_node().outputs())[0]
        real_strides = list(t.stride())
        type_strides = value.type().strides()
        self.assertEqual(real_strides, type_strides)

    def test_permute_inputs_binding(self):
        @torch.jit.script
        def foo(i, j, k):
            pass

        g = foo.graph

        idxs = []
        for i, inp in enumerate(g.inputs()):
            inp.setDebugName(f"inp{i}")
            idxs.append(i)

        permuted_idxs = list(np.random.permutation(idxs))
        g.permuteInputs(permuted_idxs)
        for i, inp in enumerate(g.inputs()):
            self.assertEqual(f"inp{permuted_idxs[i]}", inp.debugName())

    @unittest.skipIf(IS_MACOS, "Failing on MacOS only")
    def test_python_ir_utils(self):
        @torch.jit.script
        def foo(inp):
            x = inp + 1
            y = x / 2
            z = y * y
            return z

        add_node = foo.graph.findNode("aten::add")
        div_node = foo.graph.findNode("aten::div")

        with foo.graph.insert_point_guard(add_node):
            with foo.graph.insert_point_guard(div_node):
                foo.graph.insertConstant("goodbye")
            foo.graph.insertConstant("hello")
        with foo.graph.insert_point_guard(foo.graph.findNode("aten::mul")):
            foo.graph.insertConstant("hello")
        FileCheck().check("hello").check("goodbye").check("hello").run(foo.graph)

        self.assertTrue(add_node.matches(add_node.schema()))
        self.assertFalse(add_node.matches(div_node.schema()))

    def test_python_ir_utils_graph(self):
        @torch.jit.script
        def unrolled_mul(x: torch.Tensor, y: int):
            out = x
            for _ in range(y - 1):
                out = out + x
            return out

        @torch.jit.script
        def foo(x):
            return x * 4

        g = foo.graph
        muls = g.findAllNodes("aten::mul")
        scalar_muls = filter(
            lambda x: x.matches("aten::mul(Tensor self, Scalar other) -> Tensor"), muls
        )
        mul_constant_int = filter(
            lambda x: isinstance(list(x.inputs())[1].toIValue(), int), scalar_muls
        )
        for mul in mul_constant_int:
            with g.insert_point_guard(mul):
                outputs = g.insertGraph(unrolled_mul.graph, list(mul.inputs()))
                assert len(outputs) == len(list(mul.outputs()))
                for new_out, old_out in zip(outputs, g.outputs()):
                    old_out.replaceAllUsesWith(new_out)
                mul.destroy()

        FileCheck().check_not("aten::mul").check("aten::add").run(foo.graph)
        self.assertEqual(foo(torch.ones([2, 2])), torch.ones([2, 2]) * 4)
