# Owner(s): ["oncall: jit"]

import torch
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TestPythonBindings\n\n"
        "instead."
    )


class TestPythonBindings(JitTestCase):
    def test_cu_get_functions(self):
        @torch.jit.script
        def test_get_python_cu_fn(x: torch.Tensor):
            return 2 * x

        cu = torch.jit._state._python_cu
        self.assertTrue(
            "test_get_python_cu_fn" in (str(fn.name) for fn in cu.get_functions())
        )

    def test_cu_create_function(self):
        @torch.jit.script
        def fn(x: torch.Tensor):
            return 2 * x

        cu = torch._C.CompilationUnit()
        cu.create_function("test_fn", fn.graph)

        inp = torch.randn(5)

        self.assertEqual(inp * 2, cu.find_function("test_fn")(inp))
        self.assertEqual(cu.find_function("doesnt_exist"), None)
        self.assertEqual(inp * 2, cu.test_fn(inp))
        with self.assertRaises(AttributeError):
            cu.doesnt_exist(inp)

    def test_invalidation(self):
        @torch.jit.script
        def test_invalidation_fn(x: torch.Tensor):
            return 2 * x

        gr = test_invalidation_fn.graph.copy()
        n = gr.insertNode(gr.create("prim::profile"))
        v = n.output()
        # check that they work
        str((n, v))
        torch._C._jit_pass_dce(gr)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(n)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(v)

    def test_graph_iterator_keepalive(self):
        @torch.jit.script
        def test_iterator_keepalive_fn(x: torch.Tensor):
            return 2 * x

        # the list would segfault before because inlined_graph
        # is temporary and had been deleted (see issue #50454)
        n = test_iterator_keepalive_fn.inlined_graph.nodes()
        list(n)
        i = test_iterator_keepalive_fn.inlined_graph.inputs()
        list(i)
        o = test_iterator_keepalive_fn.inlined_graph.outputs()
        list(o)

    def test_aliasdb(self):
        @torch.jit.script
        def test_aliasdb_fn(x: torch.Tensor):
            return 2 * x

        gr = test_aliasdb_fn.graph.copy()
        alias_db = gr.alias_db()
        self.assertTrue("WILDCARD" in str(alias_db))
        self.assertTrue("digraph alias_db" in alias_db.to_graphviz_str())

    def test_graph_create(self):
        gr = torch._C.Graph()
        with self.assertRaises(ValueError):
            gr.create("prim::Constant", [None])

    def test_add_input(self):
        gr = torch._C.Graph()
        foo_value = gr.addInput("foo")
        assert foo_value in gr.inputs()

    def test_canonicalize(self):
        ir = """
graph(%p207 : Tensor,
      %1 : Tensor,
      %p407 : int):
  %11 : Tensor = aten::view_expand_placeholder(%1)
  %12 : Tensor = aten::pointwise_placeholder(%11, %p207, %p407)
  %13 : Tensor = aten::view_expand_placeholder(%12)
  %14 : Tensor = aten::pointwise_placeholder(%13)
  return (%14)
        """

        graph1 = torch._C.parse_ir(ir)
        graph1 = torch._C._jit_pass_canonicalize(graph1, True)

        graph2 = torch._C.parse_ir(ir)
        graph2 = torch._C._jit_pass_canonicalize(graph2)

        self.assertEqual(str(graph1), str(graph2))
        FileCheck().check("%p207").check_not("%14").run(graph1)

        graph3 = torch._C.parse_ir(ir)
        graph3 = torch._C._jit_pass_canonicalize(graph3, False)
        FileCheck().check_not("%p207").run(graph3)
