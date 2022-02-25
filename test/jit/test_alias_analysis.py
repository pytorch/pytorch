# Owner(s): ["oncall: jit"]

from torch.testing._internal.jit_utils import JitTestCase
from torch._C import parse_ir
import torch


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestAliasAnalysis(JitTestCase):
    def test_becomes_wildcard_annotations(self):
        graph_str = """
        graph(%a.1 : Tensor, %b.1 : Tensor):
            %11 : NoneType = prim::Constant()
            %8 : int = prim::Constant[value=0]()
            %7 : int = prim::Constant[value=1]()
            %x.1 : Tensor = aten::add(%a.1, %b.1, %7)
            %y.1 : Tensor[] = aten::split(%x.1, %7, %8)
            return ()
        """
        graph = parse_ir(graph_str)
        alias_db = graph.alias_db()
        split_node = graph.findNode("aten::split")
        # split input enters wildcard set, list initalized as containing wildcard set
        self.assertTrue(alias_db.may_contain_alias(next(split_node.inputs()), split_node.output()))
        # because %x.1 enters wildcard set, it now aliases other members of wildcard set (graph inputs)
        self.assertTrue(alias_db.may_contain_alias(next(split_node.inputs()), next(graph.inputs())))

    def test_nested_list_construct_not_wildcard(self):
        @torch.jit.script
        def foo(x):
            y = torch.rand([2, 2])
            return [y]

        graph = foo.graph
        graph.alias_db()
        alias_db = graph.alias_db()
        ten_construct = graph.findNode("aten::rand").output()
        output = next(graph.outputs())
        self.assertTrue(alias_db.may_contain_alias(ten_construct, output))
        self.assertFalse(alias_db.may_contain_alias(next(graph.inputs()), ten_construct))

    def test_recursive_calls(self):
        @torch.jit.script
        def foo(x, y):
            x.add_(1)
            return x + y

        @torch.jit.script
        def caller():
            a = torch.rand([2, 2])
            b = torch.ones([2, 2])
            out1 = foo(a, b)
            c = torch.rand([1])
            d = torch.ones([2])
            out2 = foo(d, c)
            return out1, out2

        isFrozen = False
        descend_function_calls = True
        alias_db = caller.graph.alias_db(isFrozen, descend_function_calls)
        func_calls = caller.graph.findAllNodes("prim::CallFunction")
        self.assertEqual(len(func_calls), 2)
        for node in func_calls:
            inps = list(node.inputs())
            self.assertTrue(alias_db.has_writers(inps[1]))
            self.assertFalse(alias_db.has_writers(inps[2]))

        class Mod(torch.nn.Module):
            def forward(self):
                a = torch.rand([2, 2])
                b = torch.ones([2, 2])
                out1 = self.foo2(a, b)
                c = torch.rand([1])
                d = torch.ones([2])
                out2 = self.foo2(d, c)
                return out1, out2

            def foo2(self, x, y):
                x.add_(1)
                return x + y

        mod = torch.jit.script(Mod())
        alias_db = mod.graph.alias_db(isFrozen, descend_function_calls)
        func_calls = mod.graph.findAllNodes("prim::CallMethod")
        self.assertEqual(len(func_calls), 2)
        for node in func_calls:
            inps = list(node.inputs())
            self.assertTrue(alias_db.has_writers(inps[1]))
            self.assertFalse(alias_db.has_writers(inps[2]))
