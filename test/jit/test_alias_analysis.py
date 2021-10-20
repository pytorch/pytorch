from torch.testing._internal.jit_utils import JitTestCase
from torch._C import parse_ir


if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestAliasAnalysis(JitTestCase):
    def test_becomes_wildcard_annotations(self):
        graph_str = """
        graph(%a.1 : Tensor, %b.1 : Tensor):
            %11 : NoneType = prim::Constant() # /private/home/eellison/pytorch/test/jit/test_alias_analysis.py:18:8
            %8 : int = prim::Constant[value=0]()
            %7 : int = prim::Constant[value=1]() # /private/home/eellison/pytorch/test/jit/test_alias_analysis.py:20:31
            %x.1 : Tensor = aten::add(%a.1, %b.1, %7) # /private/home/eellison/pytorch/test/jit/test_alias_analysis.py:19:16
            %y.1 : Tensor[] = aten::split(%x.1, %7, %8) # /private/home/eellison/pytorch/test/jit/test_alias_analysis.py:20:16
            return ()
        """
        graph = parse_ir(graph_str)
        alias_db = graph.alias_db()
        split_node = graph.findNode("aten::split")
        # split input enters wildcard set, list initalized as containing wildcard set
        self.assertTrue(alias_db.may_contain_alias(next(split_node.inputs()), split_node.output()))
        # because %x.1 enters wildcard set, it now aliases other members of wildcard set (graph inputs)
        self.assertTrue(alias_db.may_contain_alias(next(split_node.inputs()), next(graph.inputs())))
