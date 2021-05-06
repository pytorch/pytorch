import os
import sys

from torch._C import parse_ir
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

# Tests that Python slice class is supported in TorchScript
class TestIgnorableArgs(JitTestCase):
    def test_slice_ignorable_args_for_slice(self):
        graph_str = """graph():
            %15 : int = prim::Constant[value=9223372036854775807]()
            %13 : int = prim::Constant[value=0]() # test/test_jit.py:4068:19
            %10 : bool = prim::Constant[value=0]()
            %8 : NoneType = prim::Constant()
            %0 : int = prim::Constant[value=1]() # test/test_jit.py:4067:33
            %1 : int = prim::Constant[value=2]() # test/test_jit.py:4067:36
            %2 : int = prim::Constant[value=3]() # test/test_jit.py:4067:39
            %3 : int = prim::Constant[value=4]() # test/test_jit.py:4067:42
            %4 : int = prim::Constant[value=9]() # test/test_jit.py:4067:45
            %5 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %6 : int[] = prim::ListConstruct(%0, %1, %2, %3, %4, %4)
            %7 : int[][] = prim::ListConstruct(%5, %6)
            %val.1 : Tensor = aten::tensor(%7, %8, %8, %10) # test/test_jit.py:4067:18
            %16 : Tensor = aten::slice(%val.1, %13, %1, %15, %0) # test/test_jit.py:4068:19
            %20 : Tensor = aten::slice(%16, %0, %13, %0, %0) # test/test_jit.py:4068:19
            return (%20)"""
        graph = parse_ir(graph_str)
        function = self.createFunctionFromGraph(graph)
        function_copy = self.getExportImportCopy(function)
        src = str(function.code)
        # For a signature:
        # aten::slice(Tensor self, int dim, int start, int end, int step) -> Tensor
        # We ignore trailing arguments after start=2 for dim 0
        # and after end=1 for dim 1
        # because in %16, %15 and %0 are default values for the schema.
        FileCheck().check("torch.slice(torch.tensor(_0), 0, 2), 1, 0, 1)").run(src)
        self.assertEqual(function(), function_copy())
