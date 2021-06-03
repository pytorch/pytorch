import torch
from torch.testing._internal.jit_utils import JitTestCase

from torch.testing import FileCheck

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

tensor_name_dict = {
    "Float": "FloatTensor",
    "Int": "IntTensor"
}

class TestTensorPropertyPropagation(JitTestCase):
    def test_tensor_dtype_propagation(self):

        @torch.jit.script
        def foo(x, y):
            return x // y

        inputs = list(foo.graph.inputs())

        # input0, input1, result dtype
        test_cases = [
            ("Float", "Float", "Float"),
            ("Int", "Float", "Float"),
            ("Float", "Int", "Float"),
        ]

        # binary_op_list = ['add']
        # for op in binary_op_list:
        # code = '''
        #         graph(%3 : Tensor, %4 : Tensor):
        #             %2 : Tensor(*, *) = aten::{op}(%3, %4)
        #             %1 : {dtype}(*, *) = aten::relu(%2)
        #             return (%1)
        #     '''.format(op=op)

        for inp0, inp1, result in test_cases:
            # set scalar type of input tensors
            inputs[0].setType(inputs[0].type().with_scalarType(inp0))
            inputs[1].setType(inputs[1].type().with_scalarType(inp1))
            print(f"After setType graph: {foo.graph}")
            torch._C._jit_pass_propagate_tensor_property_on_graph(foo.graph)
            print(f"After propagation graph: {foo.graph}")
            FileCheck().check(f"{tensor_name_dict[result]} = aten::floor_divide").run(foo.graph)
