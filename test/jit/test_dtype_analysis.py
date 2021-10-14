from itertools import product

import torch
from torch import complex32, float32, float64, int32, int64
from torch.testing._internal.jit_utils import JitTestCase

# from torch.testing import FileCheck
# from torch.testing._internal.common_utils import make_tensor

# from textwrap import dedent

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


# XXX: Relies on Symbolic Shape Analysis, which is still a prototype
class TestDtypeAnalysis(JitTestCase):
    def setUp(self):
        self.prev_symbolic_shapes_test_enabled = (
            torch._C._jit_symbolic_shapes_test_mode_enabled()
        )
        torch._C._jit_set_symbolic_shapes_test_mode(True)

    def tearDown(self):
        torch._C._jit_set_symbolic_shapes_test_mode(
            self.prev_symbolic_shapes_test_enabled
        )

    @staticmethod
    def node_output_dtype(graph):
        # return graph.findNode(node_name).output().type().dtype()
        first_out = next(graph.outputs())
        return first_out.type().dtype()

    def prop_dtype_on_graph(self, graph, example_inputs):
        graph_inputs = list(graph.inputs())

        self.assertEqual(len(graph_inputs), len(example_inputs))
        for graph_i, example_i in zip(graph_inputs, example_inputs):
            dtype = example_i.dtype
            shape = example_i.shape
            graph_i.setType(graph_i.type().with_dtype(dtype).with_sizes(shape))

        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        torch._C._jit_pass_propagate_dtype(graph)

    def assert_dtype_equal(self, fn, in_shapes, in_dtypes):
        # Eager execution
        inputs = [self.get_rand_tensor(s, d) for s, d in zip(in_shapes, in_dtypes)]
        try:
            expected_res = fn(*inputs)
        except Exception:
            # Skip anything that doesn't execute in Eager Mode?
            return
        expected_dtype = expected_res.dtype

        # Run the Dtype Analysis
        graph = torch.jit.script(fn).graph
        self.prop_dtype_on_graph(graph, inputs)
        actual_dtype = self.node_output_dtype(graph)

        fail_text = f"Failed for shapes {in_shapes}, and dtypes {in_dtypes}"
        self.assertEqual(actual_dtype, expected_dtype, fail_text)

    def get_rand_tensor(self, shape, dtype):
        if dtype in (int32, int64):
            rand_tensor = torch.randint(0, 10, shape, dtype=dtype)
        else:
            rand_tensor = torch.rand(shape, dtype=dtype)

        # Sanity check!

        self.assertEqual(rand_tensor.dtype, dtype)
        return rand_tensor

    def test_dtype_analysis_binary(self):
        # Testing using Metatensors
        """
        functions = [
            operator.add,
            operator.mul,
            operator.truediv,
        ]
        """
        def add(x, y):
            return x + y

        functions = [
            add
        ]

        input_shapes = [
            ((2, 2), (2, 2)),  # Simple Case
            ((1, 1, 1), (1, 1)),  # Different Dim, non-zerodim
            ((), (1, 2)),  # One zerodim
            ((1, 2), ()),  # Other zerodim
            ((), ()),  # both zerodim
            # TODO: Add Scalar Cases
        ]

        input_dtypes = [
            (float32, float32),  # Simple Case
            (int32, int64),  # Size Promotion (compliated case for 0dim tensors)
            (int32, float32),  # type Promotion
            (int64, float32),  # Type promotion with size change
            (float64, complex32),  # Show we can handle complex vals as well
        ]

        for fn, in_shapes, in_dtypes in product(functions, input_shapes, input_dtypes):
            self.assert_dtype_equal(fn, in_shapes, in_dtypes)
