from itertools import product
from typing import Tuple

import torch
from torch.testing._internal.jit_utils import JitTestCase
from torch import device

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestDeviceAnalysis(JitTestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = torch.device("cpu")
        cls.cuda = torch.device("cuda")
        cls.vulkan = torch.device("vulkan")
        cls.device_types = [cls.cpu, cls.cuda, cls.vulkan]

    @staticmethod
    def node_output_device(graph):
        graph_out = list(graph.outputs())
        assert len(graph_out) == 1
        return graph_out[0].type().device()

    def prop_device_on_graph(self, graph, example_devices):
        graph_inputs = list(graph.inputs())
        torch._C._jit_pass_erase_shape_information(graph)

        self.assertEqual(len(graph_inputs), len(example_devices))
        for graph_i, device_i in zip(graph_inputs, example_devices):
            if isinstance(graph_i, torch.Tensor):
                graph_i.setType(graph_i.type().with_device(device_i))

        torch._C._jit_pass_propagate_device(graph)

    def assert_device_equal(self, fn, in_devices, expected_device):
        graph = torch.jit.script(fn).graph
        self.prop_device_on_graph(graph, in_devices)
        actual_device = self.node_output_device(graph)

        self.assertEqual(actual_device, expected_device, "Failed Verification")

    def test_device_apply(self):
        # Test if the device is properly applied to the input
        def add_self(x):
            return x + x

        graph = torch.jit.script(add_self).graph
        graph_input = next(graph.inputs())
        graph_input.setType(graph_input.type().with_device(self.cpu))
        # self.prop_device_on_graph(graph, [self.cpu])
        self.assertEqual(graph_input.type().device(), self.cpu)

    # TODO: I have fixed all the bugs with the input types being applied. However
    # I still need to fix the bugs with Device Type itself.

    def test_simple(self):
        def add_self(x):
            return x + x

        def relu_(x):
            return relu_(x)

        functions = [add_self, relu_]

        for in_device, fn in product(self.device_types, functions):
            self.assert_device_equal(fn, [in_device], in_device)

    def test_set_dtype(self):
        def set_device(x):
            return x.to("cpu")

        for in_device in self.device_types:
            self.assert_device_equal(set_device, [in_device], self.cpu)

    def test_device_arg(self):
        # Test that no device gets propagated when arg is passed in
        def set_device(x, device_name):
            return x.to(device_name)

        for in_device in self.device_types:
            self.assert_device_equal(set_device, [in_device, None], None)
