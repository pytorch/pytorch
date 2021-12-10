from itertools import product
import unittest

import torch
from torch.testing._internal.jit_utils import JitTestCase
from itertools import product

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

TEST_CUDA = torch.cuda.is_available()


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

    def prop_device_on_graph(self, graph, example_devices, in_shapes=None):
        graph_inputs = list(graph.inputs())
        torch._C._jit_pass_erase_shape_information(graph)

        self.assertEqual(len(graph_inputs), len(example_devices))
        for graph_i, device_i in zip(graph_inputs, example_devices):
            if device_i is not None:
                graph_i.setType(graph_i.type().with_device(device_i))

        if in_shapes:
            for graph_i, shapes_i in zip(graph_inputs, in_shapes):
                if shapes_i is not None:
                    graph_i.setType(graph_i.type().with_sizes(shapes_i))

            torch._C._jit_pass_propagate_shapes_on_graph(graph)

        torch._C._jit_pass_propagate_device(graph)

    def assert_device_equal(self, fn, in_devices, expected_device, in_shapes=None):
        graph = torch.jit.script(fn).graph
        print(graph)

        self.prop_device_on_graph(graph, in_devices, in_shapes)
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

    def test_simple(self):
        def add_self(x):
            return x + x

        def relu_(x):
            return torch.nn.functional.relu_(x)

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
        def set_device(x, device_name: torch.device):
            return x.to(device=device_name)

        for in_device in self.device_types:
            self.assert_device_equal(set_device, [in_device, None], None)

    def zerodim_test_core(self, device_pairs):
        # Test the support of zerodim tensors with non-zerodim tensors
        def mul(x, y):
            return x * y

        def linear(x, y):
            return torch.nn.functional.linear(x, y)

        fns = [mul, linear]

        input_shapes = [
            ((1, 2, 2), (2, 2)),  # Different dim, non-zerodim
            ((1, 2, 2), ()),  # one zerodim
            ((), ()),  # both zerodim
        ]

        for fn, shapes, devices in product(fns, input_shapes, device_pairs):
            with self.subTest(
                f"{fn.__name__} \n shapes: {shapes}, \n devices: {devices}"
            ):
                in0 = torch.random(shapes[0], device=devices[0])
                in1 = torch.random(shapes[1], device=devices[1])

                try:
                    out = fn(in0, in1)
                except Exception:
                    continue

                self.assert_device_equal(fn, devices, out.device, shapes)

    def test_zerodim_cpu(self):
        # Allow for minimal testing locally
        self.zerodim_test_core((self.cpu, self.cpu))

    @unittest.skipIf(not TEST_CUDA, "No CUDA")
    def test_zerodim_gpu(self):
        device_pairs = [
            (self.cpu, self.cuda),
            (self.cuda, self.cpu),
            (self.cuda, self.cuda),
        ]
        self.zerodim_test_core(device_pairs)

    def test_custom_device_op(self):
        # Test both of the custom functions and check that the devicetype is
        # correctly applied
        def set_cuda(x):
            return x.cuda()

        def set_cpu(x):
            return x.cpu()

        for fn, out_device in ((set_cuda, self.cuda), (set_cpu, self.cpu)):
            for in_device in self.device_types:
                with self.subTest(f"In device: {in_device}, fn: {out_device}"):
                    self.assert_device_equal(fn, [in_device], out_device)
