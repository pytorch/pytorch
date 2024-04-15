# Owner(s): ["oncall: jit"]

import unittest
from itertools import product

import torch
from torch.jit._passes._property_propagation import apply_input_props_using_example
from torch.testing._internal.common_utils import TEST_CUDA
from torch.testing._internal.jit_utils import JitTestCase

try:
    from torchvision import models
except ImportError:
    models = None

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
        cls.mkldnn = torch.device(
            "mkldnn"
        )  # MKLDNN can't mix with other device types at all
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

    def assert_device_equal(
        self, fn, in_devices, expected_device, in_shapes=None, subtest_str=""
    ):
        with self.subTest(
            f"In device: {in_devices}, expected: {expected_device}, \n {subtest_str}"
        ):
            graph = torch.jit.script(fn).graph
            self.prop_device_on_graph(graph, in_devices, in_shapes)
            actual_device = self.node_output_device(graph)

            if expected_device is None or actual_device is None:
                self.assertEqual(actual_device, expected_device)
            else:
                self.assertEqual(
                    actual_device.type, expected_device.type, "Failed Verification"
                )

    def test_device_apply(self):
        # Test if the device is properly applied to the input
        def add_self(x):
            return x + x

        graph = torch.jit.script(add_self).graph
        graph_input = next(graph.inputs())
        graph_input.setType(graph_input.type().with_device(self.cpu))
        # self.prop_device_on_graph(graph, [self.cpu])
        self.assertEqual(graph_input.type().device(), self.cpu)

    @unittest.skipIf(models is None, "Requires torchvision")
    def test_mobilenet(self):
        in_cpu = torch.randn(1, 3, 224, 224, device=self.cpu)
        in_example = in_cpu

        expected_device = self.cpu
        m = torch.jit.script(models.mobilenet_v3_small())
        m.eval()
        graph = torch.jit.freeze(m).graph
        # torch._C._jit_pass_erase_shape_information(graph)
        apply_input_props_using_example(graph, in_example)
        torch._C._jit_pass_propagate_shapes_on_graph(graph)
        torch._C._jit_pass_propagate_device(graph)

        actual_device = self.node_output_device(graph)

        if expected_device is None or actual_device is None:
            self.assertEqual(actual_device, expected_device)
        else:
            self.assertEqual(
                actual_device.type, expected_device.type, "Failed Verification"
            )

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

    def test_tensor_as_fns(self):
        def view_as_fn(x, y):
            return x.view_as(y)

        def expand_as_fn(x, y):
            return x.expand_as(y)

        def reshape_as_fn(x, y):
            return x.reshape_as(y)

        for test_fn in [view_as_fn, expand_as_fn, reshape_as_fn]:
            self.assert_device_equal(test_fn, [self.cpu, self.cpu], self.cpu)
            self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
            self.assert_device_equal(test_fn, [None, self.mkldnn], None)

        def type_as_fn(x, y):
            return x.type_as(y)

        self.assert_device_equal(type_as_fn, [self.cpu, self.cpu], self.cpu)
        self.assert_device_equal(type_as_fn, [self.cuda, None], None)
        self.assert_device_equal(type_as_fn, [None, self.mkldnn], self.mkldnn)

    def zerodim_test_core(self, device_pairs):
        # Test the support of zerodim tensors with non-zerodim tensors
        def mul(x, y):
            return x * y

        def add(x, y):
            return x + y

        fns = [mul, add]

        input_shapes = [
            ((1, 2, 2), (2, 2)),  # Different dim, non-zerodim
            ((1, 2, 2), ()),  # one zerodim
            ((), ()),  # both zerodim
        ]

        for fn, shapes, devices in product(fns, input_shapes, device_pairs):
            subtest_str = f"{fn.__name__} \n shapes: {shapes}, \n devices: {devices}"
            in0 = torch.rand(shapes[0], device=devices[0])
            in1 = torch.rand(shapes[1], device=devices[1])

            try:
                out = fn(in0, in1)
            except Exception as e:
                # Don't expect eager failures for CPU zerodim tensors
                for i in range(len(devices)):
                    if shapes[i] == () and devices[i] == self.cpu:
                        raise e

                # only expect eager failures on different devices
                if devices[0] == devices[1]:
                    raise e

                # Expect result device to be None for the failure cases.
                self.assert_device_equal(fn, devices, None, shapes, subtest_str)
                continue

            self.assert_device_equal(fn, devices, out.device, shapes, subtest_str)

            # Test that without shapes, we either get the same device or None for the device
            # Aka that the code is convservative for tensor shapes.
            graph = torch.jit.script(fn).graph
            self.prop_device_on_graph(graph, devices)
            actual_device = self.node_output_device(graph)
            self.assertTrue(
                (actual_device is None) or (actual_device.type == out.device.type)
            )

    def test_zerodim_cpu(self):
        # Allow for minimal testing locally
        self.zerodim_test_core([(self.cpu, self.cpu)])

    def test_zerodim_no_device(self):
        # If device is missing, you should never be able to infer device type.
        def mul(x, y):
            return x * y

        def add(x, y):
            return x + y

        fns = [mul, add]

        device_pairs = [
            (self.cpu, None),
            (None, self.cpu),
            (None, None),
        ]

        input_shapes = [
            ((1, 2, 2), (2, 2)),  # Different dim, non-zerodim
            ((1, 2, 2), ()),  # one zerodim
            ((), ()),  # both zerodim
        ]

        for fn, shapes, devices in product(fns, input_shapes, device_pairs):
            self.assert_device_equal(fn, devices, None, shapes)

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

        def set_mkldnn(x):
            return x.to_mkldnn()

        device_pairs = (
            (set_cuda, self.cuda),
            (set_cpu, self.cpu),
            (set_mkldnn, self.mkldnn),
        )

        for fn, out_device in device_pairs:
            for in_device in self.device_types:
                self.assert_device_equal(fn, [in_device], out_device)

    def test_device_if_propagation(self):
        def test_fn(x, y, z: bool):
            if z:
                return x + 3
            else:
                return y * 2

        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.mkldnn, self.mkldnn, None], self.mkldnn)
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None], None)

    def test_loop_simple(self):
        def test_fn(x, y, z: int):
            for _ in range(z):
                y = x
            return y

        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None], None)
        self.assert_device_equal(test_fn, [self.cpu, None, None], None)

    def test_loop_device_change(self):
        def test_fn(x, z: int):
            for _ in range(z):
                x = x.cuda()
            return x

        self.assert_device_equal(test_fn, [self.cpu, None], None)
        self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_while_change(self):
        def test_fn(x, z: int):
            while z > 0:
                x = x.cuda()
                z = 0
            return x

        self.assert_device_equal(test_fn, [self.cpu, None], None)
        self.assert_device_equal(test_fn, [self.cuda, None], self.cuda)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_nested_loops(self):
        def test_fn(x, z: int):
            for i in range(z):
                x = x.cpu()
                for _ in range(i):
                    x = x + 1

            return x

        self.assert_device_equal(test_fn, [self.cpu, None], self.cpu)
        self.assert_device_equal(test_fn, [self.cuda, None], None)
        self.assert_device_equal(test_fn, [None, None], None)

    def test_if_loop_mix(self):
        def test_fn(x, y, z: bool, a: bool):
            c = x
            while a:
                if z:
                    c = x + 3
                else:
                    c = y * 2
                a = False
            return c

        self.assert_device_equal(test_fn, [self.cpu, self.cpu, None, None], self.cpu)
        self.assert_device_equal(
            test_fn, [self.mkldnn, self.mkldnn, None, None], self.mkldnn
        )
        self.assert_device_equal(test_fn, [self.cpu, self.cuda, None, None], None)
