# Owner(s): ["module: dynamo"]
from unittest.mock import patch

import torch
import torch.fx
from torch._dynamo.graph_utils import (
    _GPU_DEVICE_TYPES,
    _graph_device_type,
    _graph_device_types,
    register_gpu_device,
)
from torch._dynamo.test_case import TestCase
from torch.testing._internal.common_utils import HardwareClassification


class GraphUtilsRegisterDeviceTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        self.original_device_types = set(_GPU_DEVICE_TYPES)

    def tearDown(self):
        _GPU_DEVICE_TYPES.clear()
        _GPU_DEVICE_TYPES.update(self.original_device_types)
        super().tearDown()

    def test_graph_device_types_excludes_default_privateuse1(self):
        # When no PrivateUse1 backend is registered, "privateuseone" must NOT
        # appear in the result (it is the sentinel, not a real device).
        device_types = _graph_device_types()
        self.assertNotIn("privateuseone", device_types)

    def test_graph_device_types_includes_privateuse1(self):
        # When a PrivateUse1 backend is registered, it is auto-discovered
        # without needing register_gpu_device().
        with patch("torch._C._get_privateuse1_backend_name", return_value="testdev"):
            device_types = _graph_device_types()
            self.assertIn("testdev", device_types)

    def test_register_gpu_device(self):
        self.assertNotIn("testdev", _GPU_DEVICE_TYPES)
        register_gpu_device("testdev")
        self.assertIn("testdev", _GPU_DEVICE_TYPES)
        self.assertIn("testdev", _graph_device_types())

    def test_graph_device_type_detects_device_in_meta(self):
        # The meta path returns the device type of the first meta value;
        # it does not filter by device_types, so "cuda" is detected directly.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.device("cuda")
        graph.output(x)
        self.assertEqual(_graph_device_type(graph), "cuda")

    def test_graph_device_type_detects_registered_device_via_to_call(self):
        register_gpu_device("testdev")
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_method("to", args=(x, "testdev"))
        self.assertEqual(_graph_device_type(graph), "testdev")

    def test_graph_device_type_detects_registered_device_via_method_call(self):
        register_gpu_device("testdev")
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.call_method("testdev", args=(x,))
        self.assertEqual(_graph_device_type(graph), "testdev")

    def test_graph_device_type_returns_cpu_without_registration(self):
        graph = torch.fx.Graph()
        # an empty graph with no device info should return "cpu"
        self.assertEqual(_graph_device_type(graph), "cpu")

    def test_graph_device_type_returns_cpu_for_none_graph(self):
        self.assertEqual(_graph_device_type(None), "cpu")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
