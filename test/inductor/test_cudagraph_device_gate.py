# Owner(s): ["module: inductor"]

import contextlib
from unittest import mock

import torch
from torch._dynamo.device_interface import (
    CudaInterface,
    device_interfaces,
    DeviceInterface,
    register_interface_for_device,
)
from torch._inductor.cudagraph_utils import (
    _graph_capture_compatible_device_type,
    check_multiple_devices_or_any_cpu_nodes,
    is_graph_capture_runtime_ready,
)
from torch._inductor.output_code import (
    cudagraph_partition_post_compile,
    cudagraph_post_compile,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import BoxedBool, GraphPartitionMap
from torch.utils._ordered_set import OrderedSet


class _FakeNode:
    name = "n"


class _FakeDevice:
    def __init__(self, device_type: str, index: int = 0) -> None:
        self.type = device_type
        self.index = index

    def __repr__(self) -> str:
        return f"{self.type}:{self.index}"


class _RegisteredInterface(DeviceInterface):
    @staticmethod
    def is_available() -> bool:
        return True


class _CaptureInterface(DeviceInterface):
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_graph_capture_supported(device=None) -> bool:
        return True


@contextlib.contextmanager
def temporary_device_interface(device_type: str, interface_cls: type[DeviceInterface]):
    had_entry = device_type in device_interfaces
    previous = device_interfaces.get(device_type)
    register_interface_for_device(device_type, interface_cls)
    try:
        yield
    finally:
        if had_entry:
            device_interfaces[device_type] = previous
        else:
            device_interfaces.pop(device_type, None)


def _make_post_compile_graph(*, device_type: str = "npu", with_partition: bool = False):
    from torch._inductor.cudagraph_utils import CudagraphCachedInfo, PlaceholderInfo

    def identity(x):
        return x

    placeholder = PlaceholderInfo(
        name="x",
        stack_trace=None,
        users=[],
        mutating_use_stack_trace=None,
    )

    class _Graph:
        pass

    graph = _Graph()
    graph.device_types = OrderedSet([device_type])
    graph.device_idxs = OrderedSet([0])
    graph.mutated_input_idxs = OrderedSet()
    graph.kernel_free_cudagraph = False
    graph.cudagraph_info = CudagraphCachedInfo(
        placeholders=(placeholder,),
        stack_traces=[None],
        user_visible_output_idxs=(0,),
        cudagraph_fail_reasons=[],
    )
    graph.fx_kwargs = {
        "is_inference": False,
        "is_backward": False,
        "static_input_idxs": (),
    }
    graph.current_callable = identity
    graph.disabled_cudagraphs_reason = None
    graph.partition_maps = None
    graph.cudagraphify_calls = []

    def recursively_apply_fns(fns):
        graph.cudagraphify_calls.extend(fns)

    graph.recursively_apply_fns = recursively_apply_fns
    if with_partition:
        graph.partition_maps = [
            GraphPartitionMap(
                id=0,
                input_index_mapping=[0],
                output_index_mapping=[[0]],
                constant_names=[],
            )
        ]
    return graph


class TestGraphCaptureCompatibleDeviceType(TestCase):
    def test_cuda(self):
        self.assertTrue(_graph_capture_compatible_device_type("cuda"))

    def test_default_privateuseone_slot_unused(self):
        with mock.patch(
            "torch._C._get_privateuse1_backend_name", return_value="privateuseone"
        ):
            self.assertFalse(_graph_capture_compatible_device_type("privateuseone"))

    def test_renamed_without_interface(self):
        with mock.patch("torch._C._get_privateuse1_backend_name", return_value="npu"):
            self.assertFalse(_graph_capture_compatible_device_type("npu"))

    def test_renamed_with_interface(self):
        with temporary_device_interface("npu", _RegisteredInterface):
            with mock.patch(
                "torch._C._get_privateuse1_backend_name", return_value="npu"
            ):
                self.assertTrue(_graph_capture_compatible_device_type("npu"))


class TestGraphCaptureRuntimeReady(TestCase):
    def test_cuda(self):
        self.assertTrue(is_graph_capture_runtime_ready({"cuda"}))
        self.assertTrue(CudaInterface.is_graph_capture_supported())

    def test_npu_without_capture_bit(self):
        with temporary_device_interface("npu", _RegisteredInterface):
            self.assertFalse(is_graph_capture_runtime_ready({"npu"}))

    def test_npu_with_capture_bit(self):
        with temporary_device_interface("npu", _CaptureInterface):
            self.assertTrue(is_graph_capture_runtime_ready({"npu"}))


class TestCudagraphPostCompileRuntimeDispatch(TestCase):
    def test_post_compile_skips_without_capture_bit(self):
        from torch._inductor import compile_fx

        graph = _make_post_compile_graph(device_type="npu")
        cudagraphs = BoxedBool(True)
        with temporary_device_interface("npu", _RegisteredInterface):
            with mock.patch.object(
                compile_fx,
                "cudagraphify",
                side_effect=AssertionError("cudagraphify must not run"),
            ):
                cudagraph_post_compile([], graph, cudagraphs, {}, None)
        self.assertFalse(cudagraphs.value)

    def test_partition_post_compile_skips_without_capture_bit(self):
        from torch._inductor import compile_fx

        graph = _make_post_compile_graph(device_type="npu", with_partition=True)
        cudagraphs = BoxedBool(True)
        with temporary_device_interface("npu", _RegisteredInterface):
            with mock.patch.object(
                compile_fx,
                "cudagraphify",
                side_effect=AssertionError("cudagraphify must not run"),
            ):
                cudagraph_partition_post_compile([], graph, cudagraphs, {}, None)
        self.assertFalse(cudagraphs.value)
        self.assertEqual(graph.cudagraphify_calls, [])

    def test_post_compile_runs_with_capture_bit(self):
        from torch._inductor import compile_fx

        calls: list[str] = []

        def npugraphify(model, static_input_idxs, **kwargs):
            calls.append("npu")
            return model

        graph = _make_post_compile_graph(device_type="npu")
        cudagraphs = BoxedBool(True)
        with temporary_device_interface("npu", _CaptureInterface):
            with mock.patch.object(compile_fx, "cudagraphify", npugraphify):
                cudagraph_post_compile([], graph, cudagraphs, {}, None)
        self.assertEqual(calls, ["npu"])
        self.assertTrue(cudagraphs.value)


class TestCudagraphDeviceGate(TestCase):
    def test_single_cuda_allowed(self):
        node = _FakeNode()
        mapping = {torch.device("cuda:0"): node}
        self.assertIsNone(check_multiple_devices_or_any_cpu_nodes(mapping))

    def test_single_xpu_skipped(self):
        node = _FakeNode()
        mapping = {torch.device("xpu:0"): node}
        msg = check_multiple_devices_or_any_cpu_nodes(mapping)
        self.assertIsNotNone(msg)
        self.assertIn("multiple devices", msg)

    def test_renamed_privateuse1_with_interface_allowed(self):
        with temporary_device_interface("npu", _RegisteredInterface):
            node = _FakeNode()
            mapping = {_FakeDevice("npu"): node}
            with mock.patch(
                "torch._C._get_privateuse1_backend_name", return_value="npu"
            ):
                self.assertIsNone(check_multiple_devices_or_any_cpu_nodes(mapping))

    def test_renamed_privateuse1_without_interface_skipped(self):
        node = _FakeNode()
        mapping = {_FakeDevice("npu"): node}
        with mock.patch("torch._C._get_privateuse1_backend_name", return_value="npu"):
            msg = check_multiple_devices_or_any_cpu_nodes(mapping)
        self.assertIsNotNone(msg)
        self.assertIn("multiple devices", msg)

    def test_default_privateuseone_skipped(self):
        node = _FakeNode()
        mapping = {torch.device("privateuseone:0"): node}
        with mock.patch(
            "torch._C._get_privateuse1_backend_name", return_value="privateuseone"
        ):
            msg = check_multiple_devices_or_any_cpu_nodes(mapping)
        self.assertIsNotNone(msg)
        self.assertIn("multiple devices", msg)


if __name__ == "__main__":
    run_tests()
