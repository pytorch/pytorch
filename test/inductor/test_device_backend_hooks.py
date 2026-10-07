# Owner(s): ["module: inductor"]
from types import SimpleNamespace
from unittest import mock

from torch._dynamo.device_interface import DeviceInterface, MtiaInterface, XpuInterface
from torch._inductor import config, ir
from torch._inductor.codegen.common import (
    _initialize_device_op_overrides,
    _uses_gpu_cpp_wrapper,
    device_op_overrides_dict,
    DeviceOpOverrides,
    register_device_op_overrides,
)
from torch._inductor.runtime.hints import DeviceProperties
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMultiProcessorCount(TestCase):
    def test_default_standard_property(self):
        class FakeInterface(DeviceInterface):
            @staticmethod
            def get_device_properties(device=None):
                return SimpleNamespace(multi_processor_count=16)

        self.assertEqual(FakeInterface.get_multi_processor_count(), 16)

    def test_missing_standard_property(self):
        class FakeInterface(DeviceInterface):
            @staticmethod
            def get_device_properties(device=None):
                return SimpleNamespace()

        with self.assertRaisesRegex(
            AttributeError, "must override get_multi_processor_count"
        ):
            FakeInterface.get_multi_processor_count()

    def test_xpu_override_without_hardware(self):
        with mock.patch.object(
            XpuInterface,
            "get_device_properties",
            side_effect=[
                SimpleNamespace(multi_processor_count=16, gpu_subslice_count=32),
                SimpleNamespace(gpu_subslice_count=32),
            ],
        ):
            self.assertEqual(XpuInterface.get_multi_processor_count(), 16)
            self.assertEqual(XpuInterface.get_multi_processor_count(), 32)

    def test_mtia_override(self):
        with mock.patch.object(
            MtiaInterface,
            "get_device_properties",
            side_effect=[
                SimpleNamespace(multi_processor_count=16),
                SimpleNamespace(),
            ],
        ):
            self.assertEqual(MtiaInterface.get_multi_processor_count(), 16)
            self.assertEqual(MtiaInterface.get_multi_processor_count(), 64)


class TestDevicePropertiesCreate(TestCase):
    def test_uses_interface_multi_processor_count(self):
        class FakeDevice:
            type = "fake"
            index = 0

        class FakeInterface:
            @staticmethod
            def get_device_properties(device):
                return SimpleNamespace()

            @staticmethod
            def get_multi_processor_count(device):
                return 32

            @staticmethod
            def get_compute_capability(device):
                return 0

        device = FakeDevice()
        DeviceProperties.create.cache_clear()
        try:
            with mock.patch(
                "torch._dynamo.device_interface.get_interface_for_device",
                return_value=FakeInterface,
            ):
                props = DeviceProperties.create(device)
            self.assertEqual(props.multi_processor_count, 32)
        finally:
            DeviceProperties.create.cache_clear()


class TestUsesGpuCppWrapper(TestCase):
    def test_builtin_devices(self):
        _initialize_device_op_overrides()
        self.assertEqual(
            {d for d in device_op_overrides_dict if _uses_gpu_cpp_wrapper(d)},
            {"cuda", "xpu"},
        )

    def test_cpu_triton_does_not_use_gpu_cpp_wrapper(self):
        with config.patch({"cpu_backend": "triton"}):
            self.assertTrue(ir.is_triton("cpu"))
            self.assertFalse(_uses_gpu_cpp_wrapper("cpu"))

    def test_out_of_tree_opt_in(self):
        class ExtensionDeviceOpOverrides(DeviceOpOverrides):
            def uses_gpu_cpp_wrapper(self) -> bool:
                return True

        name = "test_cpp_wrapper_opt_in"
        register_device_op_overrides(name, ExtensionDeviceOpOverrides())
        try:
            self.assertTrue(_uses_gpu_cpp_wrapper(name))
        finally:
            device_op_overrides_dict.pop(name, None)

    def test_out_of_tree_default(self):
        name = "test_cpp_wrapper_default"
        register_device_op_overrides(name, DeviceOpOverrides())
        try:
            self.assertFalse(_uses_gpu_cpp_wrapper(name))
        finally:
            device_op_overrides_dict.pop(name, None)

    def test_unregistered_device(self):
        self.assertFalse(_uses_gpu_cpp_wrapper("definitely_unregistered_device"))


class TestGraphLoweringCppWrapper(TestCase):
    def test_mtia_mps_single_pass_cuda_xpu_two_pass(self):
        from torch._inductor.codegen.common import init_backend_registration
        from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
        from torch._inductor.graph import GraphLowering
        from torch._inductor.utils import ValueWithLineMap
        from torch.utils._ordered_set import OrderedSet

        init_backend_registration()
        # MTIA is Triton-capable but must stay on the single-pass path.
        self.assertTrue(ir.is_triton("mtia"))

        def takes_two_pass(device_types):
            stub = object.__new__(GraphLowering)
            stub.device_types = OrderedSet(device_types)
            stub.aot_mode = True
            stub.operations = []
            wrapper = object.__new__(CppWrapperGpu)
            wrapper._lazy_kernel_names = ["k1"]
            wrapper._aot_output = "aot"
            stub.wrapper_code = wrapper
            jit = ValueWithLineMap("jit", [])
            kernel = ValueWithLineMap("kernel", [])
            with (
                mock.patch.object(GraphLowering, "validate_can_generate_cpp_wrapper"),
                mock.patch.object(GraphLowering, "codegen", return_value=(jit, kernel)),
                mock.patch.object(
                    GraphLowering, "_run_jit_variant_for_autotune"
                ) as mock_run,
                mock.patch(
                    "torch._inductor.codegen.cpp_wrapper_gpu."
                    "generate_aoti_kernel_config_header",
                    return_value="header",
                ),
                config.patch({"triton.autotune_at_compile_time": False}),
            ):
                stub.codegen_with_cpp_wrapper()
                return mock_run.called

        def jit_compile_device(device_types):
            stub = object.__new__(GraphLowering)
            stub.device_types = OrderedSet(device_types)
            stub.is_const_graph = True
            stub.constants = {}
            stub.graph_outputs = []
            stub.wrapper_code = SimpleNamespace(additional_files=[])

            class _Captured(Exception):
                pass

            captured = {}

            def fake_load(*_args, **kwargs):
                captured["device_type"] = kwargs.get("device_type")
                raise _Captured

            with mock.patch(
                "torch._inductor.codecache.CppWrapperCodeCache.load_pybinding",
                side_effect=fake_load,
            ):
                try:
                    stub._run_jit_variant_for_autotune(
                        SimpleNamespace(value="cpp"),
                        SimpleNamespace(value="kernel"),
                        list,
                        ["k1"],
                    )
                except _Captured:
                    pass
            return captured["device_type"]

        self.assertFalse(takes_two_pass(["mtia"]))
        self.assertFalse(takes_two_pass(["mps"]))
        self.assertTrue(takes_two_pass(["cuda"]))
        self.assertTrue(takes_two_pass(["xpu"]))
        self.assertFalse(takes_two_pass(["cpu", "mtia"]))
        self.assertFalse(takes_two_pass(["cpu", "mps"]))
        self.assertTrue(takes_two_pass(["cpu", "cuda"]))
        self.assertTrue(takes_two_pass(["cpu", "xpu"]))

        self.assertEqual(jit_compile_device(["cpu", "mtia"]), "cpu")
        self.assertEqual(jit_compile_device(["cpu", "mps"]), "cpu")
        self.assertEqual(jit_compile_device(["cpu", "cuda"]), "cuda")
        self.assertEqual(jit_compile_device(["cpu", "xpu"]), "xpu")


if __name__ == "__main__":
    run_tests()
