# Owner(s): ["module: inductor"]
from types import SimpleNamespace
from unittest import mock

import torch
from torch._dynamo.device_interface import (
    CpuInterface,
    CudaInterface,
    DeviceInterface,
    get_interface_for_device,
    MtiaInterface,
    XpuInterface,
)
from torch._inductor import config, ir
from torch._inductor.codegen.common import (
    _initialize_device_op_overrides,
    _uses_gpu_cpp_wrapper,
    device_op_overrides_dict,
    DeviceOpOverrides,
    register_device_op_overrides,
)
from torch._inductor.runtime.hints import DeviceProperties
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


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


@instantiate_parametrized_tests
class TestAttentionFusionDeviceHooks(TestCase):
    hw_classification = HardwareClassification.GENERIC
    # The attention-fusion hooks routed through DeviceInterface in
    # fuse_attention.py. The base DeviceInterface defaults encode the non-CUDA
    # fall-through (fp32 fusion allowed, no tf32 warning, fp32 upcast softmax
    # allowed, fused SDPA preferred over the math path); CudaInterface overrides
    # each to the CUDA-specific condition the inline checks encoded before.

    def test_base_defaults_are_non_cuda_fallthrough(self):
        self.assertTrue(DeviceInterface.is_fp32_attention_fusion_safe(torch.float32))
        self.assertFalse(DeviceInterface.should_warn_tf32_disabled())
        self.assertTrue(DeviceInterface.is_fp32_softmax_attention_fusion_safe())
        self.assertFalse(DeviceInterface.keep_attention_on_math_path())

    def test_cuda_overrides_match_inline_conditions(self):
        # keep_attention_on_math_path reproduces `query.device.type == "cuda"
        # and torch.version.hip is None`.
        self.assertEqual(
            CudaInterface.keep_attention_on_math_path(), torch.version.hip is None
        )
        # is_fp32_softmax_attention_fusion_safe rejects CUDA (was
        # `"cuda" not in str(device)` for the CUDA-resolved interface, i.e.
        # always False once dispatched through CudaInterface).
        self.assertFalse(CudaInterface.is_fp32_softmax_attention_fusion_safe())

        # cuBLASModule.fp32_precision is dispatched through __getattr__/__setattr__
        # to a C getter/setter (not a real attribute or property), so mock.patch.object
        # cannot patch it; restore it explicitly instead.
        matmul = torch.backends.cuda.matmul
        saved = matmul.fp32_precision
        try:
            matmul.fp32_precision = "tf32"
            self.assertTrue(CudaInterface.is_fp32_attention_fusion_safe(torch.float32))
            matmul.fp32_precision = "ieee"
            self.assertTrue(CudaInterface.is_fp32_attention_fusion_safe(torch.half))
            self.assertFalse(CudaInterface.is_fp32_attention_fusion_safe(torch.float32))
            matmul.fp32_precision = "ieee"
            with mock.patch.object(torch.cuda, "is_available", lambda: True), mock.patch.object(
                torch.cuda, "get_device_capability", lambda: (8, 0)
            ):
                self.assertTrue(CudaInterface.should_warn_tf32_disabled())
            with mock.patch.object(torch.cuda, "is_available", lambda: False):
                self.assertFalse(CudaInterface.should_warn_tf32_disabled())
        finally:
            matmul.fp32_precision = saved

    def test_registered_non_cuda_inherits_base_defaults(self):
        # CPU/XPU/MTIA are registered but do not override the fusion hooks, so
        # they inherit the base fall-through.
        self.assertIs(get_interface_for_device("cpu"), CpuInterface)
        self.assertIs(get_interface_for_device("xpu"), XpuInterface)
        self.assertIs(get_interface_for_device("mtia"), MtiaInterface)
        self.assertTrue(CpuInterface.is_fp32_attention_fusion_safe(torch.float32))
        self.assertTrue(CpuInterface.is_fp32_softmax_attention_fusion_safe())
        self.assertFalse(CpuInterface.keep_attention_on_math_path())

    def _match_for(self, device_type):
        device = SimpleNamespace(type=device_type)
        tensor = SimpleNamespace(dtype=torch.float32, device=device)
        node = SimpleNamespace(meta={"val": tensor})
        return SimpleNamespace(
            kwargs={"query": node, "key": node, "value": node},
            nodes=[],
        )

    @parametrize("device_type", ["cpu", "xpu", "mtia"])
    def test_sfdp_checks_registered_device_falls_through(self, device_type):
        from torch._inductor.fx_passes.fuse_attention import (
            _sfdp_extra_check,
            _sfdp_params_check,
        )

        # Registered non-CUDA devices inherit the base fall-through defaults,
        # so neither check rejects the fusion: _sfdp_params_check returns True
        # (the fp32/tf32 CUDA gate is skipped), and the fp32-upcast softmax
        # extra check does not short-circuit. This matches the pre-routing
        # `query.device.type == "cuda"` fall-through.
        match = self._match_for(device_type)
        self.assertTrue(_sfdp_params_check(match))
        self.assertTrue(_sfdp_extra_check(fp32_upcast_softmax=True)(match))

    @parametrize("device_type", ["definitely_unregistered_device", "privateuse1"])
    def test_sfdp_checks_unregistered_device_raises(self, device_type):
        from torch._inductor.fx_passes.fuse_attention import (
            _sfdp_extra_check,
            _sfdp_params_check,
        )

        # get_interface_for_device raises NotImplementedError for a device type
        # with no registered interface; the attention-fusion checks route
        # through it unconditionally, so an unregistered PrivateUse1 backend
        # surfaces that error rather than silently falling through.
        match = self._match_for(device_type)
        with self.assertRaises(NotImplementedError):
            _sfdp_params_check(match)
        with self.assertRaises(NotImplementedError):
            _sfdp_extra_check(fp32_upcast_softmax=True)(match)


if __name__ == "__main__":
    run_tests()
