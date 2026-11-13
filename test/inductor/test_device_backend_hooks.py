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
from torch._inductor.kernel.flex.flex_attention import set_float32_precision
from torch._inductor.runtime.hints import DeviceProperties
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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

    def test_base_defaults_are_non_cuda_fallthrough(self):
        self.assertTrue(DeviceInterface.is_fp32_attention_fusion_safe(torch.float32))
        self.assertFalse(DeviceInterface.should_warn_tf32_disabled())
        self.assertTrue(DeviceInterface.is_fp32_softmax_attention_fusion_safe())
        self.assertFalse(DeviceInterface.keep_attention_on_math_path())
        self.assertFalse(DeviceInterface.needs_contiguous_attn_mask())

    def test_cuda_overrides_match_inline_conditions(self):
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
            with (
                mock.patch.object(torch.cuda, "is_available", lambda: True),
                mock.patch.object(torch.cuda, "is_tf32_supported", lambda: True),
            ):
                self.assertTrue(CudaInterface.should_warn_tf32_disabled())
            with mock.patch.object(torch.cuda, "is_available", lambda: False):
                self.assertFalse(CudaInterface.should_warn_tf32_disabled())
            # CUDA rejects the fp32-upcast-softmax patterns (25-27) and keeps
            # pattern 16 on the math path; ROCm shares CudaInterface but opts
            # into fused SDPA for pattern 16, so gate on torch.version.hip.
            self.assertFalse(CudaInterface.is_fp32_softmax_attention_fusion_safe())
            self.assertEqual(
                CudaInterface.keep_attention_on_math_path(),
                torch.version.hip is None,
            )
        finally:
            matmul.fp32_precision = saved

    def test_registered_non_cuda_inherits_base_defaults(self):
        self.assertIs(get_interface_for_device("cpu"), CpuInterface)
        self.assertIs(get_interface_for_device("xpu"), XpuInterface)
        self.assertIs(get_interface_for_device("mtia"), MtiaInterface)
        self.assertTrue(CpuInterface.is_fp32_attention_fusion_safe(torch.float32))
        self.assertTrue(CpuInterface.is_fp32_softmax_attention_fusion_safe())
        self.assertFalse(CpuInterface.keep_attention_on_math_path())
        self.assertFalse(CpuInterface.needs_contiguous_attn_mask())
        self.assertTrue(XpuInterface.needs_contiguous_attn_mask())

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

        match = self._match_for(device_type)
        self.assertTrue(_sfdp_params_check(match))
        self.assertTrue(_sfdp_extra_check(fp32_upcast_softmax=True)(match))

    @parametrize("device_type", ["definitely_unregistered_device", "privateuse1"])
    def test_sfdp_checks_unregistered_device_falls_through(self, device_type):
        from torch._inductor.fx_passes.fuse_attention import (
            _sfdp_extra_check,
            _sfdp_params_check,
        )

        # Unregistered devices fall back to the base DeviceInterface defaults,
        # matching the pre-routing `== "cuda"` skip instead of raising.
        match = self._match_for(device_type)
        self.assertTrue(_sfdp_params_check(match))
        self.assertTrue(_sfdp_extra_check(fp32_upcast_softmax=True)(match))
        self.assertTrue(_sfdp_extra_check(math_path_only=True)(match))

    def test_override_changes_fusion_outcome(self):
        from torch._dynamo import device_interface as di
        from torch._inductor.fx_passes.fuse_attention import (
            _sfdp_extra_check,
            _sfdp_params_check,
        )

        # An out-of-tree backend overriding keep_attention_on_math_path drives
        # the pattern-16 fusion decision the PR is meant to enable: the override
        # rejects the pattern-16 match (math_path_only gate) while the base
        # defaults still let _sfdp_params_check and the fp32-upcast-softmax
        # gate fall through.
        class FakeBackendInterface(DeviceInterface):
            @staticmethod
            def keep_attention_on_math_path() -> bool:
                return True

        fake_device = "fakebackend"
        di.device_interfaces[fake_device] = FakeBackendInterface
        try:
            match = self._match_for(fake_device)
            self.assertTrue(_sfdp_params_check(match))
            self.assertFalse(_sfdp_extra_check(math_path_only=True)(match))
            self.assertTrue(_sfdp_extra_check(fp32_upcast_softmax=True)(match))
        finally:
            di.device_interfaces.pop(fake_device, None)


class TestFp32AttentionPrecisionFallback(TestCase):
    """Device-independent tests for get_fp32_attention_precision().

    These run on every CI job (CPU included) and lock in the base-class
    default, the "none" -> "ieee" fallback, and the unregistered-device
    fallback to base defaults.
    """

    hw_classification = HardwareClassification.GENERIC

    def test_base_interface_default_is_none(self):
        """Base DeviceInterface returns "none" so callers fall back to ieee."""
        self.assertEqual(DeviceInterface.get_fp32_attention_precision(), "none")

    def test_none_falls_back_to_ieee(self):
        """A registered interface that answers "none" must resolve to ieee."""
        # Pin a registered interface (CpuInterface inherits the base default
        # today) so this locks the "none" -> 'ieee' mapping itself rather than
        # duplicating the ambient-default route covered by
        # test_non_cuda_device_uses_base_default.
        kernel_options = {}
        with mock.patch.object(
            CpuInterface, "get_fp32_attention_precision", return_value="none"
        ):
            set_float32_precision(kernel_options, torch.float32, "cpu")
        self.assertEqual(kernel_options["FLOAT32_PRECISION"], "'ieee'")

    def test_non_cuda_device_uses_base_default(self):
        """Non-CUDA registered devices resolve through base DeviceInterface."""
        kernel_options = {}
        set_float32_precision(kernel_options, torch.float32, "cpu")
        self.assertEqual(kernel_options["FLOAT32_PRECISION"], "'ieee'")

    def test_unregistered_device_falls_back(self):
        """Unregistered device types fall back to base default (ieee)."""
        kernel_options = {}
        # "privateuseone" has no registered interface; must not raise.
        set_float32_precision(kernel_options, torch.float32, "privateuseone")
        self.assertEqual(kernel_options["FLOAT32_PRECISION"], "'ieee'")

    def test_existing_option_is_not_overwritten(self):
        """A caller-pinned FLOAT32_PRECISION must not be overwritten."""
        kernel_options = {"FLOAT32_PRECISION": "'tf32'"}
        with mock.patch.object(
            DeviceInterface, "get_fp32_attention_precision", return_value="none"
        ):
            set_float32_precision(kernel_options, torch.float32, "cpu")
        self.assertEqual(kernel_options["FLOAT32_PRECISION"], "'tf32'")

    def test_bfx9_is_ignored_for_non_fp32_dtype(self):
        """The bfx9 special case only applies to torch.float32."""
        # Pre-pin the option: the fp32-only bfx9 branch hard-sets (and would
        # clobber the pin to 'ieee'), while every other path setdefaults. This
        # stays valid on ROCm/MTIA hosts, where the final resolution is 'ieee'
        # for every dtype and an unpinned assertNotEqual("'ieee'") misfires.
        kernel_options = {"FLOAT32_PRECISION": "'tf32'"}
        with mock.patch.object(
            DeviceInterface, "get_fp32_attention_precision", return_value="bfx9"
        ):
            set_float32_precision(kernel_options, torch.bfloat16, "cpu")
        # Must not be resolved through the fp32-only bfx9 branch.
        self.assertEqual(kernel_options["FLOAT32_PRECISION"], "'tf32'")


class TestFp32AttentionPrecisionOverrides(TestCase):
    """CUDA-specific tests for get_fp32_attention_precision().

    All tests are skipped when CUDA is unavailable, so this class can live in
    a file that also runs on CPU-only CI.
    """

    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

    def test_cuda_interface_overrides_base(self, device):
        """CudaInterface overrides the base hook with the live CUDA value."""
        # cuBLASModule.fp32_precision dispatches through __setattr__ to a C
        # setter and reads back "none" on a fresh process, so set it explicitly
        # (restored in finally) rather than asserting on the ambient value.
        matmul = torch.backends.cuda.matmul
        saved = matmul.fp32_precision
        try:
            matmul.fp32_precision = "tf32"
            self.assertEqual(CudaInterface.get_fp32_attention_precision(), "tf32")
            # Must differ from the base default, proving the override is active.
            self.assertNotEqual(CudaInterface.get_fp32_attention_precision(), "none")
        finally:
            matmul.fp32_precision = saved

    def test_bfx9_falls_back_to_ieee(self, device):
        """bfx9 on fp32 resolves to ieee with a warning."""
        kernel_options = {}
        with mock.patch.object(
            CudaInterface, "get_fp32_attention_precision", return_value="bfx9"
        ):
            set_float32_precision(kernel_options, torch.float32, "cuda")
        self.assertEqual(kernel_options["FLOAT32_PRECISION"], "'ieee'")

    def test_fusion_safe_uses_interface_hook(self, device):
        """is_fp32_attention_fusion_safe must route through the new hook."""
        with mock.patch.object(
            CudaInterface, "get_fp32_attention_precision", return_value="tf32"
        ):
            self.assertTrue(CudaInterface.is_fp32_attention_fusion_safe(torch.float32))
        with mock.patch.object(
            CudaInterface, "get_fp32_attention_precision", return_value="ieee"
        ):
            self.assertFalse(CudaInterface.is_fp32_attention_fusion_safe(torch.float32))

    def test_fusion_safe_non_fp32_always_true(self, device):
        """Non-fp32 dtypes are always safe regardless of precision policy."""
        with mock.patch.object(
            CudaInterface, "get_fp32_attention_precision", return_value="ieee"
        ):
            self.assertTrue(CudaInterface.is_fp32_attention_fusion_safe(torch.bfloat16))

    def test_should_warn_tf32_disabled_uses_interface_hook(self, device):
        """should_warn_tf32_disabled must consult the hook for the bfx9 guard."""
        with mock.patch.object(
            CudaInterface, "get_fp32_attention_precision", return_value="bfx9"
        ):
            self.assertFalse(CudaInterface.should_warn_tf32_disabled())


instantiate_device_type_tests(
    TestFp32AttentionPrecisionOverrides, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
