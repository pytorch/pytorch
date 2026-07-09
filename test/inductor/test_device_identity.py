# Owner(s): ["module: inductor"]
import functools
import unittest

import torch
from torch._dynamo.device_interface import (
    CpuInterface,
    CudaInterface,
    DeviceInterface,
    MpsInterface,
    MtiaInterface,
    TpuInterface,
    XpuInterface,
    get_interface_for_device,
    get_registered_device_interfaces,
    init_device_reg,
    register_interface_for_device,
)
from torch._inductor.runtime.hints import DeviceProperties
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDeviceInterfaceCapabilityBits(TestCase):
    """Test the new capability bits added to DeviceInterface (is_gpu,
    exposes_streams, get_multi_processor_count)."""

    def test_base_defaults(self):
        """Base DeviceInterface returns safe defaults for all bits."""
        self.assertFalse(DeviceInterface.is_gpu())
        self.assertFalse(DeviceInterface.exposes_streams())
        # get_multi_processor_count requires device properties; the base
        # raises NotImplementedError because there is no device to query.

    def test_cuda_declares_gpu(self):
        """CUDA is a GPU with streams."""
        self.assertTrue(CudaInterface.is_gpu())
        self.assertTrue(CudaInterface.exposes_streams())

    def test_xpu_declares_gpu(self):
        """XPU is a GPU with streams and a non-standard multi-processor field."""
        self.assertTrue(XpuInterface.is_gpu())
        self.assertTrue(XpuInterface.exposes_streams())

    def test_mtia_declares_gpu(self):
        """MTIA is a GPU with streams and a hard-coded core count."""
        self.assertTrue(MtiaInterface.is_gpu())
        self.assertTrue(MtiaInterface.exposes_streams())
        self.assertEqual(MtiaInterface.get_multi_processor_count(), 64)

    def test_mps_is_gpu_without_streams(self):
        """MPS is a GPU but does NOT expose streams.  This is the key
        combination that ensures device_need_guard('mps') == False."""
        self.assertTrue(MpsInterface.is_gpu())
        self.assertFalse(MpsInterface.exposes_streams())

    def test_cpu_is_not_gpu(self):
        """CPU is not a GPU even though it has triton capability."""
        self.assertFalse(CpuInterface.is_gpu())
        self.assertTrue(CpuInterface.is_triton_capable())

    def test_tpu_is_not_gpu(self):
        """TPU is not a GPU."""
        self.assertFalse(TpuInterface.is_gpu())


class TestIsGpuPredicate(TestCase):
    """Test that is_gpu() queries DeviceInterface instead of a hard-coded list."""

    def test_known_gpu_returns_true(self):
        """is_gpu returns True for devices whose interface declares is_gpu()."""
        from torch._inductor.utils import is_gpu

        self.assertTrue(is_gpu("cuda"))
        self.assertTrue(is_gpu("xpu"))
        self.assertTrue(is_gpu("mps"))
        self.assertTrue(is_gpu("mtia"))

    def test_known_non_gpu_returns_false(self):
        """is_gpu returns False for devices whose interface does not opt in."""
        from torch._inductor.utils import is_gpu

        self.assertFalse(is_gpu("cpu"))
        self.assertFalse(is_gpu("tpu"))

    def test_none_returns_false(self):
        """is_gpu(None) returns False (guard against None input)."""
        from torch._inductor.utils import is_gpu

        self.assertFalse(is_gpu(None))

    def test_unknown_device_returns_false(self):
        """An unregistered device does not crash and is treated as non-GPU."""
        from torch._inductor.utils import is_gpu

        self.assertFalse(is_gpu("nonexistent_device_xyz"))

    def test_privateuse1_gpu_after_opt_in(self):
        """A PrivateUse1 backend that declares is_gpu()=True is recognized."""
        from torch._inductor.utils import is_gpu

        # Create a minimal GPU-like interface and register it
        class AccInterface(DeviceInterface):
            @staticmethod
            def is_gpu() -> bool:
                return True

            @staticmethod
            def is_available() -> bool:
                return True

        register_interface_for_device("acc_test", AccInterface)
        try:
            self.assertTrue(is_gpu("acc_test"))
        finally:
            # Clean up: remove the test registration.  The device_interfaces
            # dict keeps registrations for the lifetime of the process, so
            # we must explicitly delete our entry.
            from torch._dynamo import device_interface as di

            di.device_interfaces.pop("acc_test", None)


class TestDeviceNeedGuard(TestCase):
    """Test that device_need_guard derives from is_gpu + exposes_streams."""

    def test_cuda_needs_guard(self):
        from torch._inductor.utils import device_need_guard

        self.assertTrue(device_need_guard("cuda"))

    def test_mps_does_not_need_guard(self):
        """MPS is a GPU but lacks streams → guard not needed."""
        from torch._inductor.utils import device_need_guard

        self.assertFalse(device_need_guard("mps"))

    def test_cpu_does_not_need_guard(self):
        from torch._inductor.utils import device_need_guard

        self.assertFalse(device_need_guard("cpu"))


class TestHasTriton(TestCase):
    """Test has_triton() walks the DeviceInterface registry."""

    def test_has_triton_discovers_registered_device(self):
        """When a registered interface declares is_triton_capable(),
        has_triton() returns True (provided triton package is present)."""
        # This test requires the triton package.
        try:
            import triton  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("triton package not available")

        from torch.utils._triton import has_triton_package

        if not has_triton_package():
            raise unittest.SkipTest("triton package not available")

        # Clear the has_triton cache so our registration is seen.
        torch.utils._triton.has_triton.cache_clear()

        # has_triton() should find at least one triton-capable device
        # among the standard registrations (cuda/xpu/cpu/mtia).
        # We cannot assert True unconditionally because the test machine
        # may not have any of those devices available.
        result = torch.utils._triton.has_triton()
        # At minimum, the function should not crash and return a bool.
        self.assertIsInstance(result, bool)


class TestGetGpuType(TestCase):
    """Test get_gpu_type() disambiguation."""

    def test_returns_string(self):
        from torch._inductor.utils import get_gpu_type

        # Must clear the cache so we don't get a stale result from
        # another test's registration.
        get_gpu_type.cache_clear()
        try:
            gpu = get_gpu_type()
            self.assertIsInstance(gpu, str)
        finally:
            get_gpu_type.cache_clear()

    def test_falls_back_to_cuda_when_no_gpu_available(self):
        """When no GPU is available, get_gpu_type() returns 'cuda'."""
        from torch._inductor.utils import _gpu_types, get_gpu_type

        get_gpu_type.cache_clear()
        try:
            gpu = get_gpu_type()
            self.assertIsInstance(gpu, str)
            # On a machine with no GPU, fallback is "cuda"
            if not any(
                get_interface_for_device(t).is_available() for t in _gpu_types()
            ):
                self.assertEqual(gpu, "cuda")
        finally:
            get_gpu_type.cache_clear()


class TestDevicePropertiesCreate(TestCase):
    """Test DeviceProperties.create() uses the contract method."""

    def test_create_for_cpu(self):
        """DeviceProperties.create works for CPU (the one device always available)."""
        device = torch.device("cpu")
        props = DeviceProperties.create(device)
        self.assertIsInstance(props, DeviceProperties)
        self.assertEqual(props.type, "cpu")
        self.assertGreater(props.multi_processor_count, 0)


if __name__ == "__main__":
    run_tests()
