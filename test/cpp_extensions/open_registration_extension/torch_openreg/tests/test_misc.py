# Owner(s): ["module: PrivateUse1"]

import types
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestBackendModule(TestCase):
    def test_backend_module_name(self):
        """Test backend module name query and renaming"""
        self.assertEqual(torch._C._get_privateuse1_backend_name(), "openreg")
        # backend can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("openreg")
        with self.assertRaisesRegex(RuntimeError, "has already been set"):
            torch.utils.rename_privateuse1_backend("dev")

    def test_backend_module_registration(self):
        """Test backend module registration error handling"""

        def generate_faked_module():
            return types.ModuleType("fake_module")

        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("openreg", generate_faked_module())

    def test_backend_module_function(self):
        """Test backend module function access"""
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):
            torch.utils.backend_registration._get_custom_mod_func("func_name_")
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 2
        )

    def test_backend_module_function_error_handling(self):
        """Test error handling for backend module functions"""
        # Test non-existent function
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):
            torch.utils.backend_registration._get_custom_mod_func("non_existent_func")

        # Test valid function
        device_count = torch.utils.backend_registration._get_custom_mod_func(
            "device_count"
        )
        self.assertIsNotNone(device_count)


class TestBackendProperty(TestCase):
    def test_backend_generate_methods(self):
        """Test backend method generation"""
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            torch.utils.generate_methods_for_privateuse1_backend()

        self.assertTrue(hasattr(torch.Tensor, "is_openreg"))
        self.assertTrue(hasattr(torch.Tensor, "openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.nn.Module, "openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "openreg"))

    def test_backend_tensor_methods(self):
        """Test backend tensor methods"""
        x = torch.empty(4, 4)
        self.assertFalse(x.is_openreg)

        y = x.openreg(torch.device("openreg"))
        self.assertTrue(y.is_openreg)
        z = x.openreg(torch.device("openreg:0"))
        self.assertTrue(z.is_openreg)
        n = x.openreg(0)
        self.assertTrue(n.is_openreg)

    @unittest.skip("Need to support Parameter in openreg")
    def test_backend_module_methods(self):
        """Test backend module methods (currently skipped)"""

        class FakeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3, 3))

            def forward(self):
                pass

        module = FakeModule()
        self.assertEqual(module.x.device.type, "cpu")
        module.openreg()  # type: ignore[misc]
        self.assertEqual(module.x.device.type, "openreg")

    @unittest.skip("Need to support untyped_storage in openreg")
    def test_backend_storage_methods(self):
        """Test backend storage methods (currently skipped)"""
        x = torch.empty(4, 4)

        x_cpu = x.storage()
        self.assertFalse(x_cpu.is_openreg)
        x_openreg = x_cpu.openreg()
        self.assertTrue(x_openreg.is_openreg)

        y = torch.empty(4, 4)

        y_cpu = y.untyped_storage()
        self.assertFalse(y_cpu.is_openreg)
        y_openreg = y_cpu.openreg()
        self.assertTrue(y_openreg.is_openreg)

    def test_backend_packed_sequence_methods(self):
        """Test backend PackedSequence methods"""
        x = torch.rand(5, 3)
        y = torch.tensor([1, 1, 1, 1, 1])

        z_cpu = torch.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)

        z_openreg = z_cpu.openreg()
        self.assertTrue(z_openreg.is_openreg)

    def test_backend_packed_sequence_properties(self):
        """Test PackedSequence backend properties"""
        x = torch.rand(5, 3)
        y = torch.tensor([1, 1, 1, 1, 1])

        z_cpu = torch.nn.utils.rnn.PackedSequence(x, y)
        self.assertFalse(z_cpu.is_openreg)

        z_openreg = z_cpu.openreg()
        self.assertTrue(z_openreg.is_openreg)

        # Test that data is on correct device
        self.assertTrue(z_openreg.data.is_openreg)

    def test_backend_tensor_methods_different_devices(self):
        """Test tensor methods with different device indices"""
        x = torch.empty(4, 4)

        y0 = x.openreg(0)
        self.assertTrue(y0.is_openreg)
        self.assertEqual(y0.device.index, 0)

        y1 = x.openreg(1)
        self.assertTrue(y1.is_openreg)
        self.assertEqual(y1.device.index, 1)

        y_none = x.openreg(torch.device("openreg"))
        self.assertTrue(y_none.is_openreg)


class TestTensorType(TestCase):
    def test_backend_tensor_type(self):
        """Test tensor type string representation for different dtypes"""
        dtypes_map = {
            torch.bool: "torch.openreg.BoolTensor",
            torch.double: "torch.openreg.DoubleTensor",
            torch.float32: "torch.openreg.FloatTensor",
            torch.half: "torch.openreg.HalfTensor",
            torch.int32: "torch.openreg.IntTensor",
            torch.int64: "torch.openreg.LongTensor",
            torch.int8: "torch.openreg.CharTensor",
            torch.short: "torch.openreg.ShortTensor",
            torch.uint8: "torch.openreg.ByteTensor",
        }

        for dtype, str in dtypes_map.items():
            x = torch.empty(4, 4, dtype=dtype, device="openreg")
            self.assertTrue(x.type() == str)

    # Note that all dtype-d Tensor objects here are only for legacy reasons
    # and should NOT be used.
    @skipIfTorchDynamo()
    def test_backend_type_methods(self):
        """Test backend type methods for tensor and storage"""
        # Tensor
        tensor_cpu = torch.randn([8]).float()
        self.assertEqual(tensor_cpu.type(), "torch.FloatTensor")

        tensor_openreg = tensor_cpu.openreg()
        self.assertEqual(tensor_openreg.type(), "torch.openreg.FloatTensor")

        # Storage
        storage_cpu = tensor_cpu.storage()
        self.assertEqual(storage_cpu.type(), "torch.FloatStorage")

        tensor_openreg = tensor_cpu.openreg()
        storage_openreg = tensor_openreg.storage()
        self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")

        class CustomFloatStorage:
            @property
            def __module__(self):
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        try:
            torch.openreg.FloatStorage = CustomFloatStorage()
            self.assertEqual(storage_openreg.type(), "torch.openreg.FloatStorage")

            # test custom int storage after defining FloatStorage
            tensor_openreg = tensor_cpu.int().openreg()
            storage_openreg = tensor_openreg.storage()
            self.assertEqual(storage_openreg.type(), "torch.storage.TypedStorage")
        finally:
            torch.openreg.FloatStorage = None

    def test_backend_storage_type_consistency(self):
        """Test storage type consistency"""
        tensor = torch.randn(4, 4, device="openreg")
        storage = tensor.storage()

        # Storage should be on same device
        self.assertTrue(storage.is_openreg)

        # Test storage size
        self.assertEqual(storage.size(), tensor.numel())


class TestDispatch(TestCase):
    def test_dispatch_key_registration(self):
        """Test that PrivateUse1 dispatch keys are correctly registered"""
        core_ops = [
            "aten::empty.memory_format",
            "aten::empty_strided",
            "aten::_copy_from",
            "aten::_copy_from_and_resize",
            "aten::view",
            "aten::resize_",
            "aten::as_strided",
            "aten::_reshape_alias",
            "aten::_local_scalar_dense",
            "aten::_has_compatible_shallow_copy_type",
            "aten::set_.source_Tensor",
            "aten::set_.source_Storage",
            "aten::set_.source_Storage_storage_offset",
        ]
        for op_name in core_ops:
            self.assertTrue(
                torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1"),
                f"{op_name} should have a PrivateUse1 kernel",
            )

        extra_ops = [
            "aten::abs.out",
            "aten::quantize_per_tensor",
            "aten::_fused_sdp_choice",
            "aten::_scaled_dot_product_fused_attention_overrideable",
            "aten::_scaled_dot_product_fused_attention_overrideable_backward",
        ]
        for op_name in extra_ops:
            self.assertTrue(
                torch._C._dispatch_has_kernel_for_dispatch_key(op_name, "PrivateUse1"),
                f"{op_name} should have a PrivateUse1 kernel",
            )

        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::sub.Tensor", "PrivateUse1"
            ),
            "sub.Tensor should have a direct PrivateUse1 kernel (per-op fallback)",
        )

        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "openreg::custom_abs", "PrivateUse1"
            ),
            "openreg::custom_abs should have a PrivateUse1 kernel",
        )

        autograd_ops = [
            "openreg::custom_autograd_fn_returns_self",
            "openreg::custom_autograd_fn_aliasing",
        ]
        for op_name in autograd_ops:
            self.assertTrue(
                torch._C._dispatch_has_kernel_for_dispatch_key(
                    op_name, "AutogradPrivateUse1"
                ),
                f"{op_name} should have an AutogradPrivateUse1 kernel",
            )

        self.assertFalse(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "aten::add.Tensor", "PrivateUse1"
            ),
            "add.Tensor should not have a direct PrivateUse1 kernel (resolves via global fallback)",
        )

        self.assertTrue(
            torch._C._dispatch_has_computed_kernel_for_dispatch_key(
                "aten::add.Tensor", "PrivateUse1"
            )
        )
        self.assertTrue(
            torch._C._dispatch_has_computed_kernel_for_dispatch_key(
                "aten::mul.Tensor", "PrivateUse1"
            )
        )

    def test_cpu_fallback_numerical_correctness(self):
        """Test that CPU fallback produces numerically correct results"""
        x_cpu = torch.randn(3, 4)
        y_cpu = torch.randn(3, 4)
        x = x_cpu.to("openreg")
        y = y_cpu.to("openreg")

        result = torch.add(x, y)
        self.assertEqual(result.device.type, "openreg")
        self.assertEqual(result.cpu(), torch.add(x_cpu, y_cpu))

        result = torch.mul(x, y)
        self.assertEqual(result.device.type, "openreg")
        self.assertEqual(result.cpu(), torch.mul(x_cpu, y_cpu))

        result = torch.sub(x, y)
        self.assertEqual(result.device.type, "openreg")
        self.assertEqual(result.cpu(), torch.sub(x_cpu, y_cpu))

        z = x.clone()
        z.add_(y)
        expected = x_cpu.clone().add_(y_cpu)
        self.assertEqual(z.cpu(), expected)

        result = torch.abs(x)
        self.assertEqual(result.device.type, "openreg")
        self.assertEqual(result.cpu(), torch.abs(x_cpu))


class TestTensorSubclass(TestCase):
    def test_tensor_subclass_compatibility(self):
        """Test tensor subclass mechanisms with the OpenReg backend"""
        from torch.utils._python_dispatch import TorchDispatchMode

        class LoggingTensor(torch.Tensor):
            log = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.log.append(func)
                return super().__torch_function__(func, types, args, kwargs or {})

        x = torch.randn(3, 3, device="openreg")
        lt = x.as_subclass(LoggingTensor)
        self.assertTrue(lt.is_openreg)
        self.assertEqual(lt.device.type, "openreg")

        LoggingTensor.log.clear()
        result = lt + 1
        self.assertGreaterEqual(len(LoggingTensor.log), 1)
        self.assertEqual(result.device.type, "openreg")

        class CountingMode(TorchDispatchMode):
            def __init__(self):
                self.count = 0

            def __torch_dispatch__(self, func, types, args, kwargs=None):
                self.count += 1
                return func(*args, **(kwargs or {}))

        y = torch.randn(3, 3, device="openreg")
        with CountingMode() as mode:
            _ = y + 1
            _ = y * 2
        self.assertGreaterEqual(mode.count, 2)


if __name__ == "__main__":
    run_tests()
