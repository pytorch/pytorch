# Owner(s): ["module: cpp-extensions"]

import _codecs
import io
import os
import sys
import tempfile
import unittest
from typing import Union
from unittest.mock import patch

import numpy as np
import pytorch_openreg  # noqa: F401

import torch
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.serialization import safe_globals
from torch.testing._internal.common_utils import (
    IS_ARM64,
    skipIfTorchDynamo,
    TemporaryFileName,
    TEST_CUDA,
    TEST_XPU,
)
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


TEST_CUDA = TEST_CUDA and CUDA_HOME is not None
TEST_ROCM = TEST_CUDA and torch.version.hip is not None and ROCM_HOME is not None


def generate_faked_module():
    class _OpenRegMod:
        pass

    return _OpenRegMod()


def generate_faked_module_methods():
    def device_count() -> int:
        return 1

    def get_rng_state(
        device: Union[int, str, torch.device] = "openreg",
    ) -> torch.Tensor:
        # create a tensor using our custom device object.
        return torch.empty(4, 4, device="openreg")

    def set_rng_state(
        new_state: torch.Tensor, device: Union[int, str, torch.device] = "openreg"
    ) -> None:
        pass

    def is_available():
        return True

    def current_device():
        return 0

    torch.openreg.device_count = device_count
    torch.openreg.get_rng_state = get_rng_state
    torch.openreg.set_rng_state = set_rng_state
    torch.openreg.is_available = is_available
    torch.openreg.current_device = current_device
    torch.openreg._lazy_init = lambda: None
    torch.openreg.is_initialized = lambda: True


@unittest.skipIf(IS_ARM64, "Does not work on arm")
@unittest.skipIf(TEST_XPU, "XPU does not support cppextension currently")
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionOpenRgistration(common.TestCase):
    """Tests Open Device Registration with C++ extensions."""

    module = None

    def setUp(self):
        super().setUp()

        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        assert self.module is not None

    def tearDown(self):
        super().tearDown()

        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        torch.testing._internal.common_utils.remove_cpp_extensions_build_root()

        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                "cpp_extensions/open_registration_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

        torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
        generate_faked_module_methods()

    def test_base_device_registration(self):
        self.assertFalse(self.module.custom_add_called())
        # create a tensor using our custom device object
        device = self.module.custom_device()
        x = torch.empty(4, 4, device=device)
        y = torch.empty(4, 4, device=device)
        # Check that our device is correct.
        self.assertTrue(x.device == device)
        self.assertFalse(x.is_cpu)
        self.assertFalse(self.module.custom_add_called())
        # calls out custom add kernel, registered to the dispatcher
        z = x + y
        # check that it was called
        self.assertTrue(self.module.custom_add_called())
        z_cpu = z.to(device="cpu")
        # Check that our cross-device copy correctly copied the data to cpu
        self.assertTrue(z_cpu.is_cpu)
        self.assertFalse(z.is_cpu)
        self.assertTrue(z.device == device)
        self.assertEqual(z, z_cpu)

    def test_common_registration(self):
        # check unsupported device and duplicated registration
        with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
            torch._register_device_module("dev", generate_faked_module())
        with self.assertRaisesRegex(RuntimeError, "The runtime module of"):
            torch._register_device_module("openreg", generate_faked_module())

        # backend name can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("openreg")

        # backend name can't be renamed multiple times to different names.
        with self.assertRaisesRegex(
            RuntimeError, "torch.register_privateuse1_backend()"
        ):
            torch.utils.rename_privateuse1_backend("dev")

        # generator tensor and module can be registered only once
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            torch.utils.generate_methods_for_privateuse1_backend()

        # check whether torch.openreg have been registered correctly
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 1
        )
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.openreg"):
            torch.utils.backend_registration._get_custom_mod_func("func_name_")

        # check attributes after registered
        self.assertTrue(hasattr(torch.Tensor, "is_openreg"))
        self.assertTrue(hasattr(torch.Tensor, "openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.TypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_openreg"))
        self.assertTrue(hasattr(torch.UntypedStorage, "openreg"))
        self.assertTrue(hasattr(torch.nn.Module, "openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_openreg"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "openreg"))

    def test_open_device_generator_registration_and_hooks(self):
        device = self.module.custom_device()
        # None of our CPU operations should call the custom add function.
        self.assertFalse(self.module.custom_add_called())

        gen = torch.Generator(device=device)
        self.assertTrue(gen.device == device)

        default_gen = self.module.default_generator(0)
        self.assertTrue(
            default_gen.device.type == torch._C._get_privateuse1_backend_name()
        )

    def test_open_device_dispatchstub(self):
        # test kernels could be reused by privateuse1 backend through dispatchstub
        input_data = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        openreg_input_data = input_data.to("openreg")
        output_data = torch.abs(input_data)
        openreg_output_data = torch.abs(openreg_input_data)
        self.assertEqual(output_data, openreg_output_data.cpu())

        output_data = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        # output operand will resize flag is True in TensorIterator.
        openreg_input_data = input_data.to("openreg")
        openreg_output_data = output_data.to("openreg")
        # output operand will resize flag is False in TensorIterator.
        torch.abs(input_data, out=output_data[:, :, 0:6:2])
        torch.abs(openreg_input_data, out=openreg_output_data[:, :, 0:6:2])
        self.assertEqual(output_data, openreg_output_data.cpu())

        # output operand will resize flag is True in TensorIterator.
        # and convert output to contiguous tensor in TensorIterator.
        output_data = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        openreg_input_data = input_data.to("openreg")
        openreg_output_data = output_data.to("openreg")
        torch.abs(input_data, out=output_data[:, :, 0:6:3])
        torch.abs(openreg_input_data, out=openreg_output_data[:, :, 0:6:3])
        self.assertEqual(output_data, openreg_output_data.cpu())

    def test_open_device_quantized(self):
        input_data = torch.randn(3, 4, 5, dtype=torch.float32, device="cpu").to(
            "openreg"
        )
        quantized_tensor = torch.quantize_per_tensor(input_data, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("openreg:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)

    def test_open_device_random(self):
        # check if torch.openreg have implemented get_rng_state
        with torch.random.fork_rng(device_type="openreg"):
            pass

    def test_open_device_tensor(self):
        device = self.module.custom_device()

        # check whether print tensor.type() meets the expectation
        dtypes = {
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
        for tt, dt in dtypes.items():
            test_tensor = torch.empty(4, 4, dtype=tt, device=device)
            self.assertTrue(test_tensor.type() == dt)

        # check whether the attributes and methods of the corresponding custom backend are generated correctly
        x = torch.empty(4, 4)
        self.assertFalse(x.is_openreg)

        x = x.openreg(torch.device("openreg"))
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(x.is_openreg)

        # test different device type input
        y = torch.empty(4, 4)
        self.assertFalse(y.is_openreg)

        y = y.openreg(torch.device("openreg:0"))
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(y.is_openreg)

        # test different device type input
        z = torch.empty(4, 4)
        self.assertFalse(z.is_openreg)

        z = z.openreg(0)
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z.is_openreg)

    def test_open_device_packed_sequence(self):
        device = self.module.custom_device()  # noqa: F841
        a = torch.rand(5, 3)
        b = torch.tensor([1, 1, 1, 1, 1])
        input = torch.nn.utils.rnn.PackedSequence(a, b)
        self.assertFalse(input.is_openreg)
        input_openreg = input.openreg()
        self.assertTrue(input_openreg.is_openreg)

    def test_open_device_storage(self):
        # check whether the attributes and methods for storage of the corresponding custom backend are generated correctly
        x = torch.empty(4, 4)
        z1 = x.storage()
        self.assertFalse(z1.is_openreg)

        z1 = z1.openreg()
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z1.is_openreg)

        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            z1.openreg(torch.device("cpu"))

        z1 = z1.cpu()
        self.assertFalse(self.module.custom_add_called())
        self.assertFalse(z1.is_openreg)

        z1 = z1.openreg(device="openreg:0", non_blocking=False)
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z1.is_openreg)

        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            z1.openreg(device="cuda:0", non_blocking=False)

        # check UntypedStorage
        y = torch.empty(4, 4)
        z2 = y.untyped_storage()
        self.assertFalse(z2.is_openreg)

        z2 = z2.openreg()
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z2.is_openreg)

        # check custom StorageImpl create
        self.module.custom_storage_registry()

        z3 = y.untyped_storage()
        self.assertFalse(self.module.custom_storageImpl_called())

        z3 = z3.openreg()
        self.assertTrue(self.module.custom_storageImpl_called())
        self.assertFalse(self.module.custom_storageImpl_called())

        z3 = z3[0:3]
        self.assertTrue(self.module.custom_storageImpl_called())

    @unittest.skipIf(
        sys.version_info >= (3, 13),
        "Error: Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.",
    )
    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_open_device_storage_pin_memory(self):
        # Check if the pin_memory is functioning properly on custom device
        cpu_tensor = torch.empty(3)
        self.assertFalse(cpu_tensor.is_openreg)
        self.assertFalse(cpu_tensor.is_pinned())

        cpu_tensor_pin = cpu_tensor.pin_memory()
        self.assertTrue(cpu_tensor_pin.is_pinned())

        # Test storage pin_memory and is_pin
        cpu_storage = cpu_tensor.storage()
        self.assertFalse(cpu_storage.is_pinned("openreg"))

        cpu_storage_pinned = cpu_storage.pin_memory("openreg")
        self.assertTrue(cpu_storage_pinned.is_pinned("openreg"))

        # Test untyped storage pin_memory and is_pin
        cpu_tensor = torch.randn([3, 2, 1, 4])
        cpu_untyped_storage = cpu_tensor.untyped_storage()
        self.assertFalse(cpu_untyped_storage.is_pinned("openreg"))

        cpu_untyped_storage_pinned = cpu_untyped_storage.pin_memory("openreg")
        self.assertTrue(cpu_untyped_storage_pinned.is_pinned("openreg"))

    @unittest.skip(
        "Temporarily disable due to the tiny differences between clang++ and g++ in defining static variable in inline function"
    )
    def test_open_device_serialization(self):
        self.module.set_custom_device_index(-1)
        storage = torch.UntypedStorage(4, device=torch.device("openreg"))
        self.assertEqual(torch.serialization.location_tag(storage), "openreg")

        self.module.set_custom_device_index(0)
        storage = torch.UntypedStorage(4, device=torch.device("openreg"))
        self.assertEqual(torch.serialization.location_tag(storage), "openreg:0")

        cpu_storage = torch.empty(4, 4).storage()
        openreg_storage = torch.serialization.default_restore_location(
            cpu_storage, "openreg:0"
        )
        self.assertTrue(openreg_storage.is_openreg)

        # test tensor MetaData serialization
        x = torch.empty(4, 4).long()
        y = x.openreg()
        self.assertFalse(self.module.check_backend_meta(y))
        self.module.custom_set_backend_meta(y)
        self.assertTrue(self.module.check_backend_meta(y))

        self.module.custom_serialization_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pt")
            torch.save(y, path)
            z1 = torch.load(path)
            # loads correctly onto the openreg backend device
            self.assertTrue(z1.is_openreg)
            # loads BackendMeta data correctly
            self.assertTrue(self.module.check_backend_meta(z1))

            # cross-backend
            z2 = torch.load(path, map_location="cpu")
            # loads correctly onto the cpu backend device
            self.assertFalse(z2.is_openreg)
            # loads BackendMeta data correctly
            self.assertFalse(self.module.check_backend_meta(z2))

    def test_open_device_storage_resize(self):
        cpu_tensor = torch.randn([8])
        openreg_tensor = cpu_tensor.openreg()
        openreg_storage = openreg_tensor.storage()
        self.assertTrue(openreg_storage.size() == 8)

        # Only register tensor resize_ function.
        openreg_tensor.resize_(8)
        self.assertTrue(openreg_storage.size() == 8)

        with self.assertRaisesRegex(TypeError, "Overflow"):
            openreg_tensor.resize_(8**29)

    def test_open_device_storage_type(self):
        # test cpu float storage
        cpu_tensor = torch.randn([8]).float()
        cpu_storage = cpu_tensor.storage()
        self.assertEqual(cpu_storage.type(), "torch.FloatStorage")

        # test custom float storage before defining FloatStorage
        openreg_tensor = cpu_tensor.openreg()
        openreg_storage = openreg_tensor.storage()
        self.assertEqual(openreg_storage.type(), "torch.storage.TypedStorage")

        class CustomFloatStorage:
            @property
            def __module__(self):
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        # test custom float storage after defining FloatStorage
        try:
            torch.openreg.FloatStorage = CustomFloatStorage()
            self.assertEqual(openreg_storage.type(), "torch.openreg.FloatStorage")

            # test custom int storage after defining FloatStorage
            openreg_tensor2 = torch.randn([8]).int().openreg()
            openreg_storage2 = openreg_tensor2.storage()
            self.assertEqual(openreg_storage2.type(), "torch.storage.TypedStorage")
        finally:
            torch.openreg.FloatStorage = None

    def test_open_device_faketensor(self):
        with torch._subclasses.fake_tensor.FakeTensorMode.push():
            a = torch.empty(1, device="openreg")
            b = torch.empty(1, device="openreg:0")
            result = a + b  # noqa: F841

    def test_open_device_named_tensor(self):
        torch.empty([2, 3, 4, 5], device="openreg", names=["N", "C", "H", "W"])

    # Not an open registration test - this file is just very convenient
    # for testing torch.compile on custom C++ operators
    def test_compile_autograd_function_returns_self(self):
        x_ref = torch.randn(4, requires_grad=True)
        out_ref = self.module.custom_autograd_fn_returns_self(x_ref)
        out_ref.sum().backward()

        x_test = x_ref.detach().clone().requires_grad_(True)
        f_compiled = torch.compile(self.module.custom_autograd_fn_returns_self)
        out_test = f_compiled(x_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)

    # Not an open registration test - this file is just very convenient
    # for testing torch.compile on custom C++ operators
    @skipIfTorchDynamo("Temporary disabled due to torch._ops.OpOverloadPacket")
    def test_compile_autograd_function_aliasing(self):
        x_ref = torch.randn(4, requires_grad=True)
        out_ref = torch.ops._test_funcs.custom_autograd_fn_aliasing(x_ref)
        out_ref.sum().backward()

        x_test = x_ref.detach().clone().requires_grad_(True)
        f_compiled = torch.compile(torch.ops._test_funcs.custom_autograd_fn_aliasing)
        out_test = f_compiled(x_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)

    def test_open_device_scalar_type_fallback(self):
        z_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        z = torch.triu_indices(3, 3, device="openreg")
        self.assertEqual(z_cpu, z)

    def test_open_device_tensor_type_fallback(self):
        # create tensors located in custom device
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("openreg")
        y = torch.Tensor([1, 0, 2]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([[0, 2, 1], [1, 3, 2]])
        # Check that our device is correct.
        device = self.module.custom_device()
        self.assertTrue(x.device == device)
        self.assertFalse(x.is_cpu)

        # call sub op, which will fallback to cpu
        z = torch.sub(x, y)
        self.assertEqual(z_cpu, z)

        # call index op, which will fallback to cpu
        z_cpu = torch.Tensor([3, 1])
        y = torch.Tensor([1, 0]).long().to("openreg")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_open_device_tensorlist_type_fallback(self):
        # create tensors located in custom device
        v_openreg = torch.Tensor([1, 2, 3]).to("openreg")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_openreg, v_openreg)
        y = (v_openreg, v_openreg)
        # Check that our device is correct.
        device = self.module.custom_device()
        self.assertTrue(v_openreg.device == device)
        self.assertFalse(v_openreg.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = torch._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])

        # call _fused_adamw_ with undefined tensor.
        self.module.fallback_with_undefined_tensor()

    @skipIfTorchDynamo()
    @unittest.skipIf(
        np.__version__ < "1.25",
        "versions < 1.25 serialize dtypes differently from how it's serialized in data_legacy_numpy",
    )
    def test_open_device_numpy_serialization(self):
        """
        This tests the legacy _rebuild_device_tensor_from_numpy serialization path
        """
        device = self.module.custom_device()

        # Legacy data saved with _rebuild_device_tensor_from_numpy on f80ed0b8 via

        # with patch.object(torch._C, "_has_storage", return_value=False):
        #     x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device)
        #     x_foo = x.to(device)
        #     sd = {"x": x_foo}
        #     rebuild_func = x_foo._reduce_ex_internal(default_protocol)[0]
        #     self.assertTrue(
        #         rebuild_func is torch._utils._rebuild_device_tensor_from_numpy
        #     )
        #     with open("foo.pt", "wb") as f:
        #         torch.save(sd, f)

        data_legacy_numpy = (
            b"PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x10\x00\x12\x00archive/data.pklFB\x0e\x00ZZZZZZZZZZZZZZ\x80\x02}q\x00X\x01"
            b"\x00\x00\x00xq\x01ctorch._utils\n_rebuild_device_tensor_from_numpy\nq\x02(cnumpy.core.m"
            b"ultiarray\n_reconstruct\nq\x03cnumpy\nndarray\nq\x04K\x00\x85q\x05c_codecs\nencode\nq\x06"
            b"X\x01\x00\x00\x00bq\x07X\x06\x00\x00\x00latin1q\x08\x86q\tRq\n\x87q\x0bRq\x0c(K\x01K\x02K"
            b"\x03\x86q\rcnumpy\ndtype\nq\x0eX\x02\x00\x00\x00f4q\x0f\x89\x88\x87q\x10Rq\x11(K\x03X\x01"
            b"\x00\x00\x00<q\x12NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tq\x13b\x89h\x06X\x1c\x00\x00"
            b"\x00\x00\x00\xc2\x80?\x00\x00\x00@\x00\x00@@\x00\x00\xc2\x80@\x00\x00\xc2\xa0@\x00\x00\xc3"
            b"\x80@q\x14h\x08\x86q\x15Rq\x16tq\x17bctorch\nfloat32\nq\x18X\t\x00\x00\x00openreg:0q\x19\x89"
            b"tq\x1aRq\x1bs.PK\x07\x08\xdfE\xd6\xcaS\x01\x00\x00S\x01\x00\x00PK\x03\x04\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x00.\x00"
            b"archive/byteorderFB*\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08"
            b"\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00=\x00archive/versionFB9\x00"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00"
            b"\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZ"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0636457737946401051300000025273995036293PK\x07\x08\xee(\xcd"
            b"\x8d(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
            b"\xdfE\xd6\xcaS\x01\x00\x00S\x01\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\xa3\x01\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00"
            b"\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x16\x02\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\xee(\xcd\x8d(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x92\x02\x00\x00archive/.data/serialization_idPK\x06"
            b"\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00"
            b"\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x06\x01\x00\x00\x00\x00\x00\x008\x03\x00"
            b"\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00>\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
            b"PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00\x06\x01\x00\x008\x03\x00\x00\x00\x00"
        )
        buf_data_legacy_numpy = io.BytesIO(data_legacy_numpy)

        with safe_globals(
            [
                (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct")
                if np.__version__ >= "2.1"
                else np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                _codecs.encode,
                np.dtypes.Float32DType,
            ]
        ):
            sd_loaded = torch.load(buf_data_legacy_numpy, weights_only=True)
            buf_data_legacy_numpy.seek(0)
            # Test map_location
            sd_loaded_cpu = torch.load(
                buf_data_legacy_numpy, weights_only=True, map_location="cpu"
            )
        expected = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device
        )
        self.assertEqual(sd_loaded["x"].cpu(), expected.cpu())
        self.assertFalse(sd_loaded["x"].is_cpu)
        self.assertTrue(sd_loaded_cpu["x"].is_cpu)

    def test_open_device_cpu_serialization(self):
        torch.utils.rename_privateuse1_backend("openreg")
        device = self.module.custom_device()
        default_protocol = torch.serialization.DEFAULT_PROTOCOL

        with patch.object(torch._C, "_has_storage", return_value=False):
            x = torch.randn(2, 3)
            x_openreg = x.to(device)
            sd = {"x": x_openreg}
            rebuild_func = x_openreg._reduce_ex_internal(default_protocol)[0]
            self.assertTrue(
                rebuild_func is torch._utils._rebuild_device_tensor_from_cpu_tensor
            )
            # Test map_location
            with TemporaryFileName() as f:
                torch.save(sd, f)
                sd_loaded = torch.load(f, weights_only=True)
                # Test map_location
                sd_loaded_cpu = torch.load(f, weights_only=True, map_location="cpu")
            self.assertFalse(sd_loaded["x"].is_cpu)
            self.assertEqual(sd_loaded["x"].cpu(), x)
            self.assertTrue(sd_loaded_cpu["x"].is_cpu)

            # Test metadata_only
            with TemporaryFileName() as f:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Cannot serialize tensors on backends with no storage under skip_data context manager",
                ):
                    with torch.serialization.skip_data():
                        torch.save(sd, f)

    def test_open_device_dlpack(self):
        t = torch.randn(2, 3).to("openreg")
        capsule = torch.utils.dlpack.to_dlpack(t)
        t1 = torch.from_dlpack(capsule)
        self.assertTrue(t1.device == t.device)
        t = t.to("cpu")
        t1 = t1.to("cpu")
        self.assertEqual(t, t1)


if __name__ == "__main__":
    common.run_tests()
