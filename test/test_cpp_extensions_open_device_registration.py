# Owner(s): ["module: cpp-extensions"]

import _codecs
import io
import os
import tempfile
import types
import unittest
from typing import Union
from unittest.mock import patch

import numpy as np

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
    def device_count() -> int:
        return 1

    def get_rng_state(device: Union[int, str, torch.device] = "foo") -> torch.Tensor:
        # create a tensor using our custom device object.
        return torch.empty(4, 4, device="foo")

    def set_rng_state(
        new_state: torch.Tensor, device: Union[int, str, torch.device] = "foo"
    ) -> None:
        pass

    def is_available():
        return True

    def current_device():
        return 0

    # create a new module to fake torch.foo dynamicaly
    foo = types.ModuleType("foo")

    foo.device_count = device_count
    foo.get_rng_state = get_rng_state
    foo.set_rng_state = set_rng_state
    foo.is_available = is_available
    foo.current_device = current_device
    foo._lazy_init = lambda: None
    foo.is_initialized = lambda: True

    return foo


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

        # register torch.foo module and foo device to torch
        torch.utils.rename_privateuse1_backend("foo")
        torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
        torch._register_device_module("foo", generate_faked_module())

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
            torch._register_device_module("foo", generate_faked_module())

        # backend name can be renamed to the same name multiple times
        torch.utils.rename_privateuse1_backend("foo")

        # backend name can't be renamed multiple times to different names.
        with self.assertRaisesRegex(
            RuntimeError, "torch.register_privateuse1_backend()"
        ):
            torch.utils.rename_privateuse1_backend("dev")

        # generator tensor and module can be registered only once
        with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
            torch.utils.generate_methods_for_privateuse1_backend()

        # check whether torch.foo have been registered correctly
        self.assertTrue(
            torch.utils.backend_registration._get_custom_mod_func("device_count")() == 1
        )
        with self.assertRaisesRegex(RuntimeError, "Try to call torch.foo"):
            torch.utils.backend_registration._get_custom_mod_func("func_name_")

        # check attributes after registered
        self.assertTrue(hasattr(torch.Tensor, "is_foo"))
        self.assertTrue(hasattr(torch.Tensor, "foo"))
        self.assertTrue(hasattr(torch.TypedStorage, "is_foo"))
        self.assertTrue(hasattr(torch.TypedStorage, "foo"))
        self.assertTrue(hasattr(torch.UntypedStorage, "is_foo"))
        self.assertTrue(hasattr(torch.UntypedStorage, "foo"))
        self.assertTrue(hasattr(torch.nn.Module, "foo"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "is_foo"))
        self.assertTrue(hasattr(torch.nn.utils.rnn.PackedSequence, "foo"))

    def test_open_device_generator_registration_and_hooks(self):
        device = self.module.custom_device()
        # None of our CPU operations should call the custom add function.
        self.assertFalse(self.module.custom_add_called())

        # check generator registered before using
        with self.assertRaisesRegex(
            RuntimeError,
            "Please register a generator to the PrivateUse1 dispatch key",
        ):
            torch.Generator(device=device)

        self.module.register_generator_first()
        gen = torch.Generator(device=device)
        self.assertTrue(gen.device == device)

        # generator can be registered only once
        with self.assertRaisesRegex(
            RuntimeError,
            "Only can register a generator to the PrivateUse1 dispatch key once",
        ):
            self.module.register_generator_second()

        if self.module.is_register_hook() is False:
            self.module.register_hook()
        default_gen = self.module.default_generator(0)
        self.assertTrue(
            default_gen.device.type == torch._C._get_privateuse1_backend_name()
        )

    def test_open_device_dispatchstub(self):
        # test kernels could be reused by privateuse1 backend through dispatchstub
        input_data = torch.randn(2, 2, 3, dtype=torch.float32, device="cpu")
        foo_input_data = input_data.to("foo")
        output_data = torch.abs(input_data)
        foo_output_data = torch.abs(foo_input_data)
        self.assertEqual(output_data, foo_output_data.cpu())

        output_data = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        # output operand will resize flag is True in TensorIterator.
        foo_input_data = input_data.to("foo")
        foo_output_data = output_data.to("foo")
        # output operand will resize flag is False in TensorIterator.
        torch.abs(input_data, out=output_data[:, :, 0:6:2])
        torch.abs(foo_input_data, out=foo_output_data[:, :, 0:6:2])
        self.assertEqual(output_data, foo_output_data.cpu())

        # output operand will resize flag is True in TensorIterator.
        # and convert output to contiguous tensor in TensorIterator.
        output_data = torch.randn(2, 2, 6, dtype=torch.float32, device="cpu")
        foo_input_data = input_data.to("foo")
        foo_output_data = output_data.to("foo")
        torch.abs(input_data, out=output_data[:, :, 0:6:3])
        torch.abs(foo_input_data, out=foo_output_data[:, :, 0:6:3])
        self.assertEqual(output_data, foo_output_data.cpu())

    def test_open_device_quantized(self):
        input_data = torch.randn(3, 4, 5, dtype=torch.float32, device="cpu").to("foo")
        quantized_tensor = torch.quantize_per_tensor(input_data, 0.1, 10, torch.qint8)
        self.assertEqual(quantized_tensor.device, torch.device("foo:0"))
        self.assertEqual(quantized_tensor.dtype, torch.qint8)

    def test_open_device_random(self):
        # check if torch.foo have implemented get_rng_state
        with torch.random.fork_rng(device_type="foo"):
            pass

    def test_open_device_tensor(self):
        device = self.module.custom_device()

        # check whether print tensor.type() meets the expectation
        dtypes = {
            torch.bool: "torch.foo.BoolTensor",
            torch.double: "torch.foo.DoubleTensor",
            torch.float32: "torch.foo.FloatTensor",
            torch.half: "torch.foo.HalfTensor",
            torch.int32: "torch.foo.IntTensor",
            torch.int64: "torch.foo.LongTensor",
            torch.int8: "torch.foo.CharTensor",
            torch.short: "torch.foo.ShortTensor",
            torch.uint8: "torch.foo.ByteTensor",
        }
        for tt, dt in dtypes.items():
            test_tensor = torch.empty(4, 4, dtype=tt, device=device)
            self.assertTrue(test_tensor.type() == dt)

        # check whether the attributes and methods of the corresponding custom backend are generated correctly
        x = torch.empty(4, 4)
        self.assertFalse(x.is_foo)

        x = x.foo(torch.device("foo"))
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(x.is_foo)

        # test different device type input
        y = torch.empty(4, 4)
        self.assertFalse(y.is_foo)

        y = y.foo(torch.device("foo:0"))
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(y.is_foo)

        # test different device type input
        z = torch.empty(4, 4)
        self.assertFalse(z.is_foo)

        z = z.foo(0)
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z.is_foo)

    def test_open_device_packed_sequence(self):
        device = self.module.custom_device()
        a = torch.rand(5, 3)
        b = torch.tensor([1, 1, 1, 1, 1])
        input = torch.nn.utils.rnn.PackedSequence(a, b)
        self.assertFalse(input.is_foo)
        input_foo = input.foo()
        self.assertTrue(input_foo.is_foo)

    def test_open_device_storage(self):
        # check whether the attributes and methods for storage of the corresponding custom backend are generated correctly
        x = torch.empty(4, 4)
        z1 = x.storage()
        self.assertFalse(z1.is_foo)

        z1 = z1.foo()
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z1.is_foo)

        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            z1.foo(torch.device("cpu"))

        z1 = z1.cpu()
        self.assertFalse(self.module.custom_add_called())
        self.assertFalse(z1.is_foo)

        z1 = z1.foo(device="foo:0", non_blocking=False)
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z1.is_foo)

        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            z1.foo(device="cuda:0", non_blocking=False)

        # check UntypedStorage
        y = torch.empty(4, 4)
        z2 = y.untyped_storage()
        self.assertFalse(z2.is_foo)

        z2 = z2.foo()
        self.assertFalse(self.module.custom_add_called())
        self.assertTrue(z2.is_foo)

        # check custom StorageImpl create
        self.module.custom_storage_registry()

        z3 = y.untyped_storage()
        self.assertFalse(self.module.custom_storageImpl_called())

        z3 = z3.foo()
        self.assertTrue(self.module.custom_storageImpl_called())
        self.assertFalse(self.module.custom_storageImpl_called())

        z3 = z3[0:3]
        self.assertTrue(self.module.custom_storageImpl_called())

    @skipIfTorchDynamo("unsupported aten.is_pinned.default")
    def test_open_device_storage_pin_memory(self):
        # Check if the pin_memory is functioning properly on custom device
        cpu_tensor = torch.empty(3)
        self.assertFalse(cpu_tensor.is_foo)
        self.assertFalse(cpu_tensor.is_pinned("foo"))

        cpu_tensor_pin = cpu_tensor.pin_memory("foo")
        self.assertTrue(cpu_tensor_pin.is_pinned("foo"))

        # Test storage pin_memory and is_pin
        cpu_storage = cpu_tensor.storage()
        # We implement a dummy pin_memory of no practical significance
        # for custom device. Once tensor.pin_memory() has been called,
        # then tensor.is_pinned() will always return true no matter
        # what tensor it's called on.
        self.assertTrue(cpu_storage.is_pinned("foo"))

        cpu_storage_pinned = cpu_storage.pin_memory("foo")
        self.assertTrue(cpu_storage_pinned.is_pinned("foo"))

        # Test untyped storage pin_memory and is_pin
        cpu_tensor = torch.randn([3, 2, 1, 4])
        cpu_untyped_storage = cpu_tensor.untyped_storage()
        self.assertTrue(cpu_untyped_storage.is_pinned("foo"))

        cpu_untyped_storage_pinned = cpu_untyped_storage.pin_memory("foo")
        self.assertTrue(cpu_untyped_storage_pinned.is_pinned("foo"))

    @unittest.skip(
        "Temporarily disable due to the tiny differences between clang++ and g++ in defining static variable in inline function"
    )
    def test_open_device_serialization(self):
        self.module.set_custom_device_index(-1)
        storage = torch.UntypedStorage(4, device=torch.device("foo"))
        self.assertEqual(torch.serialization.location_tag(storage), "foo")

        self.module.set_custom_device_index(0)
        storage = torch.UntypedStorage(4, device=torch.device("foo"))
        self.assertEqual(torch.serialization.location_tag(storage), "foo:0")

        cpu_storage = torch.empty(4, 4).storage()
        foo_storage = torch.serialization.default_restore_location(cpu_storage, "foo:0")
        self.assertTrue(foo_storage.is_foo)

        # test tensor MetaData serialization
        x = torch.empty(4, 4).long()
        y = x.foo()
        self.assertFalse(self.module.check_backend_meta(y))
        self.module.custom_set_backend_meta(y)
        self.assertTrue(self.module.check_backend_meta(y))

        self.module.custom_serialization_registry()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.pt")
            torch.save(y, path)
            z1 = torch.load(path)
            # loads correctly onto the foo backend device
            self.assertTrue(z1.is_foo)
            # loads BackendMeta data correctly
            self.assertTrue(self.module.check_backend_meta(z1))

            # cross-backend
            z2 = torch.load(path, map_location="cpu")
            # loads correctly onto the cpu backend device
            self.assertFalse(z2.is_foo)
            # loads BackendMeta data correctly
            self.assertFalse(self.module.check_backend_meta(z2))

    def test_open_device_storage_resize(self):
        cpu_tensor = torch.randn([8])
        foo_tensor = cpu_tensor.foo()
        foo_storage = foo_tensor.storage()
        self.assertTrue(foo_storage.size() == 8)

        # Only register tensor resize_ function.
        foo_tensor.resize_(8)
        self.assertTrue(foo_storage.size() == 8)

        with self.assertRaisesRegex(TypeError, "Overflow"):
            foo_tensor.resize_(8**29)

    def test_open_device_storage_type(self):
        # test cpu float storage
        cpu_tensor = torch.randn([8]).float()
        cpu_storage = cpu_tensor.storage()
        self.assertEqual(cpu_storage.type(), "torch.FloatStorage")

        # test custom float storage before defining FloatStorage
        foo_tensor = cpu_tensor.foo()
        foo_storage = foo_tensor.storage()
        self.assertEqual(foo_storage.type(), "torch.storage.TypedStorage")

        class CustomFloatStorage:
            @property
            def __module__(self):
                return "torch." + torch._C._get_privateuse1_backend_name()

            @property
            def __name__(self):
                return "FloatStorage"

        # test custom float storage after defining FloatStorage
        try:
            torch.foo.FloatStorage = CustomFloatStorage()
            self.assertEqual(foo_storage.type(), "torch.foo.FloatStorage")

            # test custom int storage after defining FloatStorage
            foo_tensor2 = torch.randn([8]).int().foo()
            foo_storage2 = foo_tensor2.storage()
            self.assertEqual(foo_storage2.type(), "torch.storage.TypedStorage")
        finally:
            torch.foo.FloatStorage = None

    def test_open_device_faketensor(self):
        with torch._subclasses.fake_tensor.FakeTensorMode.push():
            a = torch.empty(1, device="foo")
            b = torch.empty(1, device="foo:0")
            result = a + b

    def test_open_device_named_tensor(self):
        torch.empty([2, 3, 4, 5], device="foo", names=["N", "C", "H", "W"])

    # Not an open registration test - this file is just very convenient
    # for testing torch.compile on custom C++ operators
    def test_compile_autograd_function_returns_self(self):
        x_ref = torch.randn(4, requires_grad=True)
        out_ref = self.module.custom_autograd_fn_returns_self(x_ref)
        out_ref.sum().backward()

        x_test = x_ref.clone().detach().requires_grad_(True)
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

        x_test = x_ref.clone().detach().requires_grad_(True)
        f_compiled = torch.compile(torch.ops._test_funcs.custom_autograd_fn_aliasing)
        out_test = f_compiled(x_test)
        out_test.sum().backward()

        self.assertEqual(out_ref, out_test)
        self.assertEqual(x_ref.grad, x_test.grad)

    def test_open_device_scalar_type_fallback(self):
        z_cpu = torch.Tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]).to(torch.int64)
        z = torch.triu_indices(3, 3, device="foo")
        self.assertEqual(z_cpu, z)

    def test_open_device_tensor_type_fallback(self):
        # create tensors located in custom device
        x = torch.Tensor([[1, 2, 3], [2, 3, 4]]).to("foo")
        y = torch.Tensor([1, 0, 2]).to("foo")
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
        y = torch.Tensor([1, 0]).long().to("foo")
        z = x[y, y]
        self.assertEqual(z_cpu, z)

    def test_open_device_tensorlist_type_fallback(self):
        # create tensors located in custom device
        v_foo = torch.Tensor([1, 2, 3]).to("foo")
        # create result tensor located in cpu
        z_cpu = torch.Tensor([2, 4, 6])
        # create tensorlist for foreach_add op
        x = (v_foo, v_foo)
        y = (v_foo, v_foo)
        # Check that our device is correct.
        device = self.module.custom_device()
        self.assertTrue(v_foo.device == device)
        self.assertFalse(v_foo.is_cpu)

        # call _foreach_add op, which will fallback to cpu
        z = torch._foreach_add(x, y)
        self.assertEqual(z_cpu, z[0])
        self.assertEqual(z_cpu, z[1])

        # call _fused_adamw_ with undefined tensor.
        self.module.fallback_with_undefined_tensor()

    @unittest.skipIf(
        np.__version__ < "1.25",
        "versions < 1.25 serialize dtypes differently from how it's serialized in data_legacy_numpy",
    )
    def test_open_device_numpy_serialization(self):
        """
        This tests the legacy _rebuild_device_tensor_from_numpy serialization path
        """
        torch.utils.rename_privateuse1_backend("foo")
        device = self.module.custom_device()
        default_protocol = torch.serialization.DEFAULT_PROTOCOL

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
            b"\x80@q\x14h\x08\x86q\x15Rq\x16tq\x17bctorch\nfloat32\nq\x18X\x05\x00\x00\x00foo:0q\x19\x89"
            b"tq\x1aRq\x1bs.PK\x07\x08\xe3\xe4\x86\xecO\x01\x00\x00O\x01\x00\x00PK\x03\x04\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x002\x00"
            b"archive/byteorderFB.\x00ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZlittlePK\x07\x08"
            b"\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00=\x00archive/versionFB9\x00"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ3\nPK\x07\x08\xd1\x9egU\x02\x00\x00"
            b"\x00\x02\x00\x00\x00PK\x03\x04\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x1e\x002\x00archive/.data/serialization_idFB.\x00ZZZZZZZZZZZZZ"
            b"ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ0636457737946401051300000027264370494161PK\x07\x08\x91\xbf"
            b"\xa7\x0c(\x00\x00\x00(\x00\x00\x00PK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00\x00\x00"
            b"\xe3\xe4\x86\xecO\x01\x00\x00O\x01\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00archive/data.pklPK\x01\x02\x00\x00\x00\x00\x08\x08\x00\x00\x00\x00"
            b"\x00\x00\x85=\xe3\x19\x06\x00\x00\x00\x06\x00\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x9f\x01\x00\x00archive/byteorderPK\x01\x02\x00\x00\x00\x00\x08\x08\x00"
            b"\x00\x00\x00\x00\x00\xd1\x9egU\x02\x00\x00\x00\x02\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x16\x02\x00\x00archive/versionPK\x01\x02\x00\x00\x00\x00\x08"
            b"\x08\x00\x00\x00\x00\x00\x00\x91\xbf\xa7\x0c(\x00\x00\x00(\x00\x00\x00\x1e\x00\x00\x00\x00"
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x92\x02\x00\x00archive/.data/serialization_idPK\x06"
            b"\x06,\x00\x00\x00\x00\x00\x00\x00\x1e\x03-\x00\x00\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00"
            b"\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x06\x01\x00\x00\x00\x00\x00\x008\x03\x00"
            b"\x00\x00\x00\x00\x00PK\x06\x07\x00\x00\x00\x00>\x04\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00"
            b"PK\x05\x06\x00\x00\x00\x00\x04\x00\x04\x00\x06\x01\x00\x008\x03\x00\x00\x00\x00"
        )
        buf_data_legacy_numpy = io.BytesIO(data_legacy_numpy)

        with safe_globals(
            [
                np.core.multiarray._reconstruct,
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
        torch.utils.rename_privateuse1_backend("foo")
        device = self.module.custom_device()
        default_protocol = torch.serialization.DEFAULT_PROTOCOL

        with patch.object(torch._C, "_has_storage", return_value=False):
            x = torch.randn(2, 3)
            x_foo = x.to(device)
            sd = {"x": x_foo}
            rebuild_func = x_foo._reduce_ex_internal(default_protocol)[0]
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


if __name__ == "__main__":
    common.run_tests()
