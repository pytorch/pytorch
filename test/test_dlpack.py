# Owner(s): ["module: tests"]

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfMPS,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    skipCUDAIfNotRocm,
    skipMeta,
)
from torch.testing._internal.common_dtype import (
    all_mps_types_and,
    all_types_and_complex_and,
)
from torch.testing._internal.common_utils import (
    IS_JETSON,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.utils.dlpack import DLDeviceType, from_dlpack, to_dlpack


# Wraps a tensor, exposing only DLPack methods:
#    - __dlpack__
#    - __dlpack_device__
#
# This is used for guaranteeing we are going through the DLPack method, and not
# something else, e.g.: CUDA array interface, buffer protocol, etc.
class TensorDLPackWrapper:
    def __init__(self, tensor):
        self.tensor = tensor

    def __dlpack__(self, *args, **kwargs):
        return self.tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self, *args, **kwargs):
        return self.tensor.__dlpack_device__(*args, **kwargs)


class TestTorchDlPack(TestCase):
    exact_dtype = True

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    @dtypesIfMPS(*all_mps_types_and(torch.bool, torch.cfloat, torch.chalf))
    def test_dlpack_capsule_conversion(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(to_dlpack(x))
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    @dtypesIfMPS(*all_mps_types_and(torch.bool, torch.cfloat, torch.chalf))
    def test_dlpack_protocol_conversion(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(x)
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    def test_dlpack_shared_storage(self, device):
        dtype = torch.bfloat16 if device.startswith("mps") else torch.float64
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(to_dlpack(x))
        z[0] = z[0] + 20.0
        self.assertEqual(z, x)

    def _dlpack_conversion_with_streams(self, stream, x):
        # DLPack protocol helps establish a correct stream order
        # (hence data dependency) at the exchange boundary.
        # DLPack manages this synchronization for us, so we don't need to
        # explicitly wait until x is populated
        if IS_JETSON:
            # DLPack protocol that establishes correct stream order
            # does not behave as expected on Jetson
            stream.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            z = from_dlpack(x)
        stream.synchronize()
        return z

    @skipMeta
    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_dlpack_conversion_with_streams(self, device, dtype):
        # Create a stream where the tensor will reside
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Do an operation in the actual stream
            x = make_tensor((5,), dtype=dtype, device=device) + 1
        z = self._dlpack_conversion_with_streams(stream, x)
        self.assertEqual(z, x)

    @skipMeta
    @onlyCUDA
    @dtypes(
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e8m0fnu,
        torch.float4_e2m1fn_x2,
    )
    def test_dlpack_conversion_with_streams_narrow_precision(self, device, dtype):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = make_tensor((5,), dtype=torch.uint8, device=device) + 1
            x = x.view(dtype)
        z = self._dlpack_conversion_with_streams(stream, x)
        self.assertEqual(z.view(torch.uint8), x.view(torch.uint8))

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    @dtypesIfMPS(*all_mps_types_and(torch.bool, torch.cfloat, torch.chalf))
    def test_from_dlpack(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        self.assertEqual(x, y)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    @dtypesIfMPS(
        *all_mps_types_and(
            torch.bool, torch.cfloat, torch.chalf, torch.uint16, torch.uint32
        )
    )
    def test_from_dlpack_noncontinguous(self, device, dtype):
        x = make_tensor((25,), dtype=dtype, device=device).reshape(5, 5)

        y1 = x[0]
        y1_dl = torch.from_dlpack(y1)
        self.assertEqual(y1, y1_dl)

        y2 = x[:, 0]
        y2_dl = torch.from_dlpack(y2)
        self.assertEqual(y2, y2_dl)

        y3 = x[1, :]
        y3_dl = torch.from_dlpack(y3)
        self.assertEqual(y3, y3_dl)

        y4 = x[1]
        y4_dl = torch.from_dlpack(y4)
        self.assertEqual(y4, y4_dl)

        y5 = x.t()
        y5_dl = torch.from_dlpack(y5)
        self.assertEqual(y5, y5_dl)

    @skipMeta
    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_dlpack_conversion_with_diff_streams(self, device, dtype):
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        # DLPack protocol helps establish a correct stream order
        # (hence data dependency) at the exchange boundary.
        # the `tensor.__dlpack__` method will insert a synchronization event
        # in the current stream to make sure that it was correctly populated.
        with torch.cuda.stream(stream_a):
            x = make_tensor((5,), dtype=dtype, device=device) + 1
            z = torch.from_dlpack(x.__dlpack__(stream=stream_b.cuda_stream))
            stream_a.synchronize()
        stream_b.synchronize()
        self.assertEqual(z, x)

    @skipMeta
    @onlyCUDA
    @dtypes(
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e8m0fnu,
        torch.float4_e2m1fn_x2,
    )
    def test_dlpack_conversion_with_diff_streams_narrow_precision(self, device, dtype):
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        with torch.cuda.stream(stream_a):
            x = make_tensor((5,), dtype=torch.uint8, device=device) + 1
            x = x.view(dtype)
            z = torch.from_dlpack(x.__dlpack__(stream=stream_b.cuda_stream))
            stream_a.synchronize()
        stream_b.synchronize()
        self.assertEqual(z.view(torch.uint8), x.view(torch.uint8))

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(
        *all_types_and_complex_and(
            torch.half,
            torch.bfloat16,
            torch.bool,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    @dtypesIfMPS(*all_mps_types_and(torch.bool, torch.cfloat, torch.chalf))
    def test_from_dlpack_dtype(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        assert x.dtype == y.dtype

    @skipMeta
    @onlyCUDA
    def test_dlpack_default_stream(self, device):
        class DLPackTensor:
            def __init__(self, tensor):
                self.tensor = tensor

            def __dlpack_device__(self):
                return self.tensor.__dlpack_device__()

            def __dlpack__(self, stream=None):
                if torch.version.hip is None:
                    assert stream == 1
                else:
                    assert stream == 0
                capsule = self.tensor.__dlpack__(stream=stream)
                return capsule

        # CUDA-based tests runs on non-default streams
        with torch.cuda.stream(torch.cuda.default_stream()):
            x = DLPackTensor(make_tensor((5,), dtype=torch.float32, device=device))
            from_dlpack(x)

    @skipMeta
    @onlyCUDA
    def test_dlpack_convert_default_stream(self, device):
        # tests run on non-default stream, so _sleep call
        # below will run on a non-default stream, causing
        # default stream to wait due to inserted syncs
        torch.cuda.default_stream().synchronize()
        # run _sleep call on a non-default stream, causing
        # default stream to wait due to inserted syncs
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            x = torch.zeros(1, device=device)
            torch.cuda._sleep(2**20)
            self.assertTrue(torch.cuda.default_stream().query())
            # ROCm uses stream 0 for default stream, CUDA uses stream 1
            default_stream_id = 0 if torch.version.hip else 1
            x.__dlpack__(stream=default_stream_id)
        # check that the default stream has work (a pending cudaStreamWaitEvent)
        self.assertFalse(torch.cuda.default_stream().query())

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_tensor_invalid_stream(self, device, dtype):
        with self.assertRaises(TypeError):
            x = make_tensor((5,), dtype=dtype, device=device)
            x.__dlpack__(stream=object())

    @skipMeta
    @onlyCUDA
    def test_dlpack_cuda_per_thread_stream(self, device):
        # Test whether we raise an error if we are trying to use per-thread default
        # stream, which is currently not supported by PyTorch.
        x = make_tensor((5,), dtype=torch.float32, device=device)

        if TEST_WITH_ROCM:
            context = self.assertRaisesRegex(
                AssertionError, r"unsupported stream on ROCm: 2"
            )
        else:
            context = self.assertRaisesRegex(
                BufferError, "per-thread default stream is not supported"
            )

        with context:
            x.__dlpack__(stream=2)

    @skipMeta
    @onlyCUDA
    @skipCUDAIfNotRocm
    def test_dlpack_invalid_rocm_streams(self, device):
        # Test that we correctly raise errors on unsupported ROCm streams.
        def test(x, stream):
            with self.assertRaisesRegex(
                AssertionError, r"unsupported stream on ROCm: \d"
            ):
                x.__dlpack__(stream=stream)

        x = make_tensor((5,), dtype=torch.float32, device=device)
        test(x, stream=1)
        test(x, stream=2)

    @skipMeta
    @onlyCUDA
    def test_dlpack_invalid_cuda_streams(self, device):
        x = make_tensor((5,), dtype=torch.float32, device=device)

        if TEST_WITH_ROCM:
            # On ROCm, stream=0 is valid (default stream).
            self.assertIsNotNone(x.__dlpack__(stream=0))
        else:
            # CUDA raises AssertionError for stream=0
            with self.assertRaisesRegex(
                AssertionError, r"unsupported stream on CUDA: \d"
            ):
                x.__dlpack__(stream=0)

    @skipMeta
    def test_dlpack_invalid_cpu_stream(self):
        x = make_tensor((5,), dtype=torch.float32, device="cpu")
        with self.assertRaisesRegex(AssertionError, r"stream should be None on cpu."):
            x.__dlpack__(stream=0)

    @skipMeta
    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_dlpack_tensor_on_different_device(self, devices):
        dev0, dev1 = devices[:2]

        with torch.device(dev0):
            x = make_tensor((5,), dtype=torch.float32, device=dev0)

        with self.assertRaisesRegex(
            BufferError, r"Can't export tensors on a different CUDA device"
        ):
            with torch.cuda.device(dev1):
                x.__dlpack__()

    # TODO: add interchange tests once NumPy 1.22 (dlpack support) is required
    @skipMeta
    def test_dlpack_export_requires_grad(self):
        x = torch.zeros(10, dtype=torch.float32, requires_grad=True)
        with self.assertRaisesRegex(BufferError, r"require gradient"):
            x.__dlpack__()

    @skipMeta
    def test_dlpack_export_is_conj(self):
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        y = torch.conj(x)
        with self.assertRaisesRegex(BufferError, r"conjugate bit"):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_export_non_strided(self):
        x = torch.sparse_coo_tensor([[0]], [1], size=(1,))
        y = torch.conj(x)
        with self.assertRaisesRegex(BufferError, r"strided"):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_normalize_strides(self):
        x = torch.rand(16)
        y = x[::3][:1]
        self.assertEqual(y.shape, (1,))
        self.assertEqual(y.stride(), (3,))
        z = from_dlpack(y)
        self.assertEqual(z.shape, (1,))
        # Stride normalization has been removed, strides should be preserved
        self.assertEqual(z.stride(), (3,))

    @skipMeta
    @onlyNativeDeviceTypes
    def test_automatically_select_in_creation(self, device):
        # Create a new tensor, and wrap it using TensorDLPackWrapper.
        tensor = torch.rand(10)
        wrap = TensorDLPackWrapper(tensor)
        # Create a new tensor from the wrapper.
        # This should identify that the wrapper class provides the DLPack methods
        # and use them for creating the new tensor, instead of iterating element
        # by element.
        new_tensor = torch.tensor(wrap)
        self.assertEqual(tensor, new_tensor)

    @skipMeta
    @skipIfTorchDynamo("__dlpack__ doesn't work with dynamo")
    @onlyNativeDeviceTypes
    def test_max_version(self, device):
        def capsule_name(kwargs):
            is_versioned = "max_version" in kwargs and kwargs["max_version"][0] >= 1
            return "dltensor_versioned" if is_versioned else "dltensor"

        def test(device, **kwargs):
            inp = make_tensor((5,), dtype=torch.float32, device=device)

            # Make sure we are actually using the (un)versioned DLPack tensor, based on the
            # informed keyword arguments.
            capsule = inp.__dlpack__(**kwargs)
            self.assertRegex(
                str(capsule), f"""capsule object "{capsule_name(kwargs)}" at"""
            )

            out = torch.from_dlpack(capsule)
            self.assertEqual(inp, out)

        # Use the DLPack 0.X version implementation, since max_version=None.
        test(device)
        # Use the DLPack 0.X version implementation.
        test(device, max_version=(0, 8))
        # Current highest DLPack version implemented.
        test(device, max_version=(1, 0))
        # Newer DLPack version.
        # Consumer should still be able to process a smaller version capsule.
        test(device, max_version=(2, 0))

    @skipMeta
    @onlyCPU
    @dtypes(
        # Note: NumPy DLPack bool support only landed in 1.25.
        *all_types_and_complex_and(
            torch.half,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    )
    def test_numpy_dlpack_protocol_conversion(self, device, dtype):
        import numpy as np

        t = make_tensor((5,), dtype=dtype, device=device)

        if hasattr(np, "from_dlpack"):
            # DLPack support only available from NumPy 1.22 onwards.
            # Here, we test having another framework (NumPy) calling our
            # Tensor.__dlpack__ implementation.
            arr = np.from_dlpack(t)
            self.assertEqual(t, arr)

        # We can't use the array created above as input to from_dlpack.
        # That's because DLPack imported NumPy arrays are read-only.
        # Thus, we need to convert it to NumPy by using the numpy() method.
        t_arr = t.numpy()

        # Transform the NumPy array back using DLPack.
        res = from_dlpack(t_arr)

        self.assertEqual(t, res)
        self.assertEqual(t.data_ptr(), res.data_ptr())

    def _test_from_dlpack(self, device, out_device=None, copy=None):
        if isinstance(device, str):
            device = torch.device(device)

        inp = make_tensor((5,), dtype=torch.float32, device=device)
        out = torch.from_dlpack(inp, device=out_device, copy=copy)

        if out_device is None:
            out_device = device
        if isinstance(out_device, str):
            out_device = torch.device(out_device)

        self.assertEqual(inp, out)
        self.assertEqual(out.device, out_device)

        # They should be moved (i.e. not copied) only if:
        #   (a) we are forcing move, i.e. copy=False
        #   (b) the output device is the same as the input one AND copy is None
        if copy is False or (copy is None and device == out_device):
            self.assertEqual(inp.data_ptr(), out.data_ptr())
        else:
            # Otherwise, inp should be copied.
            self.assertNotEqual(inp.data_ptr(), out.data_ptr())

    @skipMeta
    @onlyCUDA
    def test_copy(self, device):
        # Force-copy same device tensor.
        self._test_from_dlpack(device, copy=True)
        self._test_from_dlpack(device, out_device=device, copy=True)
        # Output should be in a different device, i.e. should have been copied.
        self._test_from_dlpack(device, out_device="cpu")
        self._test_from_dlpack(device, out_device="cpu", copy=True)

    @skipMeta
    @onlyCUDA
    def test_no_copy(self, device):
        # No copy, since tensor lives in the same device.
        self._test_from_dlpack(device)
        self._test_from_dlpack(device, copy=False)
        self._test_from_dlpack(device, out_device=device)
        self._test_from_dlpack(device, out_device=device, copy=False)

    @skipMeta
    @onlyCUDA
    def test_needs_copy_error(self, device):
        with self.assertRaisesRegex(ValueError, r"cannot move .* tensor from .*"):
            self._test_from_dlpack(device, out_device="cpu", copy=False)

    def test_dlpack_copy_fallback(self):
        """Test that copy parameter works even with producers that don't support it"""
        import numpy as np

        # Test copy=True - should work even if NumPy doesn't support copy parameter
        np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t = from_dlpack(np_array, copy=True)

        # Verify it's a copy by modifying tensor and checking NumPy unchanged
        t[0] = 999.0
        self.assertEqual(np_array[0], 1.0)

        # Test copy=None (default) - should be zero-copy view
        np_array2 = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        t2 = from_dlpack(np_array2)
        t2[0] = 999.0
        self.assertEqual(np_array2[0], 999.0)

    @skipMeta
    @onlyNativeDeviceTypes
    def test_unsupported_device_error(self, device):
        inp = make_tensor((5,), dtype=torch.float32, device=device)
        dl_device_type = DLDeviceType.kDLHexagon

        with self.assertRaisesRegex(
            BufferError, f"Unsupported device_type: {int(dl_device_type)}"
        ):
            inp.__dlpack__(max_version=(1, 0), dl_device=(dl_device_type, 0))

    @skipMeta
    @onlyCPU
    def test_dlpack_unsupported_dtype_error(self, device):
        inp = torch.quantize_per_tensor(torch.randn(()), 0.1, 10, torch.qint8)

        with self.assertRaisesRegex(
            BufferError, ".* types are not supported by dlpack"
        ):
            from_dlpack(inp)

    @skipMeta
    @onlyNativeDeviceTypes
    def test_dlpack_exchange_api(self, device):
        """Comprehensive test of all DLPack Exchange API functions using inline C++"""
        # Check that the C API capsule exists and get it
        self.assertTrue(hasattr(torch.Tensor, "__dlpack_c_exchange_api__"))
        api_capsule = torch.Tensor.__dlpack_c_exchange_api__
        self.assertEqual(
            type(api_capsule).__name__, "PyCapsule", "API should be a PyCapsule"
        )
        self.assertRegex(str(api_capsule), r'capsule object "dlpack_exchange_api"')
        tensor = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)

        source = """
        #include <torch/extension.h>
        #include <ATen/dlpack.h>
        #include <pybind11/pybind11.h>
        #include <memory>

        namespace py = pybind11;

        void test_dlpack_exchange_api(at::Tensor tensor, py::object api_obj, bool test_stream_exchange) {
            PyObject* api_capsule = api_obj.ptr();
            TORCH_CHECK(PyCapsule_IsValid(api_capsule, "dlpack_exchange_api"),
                        "Invalid or mismatched DLPack exchange API capsule");
            const DLPackExchangeAPI* api =
                static_cast<const DLPackExchangeAPI*>(
                    PyCapsule_GetPointer(api_capsule, "dlpack_exchange_api"));

            // Test 1: API structure and version
            {
                TORCH_CHECK(api != nullptr, "API pointer is NULL");
                TORCH_CHECK(api->header.version.major == DLPACK_MAJOR_VERSION,
                            "Expected major version ", DLPACK_MAJOR_VERSION,
                            ", got ", api->header.version.major);
                TORCH_CHECK(api->header.version.minor == DLPACK_MINOR_VERSION,
                            "Expected minor version ", DLPACK_MINOR_VERSION,
                            ", got ", api->header.version.minor);
                TORCH_CHECK(api->managed_tensor_allocator != nullptr,
                            "managed_tensor_allocator is NULL");
                TORCH_CHECK(api->managed_tensor_from_py_object_no_sync != nullptr,
                            "managed_tensor_from_py_object_no_sync is NULL");
                TORCH_CHECK(api->managed_tensor_to_py_object_no_sync != nullptr,
                            "managed_tensor_to_py_object_no_sync is NULL");
                TORCH_CHECK(api->dltensor_from_py_object_no_sync != nullptr,
                            "dltensor_from_py_object_no_sync is NULL");
                TORCH_CHECK(api->current_work_stream != nullptr,
                            "current_work_stream is NULL");
            }

            // Test 2: managed_tensor_allocator
            {
                DLTensor prototype;
                prototype.device.device_type = kDLCPU;
                prototype.device.device_id = 0;
                prototype.ndim = 3;
                int64_t shape[3] = {3, 4, 5};
                prototype.shape = shape;
                prototype.strides = nullptr;
                DLDataType dtype;
                dtype.code = kDLFloat;
                dtype.bits = 32;
                dtype.lanes = 1;
                prototype.dtype = dtype;
                prototype.data = nullptr;
                prototype.byte_offset = 0;

                DLManagedTensorVersioned* out_tensor = nullptr;
                int result = api->managed_tensor_allocator(
                    &prototype, &out_tensor, nullptr, nullptr);
                TORCH_CHECK(result == 0, "Allocator failed with code ", result);
                TORCH_CHECK(out_tensor != nullptr, "Allocator returned NULL");
                TORCH_CHECK(out_tensor->dl_tensor.ndim == 3,
                            "Expected ndim 3, got ", out_tensor->dl_tensor.ndim);
                TORCH_CHECK(out_tensor->dl_tensor.shape[0] == 3,
                            "Expected shape[0] = 3, got ", out_tensor->dl_tensor.shape[0]);
                TORCH_CHECK(out_tensor->dl_tensor.shape[1] == 4,
                            "Expected shape[1] = 4, got ", out_tensor->dl_tensor.shape[1]);
                TORCH_CHECK(out_tensor->dl_tensor.shape[2] == 5,
                            "Expected shape[2] = 5, got ", out_tensor->dl_tensor.shape[2]);
                TORCH_CHECK(out_tensor->dl_tensor.dtype.code == kDLFloat,
                            "Expected dtype code kDLFloat, got ",
                            out_tensor->dl_tensor.dtype.code);
                TORCH_CHECK(out_tensor->dl_tensor.dtype.bits == 32,
                            "Expected dtype bits 32, got ", out_tensor->dl_tensor.dtype.bits);
                TORCH_CHECK(out_tensor->dl_tensor.device.device_type == kDLCPU,
                            "Expected device type kDLCPU, got ",
                            out_tensor->dl_tensor.device.device_type);
                if (out_tensor->deleter) {
                    out_tensor->deleter(out_tensor);
                }
            }

            // Test 3: managed_tensor_from_py_object_no_sync
            {
                std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj(
                    THPVariable_Wrap(tensor), &Py_DecRef);
                TORCH_CHECK(py_obj.get() != nullptr, "Failed to wrap tensor to PyObject");

                DLManagedTensorVersioned* out_tensor = nullptr;
                int result = api->managed_tensor_from_py_object_no_sync(
                    py_obj.get(), &out_tensor);

                TORCH_CHECK(result == 0,
                            "from_py_object_no_sync failed with code ", result);
                TORCH_CHECK(out_tensor != nullptr,
                            "from_py_object_no_sync returned NULL");
                TORCH_CHECK(out_tensor->version.major == DLPACK_MAJOR_VERSION,
                            "Expected major version ", DLPACK_MAJOR_VERSION,
                            ", got ", out_tensor->version.major);
                TORCH_CHECK(out_tensor->version.minor == DLPACK_MINOR_VERSION,
                            "Expected minor version ", DLPACK_MINOR_VERSION,
                            ", got ", out_tensor->version.minor);
                TORCH_CHECK(out_tensor->dl_tensor.ndim == 3,
                            "Expected ndim 3, got ", out_tensor->dl_tensor.ndim);
                TORCH_CHECK(out_tensor->dl_tensor.shape[0] == 2,
                            "Expected shape[0] = 2, got ", out_tensor->dl_tensor.shape[0]);
                TORCH_CHECK(out_tensor->dl_tensor.shape[1] == 3,
                            "Expected shape[1] = 3, got ", out_tensor->dl_tensor.shape[1]);
                TORCH_CHECK(out_tensor->dl_tensor.shape[2] == 4,
                            "Expected shape[2] = 4, got ", out_tensor->dl_tensor.shape[2]);
                TORCH_CHECK(out_tensor->dl_tensor.dtype.code == kDLFloat,
                            "Expected dtype code kDLFloat, got ",
                            out_tensor->dl_tensor.dtype.code);
                TORCH_CHECK(out_tensor->dl_tensor.dtype.bits == 32,
                            "Expected dtype bits 32, got ",
                            out_tensor->dl_tensor.dtype.bits);
                TORCH_CHECK(out_tensor->dl_tensor.data != nullptr,
                            "Data pointer is NULL");

                if (out_tensor->deleter) {
                    out_tensor->deleter(out_tensor);
                }
            }

            // Test 4: managed_tensor_to_py_object_no_sync
            {
                std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj(
                    THPVariable_Wrap(tensor), &Py_DecRef);
                TORCH_CHECK(py_obj.get() != nullptr, "Failed to wrap tensor to PyObject");

                DLManagedTensorVersioned* managed_tensor = nullptr;
                int result = api->managed_tensor_from_py_object_no_sync(
                    py_obj.get(), &managed_tensor);
                TORCH_CHECK(result == 0, "from_py_object_no_sync failed");
                TORCH_CHECK(managed_tensor != nullptr,
                            "from_py_object_no_sync returned NULL");

                std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj_out(
                    nullptr, &Py_DecRef);
                PyObject* py_obj_out_raw = nullptr;
                result = api->managed_tensor_to_py_object_no_sync(
                    managed_tensor, reinterpret_cast<void**>(&py_obj_out_raw));
                py_obj_out.reset(py_obj_out_raw);

                TORCH_CHECK(result == 0,
                            "to_py_object_no_sync failed with code ", result);
                TORCH_CHECK(py_obj_out.get() != nullptr,
                            "to_py_object_no_sync returned NULL");
                TORCH_CHECK(THPVariable_Check(py_obj_out.get()),
                            "Returned PyObject is not a Tensor");

                at::Tensor result_tensor = THPVariable_Unpack(py_obj_out.get());
                TORCH_CHECK(result_tensor.dim() == 3,
                            "Expected 3 dimensions, got ", result_tensor.dim());
                TORCH_CHECK(result_tensor.size(0) == 2,
                            "Expected size(0) = 2, got ", result_tensor.size(0));
                TORCH_CHECK(result_tensor.size(1) == 3,
                            "Expected size(1) = 3, got ", result_tensor.size(1));
                TORCH_CHECK(result_tensor.size(2) == 4,
                            "Expected size(2) = 4, got ", result_tensor.size(2));
                TORCH_CHECK(result_tensor.scalar_type() == at::kFloat,
                            "Expected dtype kFloat, got ", result_tensor.scalar_type());
            }

            // Test 5: dltensor_from_py_object_no_sync (non-owning conversion)
            DLDeviceType device_type;
            int32_t device_id;
            {
                std::unique_ptr<PyObject, decltype(&Py_DecRef)> py_obj(
                    THPVariable_Wrap(tensor), &Py_DecRef);
                TORCH_CHECK(py_obj.get() != nullptr, "Failed to wrap tensor to PyObject");

                DLTensor dltensor;
                int result = api->dltensor_from_py_object_no_sync(py_obj.get(), &dltensor);
                TORCH_CHECK(result == 0,
                            "dltensor_from_py_object_no_sync failed with code ", result);
                TORCH_CHECK(dltensor.ndim == 3, "Expected ndim 3, got ", dltensor.ndim);
                TORCH_CHECK(dltensor.shape[0] == 2,
                            "Expected shape[0] = 2, got ", dltensor.shape[0]);
                TORCH_CHECK(dltensor.shape[1] == 3,
                            "Expected shape[1] = 3, got ", dltensor.shape[1]);
                TORCH_CHECK(dltensor.shape[2] == 4,
                            "Expected shape[2] = 4, got ", dltensor.shape[2]);
                TORCH_CHECK(dltensor.dtype.code == kDLFloat,
                            "Expected dtype code kDLFloat, got ", dltensor.dtype.code);
                TORCH_CHECK(dltensor.dtype.bits == 32,
                            "Expected dtype bits 32, got ", dltensor.dtype.bits);
                TORCH_CHECK(dltensor.data != nullptr, "Data pointer is NULL");

                // Capture device info for stream test
                device_type = dltensor.device.device_type;
                device_id = dltensor.device.device_id;
            }

            // Test 6: current_work_stream
            {
                if (test_stream_exchange) {
                    void* stream_out = nullptr;
                    int result = api->current_work_stream(device_type, device_id, &stream_out);
                    TORCH_CHECK(result == 0,
                                "current_work_stream failed with code ", result);
                    TORCH_CHECK(stream_out != nullptr,
                                "Expected stream to be non-NULL");
                }
            }
        }
        """

        # Load and compile the inline C++ test
        from torch.utils import cpp_extension

        module = cpp_extension.load_inline(
            name="test_dlpack_exchange_api",
            cpp_sources=[source],
            functions=["test_dlpack_exchange_api"],
            verbose=False,
            with_cuda=device.startswith("cuda"),
        )

        # Run the comprehensive C++ test
        module.test_dlpack_exchange_api(tensor, api_capsule, device.startswith("cuda"))

    @skipMeta
    @onlyCUDA
    def test_numpy_cross_device_transfer(self, device):
        """Test cross-device transfer from NumPy (CPU) to PyTorch (CUDA).

        This tests the fix for issue #169186 where torch.from_dlpack(numpy_array, device="cuda")
        would fail with "unsupported device requested" because PyTorch incorrectly asked
        NumPy to create a CUDA DLPack capsule instead of handling the device transfer itself.

        According to the DLPack spec, the consumer (PyTorch) is responsible for constructing
        the final array on the target device, not the producer (NumPy).
        """
        import numpy as np

        np_array = np.arange(10, dtype=np.float32)
        expected = torch.arange(10, dtype=torch.float32, device=device)

        # Test 1: copy=None (default) - should allow copy for cross-device
        t1 = from_dlpack(np_array, device=device)
        self.assertEqual(t1.device.type, "cuda")
        self.assertEqual(t1, expected)

        # Test 2: copy=True - explicit copy
        t2 = from_dlpack(np_array, device=device, copy=True)
        self.assertEqual(t2.device.type, "cuda")
        self.assertEqual(t2, expected)

        # Test 3: copy=False - should raise ValueError (can't do cross-device without copy)
        with self.assertRaisesRegex(
            ValueError, r"cannot move .* tensor from .* to .* without copying"
        ):
            from_dlpack(np_array, device=device, copy=False)

        # Test 4: device as string vs torch.device object (both should work)
        t_str = from_dlpack(np_array, device="cuda")
        t_obj = from_dlpack(np_array, device=torch.device("cuda"))
        self.assertEqual(t_str.device.type, "cuda")
        self.assertEqual(t_obj.device.type, "cuda")
        self.assertEqual(t_str, t_obj)

        # Test 5: Regression - CPU -> CPU should still be zero-copy (share memory)
        np_array2 = np.arange(5, dtype=np.float32)
        t_cpu = from_dlpack(np_array2, device="cpu", copy=None)
        self.assertEqual(t_cpu.device.type, "cpu")
        # Should share memory
        self.assertEqual(t_cpu.data_ptr(), torch.from_numpy(np_array2).data_ptr())
        # Mutation should affect both
        t_cpu[0] = 999
        self.assertEqual(np_array2[0], 999)

    @skipMeta
    @onlyCUDA
    @deviceCountAtLeast(2)
    def test_numpy_cross_device_multi_gpu(self, devices):
        """Test cross-device transfer to specific CUDA devices (cuda:0, cuda:1, etc)."""
        import numpy as np

        dev0, dev1 = devices[:2]
        np_array = np.arange(5, dtype=np.float32)

        # Test transfer to cuda:0
        t0 = from_dlpack(np_array, device=dev0)
        self.assertEqual(t0.device, torch.device(dev0))
        expected = torch.arange(5, dtype=torch.float32, device=dev0)
        self.assertEqual(t0, expected)

        # Test transfer to cuda:1
        t1 = from_dlpack(np_array, device=dev1)
        self.assertEqual(t1.device, torch.device(dev1))
        expected = torch.arange(5, dtype=torch.float32, device=dev1)
        self.assertEqual(t1, expected)

        # Verify they're on different devices
        self.assertNotEqual(t0.device, t1.device)


instantiate_device_type_tests(TestTorchDlPack, globals(), allow_mps=True)

if __name__ == "__main__":
    run_tests()
