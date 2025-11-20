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
    skipCUDAIfRocm,
    skipMeta,
)
from torch.testing._internal.common_dtype import (
    all_mps_types_and,
    all_types_and_complex_and,
)
from torch.testing._internal.common_utils import (
    IS_JETSON,
    run_tests,
    skipIfMPS,
    skipIfTorchDynamo,
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
    @skipIfMPS  # MPS crashes with noncontiguous now
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
    @skipCUDAIfRocm
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
            x.__dlpack__(stream=1)
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
    @skipCUDAIfRocm
    def test_dlpack_cuda_per_thread_stream(self, device):
        # Test whether we raise an error if we are trying to use per-thread default
        # stream, which is currently not supported by PyTorch.
        x = make_tensor((5,), dtype=torch.float32, device=device)
        with self.assertRaisesRegex(
            BufferError, "per-thread default stream is not supported"
        ):
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
    @skipCUDAIfRocm
    def test_dlpack_invalid_cuda_streams(self, device):
        x = make_tensor((5,), dtype=torch.float32, device=device)
        with self.assertRaisesRegex(AssertionError, r"unsupported stream on CUDA: \d"):
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
            with torch.device(dev1):
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
    def test_dlpack_byte_offset_handling(self, device, dtype):
        """Test that from_dlpack correctly converts byte_offset to storage_offset.
        
        This test manually modifies a DLPack capsule to set byte_offset,
        then verifies that from_dlpack correctly converts it to storage_offset.
        This avoids modifying toDLPack for backward compatibility.
        """
        import ctypes
        
        # Create a base tensor (no need for view since toDLPack sets byte_offset=0 anyway)
        base = make_tensor((10,), dtype=dtype, device=device)
        element_size = base.element_size()
        base_data_ptr = base.storage().data_ptr()
        original_shape = base.shape[0]
        
        # Test with different byte offsets
        for offset_elements in [1, 2, 3, 5]:
            expected_byte_offset = offset_elements * element_size
            expected_storage_offset = offset_elements
            
            # Create the expected view for comparison
            view = base[offset_elements:]
            expected_shape = view.shape
            expected_shape_size = expected_shape[0]
            
            # Get a fresh DLPack capsule from base for each iteration
            capsule = to_dlpack(base)
            
            # Manually modify byte_offset in the capsule
            class DLDevice(ctypes.Structure):
                _fields_ = [
                    ("device_type", ctypes.c_int32),
                    ("device_id", ctypes.c_int32),
                ]
            
            class DLDataType(ctypes.Structure):
                _fields_ = [
                    ("code", ctypes.c_uint8),
                    ("bits", ctypes.c_uint8),
                    ("lanes", ctypes.c_uint16),
                ]
            
            class DLTensor(ctypes.Structure):
                _fields_ = [
                    ("data", ctypes.c_void_p),
                    ("device", DLDevice),
                    ("ndim", ctypes.c_int32),
                    ("dtype", DLDataType),
                    ("shape", ctypes.POINTER(ctypes.c_int64)),
                    ("strides", ctypes.POINTER(ctypes.c_int64)),
                    ("byte_offset", ctypes.c_uint64),
                ]
            
            class DLManagedTensor(ctypes.Structure):
                _fields_ = [
                    ("dl_tensor", DLTensor),
                    ("manager_ctx", ctypes.c_void_p),
                    ("deleter", ctypes.c_void_p),
                ]
            
            # Get pointer from capsule
            capsule_name = "dltensor"
            try:
                pythonapi = ctypes.pythonapi
                pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
                
                capsule_ptr = pythonapi.PyCapsule_GetPointer(
                    ctypes.py_object(capsule),
                    capsule_name.encode('utf-8')
                )
                
                if not capsule_ptr:
                    self.skipTest("Could not get pointer from DLPack capsule")
                
                # Cast to DLManagedTensor pointer
                dl_tensor_ptr = ctypes.cast(
                    capsule_ptr,
                    ctypes.POINTER(DLManagedTensor)
                )
                
                # Verify current shape before modification
                current_shape = dl_tensor_ptr.contents.dl_tensor.shape[0]
                self.assertEqual(current_shape, original_shape, 
                               f"Initial shape should be {original_shape}")
                
                # Modify the DLPack tensor to set byte_offset
                # Keep the data pointer pointing to base, set byte_offset
                dl_tensor_ptr.contents.dl_tensor.data = base_data_ptr
                dl_tensor_ptr.contents.dl_tensor.byte_offset = expected_byte_offset
                
                # Adjust shape to match the offset view
                # Modify the shape array directly
                shape_ptr = dl_tensor_ptr.contents.dl_tensor.shape
                if shape_ptr:
                    shape_ptr[0] = expected_shape_size
                    # Verify the shape was set correctly
                    actual_shape_after = shape_ptr[0]
                    self.assertEqual(actual_shape_after, expected_shape_size,
                                   f"Shape should be {expected_shape_size}, got {actual_shape_after}")
                
                # Now test from_dlpack with the modified capsule
                result = from_dlpack(capsule)
                
                # Verify correctness
                self.assertEqual(result.shape, expected_shape, 
                               f"Shape mismatch: expected {expected_shape}, got {result.shape}")
                self.assertEqual(result, view, "Data values should match")
                self.assertEqual(
                    result.storage_offset(), 
                    expected_storage_offset,
                    f"storage_offset should be {expected_storage_offset}, "
                    f"got {result.storage_offset()}"
                )
                
            except (AttributeError, OSError, ctypes.ArgumentError) as e:
                # If ctypes access fails, skip the test
                self.skipTest(f"Could not modify DLPack capsule via ctypes: {e}")

instantiate_device_type_tests(TestTorchDlPack, globals(), allow_mps=True)

if __name__ == "__main__":
    run_tests()
