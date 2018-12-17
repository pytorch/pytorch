import unittest
import sys

import common_utils as common
from common_utils import TEST_NUMBA, TEST_NUMPY
from common_cuda import TEST_NUMBA_CUDA, TEST_CUDA, TEST_MULTIGPU

import torch

if TEST_NUMPY:
    import numpy

if TEST_NUMBA:
    import numba

if TEST_NUMBA_CUDA:
    import numba.cuda


class TestNumbaIntegration(common.TestCase):
    @unittest.skipIf(not TEST_NUMPY, "No numpy")
    @unittest.skipIf(not TEST_CUDA, "No cuda")
    def test_cuda_array_interface(self):
        """torch.Tensor exposes __cuda_array_interface__ for cuda tensors.

        An object t is considered a cuda-tensor if:
            hasattr(t, '__cuda_array_interface__')

        A cuda-tensor provides a tensor description dict:
            shape: (integer, ...) Tensor shape.
            strides: (integer, ...) Tensor strides, in bytes.
            typestr: (str) A numpy-style typestr.
            data: (int, boolean) A (data_ptr, read-only) tuple.
            version: (int) Version 0

        See:
        https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
        """

        types = [
            torch.DoubleTensor,
            torch.FloatTensor,
            torch.HalfTensor,
            torch.LongTensor,
            torch.IntTensor,
            torch.ShortTensor,
            torch.CharTensor,
            torch.ByteTensor,
        ]
        dtypes = [
            numpy.float64,
            numpy.float32,
            numpy.float16,
            numpy.int64,
            numpy.int32,
            numpy.int16,
            numpy.int8,
            numpy.uint8,
        ]
        for tp, npt in zip(types, dtypes):

            # CPU tensors do not implement the interface.
            cput = tp(10)

            self.assertFalse(hasattr(cput, "__cuda_array_interface__"))
            self.assertRaises(AttributeError, lambda: cput.__cuda_array_interface__)

            # Sparse CPU/CUDA tensors do not implement the interface
            if tp not in (torch.HalfTensor,):
                indices_t = torch.empty(1, cput.size(0), dtype=torch.long).clamp_(min=0)
                sparse_t = torch.sparse_coo_tensor(indices_t, cput)

                self.assertFalse(hasattr(sparse_t, "__cuda_array_interface__"))
                self.assertRaises(
                    AttributeError, lambda: sparse_t.__cuda_array_interface__
                )

                sparse_cuda_t = torch.sparse_coo_tensor(indices_t, cput).cuda()

                self.assertFalse(hasattr(sparse_cuda_t, "__cuda_array_interface__"))
                self.assertRaises(
                    AttributeError, lambda: sparse_cuda_t.__cuda_array_interface__
                )

            # CUDA tensors have the attribute and v0 interface
            cudat = tp(10).cuda()

            self.assertTrue(hasattr(cudat, "__cuda_array_interface__"))

            ar_dict = cudat.__cuda_array_interface__

            self.assertEqual(
                set(ar_dict.keys()), {"shape", "strides", "typestr", "data", "version"}
            )

            self.assertEqual(ar_dict["shape"], (10,))
            self.assertEqual(ar_dict["strides"], (cudat.storage().element_size(),))
            # typestr from numpy, cuda-native little-endian
            self.assertEqual(ar_dict["typestr"], numpy.dtype(npt).newbyteorder("<").str)
            self.assertEqual(ar_dict["data"], (cudat.data_ptr(), False))
            self.assertEqual(ar_dict["version"], 0)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_array_adaptor(self):
        """Torch __cuda_array_adaptor__ exposes tensor data to numba.cuda."""

        torch_dtypes = [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]

        for dt in torch_dtypes:
            if dt == torch.int8:
                # "CharTensor" numpy conversion not supported
                with self.assertRaises(TypeError):
                    torch.arange(10).to(dt).numpy()

                continue

            # CPU tensors of all types do not register as cuda arrays,
            # attempts to convert raise a type error.
            cput = torch.arange(10).to(dt)
            npt = cput.numpy()

            self.assertTrue(not numba.cuda.is_cuda_array(cput))
            with self.assertRaises(TypeError):
                numba.cuda.as_cuda_array(cput)

            # Any cuda tensor is a cuda array.
            cudat = cput.to(device="cuda")
            self.assertTrue(numba.cuda.is_cuda_array(cudat))

            numba_view = numba.cuda.as_cuda_array(cudat)
            self.assertIsInstance(numba_view, numba.cuda.devicearray.DeviceNDArray)

            # The reported type of the cuda array matches the numpy type of the cpu tensor.
            self.assertEqual(numba_view.dtype, npt.dtype)
            self.assertEqual(numba_view.strides, npt.strides)
            self.assertEqual(numba_view.shape, cudat.shape)

            # Pass back to cuda from host for all equality checks below, needed for
            # float16 comparisons, which aren't supported cpu-side.

            # The data is identical in the view.
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to("cuda"))

            # Writes to the torch.Tensor are reflected in the numba array.
            cudat[:5] = 11
            self.assertEqual(cudat, torch.tensor(numba_view.copy_to_host()).to("cuda"))

            # Strided tensors are supported.
            strided_cudat = cudat[::2]
            strided_npt = cput[::2].numpy()
            strided_numba_view = numba.cuda.as_cuda_array(strided_cudat)

            self.assertEqual(strided_numba_view.dtype, strided_npt.dtype)
            self.assertEqual(strided_numba_view.strides, strided_npt.strides)
            self.assertEqual(strided_numba_view.shape, strided_cudat.shape)

            # As of numba 0.40.0 support for strided views is ...limited...
            # Cannot verify correctness of strided view operations.

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    def test_conversion_errors(self):
        """Numba properly detects array interface for tensor.Tensor variants."""

        # CPU tensors are not cuda arrays.
        cput = torch.arange(100)

        self.assertFalse(numba.cuda.is_cuda_array(cput))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cput)

        # Sparse tensors are not cuda arrays, regardless of device.
        sparset = torch.sparse_coo_tensor(cput[None, :], cput)

        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)

        sparse_cuda_t = sparset.cuda()

        self.assertFalse(numba.cuda.is_cuda_array(sparset))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(sparset)

        # Device-status overrides gradient status.
        # CPU+gradient isn't a cuda array.
        cpu_gradt = torch.zeros(100).requires_grad_(True)

        self.assertFalse(numba.cuda.is_cuda_array(cpu_gradt))
        with self.assertRaises(TypeError):
            numba.cuda.as_cuda_array(cpu_gradt)

        # CUDA+gradient raises a RuntimeError on check or conversion.
        #
        # Use of hasattr for interface detection causes interface change in
        # python2; it swallows all exceptions not just AttributeError.
        cuda_gradt = torch.zeros(100).requires_grad_(True).cuda()

        if sys.version_info.major > 2:
            # 3+, conversion raises RuntimeError
            with self.assertRaises(RuntimeError):
                numba.cuda.is_cuda_array(cuda_gradt)
            with self.assertRaises(RuntimeError):
                numba.cuda.as_cuda_array(cuda_gradt)
        else:
            # 2, allow either RuntimeError on access or non-implementing
            # behavior to future-proof against potential changes in numba.
            try:
                was_cuda_array = numba.cuda.is_cuda_array(cuda_gradt)
                was_runtime_error = False
            except RuntimeError:
                was_cuda_array = False
                was_runtime_error = True

            self.assertFalse(was_cuda_array)

            if not was_runtime_error:
                with self.assertRaises(TypeError):
                    numba.cuda.as_cuda_array(cuda_gradt)
            else:
                with self.assertRaises(RuntimeError):
                    numba.cuda.as_cuda_array(cuda_gradt)

    @unittest.skipIf(not TEST_CUDA, "No cuda")
    @unittest.skipIf(not TEST_NUMBA_CUDA, "No numba.cuda")
    @unittest.skipIf(not TEST_MULTIGPU, "No multigpu")
    def test_active_device(self):
        """'as_cuda_array' tensor device must match active numba context."""

        # Both torch/numba default to device 0 and can interop freely
        cudat = torch.arange(10, device="cuda")
        self.assertEqual(cudat.device.index, 0)
        self.assertIsInstance(
            numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray
        )

        # Tensors on non-default device raise api error if converted
        cudat = torch.arange(10, device=torch.device("cuda", 1))

        with self.assertRaises(numba.cuda.driver.CudaAPIError):
            numba.cuda.as_cuda_array(cudat)

        # but can be converted when switching to the device's context
        with numba.cuda.devices.gpus[cudat.device.index]:
            self.assertIsInstance(
                numba.cuda.as_cuda_array(cudat), numba.cuda.devicearray.DeviceNDArray
            )


if __name__ == "__main__":
    common.run_tests()
