# -*- coding: utf-8 -*-
# Owner(s): ["module: tests"]

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests, IS_JETSON
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCUDA, dtypes, skipMeta, skipCUDAIfRocm,
    onlyNativeDeviceTypes)
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.utils.dlpack import from_dlpack, to_dlpack


class TestTorchDlPack(TestCase):
    exact_dtype = True

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_capsule_conversion(self, device, dtype):
        # DLpack does not explicitly support bool (xref dmlc/dlpack#75)
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(to_dlpack(x))
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_protocol_conversion(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        z = from_dlpack(x)
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    def test_dlpack_shared_storage(self, device):
        x = make_tensor((5,), dtype=torch.float64, device=device)
        z = from_dlpack(to_dlpack(x))
        z[0] = z[0] + 20.0
        self.assertEqual(z, x)

    @skipMeta
    @onlyCUDA
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_conversion_with_streams(self, device, dtype):
        # Create a stream where the tensor will reside
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Do an operation in the actual stream
            x = make_tensor((5,), dtype=dtype, device=device) + 1
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
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_from_dlpack(self, device, dtype):
        x = make_tensor((5,), dtype=dtype, device=device)
        y = torch.from_dlpack(x)
        self.assertEqual(x, y)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
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
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
    def test_dlpack_conversion_with_diff_streams(self, device, dtype):
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        # DLPack protocol helps establish a correct stream order
        # (hence data dependency) at the exchange boundary.
        # the `tensor.__dlpack__` method will insert a synchronization event
        # in the current stream to make sure that it was correctly populated.
        with torch.cuda.stream(stream_a):
            x = make_tensor((5,), dtype=dtype, device=device) + 1
            z = torch.from_dlpack(x.__dlpack__(stream_b.cuda_stream))
            stream_a.synchronize()
        stream_b.synchronize()
        self.assertEqual(z, x)

    @skipMeta
    @onlyNativeDeviceTypes
    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
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
                capsule = self.tensor.__dlpack__(stream)
                return capsule

        # CUDA-based tests runs on non-default streams
        with torch.cuda.stream(torch.cuda.default_stream()):
            x = DLPackTensor(make_tensor((5,), dtype=torch.float32, device=device))
            from_dlpack(x)

    @skipMeta
    @onlyCUDA
    @skipCUDAIfRocm
    def test_dlpack_convert_default_stream(self, device):
        torch.cuda.default_stream().synchronize()
        # run _sleep call on a non-default stream, causing
        # default stream to wait due to inserted syncs
        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            x = torch.zeros(1, device=device)
            torch.cuda._sleep(2**20)
            self.assertTrue(torch.cuda.default_stream().query())
            d = x.__dlpack__(1)
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
    def test_dlpack_error_on_bool_tensor(self):
        x = torch.tensor([True], dtype=torch.bool)
        with self.assertRaises(RuntimeError):
            to_dlpack(x)

    # TODO: add interchange tests once NumPy 1.22 (dlpack support) is required
    @skipMeta
    def test_dlpack_export_requires_grad(self):
        x = torch.zeros(10, dtype=torch.float32, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, r"require gradient"):
            x.__dlpack__()

    @skipMeta
    def test_dlpack_export_is_conj(self):
        x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])
        y = torch.conj(x)
        with self.assertRaisesRegex(RuntimeError, r"conjugate bit"):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_export_non_strided(self):
        x = torch.sparse_coo_tensor([[0]], [1], size=(1,))
        y = torch.conj(x)
        with self.assertRaisesRegex(RuntimeError, r"strided"):
            y.__dlpack__()

    @skipMeta
    def test_dlpack_normalize_strides(self):
        x = torch.rand(16)
        y = x[::3][:1]
        self.assertEqual(y.shape, (1,))
        self.assertEqual(y.stride(), (3,))
        z = from_dlpack(y)
        self.assertEqual(z.shape, (1,))
        # gh-83069, make sure __dlpack__ normalizes strides
        self.assertEqual(z.stride(), (1,))


instantiate_device_type_tests(TestTorchDlPack, globals())

if __name__ == '__main__':
    run_tests()
