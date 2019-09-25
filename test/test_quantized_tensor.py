import numpy as np

import torch
import io

from common_utils import TestCase, run_tests
import tempfile

class Foo(torch.nn.Module):
    def __init__(self):
        super(Foo, self).__init__()
        self.qscheme = torch.per_tensor_symmetric


class TestQuantizedTensor(TestCase):
    def test_qtensor(self):
        num_elements = 10
        r = torch.ones(num_elements, dtype=torch.float)
        scale = 1.0
        zero_point = 2
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.quint8)
        self.assertEqual(qr.q_scale(), scale)
        self.assertEqual(qr.q_zero_point(), zero_point)
        self.assertTrue(qr.is_quantized)
        self.assertFalse(r.is_quantized)
        self.assertEqual(qr.qscheme(), torch.per_tensor_affine)
        self.assertTrue(isinstance(qr.qscheme(), torch.qscheme))
        # slicing and int_repr
        int_repr = qr.int_repr()
        for num in int_repr:
            self.assertEqual(num, 3)
        for num in qr[2:].int_repr():
            self.assertEqual(num, 3)
        # dequantize
        rqr = qr.dequantize()
        for i in range(num_elements):
            self.assertEqual(r[i], rqr[i])
        # Scalar Tensor
        # item
        r = torch.ones(1, dtype=torch.float)
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.quint8)
        self.assertEqual(qr.item(), 1)
        self.assertEqual(qr[0].item(), 1)
        # assignment
        self.assertTrue(qr[0].is_quantized)
        qr[0] = 11.3  # float asignment
        self.assertEqual(qr.item(), 11)
        x = torch.ones(1, dtype=torch.float) * 15.3
        # Copying from a float Tensor
        qr[:] = x
        self.assertEqual(qr.item(), 15)
        # we can also print a qtensor
        self.assertEqual(' '.join(str(qr).split()),
                         "tensor([15.], size=(1,), dtype=torch.quint8, " +
                         "quantization_scheme=torch.per_tensor_affine, " +
                         "scale=1.0, zero_point=2)")
        empty_r = torch.ones((0, 1), dtype=torch.float)
        empty_qr = torch.quantize_per_tensor(empty_r, scale, zero_point, torch.quint8)
        self.assertEqual(' '.join(str(empty_qr).split()),
                         "tensor([], size=(0, 1), dtype=torch.quint8, " +
                         "quantization_scheme=torch.per_tensor_affine, " +
                         "scale=1.0, zero_point=2)")

    def test_qtensor_quant_dequant(self):
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        scale = 0.02
        zero_point = 2
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))

    def test_per_channel_qtensor_creation(self):
        numel = 10
        ch_axis = 0
        scales = torch.rand(numel)
        zero_points = torch.randint(0, 10, size=(numel,))
        q = torch._empty_per_channel_affine_quantized(
            [numel], scales=scales, zero_points=zero_points, axis=ch_axis, dtype=torch.quint8)
        self.assertEqual(scales, q.q_per_channel_scales())
        self.assertEqual(zero_points, q.q_per_channel_zero_points())
        self.assertEqual(ch_axis, q.q_per_channel_axis())

        # create Tensor from uint8_t Tensor, scales and zero_points
        int_tensor = torch.randint(0, 100, size=(numel,), dtype=torch.uint8)
        q = torch._make_per_channel_quantized_tensor(int_tensor, scales, zero_points, ch_axis)
        self.assertEqual(int_tensor, q.int_repr())
        self.assertEqual(scales, q.q_per_channel_scales())
        self.assertEqual(zero_points, q.q_per_channel_zero_points())
        self.assertEqual(ch_axis, q.q_per_channel_axis())

    def test_qtensor_creation(self):
        scale = 0.5
        zero_point = 10
        val = 100
        numel = 10
        q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=torch.quint8)
        self.assertEqual(scale, q.q_scale())
        self.assertEqual(zero_point, q.q_zero_point())

        # create Tensor from uint8_t Tensor, scale and zero_point
        int_tensor = torch.randint(0, 100, size=(10,), dtype=torch.uint8)
        q = torch._make_per_tensor_quantized_tensor(int_tensor, scale, zero_point)
        self.assertEqual(int_tensor, q.int_repr())
        self.assertEqual(scale, q.q_scale())
        self.assertEqual(zero_point, q.q_zero_point())

        # create via empty_like
        q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=torch.quint8)
        q_el = torch.empty_like(q)
        self.assertEqual(q.q_scale(), q_el.q_scale())
        self.assertEqual(q.q_zero_point(), q_el.q_zero_point())
        self.assertEqual(q.dtype, q_el.dtype)

        # create via empty_like but change the dtype (currently not supported)
        with self.assertRaises(RuntimeError):
            torch.empty_like(q, dtype=torch.qint8)

    def test_qtensor_dtypes(self):
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        scale = 0.2
        zero_point = 2
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.qint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.qint32)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))

    def test_qtensor_quantize_per_channel(self):
        r = torch.rand(3, 2, dtype=torch.float) * 4 - 2
        scales = torch.tensor([0.2, 0.03], dtype=torch.double)
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        axis = 1

        def quantize_c(data, scales, zero_points):
            res = torch.empty((3, 2))
            quant_min, quant_max = 0, 255
            for i in range(3):
                for j in range(2):
                    res[i][j] = np.clip(np.round(data[i][j] / scales[j]) + zero_points[j], quant_min, quant_max)
            return res
        qr = torch.quantize_per_channel(r, scales, zero_points, axis, torch.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(qr.int_repr(), quantize_c(r, scales, zero_points)))
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))

    def test_qtensor_permute(self):
        r = torch.rand(10, 30, 2, 2, dtype=torch.float) * 4 - 2
        scale = 0.02
        zero_point = 1
        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.qint8)
        qr = qr.transpose(0, 1)
        rqr = qr.dequantize()
        # compare transpose + dequantized result with orignal transposed result
        self.assertTrue(np.allclose(r.numpy().transpose([1, 0, 2, 3]), rqr.numpy(), atol=2 / scale))

        qr = torch.quantize_per_tensor(r, scale, zero_point, torch.qint8)
        qr1 = qr.permute([1, 0, 2, 3])
        qr2 = qr.transpose(0, 1)
        # compare int representation after transformations
        self.assertEqual(qr1.int_repr(), qr2.int_repr())
        self.assertEqual(qr1.q_scale(), qr2.q_scale())
        self.assertEqual(qr1.q_zero_point(), qr2.q_zero_point())
        # compare dequantized result
        self.assertEqual(qr1.dequantize(), qr2.dequantize())
        # compare permuted + dequantized result with original transposed result
        self.assertTrue(np.allclose(qr2.dequantize().numpy(), r.numpy().transpose([1, 0, 2, 3]), atol=2 / scale))
        # make permuted result contiguous
        self.assertEqual(qr2.contiguous().int_repr(), qr2.int_repr())

        # change memory format
        qlast = qr.contiguous(memory_format=torch.channels_last)
        self.assertEqual(qr.stride(), list(reversed(sorted(qr.stride()))))
        self.assertNotEqual(qlast.stride(), list(reversed(sorted(qlast.stride()))))
        self.assertEqual(qr.int_repr(), qlast.int_repr())
        self.assertEqual(qr.q_scale(), qlast.q_scale())
        self.assertEqual(qr.q_zero_point(), qlast.q_zero_point())
        self.assertEqual(qlast.dequantize(), qr.dequantize())

    def test_qtensor_per_channel_permute(self):
        r = torch.rand(20, 10, 2, 2, dtype=torch.float) * 4 - 2
        scales = torch.rand(10) * 0.02 + 0.01
        zero_points = torch.round(torch.rand(10) * 2 - 1).to(torch.long)
        qr = torch.quantize_per_channel(r, scales, zero_points, 1, torch.qint8)

        # we can't reorder the axis
        with self.assertRaises(RuntimeError):
            qr.transpose(0, 1)

        # but we can change memory format
        qlast = qr.contiguous(memory_format=torch.channels_last)
        self.assertEqual(qr.stride(), list(reversed(sorted(qr.stride()))))
        self.assertNotEqual(qlast.stride(), list(reversed(sorted(qlast.stride()))))
        self.assertEqual(qr.int_repr(), qlast.int_repr())
        self.assertEqual(scales, qlast.q_per_channel_scales())
        self.assertEqual(zero_points, qlast.q_per_channel_zero_points())
        self.assertEqual(1, qlast.q_per_channel_axis())
        self.assertEqual(qlast.dequantize(), qr.dequantize())

    def test_qtensor_load_save(self):
        scale = 0.2
        zero_point = 10
        r = torch.rand(15, 2, dtype=torch.float32) * 2
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            qr = torch.quantize_per_tensor(r, scale, zero_point, dtype)
            qrv = qr[:, 1]
            with tempfile.NamedTemporaryFile() as f:
                # Serializing and Deserializing Tensor
                torch.save((qr, qrv), f)
                f.seek(0)
                qr2, qrv2 = torch.load(f)
                self.assertEqual(qr, qr2)
                self.assertEqual(qrv, qrv2)
                self.assertEqual(qr2.storage().data_ptr(), qrv2.storage().data_ptr())

    def test_qtensor_per_channel_load_save(self):
        r = torch.rand(20, 10, dtype=torch.float) * 4 - 2
        scales = torch.rand(10) * 0.02 + 0.01
        zero_points = torch.round(torch.rand(10) * 20 + 1).to(torch.long)
        # quint32 is not supported yet
        for dtype in [torch.quint8, torch.qint8]:
            qr = torch.quantize_per_channel(r, scales, zero_points, 1, dtype)
            with tempfile.NamedTemporaryFile() as f:
                # Serializing and Deserializing Tensor
                torch.save(qr, f)
                f.seek(0)
                qr2 = torch.load(f)
                self.assertEqual(qr, qr2)

    def test_qtensor_copy(self):
        scale = 0.5
        zero_point = 10
        val = 100
        numel = 10
        # copy from same scale and zero_point
        q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=torch.quint8)
        q2 = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=torch.quint8)
        q.copy_(q2)
        self.assertEqual(q.int_repr(), q2.int_repr())
        self.assertEqual(q.q_scale(), q2.q_scale())
        self.assertEqual(q.q_zero_point(), q2.q_zero_point())
        # copying from different scale and zero_point
        scale = 3.2
        zero_point = 5
        q = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=torch.quint8)
        # check original scale and zero_points are set correctly
        self.assertEqual(q.q_scale(), scale)
        self.assertEqual(q.q_zero_point(), zero_point)
        q.copy_(q2)
        # check scale and zero_points has been copied
        self.assertEqual(q, q2)

    def test_qtensor_clone(self):
        numel = 10
        scale = 0.5
        zero_point = 10
        q2 = torch._empty_affine_quantized([numel], scale=scale, zero_point=zero_point, dtype=torch.quint8)
        q = q2.clone()
        # Check to make sure the scale and zero_point has been copied.
        self.assertEqual(q, q2)

    def test_qtensor_view(self):
        scale, zero_point, dtype = 1.0, 2, torch.quint8
        q = torch._empty_affine_quantized(1, 2, 3, scale=scale, zero_point=zero_point, dtype=dtype)
        q2 = q.view(1, 3, 2)
        self.assertEqual(q.numel(), q2.numel())
        # testing -1
        self.assertEqual(q, q2.view(1, -1, 3))

        a = torch._empty_affine_quantized([1, 2, 3, 4], scale=scale, zero_point=zero_point, dtype=dtype)
        b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
        c = a.view(1, 3, 2, 4)  # does not change tensor layout
        self.assertEqual(b.size(), c.size())
        self.assertEqual(b.q_scale(), c.q_scale())
        self.assertEqual(b.q_zero_point(), c.q_zero_point())
        # TODO: fix flaky test
        # self.assertNotEqual(b.int_repr(), c.int_repr())


        # a case can't view non-contiguos Tensor
        a = torch._empty_affine_quantized([1, 2, 3, 4], scale=scale, zero_point=zero_point, dtype=dtype)
        b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
        err_str = "view size is not compatible with input tensor's size and stride*"
        with self.assertRaisesRegex(RuntimeError, err_str):
            b.view(1, 4, 2, 3)
        # view on contiguous tensor is fine
        b.contiguous().view(1, 4, 2, 3)


    def test_qtensor_reshape(self):
        scale, zero_point, dtype = 1.0, 2, torch.quint8
        q = torch._empty_affine_quantized([3, 5], scale=scale, zero_point=zero_point, dtype=dtype)
        q2 = q.reshape([15])
        self.assertEqual(q.numel(), q2.numel())
        self.assertEqual(q2.size(), [15])
        # testing -1
        self.assertEqual(q, q2.reshape([3, -1]))

        a = torch._empty_affine_quantized([1, 2, 3, 4], scale=scale, zero_point=zero_point, dtype=dtype)
        b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
        c = a.reshape(1, 3, 2, 4)  # does not change tensor layout
        self.assertEqual(b.size(), c.size())
        self.assertEqual(b.q_scale(), c.q_scale())
        self.assertEqual(b.q_zero_point(), c.q_zero_point())
        # TODO: fix flaky test
        # self.assertNotEqual(b.int_repr(), c.int_repr())

        # we can use reshape for non-contiguous Tensor
        a = torch._empty_affine_quantized([1, 2, 3, 4], scale=scale, zero_point=zero_point, dtype=dtype)
        b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
        c = b.reshape(1, 4, 2, 3)
        self.assertEqual(b, c.reshape(1, 3, 2, 4))

    def test_qscheme_pickle(self):

        f = Foo()
        buf = io.BytesIO()
        torch.save(f, buf)

        buf.seek(0)
        f2 = torch.load(buf)

        self.assertEqual(f2.qscheme, torch.per_tensor_symmetric)

if __name__ == "__main__":
    run_tests()
