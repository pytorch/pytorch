from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import torch

from common_utils import TestCase, run_tests
import tempfile

class TestQuantizedTensor(TestCase):
    def test_qtensor(self):
        num_elements = 10
        r = torch.ones(num_elements, dtype=torch.float)
        scale = 1.0
        zero_point = 2
        qr = torch.quantize_linear(r, scale, zero_point, torch.quint8)
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
        qr = torch.quantize_linear(r, scale, zero_point, torch.quint8)
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
        self.assertEqual(str(qr),
                         "tensor([15.], size=(1,), dtype=torch.quint8, " +
                         "scale=1.0, zero_point=2)")
        empty_r = torch.ones((0, 1), dtype=torch.float)
        empty_qr = torch.quantize_linear(empty_r, scale, zero_point, torch.quint8)
        self.assertEqual(str(empty_qr),
                         "tensor([], size=(0, 1), dtype=torch.quint8, " +
                         "scale=1.0, zero_point=2)")

    def test_qtensor_quant_dequant(self):
        r = torch.rand(3, 2, dtype=torch.float) * 2 - 4
        scale = 2
        zero_point = 2
        qr = torch.quantize_linear(r, scale, zero_point, torch.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))

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
        q = torch._per_tensor_affine_qtensor(int_tensor, scale, zero_point)
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
        r = torch.rand(3, 2, dtype=torch.float) * 2 - 4
        scale = 2
        zero_point = 2
        qr = torch.quantize_linear(r, scale, zero_point, torch.qint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))
        qr = torch.quantize_linear(r, scale, zero_point, torch.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))
        qr = torch.quantize_linear(r, scale, zero_point, torch.qint32)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / scale))

    def test_qtensor_dequantize_linear(self):
        t = torch.arange(-10, 10, dtype=torch.int8)
        scale = 3
        zero_point = 2
        qt = torch._dequantize_linear(t, scale, zero_point, torch.qint8)
        qt2 = torch._per_tensor_affine_qtensor(t, scale, zero_point)
        self.assertEqual(qt, qt2.dequantize())

    def test_qtensor_per_channel_affine(self):
        r = torch.rand(3, 2, dtype=torch.float) * 2 - 4
        scales = torch.tensor([2.0, 3.0], dtype=torch.double)
        zero_points = torch.tensor([5, 10], dtype=torch.long)
        axis = [1]

        def quantize_c(data, scales, zero_points):
            res = torch.empty((3, 2))
            quant_min, quant_max = 0, 255
            for i in range(3):
                for j in range(2):
                    res[i][j] = np.clip(np.round(data[i][j] / scales[j]) + zero_points[j], quant_min, quant_max)
            return res
        qr = torch.quantize_linear_per_channel(r, scales, zero_points, axis, torch.quint8)
        rqr = qr.dequantize()
        self.assertTrue(np.allclose(qr.int_repr(), quantize_c(r, scales, zero_points)))
        self.assertTrue(np.allclose(r.numpy(), rqr.numpy(), atol=2 / np.min(scales.numpy())))

    def test_qtensor_permute(self):
        r = torch.rand(100, 30, dtype=torch.float) * 2 - 4
        scale = 2
        zero_point = 2
        qr = torch.quantize_linear(r, scale, zero_point, torch.qint8)
        qr = qr.transpose(0, 1)
        rqr = qr.dequantize()
        # compare transpose + dequantized result with orignal transposed result
        self.assertTrue(np.allclose(r.numpy().T, rqr.numpy(), atol=2 / scale))

        qr = torch.quantize_linear(r, scale, zero_point, torch.qint8)
        qr1 = qr.permute([1, 0])
        qr2 = qr.transpose(0, 1)
        # compare int representation after transformations
        self.assertTrue(torch.equal(qr1.int_repr(), qr2.int_repr()))
        self.assertTrue(qr1.q_scale() == qr2.q_scale())
        self.assertTrue(qr1.q_zero_point() == qr2.q_zero_point())
        # compare dequantized result
        self.assertTrue(np.array_equal(qr1.dequantize().numpy(), qr2.dequantize().numpy()))
        # compare permuted + dequantized result with original transposed result
        self.assertTrue(np.allclose(qr2.dequantize().numpy(), r.numpy().T, atol=2 / scale))
        # make permuted result contiguous
        self.assertTrue(torch.equal(qr2.contiguous().int_repr(), qr2.int_repr()))

    def test_qtensor_load_save(self):
        scale = 2.0
        zero_point = 10
        r = torch.ones(15, dtype=torch.float) * 2
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            qr = torch.quantize_linear(r, scale, zero_point, dtype)
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
        self.assertNotEqual(b.int_repr(), c.int_repr())


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
        self.assertNotEqual(b.int_repr(), c.int_repr())

        # we can use reshape for non-contiguous Tensor
        a = torch._empty_affine_quantized([1, 2, 3, 4], scale=scale, zero_point=zero_point, dtype=dtype)
        b = a.transpose(1, 2)  # swaps 2nd and 3rd dimension
        c = b.reshape(1, 4, 2, 3)
        self.assertEqual(b, c.reshape(1, 3, 2, 4))

if __name__ == "__main__":
    run_tests()
