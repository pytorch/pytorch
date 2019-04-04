import torch
# import unittest
from common_utils import TestCase, run_tests
from torch.autograd.gradcheck import gradgradcheck, gradcheck
import warnings


# Comment the line below to find out the CI machines having MKL-DNN build disabled
# @unittest.skipIf(not torch._C.has_mkldnn, "MKL-DNN build is disabled")
class TestMkldnn(TestCase):
    def test_conversion(self):
        for cpu_tensor in [torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu')),
                           torch.randn(1, 2, 3, 4, 5, dtype=torch.float, device=torch.device('cpu'))[:, :, :, :, 1]]:
            cpu_tensor.requires_grad_()
            mkldnn_tensor = cpu_tensor.to_mkldnn()
            cpu_tensor_1 = mkldnn_tensor.to_dense()
            self.assertEqual(cpu_tensor, cpu_tensor_1)
            self.assertEqual(mkldnn_tensor.dtype, torch.float)
            self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
            self.assertEqual(mkldnn_tensor.size(), torch.Size([1, 2, 3, 4]))
            self.assertEqual(mkldnn_tensor.numel(), cpu_tensor.numel())
            self.assertEqual(mkldnn_tensor.element_size(), cpu_tensor.element_size())
            self.assertTrue(mkldnn_tensor.data_ptr() != 0)

    def test_unsupported(self):
        # unsupported types and unsupported types with gpu
        for dtype in [torch.double, torch.half, torch.uint8, torch.int8,
                      torch.short, torch.int, torch.long]:
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cpu')).to_mkldnn()
            if torch.cuda.is_available():
                with self.assertRaises(RuntimeError) as context:
                    torch.randn(1, 2, 3, 4, dtype=dtype, device=torch.device('cuda')).to_mkldnn()
        # supported type with gpu
        if torch.cuda.is_available():
            with self.assertRaises(RuntimeError) as context:
                torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cuda')).to_mkldnn()
        # some factory functions
        for creator in [torch.empty, torch.ones, torch.zeros, torch.randn, torch.rand]:
            with self.assertRaises(RuntimeError) as context:
                creator(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'), layout=torch._mkldnn)

    def test_autograd_to_mkldnn(self):
        # MKLDNN only supports float32
        root = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)

        def func(root):
            return root.to_mkldnn().to_dense()

        # because MKLDNN only supports float32, we need to lessen the precision.
        # these numbers are just empirical results that seem to work.
        self.assertWarnsRegex(lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2),
                              'double precision floating point')
        self.assertWarnsRegex(lambda: gradgradcheck(func, [root], atol=4e-2, rtol=1e-2),
                              'double precision floating point')

    def test_autograd_from_mkldnn(self):
        # MKLDNN only supports float32
        root = torch.randn(4, 5, dtype=torch.float32).to_mkldnn().requires_grad_()

        def func(root):
            return root.to_dense()

        # because MKLDNN only supports float32, we need to lessen the precision.
        # these numbers are just empirical results that seem to work.
        self.assertWarnsRegex(lambda: gradcheck(func, [root], atol=4e-2, rtol=1e-2),
                              'double precision floating point')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertRaisesRegex(ValueError, 'MKLDNN output is not supported at gradcheck yet',
                                   lambda: gradgradcheck(func, [root], atol=4e-2, rtol=1e-2))

    def test_repr(self):
        self.assertTrue("layout=torch._mkldnn" in str(torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu')).to_mkldnn()))

if __name__ == '__main__':
    run_tests()
