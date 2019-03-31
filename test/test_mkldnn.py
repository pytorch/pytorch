import torch
from common_utils import TestCase, run_tests


class TestMkldnn(TestCase):
    def test_conversion(self):
        for cpu_tensor in [torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu')),
                           torch.randn(1, 2, 3, 4, 5, dtype=torch.float, device=torch.device('cpu'))[:,:,:,:,1]]:
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
            # compare the grad with and without mkldnn conversion
            cpu_tensor.sum().backward()
            grad_without_mkldnn = cpu_tensor.grad.clone().detach()
            cpu_tensor.grad.data.zero_()
            cpu_tensor_1.sum().backward()
            grad_with_mkldnn = cpu_tensor.grad.data
            self.assertEqual(grad_without_mkldnn, grad_with_mkldnn)

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

if __name__ == '__main__':
    run_tests()
