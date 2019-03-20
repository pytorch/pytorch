import torch
from common_utils import TestCase, run_tests


class TestMkldnn(TestCase):
    def test_conversion(self):
        cpu_tensor = torch.randn(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'))
        mkldnn_tensor = cpu_tensor.to_mkldnn()
        self.assertEqual(cpu_tensor, mkldnn_tensor.to_dense())
        self.assertEqual(mkldnn_tensor.dtype, torch.float)
        self.assertEqual(mkldnn_tensor.device, torch.device('cpu'))
        self.assertEqual(mkldnn_tensor.size(), torch.Size([1, 2, 3, 4]))

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
                creator(1, 2, 3, 4, dtype=torch.float, device=torch.device('cpu'), layout=torch.mkldnn)

if __name__ == '__main__':
    run_tests()
