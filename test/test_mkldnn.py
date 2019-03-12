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


if __name__ == '__main__':
    run_tests()
