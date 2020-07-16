import torch
import torch.cuda
from test_torch import AbstractTestCases
from torch.testing._internal.common_utils import TestCase, run_tests

class TestForeach(TestCase):
    def test_add_scalar_same_size_tensors(self):
        N = 20
        H = 20
        W = 20

        for dt in torch.testing.get_all_dtypes():
            if dt == torch.bool: 
                continue

            for d in torch.testing.get_all_device_types():
                tensors = []
                for _ in range(N):
                    tensors.append(torch.zeros(H, W, device=d, dtype=dt))

                res = torch._foreach_add(tensors, 1)
                
                for t in res: 
                    self.assertEqual(t, torch.ones(H, W, device=d, dtype=dt))

    def test_add_scalar_different_size_tensors(self):
        N = 20
        H = 20
        W = 20

        for dt in torch.testing.get_all_dtypes():
            if dt == torch.bool: 
                continue

            for d in torch.testing.get_all_device_types():
                tensors = []
                size_change = 0
                for _ in range(N):
                    tensors.append(torch.zeros(H + size_change, W + size_change, device=d, dtype=dt))
                    size_change += 1

                res = torch._foreach_add(tensors, 1)
                
                size_change = 0
                for t in res: 
                    self.assertEqual(t, torch.ones(H + size_change, W + size_change, device=d, dtype=dt))
                    size_change += 1


    def test_add_scalar_with_empty_list(self):
        for dt in torch.testing.get_all_dtypes():
            if dt == torch.bool: 
                continue

            for d in torch.testing.get_all_device_types():
                tensors = []
                with self.assertRaisesRegex(RuntimeError, r"Tensor list can't be empty."):
                    torch._foreach_add(tensors, 1)

if __name__ == '__main__':
    run_tests()
