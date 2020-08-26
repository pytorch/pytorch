import torch
import torch.cuda
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

class TestForeach(TestCase):
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_same_size_tensors(self, device, dtype):
        N = 20
        H = 20
        W = 20
        tensors = []
        for _ in range(N):
            tensors.append(torch.zeros(H, W, device=device, dtype=dtype))

        res = torch._foreach_add(tensors, 1)
        for t in res:
            if dtype == torch.bool:
                dtype = torch.int64
            self.assertEqual(t, torch.ones(H, W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_different_size_tensors(self, device, dtype):
        N = 20
        H = 20
        W = 20

        tensors = []
        size_change = 0
        for _ in range(N):
            tensors.append(torch.zeros(H + size_change, W + size_change, device=device, dtype=dtype))
            size_change += 1

        res = torch._foreach_add(tensors, 1)

        size_change = 0
        for t in res: 
            if dtype == torch.bool:
                dtype = torch.int64
            self.assertEqual(t, torch.ones(H + size_change, W + size_change, device=device, dtype=dtype))
            size_change += 1

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_empty_list(self, device, dtype):
        tensors = []
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors, 1)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_overlapping_tensors(self, device, dtype):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        expected = [torch.tensor([[[2, 2, 2]], [[2, 2, 2]]], dtype=dtype, device=device)]

        if dtype == torch.bool: 
            expected[0] = expected[0].to(torch.int64).add(1)

        res = torch._foreach_add(tensors, 1)
        self.assertEqual(res, expected)

    def test_add_scalar_with_different_tensor_dtypes(self, device):
        tensors = [torch.tensor([1], dtype=torch.float, device=device), 
                   torch.tensor([1], dtype=torch.int, device=device)]

        expected = [torch.tensor([2], dtype=torch.float, device=device), 
                    torch.tensor([2], dtype=torch.int, device=device)]

        res = torch._foreach_add(tensors, 1)
        self.assertEqual(res, expected)

    def test_add_scalar_with_different_scalar_type(self, device):
        # int tensor with float scalar
        # should go 'slow' route
        scalar = 1.1
        tensors = [torch.tensor([1], dtype=torch.int, device=device)]
        res = torch._foreach_add(tensors, scalar)
        self.assertEqual(res, [torch.tensor([2.1], device=device)])

        # float tensor with int scalar
        # should go 'fast' route
        scalar = 1
        tensors = [torch.tensor([1.1], device=device)]
        res = torch._foreach_add(tensors, scalar)
        self.assertEqual(res, [torch.tensor([2.1], device=device)])

        # bool tensor with int scalar
        # should go 'slow' route
        scalar = 1
        tensors = [torch.tensor([False], device=device)]
        res = torch._foreach_add(tensors, scalar)
        self.assertEqual(res, [torch.tensor([1], device=device)])

        # bool tensor with float scalar
        # should go 'slow' route
        scalar = 1.1
        tensors = [torch.tensor([False], device=device)]
        res = torch._foreach_add(tensors, scalar)
        self.assertEqual(res, [torch.tensor([1.1], device=device)])

instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
