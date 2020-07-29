import torch
import torch.cuda
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

devices = (torch.device('cpu'), torch.device('cuda:0'))

class TestForeach(TestCase):
    N = 20
    H = 20
    W = 20

    def get_test_data(self, device, dtype):
        tensors = []
        for _ in range(self.N):
            tensors.append(torch.ones(self.H, self.W, device=device, dtype=dtype))

        return tensors

    #
    # Test binary ops with scalar
    #
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_same_size_tensors(self, device, dtype):
        tensors = self.get_test_data(device, dtype)
        res = torch._foreach_add(tensors, 1)

        if dtype == torch.bool and device == 'cpu':
            dtype = torch.int64
            expected = torch.ones(self.H, self.W, device=device, dtype=dtype).add_(1)
        elif dtype == torch.bool and device == 'cuda:0':
            expected = torch.ones(self.H, self.W, device=device, dtype=dtype)
        else: 
            expected = torch.ones(self.H, self.W, device=device, dtype=dtype).add_(1)

        for t in res:
            self.assertEqual(t, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_different_size_tensors(self, device, dtype):
        tensors = []
        size_change = 0
        for _ in range(self.N):
            tensors.append(torch.zeros(self.H + size_change, self.W + size_change, device=device, dtype=dtype))
            size_change += 1

        res = torch._foreach_add(tensors, 1)
        size_change = 0
        for t in res: 
            if dtype == torch.bool and device == 'cpu':
                dtype = torch.int64
            self.assertEqual(t, torch.ones(self.H + size_change, self.W + size_change, device=device, dtype=dtype))
            size_change += 1

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar__same_size_tensors(self, device, dtype):
        tensors = self.get_test_data(device, dtype)
        if dtype == torch.bool:
            torch._foreach_add_(tensors, True)
            expected = torch.ones(self.H, self.W, device=device, dtype=dtype)
        else:
            torch._foreach_add_(tensors, 1)
            expected = torch.ones(self.H, self.W, device=device, dtype=dtype).add_(1)

        for t in tensors:
            self.assertEqual(t, expected)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sub_scalar_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            # Subtraction, the `-` operator, with a bool tensor is not supported.
            return

        tensors = self.get_test_data(device, dtype)
        res = torch._foreach_sub(tensors, 1)
        for t in res:
            if dtype == torch.bool and device == 'cpu':
                dtype = torch.int64
            self.assertEqual(t, torch.zeros(self.H, self.W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_sub_scalar__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            # Subtraction, the `-` operator, with a bool tensor is not supported.
            return

        tensors = self.get_test_data(device, dtype)
        torch._foreach_sub_(tensors, 1)
        for t in tensors:
            if dtype == torch.bool and device == 'cpu':
                dtype = torch.int64
            self.assertEqual(t, torch.zeros(self.H, self.W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_mul_scalar_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        tensors = self.get_test_data(device, dtype)
        res = torch._foreach_mul(tensors, 3)
        for t in res:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype).mul(3))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_div_scalar__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        tensors = self.get_test_data(device, dtype)
        torch._foreach_mul_(tensors, 3)
        for t in tensors:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype).mul(3))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_div_scalar_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return
        
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors = self.get_test_data(device, dtype)
        res = torch._foreach_div(tensors, 2)
        for t in res:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype).div(2))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_div_scalar__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors = self.get_test_data(device, dtype)
        torch._foreach_div_(tensors, 2)
        for t in tensors:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype).div(2))

    #
    # Test binary ops with list
    #
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list_same_size_tensors(self, device, dtype):
        N = 20
        H = 20
        W = 20
        tensors1 = []
        tensors2 = []
        for _ in range(N):
            tensors1.append(torch.zeros(self.H, self.W, device=device, dtype=dtype))
            tensors2.append(torch.ones(self.H, self.W, device=device, dtype=dtype))


        res = torch._foreach_add(tensors1, tensors2)
        for t in res:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_list__same_size_tensors(self, device, dtype):
        N = 20
        H = 20
        W = 20
        tensors1 = []
        tensors2 = []
        for _ in range(N):
            tensors1.append(torch.zeros(self.H, self.W, device=device, dtype=dtype))
            tensors2.append(torch.ones(self.H, self.W, device=device, dtype=dtype))


        torch._foreach_add_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype))

    #
    # Test error cases
    #
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_empty_list(self, device, dtype):
        tensors = []
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors, 1)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_overlapping_tensors(self, device, dtype):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        with self.assertRaisesRegex(RuntimeError, r"Only non overlapping and dense tensors are supported."):
            torch._foreach_add(tensors, 1)

instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
