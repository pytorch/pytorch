import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

class TestForeach(TestCase):
    N = 20
    H = 20
    W = 20

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar__same_size_tensors(self, device, dtype):
        tensors = [torch.zeros(self.H, self.W, device=device, dtype=dtype) for _ in range(self.N)]

        # inplace addition of 1 to bool fails
        if dtype == torch.bool:
            torch._foreach_add_(tensors, True)
        else:
            torch._foreach_add_(tensors, 1)

        for t in tensors:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_same_size_tensors(self, device, dtype):
        N = 20
        H = 20
        W = 20
        tensors = [torch.zeros(self.H, self.W, device=device, dtype=dtype) for _ in range(self.N)]

        res = torch._foreach_add(tensors, 1)
        for t in res:
            # bool tensor + 1 will result in int64 tensor
            if dtype == torch.bool:
                dtype = torch.int64
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_different_size_tensors(self, device, dtype):
        tensors = [torch.zeros(self.H + n, self.W + n, device=device, dtype=dtype) for n in range(self.N)]
        res = torch._foreach_add(tensors, 1)

        # bool tensor + 1 will result in int64 tensor
        if dtype == torch.bool:
            dtype = torch.int64
        self.assertEqual([torch.ones(self.H + n, self.W + n, device=device, dtype=dtype) for n in range(self.N)], torch._foreach_add(tensors, 1))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_empty_list_and_empty_tensor(self, device, dtype):
        # TODO: enable empty list case
        for tensors in [[torch.randn([0])]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)

            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_overlapping_tensors(self, device, dtype):
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]
        expected = [torch.tensor([[[2, 2, 2]], [[2, 2, 2]]], dtype=dtype, device=device)]

        # bool tensor + 1 will result in int64 tensor
        if dtype == torch.bool: 
            expected[0] = expected[0].to(torch.int64).add(1)

        res = torch._foreach_add(tensors, 1)
        self.assertEqual(res, expected)

    def test_add_scalar_with_different_tensor_dtypes(self, device):
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device), 
                   torch.tensor([1], dtype=torch.long, device=device)]

        expected = [torch.tensor([2.1], dtype=torch.float, device=device), 
                    torch.tensor([2], dtype=torch.long, device=device)]

        res = torch._foreach_add(tensors, 1)
        self.assertEqual(res, expected)

    def test_add_scalar_with_different_scalar_type(self, device):
        # int tensor with float scalar
        scalar = 1.1
        tensors = [torch.tensor([1], dtype=torch.int, device=device)]
        self.assertEqual([x + scalar for x in tensors], torch._foreach_add(tensors, scalar))

        # float tensor with int scalar
        scalar = 1
        tensors = [torch.tensor([1.1], device=device)]
        self.assertEqual([x + scalar for x in tensors], torch._foreach_add(tensors, scalar))

        # bool tensor with int scalar
        scalar = 1
        tensors = [torch.tensor([False], device=device)]
        self.assertEqual([x + scalar for x in tensors], torch._foreach_add(tensors, scalar))

        # bool tensor with float scalar
        scalar = 1.1
        tensors = [torch.tensor([False], device=device)]
        self.assertEqual([x + scalar for x in tensors], torch._foreach_add(tensors, scalar))

    def test_add_list_error_cases(self, device):
        tensors1 = []
        tensors2 = []

        # Empty lists
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
            torch._foreach_add_(tensors1, tensors2)

        # One empty list
        tensors1.append(torch.tensor([1], device=device))
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
            torch._foreach_add_(tensors1, tensors2)

        # Lists have different amount of tensors
        tensors2.append(torch.tensor([1], device=device))
        tensors2.append(torch.tensor([1], device=device))
        with self.assertRaises(RuntimeError):
            torch._foreach_add(tensors1, tensors2)
            torch._foreach_add_(tensors1, tensors2)

    def test_add_list_different_dtypes(self, device):
        tensors1 = []
        tensors2 = []
        for _ in range(self.N):
            tensors1.append(torch.zeros(self.H, self.W, device=device, dtype=torch.float))
            tensors2.append(torch.ones(self.H, self.W, device=device, dtype=torch.int))

        res = torch._foreach_add(tensors1, tensors2)
        torch._foreach_add_(tensors1, tensors2)
        self.assertEqual(res, tensors1)
        self.assertEqual(res[0], torch.ones(self.H, self.W, device=device, dtype=torch.float))

instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()



# TEST