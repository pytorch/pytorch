import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes

class TestForeach(TestCase):
    N = 20
    H = 20
    W = 20

    def get_test_data(self, device, dtype):
        tensors = []
        for _ in range(self.N):
            tensors.append(torch.ones(self.H, self.W, device=device, dtype=dtype))

        return tensors

    # Unary ops
    @dtypes(*[torch.int32, torch.half, torch.float, torch.double, torch.complex64, torch.complex128])
    def test_sqrt(self, device, dtype):
        if dtype in [torch.bool, torch.int,torch.half,]:
            return
        tensors = [torch.ones(self.H, self.W, device=device, dtype=dtype) for n in range(self.N)]

        res = torch._foreach_sqrt(tensors)
        torch._foreach_sqrt_(tensors)

        self.assertEqual([torch.sqrt(torch.ones(self.H, self.W, device=device, dtype=dtype)) for n in range(self.N)], res)
        self.assertEqual(tensors, res)

    @dtypes(*[torch.int32, torch.half, torch.float, torch.double, torch.complex64, torch.complex128])
    def test_exp(self, device, dtype):
        if dtype in [torch.bool, torch.int,torch.half,]:
            return

        tensors = [torch.ones(self.H, self.W, device=device, dtype=dtype) for n in range(self.N)]

        res = torch._foreach_exp(tensors)
        torch._foreach_exp_(tensors)

        self.assertEqual([torch.exp(torch.ones(self.H, self.W, device=device, dtype=dtype)) for n in range(self.N)], res)
        self.assertEqual(tensors, res)

    # Ops with scalar
    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar__same_size_tensors(self, device, dtype):
        tensors = [torch.zeros(self.H, self.W, device=device, dtype=dtype) for n in range(self.N)]

        # bool tensor + 1 will result in int64 tensor
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
        tensors = [torch.zeros(self.H, self.W, device=device, dtype=dtype) for n in range(self.N)]

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
    def test_mul_scalar__same_size_tensors(self, device, dtype):
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

    # Ops with list
    @dtypes(*torch.testing.get_all_dtypes())
    def test_bin_op_list_same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors1 = []
        tensors2 = []
        for _ in range(self.N):
            tensors1.append(torch.zeros(self.H, self.W, device=device, dtype=dtype))
            tensors2.append(torch.ones(self.H, self.W, device=device, dtype=dtype))

        res = torch._foreach_mul(tensors1, tensors2)
        for t in res:
            self.assertEqual(t, torch.zeros(self.H, self.W, device=device, dtype=dtype))

        res = torch._foreach_div(torch._foreach_add(tensors1, 4), torch._foreach_mul(tensors2, 2))
        for t in res:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype).mul(2))

        res = torch._foreach_add(tensors1, tensors2)
        for t in res:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype))

        res = torch._foreach_sub(res, tensors2)
        for t in res:
            self.assertEqual(t, torch.zeros(self.H, self.W, device=device, dtype=dtype))

    @dtypes(*torch.testing.get_all_dtypes())
    def test_bin_op_list_different_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors1 = []
        tensors2 = []
        size_change = 0
        for _ in range(self.N):
            tensors1.append(torch.zeros(self.H + size_change, self.W + size_change, device=device, dtype=dtype))
            tensors2.append(torch.ones(self.H + size_change, self.W + size_change, device=device, dtype=dtype))
            size_change += 1

        res = torch._foreach_mul(tensors1, tensors2)
        size_change = 0
        for t in res:
            self.assertEqual(t, torch.zeros(self.H + size_change, self.W + size_change, device=device, dtype=dtype))
            size_change += 1

        res = torch._foreach_div(torch._foreach_add(tensors1, 4), torch._foreach_mul(tensors2, 2))
        size_change = 0
        for t in res:
            self.assertEqual(t, torch.ones(self.H + size_change, self.W + size_change, device=device, dtype=dtype).mul(2))
            size_change += 1

        res = torch._foreach_add(tensors1, tensors2)
        size_change = 0
        for t in res:
            self.assertEqual(t, torch.ones(self.H + size_change, self.W + size_change, device=device, dtype=dtype))
            size_change += 1

        res = torch._foreach_sub(res, tensors2)
        size_change = 0
        for t in res:
            self.assertEqual(t, torch.zeros(self.H + size_change, self.W + size_change, device=device, dtype=dtype))
            size_change += 1

    @dtypes(*torch.testing.get_all_dtypes())
    def test_bin_op_list__same_size_tensors(self, device, dtype):
        if dtype == torch.bool:
            return

        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            # Integer division of tensors using div or / is no longer supported
            return

        tensors1 = []
        tensors2 = []
        for _ in range(self.N):
            tensors1.append(torch.zeros(self.H, self.W, device=device, dtype=dtype))
            tensors2.append(torch.ones(self.H, self.W, device=device, dtype=dtype))


        torch._foreach_add_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype))

        torch._foreach_sub_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.zeros(self.H, self.W, device=device, dtype=dtype))

        torch._foreach_mul_(tensors1, tensors2)
        for t in tensors1:
            self.assertEqual(t, torch.zeros(self.H, self.W, device=device, dtype=dtype))

        torch._foreach_div_(torch._foreach_add_(tensors1, 4), torch._foreach_add_(tensors2, 1))
        for t in tensors1:
            self.assertEqual(t, torch.ones(self.H, self.W, device=device, dtype=dtype).mul(2))

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
