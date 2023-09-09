# Owner(s): ["module: complex"]

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    dtypes,
    onlyCPU,
)
from torch.testing._internal.common_utils import TestCase, run_tests, set_default_dtype
from torch.testing._internal.common_dtype import complex_types

devices = (torch.device('cpu'), torch.device('cuda:0'))

class TestComplexTensor(TestCase):
    @dtypes(*complex_types())
    def test_to_list(self, device, dtype):
        # test that the complex float tensor has expected values and
        # there's no garbage value in the resultant list
        self.assertEqual(torch.zeros((2, 2), device=device, dtype=dtype).tolist(), [[0j, 0j], [0j, 0j]])

    @dtypes(torch.float32, torch.float64)
    def test_dtype_inference(self, device, dtype):
        # issue: https://github.com/pytorch/pytorch/issues/36834
        with set_default_dtype(dtype):
            x = torch.tensor([3., 3. + 5.j], device=device)
        self.assertEqual(x.dtype, torch.cdouble if dtype == torch.float64 else torch.cfloat)

    @onlyCPU
    @dtypes(*complex_types())
    def test_eq(self, device, dtype):
        "Test eq on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
             torch.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype)),
            (torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
             torch.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype)),
            (torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
             torch.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype)),
        ):
            actual = torch.eq(a, b)
            expected = torch.tensor([False], device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}")

            actual = torch.eq(a, a)
            expected = torch.tensor([True], device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, b, out=actual)
            expected = torch.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, a, out=actual)
            expected = torch.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}")

        # Vectorized operations
        for a, b in (
            (torch.tensor([
                -0.0610 - 2.1172j, 5.1576 + 5.4775j, complex(2.8871, nan), -6.6545 - 3.7655j, -2.7036 - 1.4470j, 0.3712 + 7.989j,
                -0.0610 - 2.1172j, 5.1576 + 5.4775j, complex(nan, -3.2650), -6.6545 - 3.7655j, -2.7036 - 1.4470j, 0.3712 + 7.989j],
                device=device, dtype=dtype),
             torch.tensor([
                -6.1278 - 8.5019j, 0.5886 + 8.8816j, complex(2.8871, nan), 6.3505 + 2.2683j, 0.3712 + 7.9659j, 0.3712 + 7.989j,
                -6.1278 - 2.1172j, 5.1576 + 8.8816j, complex(nan, -3.2650), 6.3505 + 2.2683j, 0.3712 + 7.9659j, 0.3712 + 7.989j],
                device=device, dtype=dtype)),
        ):
            actual = torch.eq(a, b)
            expected = torch.tensor([False, False, False, False, False, True,
                                    False, False, False, False, False, True],
                                    device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}")

            actual = torch.eq(a, a)
            expected = torch.tensor([True, True, False, True, True, True,
                                    True, True, False, True, True, True],
                                    device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\neq\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, b, out=actual)
            expected = torch.tensor([complex(0), complex(0), complex(0), complex(0), complex(0), complex(1),
                                    complex(0), complex(0), complex(0), complex(0), complex(0), complex(1)],
                                    device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.eq(a, a, out=actual)
            expected = torch.tensor([complex(1), complex(1), complex(0), complex(1), complex(1), complex(1),
                                    complex(1), complex(1), complex(0), complex(1), complex(1), complex(1)],
                                    device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\neq(out)\nactual {actual}\nexpected {expected}")

    @onlyCPU
    @dtypes(*complex_types())
    def test_ne(self, device, dtype):
        "Test ne on complex types"
        nan = float("nan")
        # Non-vectorized operations
        for a, b in (
            (torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
             torch.tensor([-6.1278 - 8.5019j], device=device, dtype=dtype)),
            (torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
             torch.tensor([-6.1278 - 2.1172j], device=device, dtype=dtype)),
            (torch.tensor([-0.0610 - 2.1172j], device=device, dtype=dtype),
             torch.tensor([-0.0610 - 8.5019j], device=device, dtype=dtype)),
        ):
            actual = torch.ne(a, b)
            expected = torch.tensor([True], device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}")

            actual = torch.ne(a, a)
            expected = torch.tensor([False], device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, b, out=actual)
            expected = torch.tensor([complex(1)], device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, a, out=actual)
            expected = torch.tensor([complex(0)], device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}")

        # Vectorized operations
        for a, b in (
            (torch.tensor([
                -0.0610 - 2.1172j, 5.1576 + 5.4775j, complex(2.8871, nan), -6.6545 - 3.7655j, -2.7036 - 1.4470j, 0.3712 + 7.989j,
                -0.0610 - 2.1172j, 5.1576 + 5.4775j, complex(nan, -3.2650), -6.6545 - 3.7655j, -2.7036 - 1.4470j, 0.3712 + 7.989j],
                device=device, dtype=dtype),
             torch.tensor([
                -6.1278 - 8.5019j, 0.5886 + 8.8816j, complex(2.8871, nan), 6.3505 + 2.2683j, 0.3712 + 7.9659j, 0.3712 + 7.989j,
                -6.1278 - 2.1172j, 5.1576 + 8.8816j, complex(nan, -3.2650), 6.3505 + 2.2683j, 0.3712 + 7.9659j, 0.3712 + 7.989j],
                device=device, dtype=dtype)),
        ):
            actual = torch.ne(a, b)
            expected = torch.tensor([True, True, True, True, True, False,
                                    True, True, True, True, True, False],
                                    device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}")

            actual = torch.ne(a, a)
            expected = torch.tensor([False, False, True, False, False, False,
                                    False, False, True, False, False, False],
                                    device=device, dtype=torch.bool)
            self.assertEqual(actual, expected, msg=f"\nne\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, b, out=actual)
            expected = torch.tensor([complex(1), complex(1), complex(1), complex(1), complex(1), complex(0),
                                    complex(1), complex(1), complex(1), complex(1), complex(1), complex(0)],
                                    device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}")

            actual = torch.full_like(b, complex(2, 2))
            torch.ne(a, a, out=actual)
            expected = torch.tensor([complex(0), complex(0), complex(1), complex(0), complex(0), complex(0),
                                    complex(0), complex(0), complex(1), complex(0), complex(0), complex(0)],
                                    device=device, dtype=dtype)
            self.assertEqual(actual, expected, msg=f"\nne(out)\nactual {actual}\nexpected {expected}")

instantiate_device_type_tests(TestComplexTensor, globals())

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
