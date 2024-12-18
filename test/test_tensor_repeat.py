# Owner(s): ["module: tensor"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, instantiate_parametrized_tests

class TestRepeat(TestCase):
    def test_repeat_negative_sizes(self):
        x = torch.tensor([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "negative.*repeat sizes"):
            x.repeat(-1)
        with self.assertRaisesRegex(ValueError, "negative.*repeat sizes"):
            x.repeat(2, -3)
        with self.assertRaisesRegex(ValueError, "negative.*repeat sizes"):
            x.repeat([-2])

    def test_repeat_non_integer_sizes(self):
        x = torch.tensor([1, 2, 3])
        with self.assertRaisesRegex(ValueError, "non-integer repeat sizes"):
            x.repeat(2.5)
        with self.assertRaisesRegex(ValueError, "non-integer repeat sizes"):
            x.repeat([2, 3.5])

    def test_repeat_sequence_handling(self):
        x = torch.tensor([1, 2, 3])
        self.assertEqual(x.repeat((2, 3)).shape, (2, 9))
        self.assertEqual(x.repeat([2, 3]).shape, (2, 9))
        self.assertEqual(x.repeat(torch.Size([2, 3])).shape, (2, 9))

    def test_repeat_insufficient_dims(self):
        x = torch.tensor([[1, 2], [3, 4]])
        with self.assertRaisesRegex(RuntimeError, "Invalid number of repeat dimensions"):
            x.repeat(2)

    def test_repeat_overflow(self):
        x = torch.tensor([1])
        huge_size = torch.iinfo(torch.int64).max
        with self.assertRaisesRegex(ValueError, f"Individual repeat size {huge_size} is too large"):
            x.repeat(huge_size, huge_size)

    def test_repeat_valid_cases(self):
        x = torch.tensor([1, 2, 3])
        self.assertEqual(x.repeat(2, 3).shape, (2, 9))
        self.assertEqual(x.repeat([2, 3]).shape, (2, 9))

        y = torch.tensor([[1, 2], [3, 4]])
        self.assertEqual(y.repeat(2, 1, 2).shape, (2, 2, 4))

        self.assertEqual(x.repeat(1).shape, (3,))

instantiate_parametrized_tests(TestRepeat)

if __name__ == '__main__':
    run_tests()