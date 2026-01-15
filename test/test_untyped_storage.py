import torch

from torch.testing._internal.common_utils import TestCase, run_tests


class TestUntypedStorageSetItem(TestCase):
    def test_setitem_and_getitem(self):
        storage = torch.UntypedStorage(5)

        storage[0] = 10
        storage[1] = 20
        storage[4] = 99

        self.assertEqual(storage[0], 10)
        self.assertEqual(storage[1], 20)
        self.assertEqual(storage[4], 99)

    def test_out_of_bounds(self):
        storage = torch.UntypedStorage(3)
        with self.assertRaises(RuntimeError):
            storage[3] = 1

    def test_negative_index(self):
        storage = torch.UntypedStorage(3)
        with self.assertRaises(RuntimeError):
            storage[-1] = 7


if __name__ == "__main__":
    run_tests()
