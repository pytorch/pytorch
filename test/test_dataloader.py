import math
import sys
import torch
import traceback
import unittest
from torch.utils.data import Dataset, TensorDataset, DataLoader
from common import TestCase, run_tests, TEST_NUMPY
from common_nn import TEST_CUDA


class TestTensorDataset(TestCase):

    def test_len(self):
        source = TensorDataset(torch.randn(15, 10, 2, 3, 4, 5), torch.randperm(15))
        self.assertEqual(len(source), 15)

    def test_getitem(self):
        t = torch.randn(15, 10, 2, 3, 4, 5)
        l = torch.randn(15, 10)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])

    def test_getitem_1d(self):
        t = torch.randn(15)
        l = torch.randn(15)
        source = TensorDataset(t, l)
        for i in range(15):
            self.assertEqual(t[i], source[i][0])
            self.assertEqual(l[i], source[i][1])


class ErrorDataset(Dataset):

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class TestDataLoader(TestCase):

    def setUp(self):
        self.data = torch.randn(100, 2, 3, 5)
        self.labels = torch.randperm(50).repeat(2)
        self.dataset = TensorDataset(self.data, self.labels)

    def _test_sequential(self, loader):
        batch_size = loader.batch_size
        for i, (sample, target) in enumerate(loader):
            idx = i * batch_size
            self.assertEqual(sample, self.data[idx:idx + batch_size])
            self.assertEqual(target, self.labels[idx:idx + batch_size])
        self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_shuffle(self, loader):
        found_data = {i: 0 for i in range(self.data.size(0))}
        found_labels = {i: 0 for i in range(self.labels.size(0))}
        batch_size = loader.batch_size
        for i, (batch_samples, batch_targets) in enumerate(loader):
            for sample, target in zip(batch_samples, batch_targets):
                for data_point_idx, data_point in enumerate(self.data):
                    if data_point.eq(sample).all():
                        self.assertFalse(found_data[data_point_idx])
                        found_data[data_point_idx] += 1
                        break
                self.assertEqual(target, self.labels[data_point_idx])
                found_labels[data_point_idx] += 1
            self.assertEqual(sum(found_data.values()), (i + 1) * batch_size)
            self.assertEqual(sum(found_labels.values()), (i + 1) * batch_size)
        self.assertEqual(i, math.floor((len(self.dataset) - 1) / batch_size))

    def _test_error(self, loader):
        it = iter(loader)
        errors = 0
        while True:
            try:
                it.next()
            except NotImplementedError:
                errors += 1
            except StopIteration:
                self.assertEqual(errors,
                                 math.ceil(float(len(loader.dataset)) / loader.batch_size))
                return

    def test_sequential(self):
        self._test_sequential(DataLoader(self.dataset))

    def test_sequential_batch(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_sequential_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    def test_shuffle(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True))

    def test_shuffle_batch(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True))

    def test_sequential_workers(self):
        self._test_sequential(DataLoader(self.dataset, num_workers=4))

    def test_seqential_batch_workers(self):
        self._test_sequential(DataLoader(self.dataset, batch_size=2, num_workers=4))

    def test_shuffle_workers(self):
        self._test_shuffle(DataLoader(self.dataset, shuffle=True, num_workers=4))

    def test_shuffle_batch_workers(self):
        self._test_shuffle(DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for input, target in loader:
            self.assertTrue(input.is_pinned())
            self.assertTrue(target.is_pinned())

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy(self):
        import numpy as np

        class TestDataset(torch.utils.data.Dataset):
            def __getitem__(self, i):
                return np.ones((2, 3, 4)) * i

            def __len__(self):
                return 1000

        loader = DataLoader(TestDataset(), batch_size=12)
        batch = next(iter(loader))
        self.assertIsInstance(batch, torch.DoubleTensor)
        self.assertEqual(batch.size(), torch.Size([12, 2, 3, 4]))

    def test_error(self):
        self._test_error(DataLoader(ErrorDataset(100), batch_size=2, shuffle=True))

    def test_error_workers(self):
        self._test_error(DataLoader(ErrorDataset(41), batch_size=2, shuffle=True, num_workers=4))

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_partial_workers(self):
        "check that workers exit even if the iterator is not exhausted"
        loader = iter(DataLoader(self.dataset, batch_size=2, num_workers=4, pin_memory=True))
        workers = loader.workers
        pin_thread = loader.pin_thread
        for i, sample in enumerate(loader):
            if i == 3:
                break
        del loader
        for w in workers:
            w.join(1.0)  # timeout of one second
            self.assertFalse(w.is_alive(), 'subprocess not terminated')
            self.assertEqual(w.exitcode, 0)
        pin_thread.join(1.0)
        self.assertFalse(pin_thread.is_alive())

    def test_len(self):
        def check_len(dl, expected):
            self.assertEqual(len(dl), expected)
            n = 0
            for sample in dl:
                n += 1
            self.assertEqual(n, expected)
        check_len(self.dataset, 100)
        check_len(DataLoader(self.dataset, batch_size=2), 50)
        check_len(DataLoader(self.dataset, batch_size=3), 34)


if __name__ == '__main__':
    run_tests()
