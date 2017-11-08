import math
import sys
import torch
import traceback
import unittest
from torch.utils.data import Dataset, TensorDataset, DataLoader, ConcatDataset
from torch.utils.data.dataloader import default_collate
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


class TestConcatDataset(TestCase):

    def test_concat_two_singletons(self):
        result = ConcatDataset([[0], [1]])
        self.assertEqual(2, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(1, result[1])

    def test_concat_two_non_singletons(self):
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_two_non_singletons_with_empty(self):
        # Adding an empty dataset somewhere is correctly handled
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [],
                                [5, 6, 7, 8, 9]])
        self.assertEqual(10, len(result))
        self.assertEqual(0, result[0])
        self.assertEqual(5, result[5])

    def test_concat_raises_index_error(self):
        result = ConcatDataset([[0, 1, 2, 3, 4],
                                [5, 6, 7, 8, 9]])
        with self.assertRaises(IndexError):
            # this one goes to 11
            result[11]

    def test_add_dataset(self):
        d1 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d2 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        d3 = TensorDataset(torch.rand(7, 3, 28, 28), torch.rand(7))
        result = d1 + d2 + d3
        self.assertEqual(21, len(result))
        self.assertEqual(0, (d1[0][0] - result[0][0]).abs().sum())
        self.assertEqual(0, (d2[0][0] - result[7][0]).abs().sum())
        self.assertEqual(0, (d3[0][0] - result[14][0]).abs().sum())


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
                next(it)
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

    def test_growing_dataset(self):
        dataset = [torch.ones(4) for _ in range(4)]
        dataloader_seq = DataLoader(dataset, shuffle=False)
        dataloader_shuffle = DataLoader(dataset, shuffle=True)
        dataset.append(torch.ones(4))
        self.assertEqual(len(dataloader_seq), 5)
        self.assertEqual(len(dataloader_shuffle), 5)

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

    def _test_batch_sampler(self, **kwargs):
        # [(0, 1), (2, 3, 4), (5, 6), (7, 8, 9), ...]
        batches = []
        for i in range(0, 100, 5):
            batches.append(tuple(range(i, i + 2)))
            batches.append(tuple(range(i + 2, i + 5)))

        dl = DataLoader(self.dataset, batch_sampler=batches, **kwargs)
        self.assertEqual(len(dl), 40)
        for i, (input, _target) in enumerate(dl):
            if i % 2 == 0:
                offset = i * 5 // 2
                self.assertEqual(len(input), 2)
                self.assertEqual(input, self.data[offset:offset + 2])
            else:
                offset = i * 5 // 2
                self.assertEqual(len(input), 3)
                self.assertEqual(input, self.data[offset:offset + 3])

    def test_batch_sampler(self):
        self._test_batch_sampler()
        self._test_batch_sampler(num_workers=4)

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

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_numpy_scalars(self):
        import numpy as np

        class ScalarDataset(torch.utils.data.Dataset):
            def __init__(self, dtype):
                self.dtype = dtype

            def __getitem__(self, i):
                return self.dtype()

            def __len__(self):
                return 4

        dtypes = {
            np.float64: torch.DoubleTensor,
            np.float32: torch.FloatTensor,
            np.float16: torch.HalfTensor,
            np.int64: torch.LongTensor,
            np.int32: torch.IntTensor,
            np.int16: torch.ShortTensor,
            np.int8: torch.CharTensor,
            np.uint8: torch.ByteTensor,
        }
        for dt, tt in dtypes.items():
            dset = ScalarDataset(dt)
            loader = DataLoader(dset, batch_size=2)
            batch = next(iter(loader))
            self.assertIsInstance(batch, tt)

    @unittest.skipIf(not TEST_NUMPY, "numpy unavailable")
    def test_default_colate_bad_numpy_types(self):
        import numpy as np

        # Should be a no-op
        arr = np.array(['a', 'b', 'c'])
        default_collate(arr)

        arr = np.array([[['a', 'b', 'c']]])
        self.assertRaises(TypeError, lambda: default_collate(arr))

        arr = np.array([object(), object(), object()])
        self.assertRaises(TypeError, lambda: default_collate(arr))

        arr = np.array([[[object(), object(), object()]]])
        self.assertRaises(TypeError, lambda: default_collate(arr))


class StringDataset(Dataset):
    def __init__(self):
        self.s = '12345'

    def __len__(self):
        return len(self.s)

    def __getitem__(self, ndx):
        return (self.s[ndx], ndx)


class TestStringDataLoader(TestCase):
    def setUp(self):
        self.dataset = StringDataset()

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_shuffle_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
        for batch_ndx, (s, n) in enumerate(loader):
            self.assertIsInstance(s[0], str)
            self.assertTrue(n.is_pinned())


class DictDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, ndx):
        return {
            'a_tensor': torch.Tensor(4, 2).fill_(ndx),
            'another_dict': {
                'a_number': ndx,
            },
        }


class TestDictDataLoader(TestCase):
    def setUp(self):
        self.dataset = DictDataset()

    def test_sequential_batch(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        batch_size = loader.batch_size
        for i, sample in enumerate(loader):
            idx = i * batch_size
            self.assertEqual(set(sample.keys()), {'a_tensor', 'another_dict'})
            self.assertEqual(set(sample['another_dict'].keys()), {'a_number'})

            t = sample['a_tensor']
            self.assertEqual(t.size(), torch.Size([batch_size, 4, 2]))
            self.assertTrue((t[0] == idx).all())
            self.assertTrue((t[1] == idx + 1).all())

            n = sample['another_dict']['a_number']
            self.assertEqual(n.size(), torch.Size([batch_size]))
            self.assertEqual(n[0], idx)
            self.assertEqual(n[1], idx + 1)

    @unittest.skipIf(not TEST_CUDA, "CUDA unavailable")
    def test_pin_memory(self):
        loader = DataLoader(self.dataset, batch_size=2, pin_memory=True)
        for batch_ndx, sample in enumerate(loader):
            self.assertTrue(sample['a_tensor'].is_pinned())
            self.assertTrue(sample['another_dict']['a_number'].is_pinned())


if __name__ == '__main__':
    run_tests()
