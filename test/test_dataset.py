import pickle
import tempfile
import warnings

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.utils.data import IterableDataset, RandomSampler
from torch.utils.data.datasets import \
    (CallableIterableDataset, CollateIterableDataset, BatchIterableDataset,
     ListDirFilesIterableDataset, LoadFilesFromDiskIterableDataset, SamplerIterableDataset)
from typing import List, Tuple, Dict, Any, Type


def create_temp_dir_and_files():
    # The temp dir and files within it will be released and deleted in tearDown().
    # Adding `noqa: P201` to avoid mypy's warning on not releasing the dir handle within this function.
    temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
    temp_dir_path = temp_dir.name
    temp_file1 = tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False)  # noqa: P201
    temp_file2 = tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False)  # noqa: P201
    temp_file3 = tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False)  # noqa: P201

    return (temp_dir, temp_file1.name, temp_file2.name, temp_file3.name)


class TestIterableDatasetBasic(TestCase):

    def setUp(self):
        ret = create_temp_dir_and_files()
        self.temp_dir = ret[0]
        self.temp_files = ret[1:]

    def tearDown(self):
        try:
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn("TestIterableDatasetBasic was not able to cleanup temp dir due to {}".format(str(e)))

    def test_listdirfiles_iterable_dataset(self):
        temp_dir = self.temp_dir.name
        dataset = ListDirFilesIterableDataset(temp_dir, '')
        for pathname in dataset:
            self.assertTrue(pathname in self.temp_files)

    def test_loadfilesfromdisk_iterable_dataset(self):
        temp_dir = self.temp_dir.name
        dataset1 = ListDirFilesIterableDataset(temp_dir, '')
        dataset2 = LoadFilesFromDiskIterableDataset(dataset1)

        for rec in dataset2:
            self.assertTrue(rec[0] in self.temp_files)
            self.assertTrue(rec[1].read() == open(rec[0], 'rb').read())


class IterDatasetWithoutLen(IterableDataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def __iter__(self):
        for i in self.ds:
            yield i


class IterDatasetWithLen(IterableDataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds
        self.length = len(ds)

    def __iter__(self):
        for i in self.ds:
            yield i

    def __len__(self):
        return self.length


def _fake_fn(self, data, *args, **kwargs):
    return data


class TestFunctionalIterableDataset(TestCase):

    def test_picklable(self):
        arr = range(10)
        picklable_datasets: List[Tuple[Type[IterableDataset], IterableDataset, List, Dict[str, Any]]] = [
            (CallableIterableDataset, IterDatasetWithLen(arr), [], {}),
            (CallableIterableDataset, IterDatasetWithLen(arr), [0], {'fn': _fake_fn, 'test': True}),
            (CollateIterableDataset, IterDatasetWithLen(arr), [], {}),
            (CollateIterableDataset, IterDatasetWithLen(arr), [0], {'collate_fn': _fake_fn, 'test': True}),
        ]
        for ds, d, args, kargs in picklable_datasets:
            p = pickle.dumps(ds(d, *args, **kargs))  # type: ignore

        unpicklable_datasets: List[Tuple[Type[IterableDataset], IterableDataset, List, Dict[str, Any]]] = [
            (CallableIterableDataset, IterDatasetWithLen(arr), [], {'fn': lambda x: x}),
            (CollateIterableDataset, IterDatasetWithLen(arr), [], {'collate_fn': lambda x: x}),
        ]
        for ds, d, args, kargs in unpicklable_datasets:
            with self.assertRaises(AttributeError):
                p = pickle.dumps(ds(d, *args, **kargs))  # type: ignore

    def test_callable_dataset(self):
        arr = range(10)
        ds = IterDatasetWithLen(arr)
        ds_nl = IterDatasetWithoutLen(arr)

        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        callable_ds = CallableIterableDataset(ds, fn=fn)  # type: ignore
        self.assertEqual(len(ds), len(callable_ds))
        for x, y in zip(callable_ds, ds):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        callable_ds = CallableIterableDataset(ds, torch.int, fn=fn, sum=True)  # type: ignore
        self.assertEqual(len(ds), len(callable_ds))
        for x, y in zip(callable_ds, ds):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        callable_ds_nl = CallableIterableDataset(ds_nl)  # type: ignore
        with self.assertRaises(NotImplementedError):
            len(callable_ds_nl)
        for x, y in zip(callable_ds_nl, ds_nl):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

    def test_collate_dataset(self):
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ds = IterDatasetWithLen(arrs)
        ds_nl = IterDatasetWithoutLen(arrs)

        def _collate_fn(batch):
            return torch.tensor(sum(batch), dtype=torch.float)

        collate_ds = CollateIterableDataset(ds, collate_fn=_collate_fn)
        self.assertEqual(len(ds), len(collate_ds))
        for x, y in zip(collate_ds, ds):
            self.assertEqual(x, torch.tensor(sum(y), dtype=torch.float))

        collate_ds_nl = CollateIterableDataset(ds_nl)  # type: ignore
        with self.assertRaises(NotImplementedError):
            len(collate_ds_nl)
        for x, y in zip(collate_ds_nl, ds_nl):
            self.assertEqual(x, torch.tensor(y))

    def test_batch_dataset(self):
        arrs = range(10)
        ds = IterDatasetWithLen(arrs)
        with self.assertRaises(AssertionError):
            batch_ds0 = BatchIterableDataset(ds, batch_size=0)

        # Default not drop the last batch
        batch_ds1 = BatchIterableDataset(ds, batch_size=3)
        self.assertEqual(len(batch_ds1), 4)
        batch_iter = iter(batch_ds1)
        value = 0
        for i in range(len(batch_ds1)):
            batch = next(batch_iter)
            if i == 3:
                self.assertEqual(len(batch), 1)
                self.assertEqual(batch, [9])
            else:
                self.assertEqual(len(batch), 3)
                for x in batch:
                    self.assertEqual(x, value)
                    value += 1

        # Drop the last batch
        batch_ds2 = BatchIterableDataset(ds, batch_size=3, drop_last=True)
        self.assertEqual(len(batch_ds2), 3)
        value = 0
        for batch in batch_ds2:
            self.assertEqual(len(batch), 3)
            for x in batch:
                self.assertEqual(x, value)
                value += 1

        batch_ds3 = BatchIterableDataset(ds, batch_size=2)
        self.assertEqual(len(batch_ds3), 5)
        batch_ds4 = BatchIterableDataset(ds, batch_size=2, drop_last=True)
        self.assertEqual(len(batch_ds4), 5)

        ds_nolen = IterDatasetWithoutLen(arrs)
        batch_ds_nolen = BatchIterableDataset(ds_nolen, batch_size=5)
        with self.assertRaises(NotImplementedError):
            len(batch_ds_nolen)

    def test_sampler_dataset(self):
        arrs = range(10)
        ds = IterDatasetWithLen(arrs)
        # Default SequentialSampler
        sampled_ds = SamplerIterableDataset(ds)  # type: ignore
        self.assertEqual(len(sampled_ds), 10)
        i = 0
        for x in sampled_ds:
            self.assertEqual(x, i)
            i += 1

        # RandomSampler
        random_sampled_ds = SamplerIterableDataset(ds, sampler=RandomSampler, replacement=True)  # type: ignore

        # Requires `__len__` to build SamplerDataset
        ds_nolen = IterDatasetWithoutLen(arrs)
        with self.assertRaises(AssertionError):
            sampled_ds = SamplerIterableDataset(ds_nolen)


if __name__ == '__main__':
    run_tests()
