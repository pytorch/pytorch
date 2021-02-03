import pickle
import random
import tempfile
import warnings

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.utils.data import IterableDataset, RandomSampler
from torch.utils.data.datasets import \
    (CallableIterableDataset, CollateIterableDataset, BatchIterableDataset, BucketBatchIterableDataset,
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
        arrs = list(range(10))
        ds = IterDatasetWithLen(arrs)
        with self.assertRaises(AssertionError):
            batch_ds0 = BatchIterableDataset(ds, batch_size=0)

        # Default not drop the last batch
        bs = 3
        batch_ds1 = BatchIterableDataset(ds, batch_size=bs)
        self.assertEqual(len(batch_ds1), 4)
        for i, batch in enumerate(batch_ds1):
            self.assertEqual(len(batch), 1 if i == 3 else bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        # Drop the last batch
        bs = 4
        batch_ds2 = BatchIterableDataset(ds, batch_size=bs, drop_last=True)
        self.assertEqual(len(batch_ds2), 2)
        for i, batch in enumerate(batch_ds2):
            self.assertEqual(len(batch), bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        ds_nl = IterDatasetWithoutLen(range(10))
        batch_ds_nl = BatchIterableDataset(ds_nl, batch_size=2)
        with self.assertRaises(NotImplementedError):
            len(batch_ds_nl)

    def test_bucket_batch_dataset(self):
        ds = IterDatasetWithLen(range(20))
        with self.assertRaises(AssertionError):
            BucketBatchIterableDataset(ds, batch_size=0)

        ds_nl = IterDatasetWithoutLen(range(20))
        bucket_ds_nl = BucketBatchIterableDataset(ds_nl, batch_size=7)
        with self.assertRaises(NotImplementedError):
            len(bucket_ds_nl)

        # Test Bucket Batch without sort_key
        def _helper(**kwargs):
            arrs = list(range(100))
            random.shuffle(arrs)
            ds = IterDatasetWithLen(arrs)
            bucket_ds = BucketBatchIterableDataset(ds, **kwargs)
            if kwargs["sort_key"] is None:
                # BatchDataset as reference
                ref_ds = BatchIterableDataset(ds, batch_size=kwargs['batch_size'], drop_last=kwargs['drop_last'])
                for batch, rbatch in zip(bucket_ds, ref_ds):
                    self.assertEqual(batch, rbatch)
            else:
                bucket_size = bucket_ds.bucket_size
                bucket_num = (len(ds) - 1) // bucket_size + 1
                it = iter(bucket_ds)
                for i in range(bucket_num):
                    ref = sorted(arrs[i * bucket_size: (i + 1) * bucket_size])
                    bucket: List = []
                    while len(bucket) < len(ref):
                        try:
                            batch = next(it)
                            bucket += batch
                        # If drop last, stop in advance
                        except StopIteration:
                            break
                    if len(bucket) != len(ref):
                        ref = ref[:len(bucket)]
                    # Sorted bucket
                    self.assertEqual(bucket, ref)

        _helper(batch_size=7, drop_last=False, sort_key=None)
        _helper(batch_size=7, drop_last=True, bucket_size_mul=5, sort_key=None)

        # Test Bucket Batch with sort_key
        def _sort_fn(data):
            return data

        _helper(batch_size=7, drop_last=False, bucket_size_mul=5, sort_key=_sort_fn)
        _helper(batch_size=7, drop_last=True, bucket_size_mul=5, sort_key=_sort_fn)

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
