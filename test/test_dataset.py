import tempfile
import warnings

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.utils.data import IterableDataset, RandomSampler
from torch.utils.data.datasets import \
    (CollateIterableDataset, BatchIterableDataset, ListDirFilesIterableDataset,
     LoadFilesFromDiskIterableDataset, SamplerIterableDataset, PaddedBatchIterableDataset)
from typing import Sequence, List, Dict, Union


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


class TestFunctionalIterableDataset(TestCase):
    def test_collate_dataset(self):
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        ds_len = IterDatasetWithLen(arrs)
        ds_nolen = IterDatasetWithoutLen(arrs)

        def _collate_fn(batch):
            return torch.tensor(sum(batch), dtype=torch.float)

        collate_ds = CollateIterableDataset(ds_len, collate_fn=_collate_fn)
        self.assertEqual(len(ds_len), len(collate_ds))
        ds_iter = iter(ds_len)
        for x in collate_ds:
            y = next(ds_iter)
            self.assertEqual(x, torch.tensor(sum(y), dtype=torch.float))

        collate_ds_nolen = CollateIterableDataset(ds_nolen)  # type: ignore
        with self.assertRaises(NotImplementedError):
            len(collate_ds_nolen)
        ds_nolen_iter = iter(ds_nolen)
        for x in collate_ds_nolen:
            y = next(ds_nolen_iter)
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

    def test_padded_batch_dataset(self):
        arrs = range(10)
        ds_nolen = IterDatasetWithoutLen(arrs)
        # Check valid batch size
        with self.assertRaises(AssertionError):
            PaddedBatchIterableDataset(ds_nolen, batch_size=0)
        # Check __len__ function 
        with self.assertRaises(NotImplementedError):
            len(PaddedBatchIterableDataset(ds_nolen, batch_size=2))
        # Check same type
        ds_len = IterDatasetWithLen([3, '123', torch.randn(3, 3)])
        with self.assertRaises(RuntimeError):
            pb = PaddedBatchIterableDataset(ds_len, batch_size=2)
            temp = [data for data in pb]

        def _helper(input_sizes, bs, output_sizes, *, ps=None, pv=0, dl=False):
            ds = IterDatasetWithLen([torch.randn(*s) for s in input_sizes])
            pb = PaddedBatchIterableDataset(ds, batch_size=bs, padded_shapes=ps, padded_values=pv, drop_last=dl)
            self.assertEqual(len(pb), len(output_sizes))
            it = iter(ds)
            if isinstance(pv, Sequence):
                pv = pv[0]
            for i, (output_batch, output_data_size) in enumerate(zip(pb, output_sizes)):
                # Verify Batch Size
                if i + 1 == len(pb) and not dl and len(ds) % bs > 0:
                    self.assertEqual(len(output_batch), len(ds) % bs)
                else:
                    self.assertEqual(len(output_batch), bs)
                for output_data in output_batch:
                    # Verify Original Data
                    input_data = next(it)
                    slices = [slice(0, n) for n in input_data.shape]
                    self.assertEqual(output_data[slices], input_data)
                    # Verify Padded Shape
                    self.assertEqual(output_data_size, list(output_data.shape))
                    # Verify Padded Value
                    mask = torch.ones_like(output_data, dtype=torch.bool)
                    mask[slices] = 0
                    self.assertEqual(output_data[mask].sum(), mask.sum() * pv, exact_dtype=False)

        input_sizes_0 = [(5, 5), (3, 7), (5, 3, 7), (7, 5, 5)]
        output_sizes_0 = [[5, 7], [7, 5, 7]]
        _helper(input_sizes_0, 2, output_sizes_0)
        _helper(input_sizes_0, 2, output_sizes_0, dl=True)
        # Each element should have same dimensions
        with self.assertRaises(RuntimeError):
            _helper(input_sizes_0, 3, output_sizes_0)

        input_sizes_1 = [(3, 5), (5, 7), (7, 3), (7, 5)]
        # Not specify padded shape and value
        _helper(input_sizes_1, 2, [(5, 7), (7, 5)])
        _helper(input_sizes_1, 3, [(7, 7), (7, 5)])
        # Specify padded value, and drop last batch
        _helper(input_sizes_1, 2, [(5, 7), (7, 5)], pv=1, dl=True)
        _helper(input_sizes_1, 3, [(7, 7)], pv=1, dl=True)
        # Specify padded shape and value
        _helper(input_sizes_1, 3, [(8, 8), (8, 8)], ps=torch.Size([8, 8]), pv=1)
        _helper(input_sizes_1, 3, [(8, 8), (8, 8)], ps=[8, 8], pv=[1])
        with self.assertRaises(RuntimeError):
            _helper(input_sizes_1, 2, [(6, 6), (6, 6)], ps=torch.Size([6, 6]))

        # Create Nested Tensor List/Map Dataset
        def _create_nested_ds(input_sizes):
            is_list = True if isinstance(input_sizes[0], Sequence) else False
            nested_lists: List[Union[List, Dict]] = []
            for sizes in input_sizes:
                if is_list:
                    lists = [torch.randn(*s) for s in sizes]
                    nested_lists.append(lists)
                else:
                    maps = {key: [torch.randn(*s) for s in v]
                            if isinstance(v, list)
                            else torch.randn(*v)
                            for key, v in sizes.items()}
                    nested_lists.append(maps)
            ds = IterDatasetWithLen(nested_lists)
            # Flag for determine the element is list or dict
            ds.is_list = is_list  # type: ignore
            return ds

        def _nested_helper(ds, bs, output_sizes, *, ps=None, pv=0, dl=False):
            pb = PaddedBatchIterableDataset(ds, batch_size=bs, padded_shapes=ps, padded_values=pv, drop_last=dl)
            self.assertEqual(len(pb), len(output_sizes))
            it = iter(ds)
            for i, (output_lists, output_batch_sizes) in enumerate(zip(pb, output_sizes)):
                transposed = zip(*output_lists) if ds.is_list else zip(*output_lists.values())  # type: ignore
                for output_list in transposed:
                    input_list = next(it) if ds.is_list else next(it).values()
                    for output_data, input_data, output_data_size in zip(output_list, input_list, output_batch_sizes):
                        # Verify Original Data
                        slices = [slice(0, n) for n in input_data.shape]
                        self.assertEqual(output_data[slices], input_data)
                        # Verify Padded Shape
                        self.assertEqual(output_data_size, list(output_data.shape))
                        # Verify Padded Value
                        mask = torch.ones_like(output_data, dtype=torch.bool)
                        mask[slices] = 0
                        self.assertEqual(output_data[mask].sum(), mask.sum() * pv, exact_dtype=False)

        # Nested List
        list_ds = _create_nested_ds([[(3, 5), (7, 9)],
                                     [(3, 3), (11, 5)],
                                     [(5, 1), (7, 11)],
                                     [(1, 3), (9, 5)]])
        # Not specify padded shape
        _nested_helper(list_ds, 2, [[(3, 5), (11, 9)], [(5, 3), (9, 11)]])
        _nested_helper(list_ds, 3, [[(5, 5), (11, 11)], [(1, 3), (9, 5)]])
        # Specify padded shape
        _nested_helper(list_ds, 2, [[(8, 8), (11, 11)], [(8, 8), (11, 11)]], ps=[torch.Size([8, 8]), torch.Size([11, 11])])
        _nested_helper(list_ds, 2, [[(8, 8), (11, 11)], [(8, 8), (11, 11)]], ps=[[8, 8], [11, 11]])
        # Broadcast padded shape
        _nested_helper(list_ds, 2, [[(11, 11), (11, 11)], [(11, 11), (11, 11)]], ps=[[11, 11]])

        # Nested Map
        map_ds = _create_nested_ds([{'k1': (3, 5), 'k2': (7, 9)},
                                    {'k1': (3, 3), 'k2': (11, 5)},
                                    {'k1': (5, 1), 'k2': (7, 11)},
                                    {'k1': (1, 3), 'k2': (9, 5)}])

        # Not specify padded shape
        _nested_helper(map_ds, 2, [[(3, 5), (11, 9)], [(5, 3), (9, 11)]])
        _nested_helper(map_ds, 3, [[(5, 5), (11, 11)], [(1, 3), (9, 5)]])
        _nested_helper(map_ds, 3, [[(5, 5), (11, 11)]], dl=True)
        #  Specify padded shape
        _nested_helper(map_ds, 2, [[(8, 8), (11, 11)], [(8, 8), (11, 11)]], ps=[torch.Size([8, 8]), torch.Size([11, 11])])
        # Broadcast padded shape
        _nested_helper(map_ds, 2, [[(11, 11), (11, 11)], [(11, 11), (11, 11)]], ps=[[11, 11]])

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
