import pickle
import tempfile
import warnings

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.utils.data import IterableDataset, RandomSampler
from torch.utils.data.datasets import \
    (CallableIterableDataset, CollateIterableDataset, BatchIterableDataset,
     ListDirFilesIterableDataset, LoadFilesFromDiskIterableDataset, SamplerIterableDataset,
     PaddedBatchIterableDataset)
from typing import List, Tuple, Dict, Any, Type, Sequence, Mapping, Union


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
    def test_picklable(self):
        arr = range(10)
        picklable_datasets: List[Tuple[Type[IterableDataset], IterableDataset, Dict[str, Any]]] = [
            (CallableIterableDataset, IterDatasetWithLen(arr), {}),
            (CollateIterableDataset, IterDatasetWithLen(arr), {}),
        ]
        for ds, d, kargs in picklable_datasets:
            p = pickle.dumps(ds(d, **kargs))  # type: ignore

        unpicklable_datasets: List[Tuple[Type[IterableDataset], IterableDataset, Dict[str, Any]]] = [
            (CallableIterableDataset, IterDatasetWithLen(arr), {'fn': lambda x: x}),
            (CollateIterableDataset, IterDatasetWithLen(arr), {'collate_fn': lambda x: x}),
        ]
        for ds, d, kargs in unpicklable_datasets:
            with self.assertRaises(AttributeError):
                p = pickle.dumps(ds(d, **kargs))  # type: ignore

    def test_callable_dataset(self):
        arr = range(10)
        ds_len = IterDatasetWithLen(arr)
        ds_nolen = IterDatasetWithoutLen(arr)

        def fn(item):
            return torch.tensor(item, dtype=torch.float)

        callable_ds = CallableIterableDataset(ds_len, fn=fn)  # type: ignore
        self.assertEqual(len(ds_len), len(callable_ds))
        ds_iter = iter(ds_len)
        for x in callable_ds:
            y = next(ds_iter)
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        callable_ds_nolen = CallableIterableDataset(ds_nolen)  # type: ignore
        with self.assertRaises(NotImplementedError):
            len(callable_ds_nolen)
        ds_nolen_iter = iter(ds_nolen)
        for x in callable_ds_nolen:
            y = next(ds_nolen_iter)
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

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
            for data in pb:
                pass

        def _create_ds(input_sizes):

            # Recursive create Tensors
            def _helper(size_list):
                if isinstance(size_list, tuple):
                    return torch.randn(*size_list)
                data: List[Union[torch.Tensor, List, Dict]] = []
                for s in size_list:
                    if isinstance(s, tuple):
                        data.append(torch.randn(*s))
                    elif isinstance(s, list):
                        data.append(_helper(s))
                    elif isinstance(s, dict):
                        data.append({k: _helper(s[k]) for k in s})
                return data

            return IterDatasetWithLen(_helper(input_sizes))

        def _test_batch_padded(ds, batch_size, exp_pshapes, exp_pvalues, **kwargs):
            pb_ds = PaddedBatchIterableDataset(ds, batch_size=batch_size, **kwargs)
            self.assertEqual(len(pb_ds), len(exp_pshapes))

            # Mimic PaddedBatch
            def _collate_fn(batch):
                elem = batch[0]
                if isinstance(elem, (torch.Tensor, float, int, complex, str)):
                    return batch
                elif isinstance(elem, Mapping):
                    return {key: _collate_fn([d[key] for d in batch]) for key in elem}
                elif isinstance(elem, Sequence):
                    return [_collate_fn(d) for d in zip(*batch)]
                else:
                    return batch

            # Use Batch + Callable as input reference
            dl = kwargs['drop_last'] if 'drop_last' in kwargs else False
            b_ds = CallableIterableDataset(BatchIterableDataset(ds, batch_size=batch_size, drop_last=dl),  # type: ignore
                                           fn=_collate_fn)

            # Helper function to check batch recursively
            def _helper(ref_batch, output_batch, exp_padded_shape, exp_padded_value):
                # Batch is a list
                if isinstance(output_batch, Sequence):
                    elem = output_batch[0]
                    # Batch of Tensors -> Check shape/value
                    if isinstance(elem, torch.Tensor):
                        # Check Batch Size
                        self.assertEqual(len(ref_batch), len(output_batch))
                        for ref_data, output_data in zip(ref_batch, output_batch):
                            # Verify Original Data
                            slices = [slice(0, n) for n in ref_data.shape]
                            self.assertEqual(output_data[slices], ref_data)
                            # Verify Padded Shape
                            self.assertEqual(exp_padded_shape, tuple(output_data.shape))
                            # Verify Padded Shape
                            mask = torch.ones_like(output_data, dtype=torch.bool)
                            mask[slices] = 0
                            self.assertTrue(torch.eq(output_data[mask], exp_padded_value).all())
                    # Nested List
                    else:
                        for rb, ob, ps, pv in zip(ref_batch, output_batch, exp_padded_shape, exp_padded_value):
                            _helper(rb, ob, ps, pv)
                # Batch is a dict
                elif isinstance(output_batch, Mapping):
                    for rb, ob, ps, pv in zip(ref_batch.values(), output_batch.values(), exp_padded_shape, exp_padded_value):
                        _helper(rb, ob, ps, pv)
                else:
                    self.assertTrue(False, msg="Type is not supported.")

            for ref_batch, output_batch, exp_padded_shape in zip(b_ds, pb_ds, exp_pshapes):
                _helper(ref_batch, output_batch, exp_padded_shape, exp_pvalues)

        # Tensor
        tensor_shapes_0 = [(3, 5), (5, 7), (7, 3), (5, 5)]
        tensor_shapes_1 = [(5, 5), (3, 7), (5, 3, 7), (7, 5, 5)]
        # List of Tensor
        list_tensor_shapes = [[(3, 5), (7, 9)],
                              [(3, 3), (11, 5)],
                              [(5, 1), (7, 11)],
                              [(1, 3), (9, 5)]]
        # dict of Tensor
        map_tensor_shapes = [{'k1': (3, 5), 'k2': (7, 9)},
                             {'k1': (3, 3), 'k2': (11, 5)},
                             {'k1': (5, 1), 'k2': (7, 11)},
                             {'k1': (1, 3), 'k2': (9, 5)}]
        # Nested List of List
        nested_ll_shapes = [[(3, 5), [(7, 9), (1, 2)]],
                            [(3, 3), [(11, 5), (3, 2)]],
                            [(5, 1), [(7, 11), (5, 3)]],
                            [(1, 3), [(9, 5), (2, 2)]]]
        # Nested Map of List
        nested_ml_shapes = [{'k1': (3, 5), 'k2': [(7, 9), (1, 2)]},
                            {'k1': (3, 3), 'k2': [(11, 5), (3, 2)]},
                            {'k1': (5, 1), 'k2': [(7, 11), (5, 3)]},
                            {'k1': (1, 3), 'k2': [(9, 5), (2, 2)]}]
        # Nested different types
        nested_dt_shapes = [[{'k': (3, 5)}, [(7, 9), (1, 2)]],
                            [{'k': (3, 3)}, [(11, 5), (3, 2)]],
                            [{'k': (5, 1)}, [(7, 11), (5, 3)]],
                            [{'k': (1, 3)}, [(9, 5), (2, 2)]]]

        # input_shape, exp_shape, exp_value, batch_size, {padded_shape, padded_value, drop_last}, Error
        test_cases: List[Tuple[List, List, Union[int, List], int, Dict, Any]] = [
            # ===== Tensor =====
            (
                tensor_shapes_0,
                [(5, 7), (7, 5)],  # expected output shape
                0,  # expected output value
                2, {}, None,
            ),
            (
                tensor_shapes_0,
                [(7, 7), (5, 5)],  # expected output shape
                0,  # expected output value
                3, {}, None,
            ),
            (
                tensor_shapes_0,
                [(8, 8), (8, 8)],  # expected output shape
                1,  # expected output value
                2, {'padded_shapes': torch.Size([8, 8]), 'padded_values': 1}, None,
            ),
            (
                tensor_shapes_0,
                [(8, 8)],  # expected output shape
                0,  # expected output value
                3, {'padded_shapes': [8, 8], 'drop_last': True}, None,
            ),
            (
                tensor_shapes_0,
                [(7, 7), (5, 5)],  # expected output shape
                0,  # expected output value
                2, {'padded_shapes': [6, 6]}, RuntimeError,  # Tensor bigger than padded shape

            ),
            (
                tensor_shapes_1,
                [(), ()],  # output_shapeexpected output shape
                0,  # expected output value
                3, {}, RuntimeError,  # Inequal shape dimensions within batch
            ),
            # ===== List of tensors =====
            (
                list_tensor_shapes,
                [[(3, 5), (11, 9)],
                 [(5, 3), (9, 11)]],  # expected output shape
                [1, 0],  # expected output value
                2, {'padded_values': [1, 0], 'drop_last': False}, None,
            ),
            # Specify Shape and Value
            (
                list_tensor_shapes,
                [[(8, 8), (11, 11)],
                 [(8, 8), (11, 11)]],  # expected output shape
                [0, 1],  # expected output value
                2, {'padded_shapes': [torch.Size([8, 8]), torch.Size([11, 11])], 'padded_values': [0, 1]}, None,
            ),
            # Broadcast Shape and Value
            (
                list_tensor_shapes,
                [[(11, 11), (11, 11)],
                 [(11, 11), (11, 11)]],  # expected output shape
                [1, 1],  # expected output value
                2, {'padded_shapes': [(11, 11)], 'padded_values': 1}, None,
            ),
            # ===== Map of tensors =====
            (
                map_tensor_shapes,
                [[(3, 5), (11, 9)],
                 [(5, 3), (9, 11)]],  # expected output shape
                [0, 0],  # expected output value
                2, {}, None,
            ),
            # Specify Shape and Value
            (
                map_tensor_shapes,
                [[(8, 8), (11, 11)],
                 [(8, 8), (11, 11)]],  # expected output shape
                [0, 1],  # expected output value
                2, {'padded_shapes': [[8, 8], [11, 11]], 'padded_values': [0, 1]}, None,
            ),
            # Broadcast Shape and Value
            (
                map_tensor_shapes,
                [[(11, 11), (11, 11)],
                 [(11, 11), (11, 11)]],  # expected output shape
                [1, 1],  # expected output value
                2, {'padded_shapes': [[11, 11]], 'padded_values': [1]}, None,
            ),
            # ===== Nested List of List =====
            (
                nested_ll_shapes,
                [[(3, 5), [(11, 9), (3, 2)]],
                 [(5, 3), [(9, 11), (5, 3)]]],  # expected output shape
                [0, [0, 0]],  # expected output value
                2, {}, None,
            ),
            # Specify Nested Shape and Value
            (
                nested_ll_shapes,
                [[(8, 8), [(11, 11), (5, 5)]],
                 [(8, 8), [(11, 11), (5, 5)]]],  # expected output shape
                [0, [1, 0]],  # expected output value
                2, {'padded_shapes': [[8, 8], [[11, 11], [5, 5]]], 'padded_values': [0, [1, 0]]}, None,
            ),
            # Broadcast Shape and Value
            (
                nested_ll_shapes,
                [[(8, 8), [(11, 11), (11, 11)]],
                 [(8, 8), [(11, 11), (11, 11)]]],  # expected output shape
                [1, [1, 1]],  # expected output value
                2, {'padded_shapes': [[8, 8], [[11, 11]]], 'padded_values': [1]}, None,
            ),
            # ===== Nested Map of List ======
            (
                nested_ml_shapes,
                [[(3, 5), [(11, 9), (3, 2)]],
                 [(5, 3), [(9, 11), (5, 3)]]],  # expected output shape
                [0, [0, 0]],  # expected output value
                2, {}, None,
            ),
            # Specify Nested Shape and Value
            (
                nested_ml_shapes,
                [[(8, 8), [(11, 11), (5, 5)]],
                 [(8, 8), [(11, 11), (5, 5)]]],  # expected output shape
                [0, [1, 0]],  # expected output value
                2, {'padded_shapes': [[8, 8], [[11, 11], [5, 5]]], 'padded_values': [0, [1, 0]]}, None,
            ),
            # Broadcast Shape and Value
            (
                nested_ml_shapes,
                [[(8, 8), [(11, 11), (11, 11)]],
                 [(8, 8), [(11, 11), (11, 11)]]],  # expected output shape
                [1, [1, 1]],  # expected output value
                2, {'padded_shapes': [[8, 8], [[11, 11]]], 'padded_values': [1]}, None,
            ),
            # ===== Nested Different Types ======
            (
                nested_dt_shapes,
                [[[(3, 5)], [(11, 9), (3, 2)]],
                 [[(5, 3)], [(9, 11), (5, 3)]]],  # expected output shape
                [[0], [0, 0]],  # expected output value
                2, {}, None,
            ),
        ]

        # Run through all test cases
        for input_shape, exp_shapes, exp_values, batch_size, kwargs, error in test_cases:
            ds = _create_ds(input_shape)
            if error:
                with self.assertRaises(error):
                    pb_ds = PaddedBatchIterableDataset(ds, batch_size=batch_size, **kwargs)
                    for data in pb_ds:
                        pass
            else:
                _test_batch_padded(ds, batch_size, exp_shapes, exp_values, **kwargs)

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
