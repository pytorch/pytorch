import os
import pickle
import random
import tempfile
import warnings
import tarfile
import zipfile

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.utils.data import IterDataPipe, RandomSampler
from typing import List, Tuple, Dict, Any, Type

import torch.utils.data.datapipes as dp

def create_temp_dir_and_files():
    # The temp dir and files within it will be released and deleted in tearDown().
    # Adding `noqa: P201` to avoid mypy's warning on not releasing the dir handle within this function.
    temp_dir = tempfile.TemporaryDirectory()  # noqa: P201
    temp_dir_path = temp_dir.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.txt') as f:
        temp_file1_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.byte') as f:
        temp_file2_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False, suffix='.empty') as f:
        temp_file3_name = f.name

    with open(temp_file1_name, 'w') as f1:
        f1.write('0123456789abcdef')
    with open(temp_file2_name, 'wb') as f2:
        f2.write(b"0123456789abcdef")

    temp_sub_dir = tempfile.TemporaryDirectory(dir=temp_dir_path)  # noqa: P201
    temp_sub_dir_path = temp_sub_dir.name
    with tempfile.NamedTemporaryFile(dir=temp_sub_dir_path, delete=False, suffix='.txt') as f:
        temp_sub_file1_name = f.name
    with tempfile.NamedTemporaryFile(dir=temp_sub_dir_path, delete=False, suffix='.byte') as f:
        temp_sub_file2_name = f.name

    with open(temp_sub_file1_name, 'w') as f1:
        f1.write('0123456789abcdef')
    with open(temp_sub_file2_name, 'wb') as f2:
        f2.write(b"0123456789abcdef")

    return [(temp_dir, temp_file1_name, temp_file2_name, temp_file3_name),
            (temp_sub_dir, temp_sub_file1_name, temp_sub_file2_name)]

class TestIterableDataPipeBasic(TestCase):

    def setUp(self):
        ret = create_temp_dir_and_files()
        self.temp_dir = ret[0][0]
        self.temp_files = ret[0][1:]
        self.temp_sub_dir = ret[1][0]
        self.temp_sub_files = ret[1][1:]

    def tearDown(self):
        try:
            self.temp_sub_dir.cleanup()
            self.temp_dir.cleanup()
        except Exception as e:
            warnings.warn("TestIterableDatasetBasic was not able to cleanup temp dir due to {}".format(str(e)))

    def test_listdirfiles_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        datapipe = dp.iter.ListDirFiles(temp_dir, '')

        count = 0
        for pathname in datapipe:
            count = count + 1
            self.assertTrue(pathname in self.temp_files)
        self.assertEqual(count, len(self.temp_files))

        count = 0
        datapipe = dp.iter.ListDirFiles(temp_dir, '', recursive=True)
        for pathname in datapipe:
            count = count + 1
            self.assertTrue((pathname in self.temp_files) or (pathname in self.temp_sub_files))
        self.assertEqual(count, len(self.temp_files) + len(self.temp_sub_files))


    def test_loadfilesfromdisk_iterable_datapipe(self):
        # test import datapipe class directly
        from torch.utils.data.datapipes.iter import ListDirFiles, LoadFilesFromDisk

        temp_dir = self.temp_dir.name
        datapipe1 = ListDirFiles(temp_dir, '')
        datapipe2 = LoadFilesFromDisk(datapipe1)

        count = 0
        for rec in datapipe2:
            count = count + 1
            self.assertTrue(rec[0] in self.temp_files)
            self.assertTrue(rec[1].read() == open(rec[0], 'rb').read())
        self.assertEqual(count, len(self.temp_files))


    def test_readfilesfromtar_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_tarfile_pathname = os.path.join(temp_dir, "test_tar.tar")
        with tarfile.open(temp_tarfile_pathname, "w:gz") as tar:
            tar.add(self.temp_files[0])
            tar.add(self.temp_files[1])
            tar.add(self.temp_files[2])
        datapipe1 = dp.iter.ListDirFiles(temp_dir, '*.tar')
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)
        datapipe3 = dp.iter.ReadFilesFromTar(datapipe2)
        # read extracted files before reaching the end of the tarfile
        count = 0
        for rec, temp_file in zip(datapipe3, self.temp_files):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0]), os.path.basename(temp_file))
            self.assertEqual(rec[1].read(), open(temp_file, 'rb').read())
        self.assertEqual(count, len(self.temp_files))
        # read extracted files after reaching the end of the tarfile
        count = 0
        data_refs = []
        for rec in datapipe3:
            count = count + 1
            data_refs.append(rec)
        self.assertEqual(count, len(self.temp_files))
        for i in range(0, count):
            self.assertEqual(os.path.basename(data_refs[i][0]), os.path.basename(self.temp_files[i]))
            self.assertEqual(data_refs[i][1].read(), open(self.temp_files[i], 'rb').read())


    def test_readfilesfromzip_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_zipfile_pathname = os.path.join(temp_dir, "test_zip.zip")
        with zipfile.ZipFile(temp_zipfile_pathname, 'w') as myzip:
            myzip.write(self.temp_files[0])
            myzip.write(self.temp_files[1])
            myzip.write(self.temp_files[2])
        datapipe1 = dp.iter.ListDirFiles(temp_dir, '*.zip')
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)
        datapipe3 = dp.iter.ReadFilesFromZip(datapipe2)
        # read extracted files before reaching the end of the zipfile
        count = 0
        for rec, temp_file in zip(datapipe3, self.temp_files):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0]), os.path.basename(temp_file))
            self.assertEqual(rec[1].read(), open(temp_file, 'rb').read())
        self.assertEqual(count, len(self.temp_files))
        # read extracted files before reaching the end of the zipile
        count = 0
        data_refs = []
        for rec in datapipe3:
            count = count + 1
            data_refs.append(rec)
        self.assertEqual(count, len(self.temp_files))
        for i in range(0, count):
            self.assertEqual(os.path.basename(data_refs[i][0]), os.path.basename(self.temp_files[i]))
            self.assertEqual(data_refs[i][1].read(), open(self.temp_files[i], 'rb').read())


class IDP_NoLen(IterDataPipe):
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp

    def __iter__(self):
        for i in self.input_dp:
            yield i


class IDP(IterDataPipe):
    def __init__(self, input_dp):
        super().__init__()
        self.input_dp = input_dp
        self.length = len(input_dp)

    def __iter__(self):
        for i in self.input_dp:
            yield i

    def __len__(self):
        return self.length


def _fake_fn(self, data, *args, **kwargs):
    return data


class TestFunctionalIterDataPipe(TestCase):

    def test_picklable(self):
        arr = range(10)
        picklable_datapipes: List[Tuple[Type[IterDataPipe], IterDataPipe, List, Dict[str, Any]]] = [
            (dp.iter.Callable, IDP(arr), [], {}),
            (dp.iter.Callable, IDP(arr), [0], {'fn': _fake_fn, 'test': True}),
            (dp.iter.Collate, IDP(arr), [], {}),
            (dp.iter.Collate, IDP(arr), [0], {'collate_fn': _fake_fn, 'test': True}),
        ]
        for dpipe, input_dp, args, kargs in picklable_datapipes:
            p = pickle.dumps(dpipe(input_dp, *args, **kargs))  # type: ignore

        unpicklable_datapipes: List[Tuple[Type[IterDataPipe], IterDataPipe, List, Dict[str, Any]]] = [
            (dp.iter.Callable, IDP(arr), [], {'fn': lambda x: x}),
            (dp.iter.Collate, IDP(arr), [], {'collate_fn': lambda x: x}),
        ]
        for dpipe, input_dp, args, kargs in unpicklable_datapipes:
            with self.assertRaises(AttributeError):
                p = pickle.dumps(dpipe(input_dp, *args, **kargs))  # type: ignore

    def test_callable_datapipe(self):
        arr = range(10)
        input_dp = IDP(arr)
        input_dp_nl = IDP_NoLen(arr)

        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        callable_dp = dp.iter.Callable(input_dp, fn=fn)  # type: ignore
        self.assertEqual(len(input_dp), len(callable_dp))
        for x, y in zip(callable_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        callable_dp = dp.iter.Callable(input_dp, torch.int, fn=fn, sum=True)  # type: ignore
        self.assertEqual(len(input_dp), len(callable_dp))
        for x, y in zip(callable_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        callable_dp_nl = dp.iter.Callable(input_dp_nl)  # type: ignore
        with self.assertRaises(NotImplementedError):
            len(callable_dp_nl)
        for x, y in zip(callable_dp_nl, input_dp_nl):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

    def test_collate_datapipe(self):
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        input_dp = IDP(arrs)
        input_dp_nl = IDP_NoLen(arrs)

        def _collate_fn(batch):
            return torch.tensor(sum(batch), dtype=torch.float)

        collate_dp = dp.iter.Collate(input_dp, collate_fn=_collate_fn)
        self.assertEqual(len(input_dp), len(collate_dp))
        for x, y in zip(collate_dp, input_dp):
            self.assertEqual(x, torch.tensor(sum(y), dtype=torch.float))

        collate_dp_nl = dp.iter.Collate(input_dp_nl)  # type: ignore
        with self.assertRaises(NotImplementedError):
            len(collate_dp_nl)
        for x, y in zip(collate_dp_nl, input_dp_nl):
            self.assertEqual(x, torch.tensor(y))

    def test_batch_datapipe(self):
        arrs = list(range(10))
        input_dp = IDP(arrs)
        with self.assertRaises(AssertionError):
            batch_dp0 = dp.iter.Batch(input_dp, batch_size=0)

        # Default not drop the last batch
        bs = 3
        batch_dp1 = dp.iter.Batch(input_dp, batch_size=bs)
        self.assertEqual(len(batch_dp1), 4)
        for i, batch in enumerate(batch_dp1):
            self.assertEqual(len(batch), 1 if i == 3 else bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        # Drop the last batch
        bs = 4
        batch_dp2 = dp.iter.Batch(input_dp, batch_size=bs, drop_last=True)
        self.assertEqual(len(batch_dp2), 2)
        for i, batch in enumerate(batch_dp2):
            self.assertEqual(len(batch), bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        input_dp_nl = IDP_NoLen(range(10))
        batch_dp_nl = dp.iter.Batch(input_dp_nl, batch_size=2)
        with self.assertRaises(NotImplementedError):
            len(batch_dp_nl)

    def test_bucket_batch_datapipe(self):
        input_dp = IDP(range(20))
        with self.assertRaises(AssertionError):
            dp.iter.BucketBatch(input_dp, batch_size=0)

        input_dp_nl = IDP_NoLen(range(20))
        bucket_dp_nl = dp.iter.BucketBatch(input_dp_nl, batch_size=7)
        with self.assertRaises(NotImplementedError):
            len(bucket_dp_nl)

        # Test Bucket Batch without sort_key
        def _helper(**kwargs):
            arrs = list(range(100))
            random.shuffle(arrs)
            input_dp = IDP(arrs)
            bucket_dp = dp.iter.BucketBatch(input_dp, **kwargs)
            if kwargs["sort_key"] is None:
                # BatchDataset as reference
                ref_dp = dp.iter.Batch(input_dp, batch_size=kwargs['batch_size'], drop_last=kwargs['drop_last'])
                for batch, rbatch in zip(bucket_dp, ref_dp):
                    self.assertEqual(batch, rbatch)
            else:
                bucket_size = bucket_dp.bucket_size
                bucket_num = (len(input_dp) - 1) // bucket_size + 1
                it = iter(bucket_dp)
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

    def test_sampler_datapipe(self):
        arrs = range(10)
        input_dp = IDP(arrs)
        # Default SequentialSampler
        sampled_dp = dp.iter.Sampler(input_dp)  # type: ignore
        self.assertEqual(len(sampled_dp), 10)
        i = 0
        for x in sampled_dp:
            self.assertEqual(x, i)
            i += 1

        # RandomSampler
        random_sampled_dp = dp.iter.Sampler(input_dp, sampler=RandomSampler, replacement=True)  # type: ignore

        # Requires `__len__` to build SamplerDataset
        input_dp_nolen = IDP_NoLen(arrs)
        with self.assertRaises(AssertionError):
            sampled_dp = dp.iter.Sampler(input_dp_nolen)


if __name__ == '__main__':
    run_tests()
