import os
import pickle
import random
import tempfile
import warnings
import tarfile
import zipfile
import numpy as np
from PIL import Image
from unittest import skipIf

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.utils.data import IterDataPipe, RandomSampler, DataLoader
from typing import List, Tuple, Dict, Any, Type

import torch.utils.data.datapipes as dp
from torch.utils.data.datapipes.utils.decoder import (
    basichandlers as decoder_basichandlers,
    imagehandler as decoder_imagehandler)

try:
    import torchvision.transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")


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


    def test_routeddecoder_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_pngfile_pathname = os.path.join(temp_dir, "test_png.png")
        img = Image.new('RGB', (2, 2), color='red')
        img.save(temp_pngfile_pathname)
        datapipe1 = dp.iter.ListDirFiles(temp_dir, ['*.png', '*.txt'])
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)
        datapipe3 = dp.iter.RoutedDecoder(datapipe2, handlers=[decoder_imagehandler('rgb')])
        datapipe3.add_handler(decoder_basichandlers)

        for rec in datapipe3:
            ext = os.path.splitext(rec[0])[1]
            if ext == '.png':
                expected = np.array([[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]], dtype=np.single)
                self.assertTrue(np.array_equal(rec[1], expected))
            else:
                self.assertTrue(rec[1] == open(rec[0], 'rb').read().decode('utf-8'))


    def test_groupbykey_iterable_datapipe(self):
        temp_dir = self.temp_dir.name
        temp_tarfile_pathname = os.path.join(temp_dir, "test_tar.tar")
        file_list = [
            "a.png", "b.png", "c.json", "a.json", "c.png", "b.json", "d.png",
            "d.json", "e.png", "f.json", "g.png", "f.png", "g.json", "e.json",
            "h.txt", "h.json"]
        with tarfile.open(temp_tarfile_pathname, "w:gz") as tar:
            for file_name in file_list:
                file_pathname = os.path.join(temp_dir, file_name)
                with open(file_pathname, 'w') as f:
                    f.write('12345abcde')
                tar.add(file_pathname)

        datapipe1 = dp.iter.ListDirFiles(temp_dir, '*.tar')
        datapipe2 = dp.iter.LoadFilesFromDisk(datapipe1)
        datapipe3 = dp.iter.ReadFilesFromTar(datapipe2)
        datapipe4 = dp.iter.GroupByKey(datapipe3, group_size=2)

        expected_result = [("a.png", "a.json"), ("c.png", "c.json"), ("b.png", "b.json"), ("d.png", "d.json"), (
            "f.png", "f.json"), ("g.png", "g.json"), ("e.png", "e.json"), ("h.json", "h.txt")]

        count = 0
        for rec, expected in zip(datapipe4, expected_result):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0][0]), expected[0])
            self.assertEqual(os.path.basename(rec[1][0]), expected[1])
            self.assertEqual(rec[0][1].read(), b'12345abcde')
            self.assertEqual(rec[1][1].read(), b'12345abcde')
        self.assertEqual(count, 8)


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


def _fake_fn(data, *args, **kwargs):
    return data

def _fake_filter_fn(data, *args, **kwargs):
    return data >= 5

def _worker_init_fn(worker_id):
    random.seed(123)


class TestFunctionalIterDataPipe(TestCase):

    def test_picklable(self):
        arr = range(10)
        picklable_datapipes: List[Tuple[Type[IterDataPipe], IterDataPipe, Tuple, Dict[str, Any]]] = [
            (dp.iter.Map, IDP(arr), (), {}),
            (dp.iter.Map, IDP(arr), (_fake_fn, (0, ), {'test': True}), {}),
            (dp.iter.Collate, IDP(arr), (), {}),
            (dp.iter.Collate, IDP(arr), (_fake_fn, (0, ), {'test': True}), {}),
            (dp.iter.Filter, IDP(arr), (_fake_filter_fn, (0, ), {'test': True}), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in picklable_datapipes:
            p = pickle.dumps(dpipe(input_dp, *dp_args, **dp_kwargs))  # type: ignore

        unpicklable_datapipes: List[Tuple[Type[IterDataPipe], IterDataPipe, Tuple, Dict[str, Any]]] = [
            (dp.iter.Map, IDP(arr), (lambda x: x, ), {}),
            (dp.iter.Collate, IDP(arr), (lambda x: xi, ), {}),
            (dp.iter.Filter, IDP(arr), (lambda x: x >= 5, ), {}),
        ]
        for dpipe, input_dp, dp_args, dp_kwargs in unpicklable_datapipes:
            with warnings.catch_warnings(record=True) as wa:
                datapipe = dpipe(input_dp, *dp_args, **dp_kwargs)
                self.assertEqual(len(wa), 1)
                self.assertRegex(str(wa[0].message), r"^Lambda function is not supported for pickle")
                with self.assertRaises(AttributeError):
                    p = pickle.dumps(datapipe)  # type: ignore

    def test_concat_datapipe(self):
        input_dp1 = IDP(range(10))
        input_dp2 = IDP(range(5))

        with self.assertRaisesRegex(ValueError, r"Expected at least one DataPipe"):
            dp.iter.Concat()

        with self.assertRaisesRegex(TypeError, r"Expected all inputs to be `IterDataPipe`"):
            dp.iter.Concat(input_dp1, ())

        concat_dp = input_dp1.concat(input_dp2)
        self.assertEqual(len(concat_dp), 15)
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

        # Test Reset
        self.assertEqual(list(concat_dp), list(range(10)) + list(range(5)))

        input_dp_nl = IDP_NoLen(range(5))
        concat_dp = input_dp1.concat(input_dp_nl)
        with self.assertRaises(NotImplementedError):
            len(concat_dp)

        self.assertEqual(list(d for d in concat_dp), list(range(10)) + list(range(5)))

    def test_map_datapipe(self):
        input_dp = IDP(range(10))

        def fn(item, dtype=torch.float, *, sum=False):
            data = torch.tensor(item, dtype=dtype)
            return data if not sum else data.sum()

        map_dp = input_dp.map(fn)
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

        map_dp = input_dp.map(fn=fn, fn_args=(torch.int, ), fn_kwargs={'sum': True})
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        from functools import partial
        map_dp = input_dp.map(partial(fn, dtype=torch.int, sum=True))
        self.assertEqual(len(input_dp), len(map_dp))
        for x, y in zip(map_dp, input_dp):
            self.assertEqual(x, torch.tensor(y, dtype=torch.int).sum())

        input_dp_nl = IDP_NoLen(range(10))
        map_dp_nl = input_dp_nl.map()
        with self.assertRaises(NotImplementedError):
            len(map_dp_nl)
        for x, y in zip(map_dp_nl, input_dp_nl):
            self.assertEqual(x, torch.tensor(y, dtype=torch.float))

    def test_collate_datapipe(self):
        arrs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        input_dp = IDP(arrs)

        def _collate_fn(batch):
            return torch.tensor(sum(batch), dtype=torch.float)

        collate_dp = input_dp.collate(collate_fn=_collate_fn)
        self.assertEqual(len(input_dp), len(collate_dp))
        for x, y in zip(collate_dp, input_dp):
            self.assertEqual(x, torch.tensor(sum(y), dtype=torch.float))

        input_dp_nl = IDP_NoLen(arrs)
        collate_dp_nl = input_dp_nl.collate()
        with self.assertRaises(NotImplementedError):
            len(collate_dp_nl)
        for x, y in zip(collate_dp_nl, input_dp_nl):
            self.assertEqual(x, torch.tensor(y))

    def test_batch_datapipe(self):
        arrs = list(range(10))
        input_dp = IDP(arrs)
        with self.assertRaises(AssertionError):
            input_dp.batch(batch_size=0)

        # Default not drop the last batch
        bs = 3
        batch_dp = input_dp.batch(batch_size=bs)
        self.assertEqual(len(batch_dp), 4)
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), 1 if i == 3 else bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        # Drop the last batch
        bs = 4
        batch_dp = input_dp.batch(batch_size=bs, drop_last=True)
        self.assertEqual(len(batch_dp), 2)
        for i, batch in enumerate(batch_dp):
            self.assertEqual(len(batch), bs)
            self.assertEqual(batch, arrs[i * bs: i * bs + len(batch)])

        input_dp_nl = IDP_NoLen(range(10))
        batch_dp_nl = input_dp_nl.batch(batch_size=2)
        with self.assertRaises(NotImplementedError):
            len(batch_dp_nl)

    def test_bucket_batch_datapipe(self):
        input_dp = IDP(range(20))
        with self.assertRaises(AssertionError):
            input_dp.bucket_batch(batch_size=0)

        input_dp_nl = IDP_NoLen(range(20))
        bucket_dp_nl = input_dp_nl.bucket_batch(batch_size=7)
        with self.assertRaises(NotImplementedError):
            len(bucket_dp_nl)

        # Test Bucket Batch without sort_key
        def _helper(**kwargs):
            arrs = list(range(100))
            random.shuffle(arrs)
            input_dp = IDP(arrs)
            bucket_dp = input_dp.bucket_batch(**kwargs)
            if kwargs["sort_key"] is None:
                # BatchDataset as reference
                ref_dp = input_dp.batch(batch_size=kwargs['batch_size'], drop_last=kwargs['drop_last'])
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

    def test_filter_datapipe(self):
        input_ds = IDP(range(10))

        def _filter_fn(data, val, clip=False):
            if clip:
                return data >= val
            return True

        filter_dp = input_ds.filter(filter_fn=_filter_fn, fn_args=(5, ))
        for data, exp in zip(filter_dp, range(10)):
            self.assertEqual(data, exp)

        filter_dp = input_ds.filter(filter_fn=_filter_fn, fn_kwargs={'val': 5, 'clip': True})
        for data, exp in zip(filter_dp, range(5, 10)):
            self.assertEqual(data, exp)

        with self.assertRaises(NotImplementedError):
            len(filter_dp)

        def _non_bool_fn(data):
            return 1

        filter_dp = input_ds.filter(filter_fn=_non_bool_fn)
        with self.assertRaises(ValueError):
            temp = list(d for d in filter_dp)

    def test_sampler_datapipe(self):
        input_dp = IDP(range(10))
        # Default SequentialSampler
        sampled_dp = dp.iter.Sampler(input_dp)  # type: ignore
        self.assertEqual(len(sampled_dp), 10)
        for i, x in enumerate(sampled_dp):
            self.assertEqual(x, i)

        # RandomSampler
        random_sampled_dp = dp.iter.Sampler(input_dp, sampler=RandomSampler, sampler_kwargs={'replacement': True})  # type: ignore

        # Requires `__len__` to build SamplerDataPipe
        input_dp_nolen = IDP_NoLen(range(10))
        with self.assertRaises(AssertionError):
            sampled_dp = dp.iter.Sampler(input_dp_nolen)

    def test_shuffle_datapipe(self):
        exp = list(range(20))
        input_ds = IDP(exp)

        with self.assertRaises(AssertionError):
            shuffle_dp = input_ds.shuffle(buffer_size=0)

        for bs in (5, 20, 25):
            shuffle_dp = input_ds.shuffle(buffer_size=bs)
            self.assertEqual(len(shuffle_dp), len(input_ds))

            random.seed(123)
            res = list(d for d in shuffle_dp)
            self.assertEqual(sorted(res), exp)

            # Test Deterministic
            for num_workers in (0, 1):
                random.seed(123)
                dl = DataLoader(shuffle_dp, num_workers=num_workers, worker_init_fn=_worker_init_fn)
                dl_res = list(d for d in dl)
                self.assertEqual(res, dl_res)

        shuffle_dp_nl = IDP_NoLen(range(20)).shuffle(buffer_size=5)
        with self.assertRaises(NotImplementedError):
            len(shuffle_dp_nl)

    @skipIfNoTorchVision
    def test_transforms_datapipe(self):
        torch.set_default_dtype(torch.float)
        # A sequence of numpy random numbers representing 3-channel images
        w = h = 32
        inputs = [np.random.randint(0, 255, (h, w, 3), dtype=np.uint8) for i in range(10)]
        tensor_inputs = [torch.tensor(x, dtype=torch.float).permute(2, 0, 1) / 255. for x in inputs]

        input_dp = IDP(inputs)
        # Raise TypeError for python function
        with self.assertRaisesRegex(TypeError, r"`transforms` are required to be"):
            input_dp.transforms(_fake_fn)

        # transforms.Compose of several transforms
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Pad(1, fill=1, padding_mode='constant'),
        ])
        tsfm_dp = input_dp.transforms(transforms)
        self.assertEqual(len(tsfm_dp), len(input_dp))
        for tsfm_data, input_data in zip(tsfm_dp, tensor_inputs):
            self.assertEqual(tsfm_data[:, 1:(h + 1), 1:(w + 1)], input_data)

        # nn.Sequential of several transforms (required to be instances of nn.Module)
        input_dp = IDP(tensor_inputs)
        transforms = nn.Sequential(
            torchvision.transforms.Pad(1, fill=1, padding_mode='constant'),
        )
        tsfm_dp = input_dp.transforms(transforms)
        self.assertEqual(len(tsfm_dp), len(input_dp))
        for tsfm_data, input_data in zip(tsfm_dp, tensor_inputs):
            self.assertEqual(tsfm_data[:, 1:(h + 1), 1:(w + 1)], input_data)

        # Single transform
        input_dp = IDP_NoLen(inputs)
        transform = torchvision.transforms.ToTensor()
        tsfm_dp = input_dp.transforms(transform)
        with self.assertRaises(NotImplementedError):
            len(tsfm_dp)
        for tsfm_data, input_data in zip(tsfm_dp, tensor_inputs):
            self.assertEqual(tsfm_data, input_data)

    def test_zip_datapipe(self):
        with self.assertRaises(TypeError):
            dp.iter.Zip(IDP(range(10)), list(range(10)))

        zipped_dp = dp.iter.Zip(IDP(range(10)), IDP_NoLen(range(5)))
        with self.assertRaises(NotImplementedError):
            len(zipped_dp)
        exp = list((i, i) for i in range(5))
        self.assertEqual(list(d for d in zipped_dp), exp)

        zipped_dp = dp.iter.Zip(IDP(range(10)), IDP(range(5)))
        self.assertEqual(len(zipped_dp), 5)
        self.assertEqual(list(zipped_dp), exp)
        # Reset
        self.assertEqual(list(zipped_dp), exp)


if __name__ == '__main__':
    run_tests()
