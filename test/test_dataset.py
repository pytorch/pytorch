import tempfile
import tarfile
import zipfile
import warnings
import os
import numpy as np
from PIL import Image

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests)
from torch.utils.data import IterableDataset, RandomSampler
from torch.utils.data.datasets import \
    (CollateIterableDataset, BatchIterableDataset, SamplerIterableDataset)

from torch.utils.data.datasets.decoder import (
    basichandlers as decoder_basichandlers,
    imagehandler as decoder_imagehandler)

from torch.utils.data.datasets import (
    ListDirFilesIterableDataset, LoadFilesFromDiskIterableDataset, ReadFilesFromTarIterableDataset,
    ReadFilesFromZipIterableDataset, RoutedDecoderIterableDataset, GroupByFilenameIterableDataset)

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

    return (temp_dir, temp_file1_name, temp_file2_name, temp_file3_name)


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

    def test_readfilesfromtar_iterable_dataset(self):
        temp_dir = self.temp_dir.name
        temp_tarfile_pathname = os.path.join(temp_dir, "test_tar.tar")
        with tarfile.open(temp_tarfile_pathname, "w:gz") as tar:
            tar.add(self.temp_files[0])
            tar.add(self.temp_files[1])
            tar.add(self.temp_files[2])
        dataset1 = ListDirFilesIterableDataset(temp_dir, '*.tar')
        dataset2 = LoadFilesFromDiskIterableDataset(dataset1)
        dataset3 = ReadFilesFromTarIterableDataset(dataset2)
        # read extracted files before reaching the end of the tarfile
        count = 0
        for rec, temp_file in zip(dataset3, self.temp_files):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0]), os.path.basename(temp_file))
            self.assertEqual(rec[1].read(), open(temp_file, 'rb').read())
        self.assertEqual(count, len(self.temp_files))
        # read extracted files after reaching the end of the tarfile
        count = 0
        data_refs = []
        for rec in dataset3:
            count = count + 1
            data_refs.append(rec)
        self.assertEqual(count, len(self.temp_files))
        for i in range(0, count):
            self.assertEqual(os.path.basename(data_refs[i][0]), os.path.basename(self.temp_files[i]))
            self.assertEqual(data_refs[i][1].read(), open(self.temp_files[i], 'rb').read())

    def test_readfilesfromzip_iterable_dataset(self):
        temp_dir = self.temp_dir.name
        temp_zipfile_pathname = os.path.join(temp_dir, "test_zip.zip")
        with zipfile.ZipFile(temp_zipfile_pathname, 'w') as myzip:
            myzip.write(self.temp_files[0])
            myzip.write(self.temp_files[1])
            myzip.write(self.temp_files[2])
        dataset1 = ListDirFilesIterableDataset(temp_dir, '*.zip')
        dataset2 = LoadFilesFromDiskIterableDataset(dataset1)
        dataset3 = ReadFilesFromZipIterableDataset(dataset2)
        # read extracted files before reaching the end of the zipfile
        count = 0
        for rec, temp_file in zip(dataset3, self.temp_files):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0]), os.path.basename(temp_file))
            self.assertEqual(rec[1].read(), open(temp_file, 'rb').read())
        self.assertEqual(count, len(self.temp_files))
        # read extracted files before reaching the end of the zipile
        count = 0
        data_refs = []
        for rec in dataset3:
            count = count + 1
            data_refs.append(rec)
        self.assertEqual(count, len(self.temp_files))
        for i in range(0, count):
            self.assertEqual(os.path.basename(data_refs[i][0]), os.path.basename(self.temp_files[i]))
            self.assertEqual(data_refs[i][1].read(), open(self.temp_files[i], 'rb').read())

    def test_routeddecoder_iterable_dataset(self):
        temp_dir = self.temp_dir.name
        temp_pngfile_pathname = os.path.join(temp_dir, "test_png.png")
        img = Image.new('RGB', (2, 2), color='red')
        img.save(temp_pngfile_pathname)
        dataset1 = ListDirFilesIterableDataset(temp_dir, ['*.png', '*.txt'])
        dataset2 = LoadFilesFromDiskIterableDataset(dataset1)
        dataset3 = RoutedDecoderIterableDataset(dataset2, decoders=[decoder_imagehandler('rgb')])
        dataset3.add_decoder(decoder_basichandlers)

        for rec in dataset3:
            ext = os.path.splitext(rec[0])[1]
            if ext == '.png':
                expected = np.array([[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]], dtype=np.single)
                self.assertTrue(np.array_equal(rec[1], expected))
            else:
                self.assertTrue(rec[1] == open(rec[0], 'rb').read().decode('utf-8'))

    def test_groupbyfilename_iterable_dataset(self):
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

        dataset1 = ListDirFilesIterableDataset(temp_dir, '*.tar')
        dataset2 = LoadFilesFromDiskIterableDataset(dataset1)
        dataset3 = ReadFilesFromTarIterableDataset(dataset2)
        dataset4 = GroupByFilenameIterableDataset(dataset3, group_size=2)

        expected_result = [("a.png", "a.json"), ("c.png", "c.json"), ("b.png", "b.json"), ("d.png", "d.json"), (
            "f.png", "f.json"), ("g.png", "g.json"), ("e.png", "e.json"), ("h.json", "h.txt")]

        count = 0
        for rec, expected in zip(dataset4, expected_result):
            count = count + 1
            self.assertEqual(os.path.basename(rec[0][0]), expected[0])
            self.assertEqual(os.path.basename(rec[1][0]), expected[1])
            self.assertEqual(rec[0][1].read(), b'12345abcde')
            self.assertEqual(rec[1][1].read(), b'12345abcde')
        self.assertEqual(count, 8)


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
