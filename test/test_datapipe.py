import tempfile
import warnings

from torch.testing._internal.common_utils import (TestCase, run_tests)

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

if __name__ == '__main__':
    run_tests()
