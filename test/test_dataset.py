import tempfile
import warnings

from torch.testing._internal.common_utils import (TestCase, run_tests)

from torch.utils.data.datasets import (ListDirFilesIterableDataset)

def create_temp_dir_and_files():
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    temp_file1 = tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False)
    temp_file2 = tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False)
    temp_file3 = tempfile.NamedTemporaryFile(dir=temp_dir_path, delete=False)

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

if __name__ == '__main__':
    run_tests()
