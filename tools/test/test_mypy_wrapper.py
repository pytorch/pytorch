import unittest
from pathlib import PurePosixPath

from tools import mypy_wrapper


class TestMypyWrapper(unittest.TestCase):
    def test_config_files(self):
        self.assertEqual(mypy_wrapper.config_files(), {
            'mypy.ini',
            'mypy-strict.ini',
        })

    def test_glob_can_match_individual_files(self):
        self.assertTrue(mypy_wrapper.glob(
            pattern='test/test_torch.py',
            filename=PurePosixPath('test/test_torch.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='test/test_torch.py',
            filename=PurePosixPath('test/test_testing.py'),
        ))

    def test_glob_dir_matters(self):
        self.assertFalse(mypy_wrapper.glob(
            pattern='tools/codegen/utils.py',
            filename=PurePosixPath('torch/nn/modules.py'),
        ))
        self.assertTrue(mypy_wrapper.glob(
            pattern='setup.py',
            filename=PurePosixPath('setup.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='setup.py',
            filename=PurePosixPath('foo/setup.py'),
        ))
        self.assertTrue(mypy_wrapper.glob(
            pattern='foo/setup.py',
            filename=PurePosixPath('foo/setup.py'),
        ))

    def test_glob_can_match_dirs(self):
        self.assertTrue(mypy_wrapper.glob(
            pattern='torch',
            filename=PurePosixPath('torch/random.py'),
        ))
        self.assertTrue(mypy_wrapper.glob(
            pattern='torch',
            filename=PurePosixPath('torch/nn/cpp.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='torch',
            filename=PurePosixPath('tools/fast_nvcc/fast_nvcc.py'),
        ))

    def test_glob_can_match_wildcards(self):
        self.assertTrue(mypy_wrapper.glob(
            pattern='tools/autograd/*.py',
            filename=PurePosixPath('tools/autograd/gen_autograd.py'),
        ))
        self.assertFalse(mypy_wrapper.glob(
            pattern='tools/autograd/*.py',
            filename=PurePosixPath('tools/autograd/deprecated.yaml'),
        ))

if __name__ == '__main__':
    unittest.main()
