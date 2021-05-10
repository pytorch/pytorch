import unittest

from tools import mypy_wrapper


class TestMypyWrapper(unittest.TestCase):
    def test_config_files(self) -> None:
        self.assertEqual(mypy_wrapper.config_files(), {
            'mypy.ini',
            'mypy-strict.ini',
        })

    def test_is_match_can_match_individual_files(self) -> None:
        self.assertTrue(mypy_wrapper.is_match(
            pattern='test/test_torch.py',
            filename='test/test_torch.py',
        ))
        self.assertFalse(mypy_wrapper.is_match(
            pattern='test/test_torch.py',
            filename='test/test_testing.py',
        ))

    def test_is_match_dir_matters(self) -> None:
        self.assertFalse(mypy_wrapper.is_match(
            pattern='tools/codegen/utils.py',
            filename='torch/nn/modules.py',
        ))
        self.assertTrue(mypy_wrapper.is_match(
            pattern='setup.py',
            filename='setup.py',
        ))
        self.assertFalse(mypy_wrapper.is_match(
            pattern='setup.py',
            filename='foo/setup.py',
        ))
        self.assertTrue(mypy_wrapper.is_match(
            pattern='foo/setup.py',
            filename='foo/setup.py',
        ))

    def test_is_match_can_match_dirs(self) -> None:
        self.assertTrue(mypy_wrapper.is_match(
            pattern='torch',
            filename='torch/random.py',
        ))
        self.assertTrue(mypy_wrapper.is_match(
            pattern='torch',
            filename='torch/nn/cpp.py',
        ))
        self.assertFalse(mypy_wrapper.is_match(
            pattern='torch',
            filename='tools/fast_nvcc/fast_nvcc.py',
        ))

    def test_is_match_can_match_wildcards(self) -> None:
        self.assertTrue(mypy_wrapper.is_match(
            pattern='tools/autograd/*.py',
            filename='tools/autograd/gen_autograd.py',
        ))
        self.assertFalse(mypy_wrapper.is_match(
            pattern='tools/autograd/*.py',
            filename='tools/autograd/deprecated.yaml',
        ))

    def test_is_match_wildcards_dont_expand_or_collapse(self) -> None:
        self.assertFalse(mypy_wrapper.is_match(
            pattern='benchmarks/instruction_counts/*.py',
            filename='benchmarks/instruction_counts/core/utils.py',
        ))
        self.assertFalse(mypy_wrapper.is_match(
            pattern='benchmarks/instruction_counts/*/*.py',
            filename='benchmarks/instruction_counts/main.py',
        ))


if __name__ == '__main__':
    unittest.main()
