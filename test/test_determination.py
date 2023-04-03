# Owner(s): ["module: ci"]

import os

import run_test
from torch.testing._internal.common_utils import TestCase, run_tests


class DummyOptions:
    verbose = False


class DeterminationTest(TestCase):
    # Test determination on a subset of tests
    TESTS = [
        "test_nn",
        "test_jit_profiling",
        "test_jit",
        "test_torch",
        "test_cpp_extensions_aot_ninja",
        "test_cpp_extensions_aot_no_ninja",
        "test_utils",
        "test_determination",
        "test_quantization",
    ]

    @classmethod
    def determined_tests(cls, changed_files):
        changed_files = [os.path.normpath(path) for path in changed_files]
        return [
            test
            for test in cls.TESTS
            if run_test.should_run_test(run_test.TARGET_DET_LIST, test, changed_files, DummyOptions())
        ]

    def test_target_det_list_is_sorted(self):
        # We keep TARGET_DET_LIST sorted to minimize merge conflicts
        # but most importantly to allow us to comment on the absence
        # of a test. It would be very difficult to add a file right
        # next to a comment that says to keep it out of the list.
        self.assertListEqual(run_test.TARGET_DET_LIST, sorted(run_test.TARGET_DET_LIST))

    def test_config_change_only(self):
        """CI configs trigger all tests"""
        self.assertEqual(
            self.determined_tests([".ci/pytorch/test.sh"]), self.TESTS
        )

    def test_run_test(self):
        """run_test.py is imported by determination tests"""
        self.assertEqual(
            self.determined_tests(["test/run_test.py"]), ["test_determination"]
        )

    def test_non_code_change(self):
        """Non-code changes don't trigger any tests"""
        self.assertEqual(
            self.determined_tests(["CODEOWNERS", "README.md", "docs/doc.md"]), []
        )

    def test_cpp_file(self):
        """CPP files trigger all tests"""
        self.assertEqual(
            self.determined_tests(["aten/src/ATen/native/cpu/Activation.cpp"]),
            self.TESTS,
        )

    def test_test_file(self):
        """Test files trigger themselves and dependent tests"""
        self.assertEqual(
            self.determined_tests(["test/test_jit.py"]), ["test_jit_profiling", "test_jit"]
        )
        self.assertEqual(
            self.determined_tests(["test/jit/test_custom_operators.py"]),
            ["test_jit_profiling", "test_jit"],
        )
        self.assertEqual(
            self.determined_tests(["test/quantization/eager/test_quantize_eager_ptq.py"]),
            ["test_quantization"],
        )

    def test_test_internal_file(self):
        """testing/_internal files trigger dependent tests"""
        self.assertEqual(
            self.determined_tests(["torch/testing/_internal/common_quantization.py"]),
            [
                "test_jit_profiling",
                "test_jit",
                "test_quantization",
            ],
        )

    def test_torch_file(self):
        """Torch files trigger dependent tests"""
        self.assertEqual(
            # Many files are force-imported to all tests,
            # due to the layout of the project.
            self.determined_tests(["torch/onnx/utils.py"]),
            self.TESTS,
        )
        self.assertEqual(
            self.determined_tests(
                [
                    "torch/autograd/_functions/utils.py",
                    "torch/autograd/_functions/utils.pyi",
                ]
            ),
            ["test_utils"],
        )
        self.assertEqual(
            self.determined_tests(["torch/utils/cpp_extension.py"]),
            [
                "test_cpp_extensions_aot_ninja",
                "test_cpp_extensions_aot_no_ninja",
                "test_utils",
                "test_determination",
            ],
        )

    def test_caffe2_file(self):
        """Caffe2 files trigger dependent tests"""
        self.assertEqual(self.determined_tests(["caffe2/python/brew_test.py"]), [])
        self.assertEqual(
            self.determined_tests(["caffe2/python/context.py"]), self.TESTS
        )

    def test_new_folder(self):
        """New top-level Python folder triggers all tests"""
        self.assertEqual(self.determined_tests(["new_module/file.py"]), self.TESTS)

    def test_new_test_script(self):
        """New test script triggers nothing (since it's not in run_tests.py)"""
        self.assertEqual(self.determined_tests(["test/test_new_test_script.py"]), [])


if __name__ == "__main__":
    run_tests()
