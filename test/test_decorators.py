import unittest
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU, onlyCUDA, onlyOn, skipIfRocm, dtypes
from torch.testing._internal.common_utils import TestCase, run_tests, expectedFailureMeta, skipIfNoLapack, TEST_NUMPY

class TestDecorators(TestCase):

    @expectedFailureMeta
    def test_expected_to_fail(self):
        self.assertEqual(1, 2)

    def run_single_test(self, test_method):
        test_suite = unittest.defaultTestLoader.loadTestsFromTestCase(type("TempTest", (TestDecorators,), {
            'single_test': test_method
        }))

        class TestExecutionChecker(unittest.TestResult):
            executed = False

            def addExpectedFailure(self, test, err):
                self.executed = True

        result = TestExecutionChecker()
        test_suite.run(result)
        return result.executed

    def test_check_failure(self):
        # Run the test_expected_to_fail
        result = self.run_single_test(self.test_expected_to_fail)

        # Check if the test was executed and registered as expected failure
        self.assertTrue(result)

    @onlyCPU
    def test_only_cpu(self):
        self.assertEqual(str(self.device), 'cpu')

    @onlyCUDA
    def test_only_cuda(self):
        self.assertIn("cuda", str(self.device).lower())

    @onlyOn('cpu', 'cuda')
    def test_only_on_cpu_or_cuda(self):
        self.assertIn(str(self.device), ['cpu', 'cuda'])

    @skipIfRocm
    def test_skip_if_rocm(self):
        self.assertNotIn("rocm", str(self.device).lower())

    @skipIfNoLapack
    def test_skip_if_no_lapack(self):
        import torch
        self.assertTrue(torch._C.has_lapack)

    @TEST_NUMPY
    def test_test_numpy(self):
        import numpy as np
        self.assertIsNotNone(np)

    @dtypes(torch.float32, torch.float64)
    def test_dtypes(self, dtype):
        expected_dtypes = (torch.float32, torch.float64)
        self.assertIn(dtype, expected_dtypes)

if __name__ == '__main__':
    run_tests()
