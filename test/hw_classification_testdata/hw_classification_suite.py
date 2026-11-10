"""One test class per kind of label, for TestHardwareClassifications in
test_testing.py, which runs this file with PYTORCH_TEST_EXCLUDE_HW_CLASSIFICATION
set. The labels are test data: the bodies are empty, UnclassifiedTest has no
label on purpose, and the device labels don't describe what the tests do."""

from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class GenericTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_generic(self):
        pass


class CpuTest(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_cpu(self):
        pass


class CudaTest(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cuda(self):
        pass


class UnclassifiedTest(TestCase):
    def test_unclassified(self):
        pass


if __name__ == "__main__":
    run_tests()
