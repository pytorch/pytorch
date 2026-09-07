# Owner(s): ["module: ci"]

from torch.testing._internal.common_utils import (
    HardwareClassification,
    TestCase,
    run_tests,
)


# these tests could eventually be changed to fail if the import/init
# time is greater than a certain threshold, but for now we just use them
# as a way to track the duration of `import torch`.
class TestImportTime(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_time_import_torch(self):
        TestCase.runWithPytorchAPIUsageStderr("import torch")

    def test_time_device_count(self):
        TestCase.runWithPytorchAPIUsageStderr(
            "import torch;torch.accelerator.device_count()",
        )


if __name__ == "__main__":
    run_tests()
