from torch.testing._internal.common_utils import TestCase, run_tests


# these tests could eventually be changed to fail if the import/init
# time is greater than a certain threshold, but for now we just use them
# as a way to track the duration of `import torch` in our ossci-metrics
# S3 bucket (see tools/print_test_stats.py)
class TestImportTime(TestCase):
    def test_time_import_torch(self):
        TestCase.runWithPytorchAPIUsageStderr('import torch')

    def test_time_cuda_device_count(self):
        TestCase.runWithPytorchAPIUsageStderr(
            'import torch; torch.cuda.device_count()',
        )


if __name__ == '__main__':
    run_tests()
