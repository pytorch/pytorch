import subprocess

from torch.testing._internal.common_utils import TestCase, run_tests


class TestImportTime(TestCase):
    # this test could eventually be changed to fail if the import time
    # is greater than a certain threshold, but for now we just use it as
    # a way to track the duration of `import torch` in our ossci-metrics
    # S3 bucket (see tools/print_test_stats.py)
    def test_time_import_torch(self):
        subprocess.run(['python3', '-c', 'import torch'])


if __name__ == '__main__':
    run_tests()
