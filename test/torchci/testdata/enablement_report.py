"""A run_tests script with one passing test (TestReportEnablement)."""

from torch.testing._internal.common_utils import run_tests, TestCase


class TestOne(TestCase):
    def test_pass(self):
        pass


if __name__ == "__main__":
    run_tests()
