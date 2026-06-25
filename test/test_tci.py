# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.tci import schedule


class TestTci(TestCase):
    def test_schedule_marker_strings(self):
        self.assertEqual(str(schedule.pull), "tci.schedule:pull")
        self.assertEqual(str(schedule.periodic), "tci.schedule:periodic")


if __name__ == "__main__":
    run_tests()
