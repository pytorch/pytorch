# Owner(s): ["module: unknown"]
import tempfile

from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.waitcounterfilelock import WaitCounterFileLock


class TestWaitCounterFileLock(TestCase):
    def test_checkpoint_trigger(self):
        _, p = tempfile.mkstemp()
        with WaitCounterFileLock(p):
            pass


if __name__ == "__main__":
    run_tests()
