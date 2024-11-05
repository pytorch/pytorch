# Owner(s): ["module: unknown"]
import torch
import tempfile

from torch.utils.waitcounterfilelock import WaitCounterFileLock
from torch.testing._internal.common_utils import run_tests, TestCase



class TestWaitCounterFileLock(TestCase):
    def test_checkpoint_trigger(self):
        _, p = tempfile.mkstemp()
        with WaitCounterFileLock(p):
            pass

if __name__ == "__main__":
    run_tests()
