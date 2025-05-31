# Owner(s): ["oncall: distributed"]

from torch.testing._internal.common_distributed import MultiProcContinousTest
from torch.testing._internal.common_utils import run_tests


class TestTemplate(MultiProcContinousTest):
    def testABC(self):
        print(f"rank {self.rank} of {self.world_size} testing ABC")

    def testDEF(self):
        print(f"rank {self.rank} of {self.world_size} testing DEF")


if __name__ == "__main__":
    run_tests()
