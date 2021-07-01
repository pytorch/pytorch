import sys

from torch.testing._internal.common_utils import run_tests

if __name__ == '__main__':
    sys.argv.append("--jit_executor=legacy")
    run_tests()
