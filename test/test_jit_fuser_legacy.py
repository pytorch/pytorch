import sys
sys.argv.append("--jit_executor=legacy")
from test_jit_fuser import *  # noqa: F403

if __name__ == '__main__':
    run_tests()
