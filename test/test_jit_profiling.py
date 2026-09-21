# Owner(s): ["oncall: jit"]

import sys
sys.argv.append("--jit-executor=profiling")
from test_jit import *  # noqa: F403

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
