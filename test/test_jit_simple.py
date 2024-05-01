# Owner(s): ["oncall: jit"]

import sys
sys.argv.append("--jit-executor=simple")
from test_jit import *  # noqa: F403

if __name__ == '__main__':
    run_tests()
