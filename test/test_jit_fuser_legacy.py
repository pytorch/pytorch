# Owner(s): ["oncall: jit"]

import sys
sys.argv.append("--jit-executor=legacy")
from test_jit_fuser import run_tests

if __name__ == '__main__':
    run_tests()
