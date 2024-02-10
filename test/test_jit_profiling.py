# Owner(s): ["oncall: jit"]

import sys
sys.argv.append("--jit-executor=profiling")
from test_jit import run_tests

if __name__ == '__main__':
    run_tests()
