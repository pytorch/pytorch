import sys
sys.argv.append("--ge_config=profiling")
from test_jit_cuda_fuser import *

if __name__ == '__main__':
    run_tests()
