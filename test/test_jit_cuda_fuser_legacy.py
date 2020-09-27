import sys
sys.argv.append("--ge_config=legacy")

import os
os.environ['PYTORCH_CUDA_FUSER_DISABLE_FALLBACK'] = '1'
os.environ['PYTORCH_CUDA_FUSER_DISABLE_FMA'] = '1'
os.environ['PYTORCH_CUDA_FUSER_JIT_OPT_LEVEL'] = '0'

from test_jit_cuda_fuser import *

if __name__ == '__main__':
    run_tests()
