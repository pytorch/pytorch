import torch
from test_jit_fuser import *


if __name__ == "__main__":
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_set_texpr_fuser_enabled(True)
    torch._C._jit_register_tensorexpr_fuser()
    run_tests()
