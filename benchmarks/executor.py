import torch
import argparse

def set_mode():
    parser = argparse.ArgumentParser("Runs a simple gru benchmark")
    parser.add_argument("--mode", type=str, default="se")
    parser.add_argument("--test", type=str, required=True)
    pargs = parser.parse_args()
    if pargs.mode == 'te':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
        # torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
    elif pargs.mode == 'le':
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_bailout_depth(20)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif pargs.mode == 'pe':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_set_bailout_depth(20)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif pargs.mode == 'se':
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_bailout_depth(20)
    else:
        raise AssertionError("Unexpected Mode")

    return (pargs.mode, pargs.test)

