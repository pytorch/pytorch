import contextlib

import torch

@contextlib.contextmanager
def optimized_execution(should_optimize):
    """
    A context manager that controls whether the JIT's executor will run
    optimizations before executing a function.
    """
    stored_flag = torch._C._get_graph_executor_optimize()
    torch._C._set_graph_executor_optimize(should_optimize)
    try:
        yield
    finally:
        torch._C._set_graph_executor_optimize(stored_flag)

@contextlib.contextmanager
def fuser(name):
    """
    A context manager that facilitates switching between
    backend fusers.

    Valid names:
    * ``fuser0`` - enables only legacy fuser
    * ``fuser1`` - enables only NNC
    * ``fuser2`` - enables only nvFuser
    """
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()
    if name == 'fuser0':  # legacy fuser
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(False)
    elif name == 'fuser1':  # NNC
        old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        old_profiling_mode = torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)
    elif name == 'fuser2':  # nvFuser
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
    else:
        raise Exception("unrecognized fuser option")
    try:
        yield
    finally:
        if name == 'fuser1':  # NNC
            torch._C._jit_set_profiling_executor(old_profiling_executor)
            torch._C._jit_set_profiling_mode(old_profiling_mode)
        # recover the previous values
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)


last_executed_optimized_graph = torch._C._last_executed_optimized_graph


def _graph_for(self, *args, **kwargs):
    self(*args, **kwargs)
    return last_executed_optimized_graph()
