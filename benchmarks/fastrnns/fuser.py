import torch

def set_fuser(fuser_name, executor_name):
    assert fuser_name in ['te', 'old', 'none', 'default']
    if fuser_name == 'te':
        torch._C._jit_set_profiling_executor(True)
        torch._C._get_graph_executor_optimize(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
    elif fuser_name == 'old':
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == 'none':
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
    elif fuser_name == 'default':
        pass

    # --executor overrides settings of --fuser
    if executor_name == 'profiling':
        torch._C._jit_set_profiling_executor(True)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == 'simple':
        torch._C._get_graph_executor_optimize(False)
    elif executor_name == 'legacy':
        torch._C._jit_set_profiling_executor(False)
        torch._C._get_graph_executor_optimize(True)
    elif executor_name == 'default':
        pass
