import torch

import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.debug.metrics as metrics
import lazy_tensor_core.core.lazy_model as ltm

torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)

class lazy_execute(object):
    def __init__(self, enabled=True):
        self._enabled = enabled
        
    def __enter__(self):
        lazy_tensor_core._LAZYC._ltc_enable_lazy_mode()
               
    def __exit__(self, type, value, traceback) : 
        lazy_tensor_core._LAZYC._ltc_sync_live_tensors(lazy_tensor_core._LAZYC._ltc_get_default_device(), [], False)
        lazy_tensor_core._LAZYC._ltc_disable_lazy_mode()
        ltm.mark_step()
        ltm.wait_device_ops()

lazy_tensor_core._LAZYC._ltc_disable_lazy_mode()
                            
def f(x):
    return x.relu()

def run_model() :
    torch.manual_seed(42) 
    tests = 1

    # initialize tensor on `cuda`
    inputs = [torch.randn(8, 8, device='cuda') for _ in range(tests) ]
 
    for step in range(tests) :
        print(inputs[step])
        print(inputs[step].device)  # tensor started on `cuda`
        with lazy_execute() : 
            loss = f(inputs[step])  # within lazy context, all tensors are lazily evaluated
            print(loss.device)
        # tensor should still reside on `cuda` device, we just need to
        # move/access eager tensor wrapped inside LTCTensorImpl
        print(loss.device)
        print(loss)
        print(metrics.metrics_report())

if __name__ == "__main__" :
    run_model()
