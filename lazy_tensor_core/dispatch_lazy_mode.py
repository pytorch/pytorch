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
    inputs = [torch.randn(8, 8, device='cuda') for _ in range(tests) ]
 
    with torch.autograd.profiler.emit_nvtx() :
        for step in range(tests) :
            torch.cuda.nvtx.range_push("STEP: " + str(step))
            print(inputs[step])
            print(inputs[step].device)
            with lazy_execute() : 
                loss = f(inputs[step])
                print(loss.device)
            torch.cuda.nvtx.range_pop()
            print(loss.device)
            print(loss)
            print(metrics.metrics_report())

if __name__ == "__main__" :
    run_model()
