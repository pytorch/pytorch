
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.log_compilation_metrics = False
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm_backends = 'CUTLASS'
torch._inductor.config.cuda.version = '12.1'
torch._inductor.config.cuda.cutlass_dir = '/data/users/jezng/pytorch/test/inductor/../../third_party/cutlass/'
torch._inductor.config.cuda.cutlass_max_profiling_configs = 4
torch._inductor.config.trace.enabled = True
torch._inductor.config.trace.debug_dir = '/data/users/jezng/pytorch/tmp/test_code'




isolate_fails_code_str = None



# torch version: 2.3.0a0+git9cfb20e
# torch cuda version: 12.1
# torch git version: 9cfb20e69188216245950b30ddb8a969847529ff


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Mon_Apr__3_17:16:06_PDT_2023 
# Cuda compilation tools, release 12.1, V12.1.105 
# Build cuda_12.1.r12.1/compiler.32688072_0 

# GPU Hardware Info: 
# NVIDIA H100 : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        mm = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        mul = torch.ops.aten.mul.Tensor(arg2_1, 3.3);  arg2_1 = None
        sub = torch.ops.aten.sub.Tensor(mm, mul);  mm = mul = None
        return (sub,)
        
def load_args(reader):
    buf0 = reader.storage(None, 16777216, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (2048, 4096), dtype=torch.float16, is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 4194304, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf1, (4096, 512), dtype=torch.float16, is_leaf=True)  # arg1_1
    buf2 = reader.storage(None, 2097152, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf2, (2048, 512), dtype=torch.float16, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
