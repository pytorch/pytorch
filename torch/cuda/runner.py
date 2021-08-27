import argparse
from functools import reduce
import operator
from operator import itemgetter

import torch
import torch.nn.functional as F

# Enable NVFuser
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_bailout_depth(20)


parser = argparse.ArgumentParser(description='Fusion Benchmark Runner')
parser.add_argument('--warmup-trials', default=5, type=int, help='Number of trials to not measure.')
parser.add_argument('--trials', default=100, type=int, help='Number of trials to average execution time over.')
parser.add_argument('--fp16', default=False, action='store_true', help='FP16 Precision.')

args = parser.parse_args()

def scaled_masked_softmax(x, mask, scale):
    x_masked = x.clone() * scale
    x_masked[mask] = -float(10000.0)

    return F.softmax(x_masked, dim=1).sum()

def upper_triang_scaled_masked_softmax(x, mask, scale):
    d1, d2 = x.size()
    masked_indices = torch.triu_indices(d1, d2)
    x_masked = x.clone() * scale
    x_masked[masked_indices[0], masked_indices[1]] = -float(10000.0)
    return F.softmax(x_masked, dim=1).sum()
def clear_l2_cache() :
    t0 = torch.empty(1024*1024*50, dtype=torch.float, device='cuda', requires_grad=False)
    t1 = t0.clone()


op_modules = [scaled_masked_softmax, upper_triang_scaled_masked_softmax]

# Keep runs consistent
torch.cuda.manual_seed(111)
data_type = torch.float16 if args.fp16 else torch.float32

op_impls = []
for mod in op_modules :
    op_impls.append(('Eager', mod))
    op_impls.append(('NVFuser', mod))
  
# Create Cuda Timing Events
start_evt_fwd = torch.cuda.Event(enable_timing=True)
stop_evt_fwd = torch.cuda.Event(enable_timing=True)
start_evt_bwd = None
stop_evt_bwd = None
if not args.inference :
    start_evt_bwd = torch.cuda.Event(enable_timing=True)
    stop_evt_bwd = torch.cuda.Event(enable_timing=True)
input_shape = (10,10,128,128)
mask_shape = (10,1,128,1)
scale = torch.randn(1,device="cuda", dtype=data_type, requires_grad=True)
inputs = torch.randn(input_shape, device="cuda", dtype=data_type, requires_grad=True)
mask = torch.randn(mask_shape, device="cuda", dtype=data_type, requires_grad=True) < 1
result = ''

# Setup Data Tensors

params = [scale, inputs, mask]
# Loop over model implemenatations
for impl in op_impls :
    if impl[0] == 'NVFuser' :
        model = torch.jit.script(impl[1])
    else :
        model = impl[1]

    if args.fp16 :
        model.half()
    model.cuda()

    elapsed_time_fwd = 0.0
    elapsed_time_bwd = 0.0
    for cnt in range(0, args.trials + args.warmup_trials) :
        # Setup Step
        inputs.grad = None
        for p in params:
            p.grad = None
        clear_l2_cache()

        # Time forward
        start_evt_fwd.record()
        out = model(inputs, mask, scale)
        stop_evt_fwd.record()

        # Time backward (if enabled)
        start_evt_bwd.record()
        out.backward(grads)
        stop_evt_bwd.record()

        # Collect timing results
        if cnt >= args.warmup_trials :
            torch.cuda.synchronize()
            elapsed_time_fwd += start_evt_fwd.elapsed_time(stop_evt_fwd)
            elapsed_time_bwd += start_evt_bwd.elapsed_time(stop_evt_bwd)

    fwd_time = elapsed_time_fwd / args.trials
    print(impl[0],impl[1],'foward time:', fwd_time)

    fwd_time = elapsed_time_bwd / args.trials
    print(impl[0],impl[1],'foward time:', bwd_time)
