"""
step 1: pip install deepspeed
step 2: NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 run_deepspeed.py 2>&1 | tee run_deepspeed.log
step 3: modify DEEPSPEED_CONFIG_JSON if needed, like zero stage 0, 1, 2
"""
import logging
import deepspeed
import os
from typing import Callable, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
import sys
import pdb

logging.basicConfig(level=logging.INFO)
NUM_ITERS = 2
PROFILE_SAVE_DIR = "./profiles"
DEEPSPEED_CONFIG_FILE = "ds_config.json"

# deep speed configs
# https://www.deepspeed.ai/docs/config-json/#zero-optimizations-for-fp16-training
# zero stage 0: disabled
# zero stage 1: optimizer state partitioning
# zero stage 2: optimizer+gradient state partitioning
# zero stage 3: optimizer+gradient+parameter partitioning
# overlap_comm: overlap the reduction of the gradients with backward computation
DEEPSPEED_CONFIG_JSON = """
{
  "train_batch_size": 8,
  "fp16": {
    "enabled": true,
    "auto_cast": false
  },
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "session_params": {
    "zero_optimization": {
      "stage": 3
    },
    "overlap_comm": true
  }
}
"""



class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def init() -> Tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    model = nn.Transformer(d_model=1024, nhead=8, num_encoder_layers=2, num_decoder_layers=2, device="cuda")

    # create deepspeed config file
    with open(DEEPSPEED_CONFIG_FILE, "w") as config_file:
        config_file.write(DEEPSPEED_CONFIG_JSON)
    
    # wrap nn.module with deepspeed
    wrapped_model, optim, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=DEEPSPEED_CONFIG_FILE)

    # print model
    rank = dist.get_rank()
    if rank == 0:
        print("wrapped model:\n", wrapped_model)
    
    return wrapped_model, optim

def run():
    wrapped_model, optim = init()
    # if dist.get_rank() == 0:
    #     ForkedPdb().set_trace()
    
    torch.manual_seed(dist.get_rank() + 1)
    src = torch.randn((10, 1, 1024), device="cuda", dtype=torch.half)
    tgt = torch.randn((20, 1, 1024), device="cuda", dtype=torch.half)
    def inner():
        for _ in range(NUM_ITERS):
            loss = wrapped_model(src, tgt).sum()
            wrapped_model.backward(loss)
            wrapped_model.step()
    # inner()
    benchmark_with_profiler(inner)

def benchmark_with_profiler(
    benchmark_fn: Callable,
    *benchmark_fn_args,
    **benchmark_fn_kwargs,
) -> None:
    """
    PyTorch profiler:
    - Tutorial: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    - API: https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    """
    wait, warmup, active = 0, 1, 2
    num_steps = wait + warmup + active
    rank = get_rank()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1, skip_first=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_SAVE_DIR)
        if not rank  # only save on rank 0
        else None,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,  # incurs an additional overhead; disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models at the moment
        experimental_config=torch.profiler._ExperimentalConfig(
            enable_cuda_sync_events=True
        ),
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            benchmark_fn(*benchmark_fn_args, **benchmark_fn_kwargs)
            if rank is None or rank == 0:
                prof.step()  # notify the profiler at end of each step

def get_rank() -> Optional[int]:
    try:
        rank = torch.distributed.get_rank()
    except RuntimeError:
        rank = None
    return rank

def main():
    # use nccl backend by default
    deepspeed.init_distributed(dist_backend='nccl', verbose=True)
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    run()

if __name__ == "__main__":
    main()
