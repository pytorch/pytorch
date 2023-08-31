"""
torchrun --standalone --nproc_per_node=2 run_fsdp.py
If you want slower (and hence more visible) collective kernels, then you can
run the following (i.e. with the env var):
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc_per_node=2 run_fsdp.py
"""
import logging
import os
from typing import Callable, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed._composable import checkpoint

NUM_ITERS = 2
logging.basicConfig(level=logging.INFO)
PROFILE_SAVE_DIR = "./profiles"
def init() -> Tuple[nn.Module, torch.optim.Optimizer]:
    torch.manual_seed(0)
    rank = dist.get_rank()
    model = nn.Transformer(d_model=1024, nhead=8, device="cuda")
    for module in model.modules():
        if isinstance(module, (nn.TransformerDecoderLayer, nn.TransformerEncoderLayer)):
            checkpoint(module)
    policy = ModuleWrapPolicy({nn.TransformerEncoderLayer, nn.TransformerDecoderLayer})
    fsdp_model = FSDP(
        model,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=policy,
        use_orig_params=True,
    )
    if rank == 0:
        print(model)
    optim = torch.optim.SGD(fsdp_model.parameters(), lr=1e-2)
    return fsdp_model, optim
def run():
    fsdp_model, optim = init()
    torch.manual_seed(dist.get_rank() + 1)
    src = torch.randn((10, 1, 1024), device="cuda")
    tgt = torch.randn((20, 1, 1024), device="cuda")
    def inner():
        for _ in range(NUM_ITERS):
            optim.zero_grad()
            loss = fsdp_model(src, tgt).sum()
            loss.backward()
            optim.step()
    # inner()
    benchmark_with_profiler(inner)
    # inner()
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
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    run()
if __name__ == "__main__":
    main()

