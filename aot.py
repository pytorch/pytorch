import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from functorch.compile import aot_function
from functorch.compile import clear_compile_cache

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def broadcast(x):
    # torch.add(x, 1) # Just to see if __torch_dispatch__ functions.
    dist.broadcast(x, 0)
    print(f"{os.getpid()} broadcast: {x}")
    return x

# The compiler_fn is called after the forward and backward graphs are extracted.
# Here, we just print the code in the compiler_fn. Return of this function is a callable.
def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    return fx_module

def demo_basic(rank, world_size):
    # without flushing, mp won't print this message if crash.
    print(f"Running basic DDP example on rank {rank}, and process id: {os.getpid()}", flush=True)
    setup(rank, world_size)

    # Pass on the compiler_fn to the aot_function API
    aot_print_fn = aot_function(broadcast, fw_compiler=compiler_fn, bw_compiler=compiler_fn)

    # Run the aot_print_fn once to trigger the compilation and print the graphs
    device = torch.device("cuda", dist.get_rank())
    x = torch.zeros(2, 3).to(device)
    if device.index == 0:
        x = torch.ones(2, 3).to(device)
    res = aot_print_fn(x)
    ref = broadcast(x)
    assert torch.allclose(ref, res)

    clear_compile_cache()

    cleanup()


def run_demo(demo_fn, world_size):
    # it won't print exception messages in the child process.
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    print(f"main process id: {os.getpid()}")
    n_gpus = torch.cuda.device_count()
    # assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)
    # demo_basic(0,1)
