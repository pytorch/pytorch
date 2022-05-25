import copy
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
    dist.broadcast(x, 0)
    print(f"{os.getpid()} broadcast: {x}")
    return x

def all_reduce(x):
    # AOT in general don't support in-place functions since
    # it actually executes the function twice: one for shape inference,
    # and one for tracing during the first time.
    # So, let's work around by clone the input.
    xc = x.clone()
    dist.all_reduce(xc)
    print(f"{os.getpid()} all_reduce: {xc}")
    return xc

def all_gather(xs, x):
    dist.all_gather(xs, x)
    print(f"{os.getpid()} all_gather: {xs}")
    return xs

# The compiler_fn is called after the forward and backward graphs are extracted.
# Here, we just print the code in the compiler_fn. Return of this function is a callable.
def compiler_fn(fx_module: torch.fx.GraphModule, _):
    print(fx_module.code)
    with torch.jit.fuser("fuser2"):
        return torch.jit.script(fx_module)

def demo_basic(rank, world_size):
    # without flushing, mp won't print this message if crash.
    print(f"Running basic DDP example on rank {rank}, and process id: {os.getpid()}", flush=True)
    setup(rank, world_size)

    device = torch.device("cuda", dist.get_rank())
    x = torch.full((2, 3), dist.get_rank() + 1).to(device)

    # broadcast
    aot_print_fn = aot_function(broadcast, fw_compiler=compiler_fn, bw_compiler=compiler_fn)
    res = aot_print_fn(copy.deepcopy(x))
    ref = broadcast(copy.deepcopy(x))
    assert torch.allclose(ref, res)

    #all_reduce
    aot_print_fn = aot_function(all_reduce, fw_compiler=compiler_fn, bw_compiler=compiler_fn)
    res = aot_print_fn(copy.deepcopy(x))
    ref = all_reduce(copy.deepcopy(x))
    assert torch.allclose(ref, res)

    #all_gather
    xs = [torch.zeros(2,3, dtype=torch.int64).to(device) for _ in range(dist.get_world_size())]
    aot_print_fn = aot_function(all_gather, fw_compiler=compiler_fn, bw_compiler=compiler_fn)
    ress = aot_print_fn(copy.deepcopy(xs), copy.deepcopy(x))
    refs = all_gather(copy.deepcopy(xs), copy.deepcopy(x))
    assert all([torch.allclose(ref, res) for ref, res in zip(refs, ress)])

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
