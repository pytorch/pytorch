import torch
import torch.fx as fx
from functorch import make_fx
from torch._functorch.compile_utils import fx_graph_cse
from torch.profiler import profile, ProfilerActivity

def profile_it(f, inp):
    # 1. Warm-up to initialize CUDA context and internal buffers
    for _ in range(10):
        _ = f(inp)
    torch.cuda.synchronize()

    # 2. Benchmark with proper synchronization to capture GPU execution time
    itr = 20
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(itr):
            f(inp)
            torch.cuda.synchronize() 

    timing = prof.key_averages()
    # FIXED: Accessing device_time_total to accumulate total GPU processing time
    cuda_time_total = sum(e.device_time_total for e in timing)
    return cuda_time_total / itr

def profile_function(name, f, inp):
    fx_g = make_fx(f)(inp)
    new_g_graph = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_g_graph)
    
    avg_cuda_time_f = profile_it(fx_g, inp)
    avg_cuda_time_g = profile_it(new_g, inp)
    num_node_decrease = len(fx_g.graph.nodes) - len(new_g.graph.nodes)

    # Output results in a clear table format
    print(f"{name:<15}, {avg_cuda_time_f/1e3:>12.2f}, {avg_cuda_time_g/1e3:>12.2f}, {num_node_decrease:>10}, {len(fx_g.graph.nodes):>10}")

# --- Execution ---
print(f"{'Function':<15}, {'Original(us)':>12}, {'CSE(us)':>12}, {'Nodes Red.':>10}, {'Total Nodes':>10}")
print("-" * 75)

g_gpu = torch.Generator(device="cuda")
g_gpu.manual_seed(2147483647)
inp = torch.randn(2**20, device="cuda", generator=g_gpu)

# Tests
profile_function("f1", lambda x: x.cos().cos(), inp)
profile_function("fsum", lambda x: x.sum() + x.sum() + x.sum() + x.sum(), inp)
profile_function("fconcat", lambda x: torch.cat((x, x)) + torch.cat((x, x)), inp)
profile_function("fsum2", lambda x: (lambda a: [a := a + x.sum() for _ in range(30)][-1])(x.sum()), inp)
profile_function("fsummulti", lambda x: (lambda a: [a := (a + x.sum()) * x.sum() for _ in range(3)][-1])(0), inp)
profile_function("fsummulti2", lambda x: (lambda a: [a := (a + x.sum()) * x.sum() for _ in range(30)][-1])(0), inp)
profile_function("fcos", lambda x: (lambda a: [a := a + x.cos() for _ in range(3)][-1])(0), inp)
profile_function("fcos2", lambda x: (lambda a: [a := a + x.cos() for _ in range(30)][-1])(0), inp)
