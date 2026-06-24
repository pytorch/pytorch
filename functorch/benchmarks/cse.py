import torch
import torch.fx as fx

try:
    # Newer PyTorch exposes make_fx under torch.func
    from torch.func import make_fx
except Exception:
    # Fallback for older packaging
    from functorch import make_fx

from torch._functorch.compile_utils import fx_graph_cse
from torch.profiler import profile, ProfilerActivity


def profile_it(f, inp):
    """Profile function f on inp and return the average device time per iteration in microseconds.

    Improvements over the original:
    - Warm up the CUDA context and kernels to avoid measuring one-time setup costs.
    - Synchronize after warm-up and after each measured iteration to ensure we capture GPU execution time.
    - Increase iteration count for more stable averages.
    - Use a compatibility fallback for profiler timing attributes, skipping events
      that don't expose either attribute instead of failing the whole run.
    """
    # This benchmark requires CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark script")

    # 1. Warm-up to initialize CUDA context and internal buffers
    for _ in range(10):
        _ = f(inp)
    torch.cuda.synchronize()

    # 2. Benchmark with proper synchronization to capture GPU execution time
    itr = 20
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(itr):
            f(inp)
            # Ensures the GPU work finishes before proceeding to the next iteration
            torch.cuda.synchronize()

    timing = prof.key_averages()


    def _dev_time(e):
        if hasattr(e, "device_time_total"):
            return e.device_time_total
        elif hasattr(e, "cuda_time_total"):
            return e.cuda_time_total
        else:
            return None

    times = [_dev_time(e) for e in timing]
    valid_times = [t for t in times if t is not None]

    if not valid_times:

        raise RuntimeError(
            "Profiler timing attributes not found on any events (expected device_time_total or cuda_time_total)"
        )

    cuda_time_total = sum(valid_times)
    #profiler reports in microseconds
    return cuda_time_total / itr


def profile_function(name, f, inp):
    fx_g = make_fx(f)(inp)

    # Applying common-subexpression-elimination (CSE) to the fx graph
    new_g_graph = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_g_graph)

    # script_f = torch.jit.script(fx_g)
    # script_g = torch.jit.script(new_g)
    # avg_cuda_time_f = profile_it(script_f, inp)
    # avg_cuda_time_g = profile_it(script_g, inp)

    avg_cuda_time_f = profile_it(fx_g, inp)
    avg_cuda_time_g = profile_it(new_g, inp)
    num_node_decrease = len(fx_g.graph.nodes) - len(new_g.graph.nodes)

    # Print results in milliseconds with a clear header elsewhere
    print(
        f"{name:<15} {avg_cuda_time_f / 1e3:12.3f} {avg_cuda_time_g / 1e3:12.3f} {num_node_decrease:10d} {len(fx_g.graph.nodes):10d}"
    )


if __name__ == "__main__":
    # --- Execution ---
    print(
        f"{'Function':<15} {'Original(ms)':>12} {'CSE(ms)':>12} {'Nodes Red.':>10} {'Total Nodes':>10}"
    )
    print("-" * 75)

    g_gpu = torch.Generator(device="cuda")
    g_gpu.manual_seed(2147483647)
    inp = torch.randn(2**20, device="cuda", generator=g_gpu)

    def f1(x):
        return x.cos().cos()

    profile_function("f1", f1, inp)

    def fsum(x):
        a = x.sum()
        b = x.sum()
        c = x.sum()
        d = x.sum()
        return a + b + c + d

    profile_function("fsum", fsum, inp)

    def fconcat(x):
        a = torch.cat((x, x))
        b = torch.cat((x, x))
        return a + b

    profile_function("fconcat", fconcat, inp)

    def fsum2(x):
        a = x.sum()
        for _ in range(30):
            a = a + x.sum()
        return a

    profile_function("fsum2", fsum2, inp)

    def fsummulti(x):
        a = 0
        for _ in range(3):
            a = a + x.sum()
            a = a * x.sum()
        return a

    profile_function("fsummulti", fsummulti, inp)

    def fsummulti2(x):
        a = 0
        for _ in range(30):
            a = a + x.sum()
            a = a * x.sum()
        return a

    profile_function("fsummulti2", fsummulti2, inp)

    def fcos(x):
        a = 0
        for _ in range(3):
            a = a + x.cos()
        return a

    profile_function("fcos", fcos, inp)

    def fcos2(x):
        a = 0
        for _ in range(30):
            a = a + x.cos()
        return a

    profile_function("fcos2", fcos2, inp)
