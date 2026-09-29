import torch
import torch.fx as fx
from functorch import make_fx
from torch._functorch.compile_utils import fx_graph_cse
from torch.profiler import profile, ProfilerActivity


def profile_it(f, inp):
    # Warm up the GPU before measuring.
    # The first few calls can include setup work that affects the timing.
    for _ in range(5):
        f(inp)

    itr = 5

    # Record CUDA activity while running the function five times.
    # record_shapes also asks the profiler to collect input-shape information.
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(itr):
            f(inp)

    # Group matching profiler events together.
    timing = prof.key_averages()

    # Add up their reported CUDA times and average across the five runs.
    # These values are in microseconds.
    #
    # This measures accumulated profiler time, not end-to-end wall-clock
    # latency. Summing inclusive event times can also double-count work
    # if overlapping parent/child events are included.
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total

    return cuda_time_total / itr


def profile_function(name, f, inp):
    # Trace the function using this example input.
    # The result is an executable FX graph of PyTorch operations.
    #
    # Fixed Python loops in these examples are unrolled during tracing:
    # their repeated operations become individual graph nodes.
    fx_g = make_fx(f)(inp)

    # Apply common subexpression elimination.
    # For example, several identical x.sum() nodes can share one result.
    new_g = fx_graph_cse(fx_g.graph)

    # Wrap the optimized graph in a module so we can execute it.
    new_g = fx.GraphModule(fx_g, new_g)

    # Benchmark the FX modules directly.
    # TorchScript already performs some CSE, which would make it harder
    # to isolate the effect of the optimization we applied above.
    #
    # script_f = torch.jit.script(fx_g)
    # script_g = torch.jit.script(new_g)
    # avg_cuda_time_f = profile_it(script_f, inp)
    # avg_cuda_time_g = profile_it(script_g, inp)

    avg_cuda_time_f = profile_it(fx_g, inp)
    avg_cuda_time_g = profile_it(new_g, inp)

    # Count how many graph nodes the optimization removed.
    # Graph nodes also include things such as the input and output nodes,
    # so this is not a count of GPU kernels.
    num_node_decrease = len(fx_g.graph.nodes) - len(new_g.graph.nodes)

    # Output columns:
    # name, original GPU time, optimized GPU time,
    # nodes removed, original node count
    print(
        f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, "
        f"{num_node_decrease}, {len(fx_g.graph.nodes)}"
    )


# Use a seeded CUDA generator so the input is reproducible.
g_gpu = torch.Generator(device="cuda")
g_gpu.manual_seed(2147483647)

# Create 1,048,576 random values directly on the GPU.
# This script requires CUDA support and an available CUDA device.
inp = torch.randn(2**20, device="cuda", generator=g_gpu)


def f1(x):
    # This computes cos(cos(x)).
    # Both operations are cosine, but their inputs are different:
    # the second one uses the result of the first.
    # CSE should therefore keep both operations.
    return x.cos().cos()


profile_function("f1", f1, inp)


def fsum(x):
    # Compute the exact same sum four times.
    # CSE can calculate it once and reuse that scalar result.
    a = x.sum()
    b = x.sum()
    c = x.sum()
    d = x.sum()

    # The additions remain; CSE itself does not rewrite this as 4 * a.
    return a + b + c + d


profile_function("fsum", fsum, inp)


def fconcat(x):
    # Both concatenations have the same inputs in the same order.
    # CSE can reuse one concatenated tensor for both a and b.
    a = torch.cat((x, x))
    b = torch.cat((x, x))
    return a + b


profile_function("fconcat", fconcat, inp)


def fsum2(x):
    # There are 31 identical sums: one here and 30 inside the loop.
    a = x.sum()

    for _ in range(30):
        # x never changes, so its sum can be reused.
        # a does change, so the chain of additions still has to run.
        a = a + x.sum()

    return a


profile_function("fsum2", fsum2, inp)


def fsummulti(x):
    a = 0

    for _ in range(3):
        # Six identical sums in total: two per iteration.
        # CSE can share their result across both operations
        # and across all three iterations.
        a = a + x.sum()
        a = a * x.sum()

    return a


profile_function("fsummulti", fsummulti, inp)


def fsummulti2(x):
    # The same pattern as fsummulti, with more repeated work.
    a = 0

    for _ in range(30):
        # There are 60 identical sum calls available for CSE to merge.
        a = a + x.sum()
        a = a * x.sum()

    # Repeated multiplication can make a extremely large and may
    # overflow. This benchmark does not check output correctness.
    return a


profile_function("fsummulti2", fsummulti2, inp)


def fcos(x):
    a = 0

    for _ in range(3):
        # Each call computes cosine for every element of the same x.
        # CSE can compute that tensor once and reuse it three times.
        a = a + x.cos()

    return a


profile_function("fcos", fcos, inp)


def fcos2(x):
    a = 0

    for _ in range(30):
        # Same idea, now with 30 identical cosine calls.
        # CSE can remove the duplicate cosine calculations,
        # while keeping the additions that accumulate their results.
        a = a + x.cos()

    return a


profile_function("fcos2", fcos2, inp)
