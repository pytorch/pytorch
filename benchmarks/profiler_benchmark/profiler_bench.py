import torch
import timeit

def run_profiler_benchmark():
    """
    Run a tight loop with a single matmul under the autograd profiler
    and report the latency. Useful for profiling changes to the profiler
    that may affect its performance.
    """
    # Run a bunch of iterations of a single op under the profiler.
    with torch.autograd.profiler.profile():
        for i in range(1000):
            torch.mm(torch.rand(3, 3), torch.randn(3, 3))


if __name__ == '__main__':
    n_iters = 100
    latencies = timeit.repeat(run_profiler_benchmark, repeat=n_iters, number=1)
    avg = torch.mean(torch.tensor(latencies, dtype=float))
    print("Iters: {} Profiler Latency: {} s".format(n_iters, avg))
