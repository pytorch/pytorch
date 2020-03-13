import torch
import time

def run_profiler_benchmark(n_iters):
    """
    Run a tight loop with a single matmul under the autograd profiler
    and report the latency. Useful for profiling changes to the profiler
    that may affect its performance.
    """
    latencies = []
    for i in range(n_iters):
        start = time.time()
        # Run a bunch of iterations of a single op under the profiler.
        with torch.autograd.profiler.profile():
            for i in range(1000):
                torch.mm(torch.rand(3, 3), torch.randn(3, 3))
        end = time.time()
        latencies.append(end - start)
    avg = torch.mean(torch.tensor(latencies, dtype=float))
    print("Iters: {} Profiler Latency: {} s".format(n_iters, avg))


if __name__ == '__main__':
    run_profiler_benchmark(100)
