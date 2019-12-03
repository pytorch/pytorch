import timeit
import torch
import math

total_time_each_loop = 0.01  # seconds
max_loop_size = 1000
num_loops = 10

def seconds_to_gs(numel, seconds):
    return numel / seconds / (1024 * 1024 * 1024)

def get_loop_size(f):
    start = timeit.default_timer()
    f()
    torch.cuda.synchronize()
    end = timeit.default_timer()
    elapsed = end - start
    return min(max_loop_size, max(1, int(total_time_each_loop / elapsed)))

def time_one_loop(f):
    def timer():
        loop_size = get_loop_size(f)
        start = timeit.default_timer()
        for _ in range(loop_size):
            f()
        end = timeit.default_timer()
        return (end - start) / loop_size
    return timer

def time_one_loop_cuda(f):
    def timer():
        loop_size = get_loop_size(f)
        torch.cuda.synchronize()
        start = timeit.default_timer()
        for _ in range(loop_size):
            f()
            torch.cuda.synchronize()
        end = timeit.default_timer()
        return (end - start) / loop_size
    return timer

def time_func(one_loop_timer, numel):
    min_elapsed = math.inf
    for _ in range(num_loops):
        elapsed = one_loop_timer()
        min_elapsed = min(min_elapsed, elapsed)
    return seconds_to_gs(numel, min_elapsed)
