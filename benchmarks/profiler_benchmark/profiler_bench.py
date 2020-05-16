import itertools
import statistics
import timeit
import torch

profiling_enabled = None
profiling_tensor_size = None
TENSOR_SIZES = [1, 32, 128, 256, 512]
INTERNAL_ITER = 256
PARALLEL_TASKS_NUM = 4
N = 100

def loop_workload(x):
    for i in range(INTERNAL_ITER):
        x = torch.mm(x, x)
    return x

traced_loop_workload = None
def run_profiler_benchmark_loop():
    x = torch.rand(profiling_tensor_size, profiling_tensor_size)
    if profiling_enabled:
        with torch.autograd.profiler.profile() as prof:
            traced_loop_workload(x)
    else:
        traced_loop_workload(x)

def parallel_task(x):
    for i in range(int(INTERNAL_ITER / PARALLEL_TASKS_NUM)):
        x = torch.mm(x, x)
    return x

def parallel_workload(x):
    futs = []
    for i in range(PARALLEL_TASKS_NUM):
        futs.append(torch.jit._fork(parallel_task, x))
    for i in range(PARALLEL_TASKS_NUM):
        torch.jit._wait(futs[i])
    return x

traced_parallel_workload = None
def run_profiler_benchmark_parallel():
    x = torch.rand(profiling_tensor_size, profiling_tensor_size)
    if profiling_enabled:
        with torch.autograd.profiler.profile() as prof:
            traced_parallel_workload(x)
    else:
        traced_parallel_workload(x)

if __name__ == '__main__':
    for workload_name in ["loop", "parallel"]:
        print("Payload: {}; {} iterations, N = {}\n".format(
            workload_name, INTERNAL_ITER, N))
        for params in itertools.product(TENSOR_SIZES, [False, True]):
            profiling_tensor_size = params[0]
            profiling_enabled = params[1]

            print("Profiling {}, tensor size {}x{}".format(
                "enabled " if profiling_enabled else "disabled",
                profiling_tensor_size, profiling_tensor_size))

            x = torch.rand(profiling_tensor_size, profiling_tensor_size)
            workload = None
            if workload_name == "loop":
                workload = run_profiler_benchmark_loop
                traced_loop_workload = torch.jit.trace(loop_workload, x)
            elif workload_name == "parallel":
                workload = run_profiler_benchmark_parallel
                traced_parallel_workload = torch.jit.trace(
                    parallel_workload, x)

            runtimes = timeit.repeat(workload, repeat=N, number=1)
            avg_time = statistics.mean(runtimes) * 1000.0
            stddev_time = statistics.stdev(runtimes) * 1000.0
            print("\tavg. time: {:.3f} ms, stddev: {:.3f} ms".format(
                avg_time, stddev_time))
            print("\ttime per iteration: {:.3f} ms\n".format(
                avg_time / INTERNAL_ITER))
