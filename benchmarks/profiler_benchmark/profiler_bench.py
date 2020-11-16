import argparse
import statistics
import sys
import timeit
import torch

from torch.utils._benchmark import Timer

PARALLEL_TASKS_NUM = 4
INTERNAL_ITER = None
def loop_workload(x):
    for i in range(INTERNAL_ITER):
        x = torch.mm(x, x)
    return x

def parallel_workload(x):
    def parallel_task(x):
        for i in range(int(INTERNAL_ITER / PARALLEL_TASKS_NUM)):
            x = torch.mm(x, x)
        return x
    futs = []
    for i in range(PARALLEL_TASKS_NUM):
        futs.append(torch.jit._fork(parallel_task, x))
    for i in range(PARALLEL_TASKS_NUM):
        torch.jit._wait(futs[i])
    return x


if __name__ == '__main__':
    torch._C._set_graph_executor_optimize(False)
    parser = argparse.ArgumentParser(
        description='Profiler benchmark')

    parser.add_argument('--with_cuda', action='store_true')
    parser.add_argument('--with_stack', action='store_true')
    parser.add_argument('--use_script', action='store_true')
    parser.add_argument('--profiling_tensor_size', default=1, type=int)
    parser.add_argument('--workload', default='loop', type=str)
    parser.add_argument('--internal_iter', default=256, type=int)
    parser.add_argument('--n', default=100, type=int)
    parser.add_argument('--use_timer', action='store_true')
    parser.add_argument('--timer_min_run_time', default=100, type=int)

    args = parser.parse_args()

    if args.with_cuda and not torch.cuda.is_available():
        print("No CUDA available")
        sys.exit()

    print("Payload: {}; {} iterations, N = {}\n".format(
        args.workload, args.internal_iter, args.n))
    INTERNAL_ITER = args.internal_iter

    for profiling_enabled in [False, True]:
        print("Profiling {}, tensor size {}x{}, use cuda: {}, with stacks: {}, use script: {}".format(
            "enabled" if profiling_enabled else "disabled",
            args.profiling_tensor_size,
            args.profiling_tensor_size,
            args.with_cuda,
            args.with_stack,
            args.use_script))

        input_x = torch.rand(
            args.profiling_tensor_size,
            args.profiling_tensor_size)

        if args.with_cuda:
            input_x = input_x.cuda()

        workload = None
        assert args.workload in ["loop", "parallel"]
        if args.workload == "loop":
            workload = loop_workload
        else:
            workload = parallel_workload

        if args.use_script:
            traced_workload = torch.jit.trace(workload, (input_x,))
            workload = traced_workload

        if profiling_enabled:
            def payload():
                x = None
                with torch.autograd.profiler.profile(
                        use_cuda=args.with_cuda,
                        with_stack=args.with_stack) as prof:
                    x = workload(input_x)
                return x
        else:
            def payload():
                return workload(input_x)

        if args.use_timer:
            t = Timer(
                "payload()",
                globals={"payload": payload},
                timer=timeit.default_timer,
            ).blocked_autorange(min_run_time=args.timer_min_run_time)
            print(t)
        else:
            runtimes = timeit.repeat(payload, repeat=args.n, number=1)
            avg_time = statistics.mean(runtimes) * 1000.0
            stddev_time = statistics.stdev(runtimes) * 1000.0
            print("\tavg. time: {:.3f} ms, stddev: {:.3f} ms".format(
                avg_time, stddev_time))
            if args.workload == "loop":
                print("\ttime per iteration: {:.3f} ms".format(
                    avg_time / args.internal_iter))
        print()
