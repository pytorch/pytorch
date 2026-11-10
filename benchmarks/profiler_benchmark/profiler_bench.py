import argparse
import sys
import timeit

import torch
from torch.utils.benchmark import Timer


INTERNAL_ITER = None


def loop_workload(x):
    for i in range(INTERNAL_ITER):
        x = torch.mm(x, x)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiler benchmark")

    parser.add_argument("--with-cuda", "--with_cuda", action="store_true")
    parser.add_argument("--with-stack", "--with_stack", action="store_true")
    parser.add_argument("--use-kineto", "--use_kineto", action="store_true")
    parser.add_argument(
        "--profiling-tensor-size", "--profiling_tensor_size", default=1, type=int
    )
    parser.add_argument("--workload", default="loop", choices=["loop"])
    parser.add_argument("--internal-iter", "--internal_iter", default=256, type=int)
    parser.add_argument(
        "--timer-min-run-time", "--timer_min_run_time", default=10, type=int
    )
    parser.add_argument("--cuda-only", "--cuda_only", action="store_true")

    args = parser.parse_args()

    if args.with_cuda and not torch.cuda.is_available():
        print("No CUDA available")
        sys.exit()

    print(
        f"Payload: {args.workload}, {args.internal_iter} iterations; timer min. runtime = {args.timer_min_run_time}\n"
    )
    INTERNAL_ITER = args.internal_iter

    for profiling_enabled in [False, True]:
        print(
            "Profiling {}, tensor size {}x{}, use cuda: {}, use kineto: {}, with stacks: {}".format(
                "enabled" if profiling_enabled else "disabled",
                args.profiling_tensor_size,
                args.profiling_tensor_size,
                args.with_cuda,
                args.use_kineto,
                args.with_stack,
            )
        )

        input_x = torch.rand(args.profiling_tensor_size, args.profiling_tensor_size)

        if args.with_cuda:
            input_x = input_x.cuda()

        workload = loop_workload

        if profiling_enabled:

            def payload():
                x = None
                with torch.autograd.profiler.profile(
                    use_device="cuda" if args.with_cuda else None,
                    with_stack=args.with_stack,
                    use_kineto=args.use_kineto,
                    use_cpu=not args.cuda_only,
                ):
                    x = workload(input_x)
                return x

        else:

            def payload():
                return workload(input_x)

        t = Timer(
            "payload()",
            globals={"payload": payload},
            timer=timeit.default_timer,
        ).blocked_autorange(min_run_time=args.timer_min_run_time)
        print(t)
