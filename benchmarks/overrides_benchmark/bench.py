import torch
import time
import argparse

from common import SubTensor, WithTorchFunction, SubWithTorchFunction

NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000
NUM_ARGS_IN_LIST = 2


def bench(f, *args):
    bench_times = []
    for _ in range(NUM_REPEAT_OF_REPEATS):
        time_start = time.time()
        for _ in range(NUM_REPEATS):
            f(*args)
        bench_times.append(time.time() - time_start)

    bench_time = float(torch.min(torch.Tensor(bench_times))) / 1000
    bench_std = float(torch.std(torch.Tensor(bench_times))) / 1000

    return bench_time, bench_std


def main():
    global NUM_REPEATS
    global NUM_REPEAT_OF_REPEATS
    global NUM_ARGS_IN_LIST

    parser = argparse.ArgumentParser(
        description="Run the __torch_function__ benchmarks."
    )
    parser.add_argument(
        "--nreps",
        "-n",
        type=int,
        default=NUM_REPEATS,
        help="The number of repeats for one measurement.",
    )
    parser.add_argument(
        "--nrepreps",
        "-m",
        type=int,
        default=NUM_REPEAT_OF_REPEATS,
        help="The number of measurements.",
    )
    parser.add_argument(
        "--nlist",
        "-l",
        type=int,
        default=NUM_REPEAT_OF_REPEATS,
        help="The number of tensors in a tensor list.",
    )
    args = parser.parse_args()

    NUM_REPEATS = args.nreps
    NUM_REPEAT_OF_REPEATS = args.nrepreps
    NUM_ARGS_IN_LIST = args.nlist

    types = torch.Tensor, SubTensor, WithTorchFunction, SubWithTorchFunction

    for t in types:
        tensor_1 = t(1)
        tensor_2 = t(2)

        bench_min, bench_std = bench(torch.add, tensor_1, tensor_2)
        print(
            "Type {0} had a minimum time of {1} us"
            " and a standard deviation of {2} us for torch.add.".format(
                t.__name__, (10 ** 6 * bench_min), (10 ** 6) * bench_std
            )
        )

        tensor_list = [tensor_1] * NUM_ARGS_IN_LIST
        bench_min, bench_std = bench(torch.cat, tensor_list)
        print(
            "Type {0} had a minimum time of {1} us"
            " and a standard deviation of {2} us for torch.cat.".format(
                t.__name__, (10 ** 6 * bench_min), (10 ** 6) * bench_std
            )
        )


if __name__ == "__main__":
    main()
