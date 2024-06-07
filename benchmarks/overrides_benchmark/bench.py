import argparse
import time

from common import SubTensor, SubWithTorchFunction, WithTorchFunction

import torch

NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000


def bench(t1, t2):
    bench_times = []
    for _ in range(NUM_REPEAT_OF_REPEATS):
        time_start = time.time()
        for _ in range(NUM_REPEATS):
            torch.add(t1, t2)
        bench_times.append(time.time() - time_start)

    bench_time = float(torch.min(torch.tensor(bench_times))) / 1000
    bench_std = float(torch.std(torch.tensor(bench_times))) / 1000

    return bench_time, bench_std


def main():
    global NUM_REPEATS
    global NUM_REPEAT_OF_REPEATS

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
    args = parser.parse_args()

    NUM_REPEATS = args.nreps
    NUM_REPEAT_OF_REPEATS = args.nrepreps

    types = torch.tensor, SubTensor, WithTorchFunction, SubWithTorchFunction

    for t in types:
        tensor_1 = t([1.0])
        tensor_2 = t([2.0])

        bench_min, bench_std = bench(tensor_1, tensor_2)
        print(
            f"Type {t.__name__} had a minimum time of {10**6 * bench_min} us"
            f" and a standard deviation of {(10**6) * bench_std} us."
        )


if __name__ == "__main__":
    main()
