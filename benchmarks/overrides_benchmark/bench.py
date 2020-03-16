import torch
import time

from common import SubTensor, WithTorchFunction, SubWithTorchFunction

NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000


def bench(t1, t2):
    bench_times = []
    for _ in range(NUM_REPEAT_OF_REPEATS):
        time_start = time.time()
        for _ in range(NUM_REPEATS):
            torch.add(t1, t2)
        bench_times.append(time.time() - time_start)

    bench_time = float(torch.min(torch.Tensor(bench_times))) / 1000
    bench_std = float(torch.std(torch.Tensor(bench_times))) / 1000

    return bench_time, bench_std


def main():
    types = [torch.Tensor, SubTensor, WithTorchFunction, SubWithTorchFunction]

    for t in types:
        tensor_1 = t(1)
        tensor_2 = t(2)

        bench_min, bench_std = bench(tensor_1, tensor_2)
        print(
            "Type {0} had a minimum time of {1} μs"
            " and a standard deviation of {2} μs.".format(
                t.__name__, (10 ** 6) * bench_min, (10 ** 6) * bench_std,
            )
        )


if __name__ == "__main__":
    main()
