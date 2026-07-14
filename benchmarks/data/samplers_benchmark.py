#!/usr/bin/env python3

import time
from collections.abc import Iterable, Iterator
from typing import Union

import numpy as np
from tabulate import tabulate

from torch.utils.data import BatchSampler, Sampler, SequentialSampler


class NewBatchSampler(Sampler[list[int]]):
    """Alternative implementation of BatchSampler for benchmarking purposes."""

    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                f"drop_last should be a boolean value, but got drop_last={drop_last}"
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


def main():
    """Run benchmark with specified parameters."""
    DATA_SIZE = 99999
    AVG_TIMES = 10
    BATCH_SIZES = [4, 8, 64, 640, 6400, 64000]
    DROP_LAST_OPTIONS = [True, False]

    results = []

    # Set up samplers here, ensure right args are passed in
    baselineSampler = BatchSampler
    testSampler = NewBatchSampler

    for batch_size in BATCH_SIZES:
        for drop_last in DROP_LAST_OPTIONS:
            print(f"Benchmarking with batch_size={batch_size}, drop_last={drop_last}")

            # Benchmark baselineSampler
            original_times = []
            for _ in range(AVG_TIMES):
                start = time.perf_counter()
                for _ in baselineSampler(
                    sampler=SequentialSampler(range(DATA_SIZE)),
                    batch_size=batch_size,
                    drop_last=drop_last,
                ):
                    pass
                end = time.perf_counter()
                original_times.append(end - start)
                time.sleep(0.1)

            original_avg = float(np.mean(original_times))

            # Benchmark testSampler
            new_times = []
            for _ in range(AVG_TIMES):
                start = time.perf_counter()
                for _ in testSampler(
                    sampler=SequentialSampler(range(DATA_SIZE)),
                    batch_size=batch_size,
                    drop_last=drop_last,
                ):
                    pass
                end = time.perf_counter()
                new_times.append(end - start)
                time.sleep(0.1)  # Small delay to reduce system load

            new_avg = float(np.mean(new_times))

            # Calculate speedup
            if original_avg > 0 and new_avg > 0:
                speedup = (original_avg - new_avg) / original_avg * 100
                speedup_str = f"{speedup:.2f}%"
            else:
                speedup_str = "N/A"

            print(f"Speedup: {speedup_str}\n")

            results.append(
                [
                    batch_size,
                    drop_last,
                    f"{original_avg:.4f}",
                    f"{new_avg:.4f}",
                    speedup_str,
                ]
            )

    # Print results in a table
    headers = ["Batch Size", "Drop Last", "Original (s)", "New (s)", "Speedup"]
    print("\nBenchmark Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()
