"""Compare default collation with reusable shared input slots on synthetic tensors."""

import argparse
import statistics
import time

import torch
from torch.utils.data import DataLoader


def measure(sample, args, pooled):
    batch_size = 2
    batches = args.warmup + args.steps
    batch_bytes = sample.numel() * batch_size
    loader = DataLoader(
        [sample] * (batches * batch_size),
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shared_memory_pool=(args.slots, batch_bytes) if pooled else None,
    )
    iterator = iter(loader)
    for _ in range(args.warmup):
        next(iterator)
    start = time.perf_counter()
    for _ in range(args.steps):
        batch = next(iterator)
    elapsed = time.perf_counter() - start
    if not batch.is_pinned():
        raise RuntimeError("the benchmark requires CUDA pinning")
    return 1000 * elapsed / args.steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-mib", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("all arguments must be positive")
    if not torch.cuda.is_available():
        parser.error("a CUDA device is required")
    sample = torch.zeros(args.batch_mib * 2**19, dtype=torch.uint8)
    results = {False: [], True: []}
    for repeat in range(args.repeats):
        for pooled in (False, True) if repeat % 2 == 0 else (True, False):
            results[pooled].append(measure(sample, args, pooled))
    for pooled, times in results.items():
        median = statistics.median(times)
        print(f"pooled={pooled} ms/pinned batch {times} median={median:.3f}")


if __name__ == "__main__":
    main()
