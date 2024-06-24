#!/usr/bin/env python3
#
# Computes difference between measurements produced by ./benchmark.py.
#

import argparse
import json

import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="PyTorch distributed benchmark diff")
    parser.add_argument("file", nargs=2)
    args = parser.parse_args()

    if len(args.file) != 2:
        raise RuntimeError("Must specify 2 files to diff")

    ja = load(args.file[0])
    jb = load(args.file[1])

    keys = (set(ja.keys()) | set(jb.keys())) - {"benchmark_results"}
    print(f"{'':20s} {'baseline':>20s}      {'test':>20s}")
    print(f"{'':20s} {'-' * 20:>20s}      {'-' * 20:>20s}")
    for key in sorted(keys):
        va = str(ja.get(key, "-"))
        vb = str(jb.get(key, "-"))
        print(f"{key + ':':20s} {va:>20s}  vs  {vb:>20s}")
    print("")

    ba = ja["benchmark_results"]
    bb = jb["benchmark_results"]
    for ra, rb in zip(ba, bb):
        if ra["model"] != rb["model"]:
            continue
        if ra["batch_size"] != rb["batch_size"]:
            continue

        model = ra["model"]
        batch_size = int(ra["batch_size"])
        name = f"{model} with batch size {batch_size}"
        print(f"Benchmark: {name}")

        # Print header
        print("")
        print(f"{'':>10s}", end="")  # noqa: E999
        for _ in [75, 95]:
            print(
                f"{'sec/iter':>16s}{'ex/sec':>10s}{'diff':>10s}", end=""
            )  # noqa: E999
        print("")

        # Print measurements
        for i, (xa, xb) in enumerate(zip(ra["result"], rb["result"])):
            # Ignore round without ddp
            if i == 0:
                continue
            # Sanity check: ignore if number of ranks is not equal
            if len(xa["ranks"]) != len(xb["ranks"]):
                continue

            ngpus = len(xa["ranks"])
            ma = sorted(xa["measurements"])
            mb = sorted(xb["measurements"])
            print(f"{ngpus:>4d} GPUs:", end="")  # noqa: E999
            for p in [75, 95]:
                va = np.percentile(ma, p)
                vb = np.percentile(mb, p)
                # We're measuring time, so lower is better (hence the negation)
                delta = -100 * ((vb - va) / va)
                print(
                    f"  p{p:02d}: {vb:8.3f}s {int(batch_size / vb):7d}/s {delta:+8.1f}%",
                    end="",
                )  # noqa: E999
            print("")
        print("")


if __name__ == "__main__":
    main()
