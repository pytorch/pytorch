#!/usr/bin/env python3
"""Pair A (forced MPSGraph) vs B (Metal) softmax bench results.

speedup = A_us / B_us  (>1 => Metal FASTER = good; <1 => Metal slower = regression)
Per cell we take the median over reps of A and of B independently, then ratio.
A cell is a REGRESSION if speedup < REG_THRESH (default 0.95).
"""

import glob
import json
import os
import sys


OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_out"
REG = float(os.environ.get("REG_THRESH", "0.95"))


def load(prefix):
    cells = {}  # (shape,dtype,kind) -> list of median_us
    for f in sorted(glob.glob(os.path.join(OUT, prefix + "_*.json"))):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception as e:
            print("skip", f, e, file=sys.stderr)
            continue
        for r in d["results"]:
            if "error" in r:
                continue
            k = (r["shape"], r["dtype"], r["kind"])
            cells.setdefault(k, []).append(r["median_us"])
    return cells


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


A = load("A")
B = load("B")
keys = sorted(set(A) & set(B))
rows = []
for k in keys:
    a = med(A[k])
    b = med(B[k])
    sp = a / b if b > 0 else float("inf")
    rows.append((k, a, b, sp))

regs = [r for r in rows if r[3] < REG]
regs.sort(key=lambda r: r[3])

print("=== ALL CELLS (speedup = MPSGraph_us / Metal_us; <1 = Metal slower) ===")
print(
    f"{'shape':<22} {'dt':<5} {'kind':<7} {'A(MG)us':>9} {'B(Mtl)us':>9} {'speedup':>7}"
)
for k, a, b, sp in sorted(rows, key=lambda r: r[3]):
    flag = "  REG" if sp < REG else ""
    print(f"{k[0]:<22} {k[1]:<5} {k[2]:<7} {a:>9.2f} {b:>9.2f} {sp:>7.3f}{flag}")

print()
print(f"=== REGRESSIONS (speedup < {REG:.2f}): {len(regs)} ===")
for k, a, b, sp in regs:
    print(
        f"{k[0]:<22} {k[1]:<5} {k[2]:<7}  Metal={b:.2f}us MPSGraph={a:.2f}us  "
        f"{sp:.3f}x  (Metal +{b - a:.2f}us)"
    )
print()
print(f"TOTAL CELLS={len(rows)}  REGRESSIONS={len(regs)}")
