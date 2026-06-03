"""Isolated-subprocess benchmark for MPS loss ops.

Each (op, shape, dtype, reduction) cell runs in a fresh Python subprocess —
eliminating MPSGraph cache pollution and thermal accumulation across cells
that bias the in-process bench_loss_ops.py. Used to measure the loss-ops
PR against an MPSGraph baseline build of the same upstream commit.

Usage:
    python bench_loss_ops_isolated.py \\
        --pr-py /path/to/pr/.venv/bin/python \\
        --baseline-py /path/to/baseline/.venv/bin/python

Default shapes cover small (4K) to LLM training (8×2048×4096) scale.
"""

import argparse
import json
import subprocess
import sys


# In-subprocess timing block — runs when invoked with --cell
_CELL_SCRIPT = """
import sys, json, ast
import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer

op = sys.argv[2]
shape = tuple(ast.literal_eval(sys.argv[3]))
dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[sys.argv[4]]
red = sys.argv[5]

torch.manual_seed(0)
DEV = "mps"
x = torch.randn(shape, device=DEV, dtype=dtype)
y = torch.randn(shape, device=DEV, dtype=dtype)
if op == "bce":
    x = x.sigmoid().detach()
    y = (y > 0).to(dtype)
xg = x.detach().clone().requires_grad_(True)
w = torch.rand_like(x) if red == "none" else None

stmts = {
    "mse":       ("F.mse_loss(x, y, reduction=red)",
                   "xg.grad=None; F.mse_loss(xg, y, reduction='none').backward(w)" if red == "none"
                   else "xg.grad=None; F.mse_loss(xg, y, reduction=red).backward()"),
    "smooth_l1": ("F.smooth_l1_loss(x, y, reduction=red, beta=1.0)",
                   "xg.grad=None; F.smooth_l1_loss(xg, y, reduction='none', beta=1.0).backward(w)" if red == "none"
                   else "xg.grad=None; F.smooth_l1_loss(xg, y, reduction=red, beta=1.0).backward()"),
    "huber":     ("F.huber_loss(x, y, reduction=red, delta=1.0)",
                   "xg.grad=None; F.huber_loss(xg, y, reduction='none', delta=1.0).backward(w)" if red == "none"
                   else "xg.grad=None; F.huber_loss(xg, y, reduction=red, delta=1.0).backward()"),
    "bce":       ("F.binary_cross_entropy(x, y, reduction=red)",
                   "xg.grad=None; F.binary_cross_entropy(xg, y, reduction='none').backward(w)" if red == "none"
                   else "xg.grad=None; F.binary_cross_entropy(xg, y, reduction=red).backward()"),
}
fwd_stmt, bwd_stmt = stmts[op]
g = {"F": F, "x": x, "y": y, "xg": xg, "red": red, "w": w, "torch": torch}

# Warmup
for _ in range(15):
    exec(bwd_stmt, g)
torch.mps.synchronize()

fwd_m = Timer(stmt=fwd_stmt, globals=g).blocked_autorange(min_run_time=0.2).median * 1e6
fwdbwd_m = Timer(stmt=bwd_stmt, globals=g).blocked_autorange(min_run_time=0.4).median * 1e6
print(json.dumps({"op": op, "shape": str(shape), "dtype": sys.argv[4], "red": red,
                  "fwd_us": fwd_m, "fwdbwd_us": fwdbwd_m,
                  "torch_version": torch.__version__}))
"""

SHAPES = [(8, 2048, 4096), (16, 512, 4096), (1048576,), (262144,), (65536,), (4096,)]
OPS = ["mse", "smooth_l1", "huber", "bce"]
DTYPES = ["f32", "f16", "bf16"]
REDS = ["none", "mean", "sum"]


def run_cell(py, op, shape, dtype, red):
    proc = subprocess.run(
        [py, __file__, "--cell", op, str(shape), dtype, red],
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        return json.loads(proc.stdout.strip().split("\n")[-1])
    except Exception as e:
        return {"error": f"{e}: stdout={proc.stdout[:200]} stderr={proc.stderr[:200]}"}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cell":
        # Subprocess mode: run one cell and emit a JSON line.
        exec(_CELL_SCRIPT, {"__name__": "__main__"})
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--pr-py", required=True, help="path to PR venv's python")
    ap.add_argument(
        "--baseline-py", required=True, help="path to baseline venv's python (MPSGraph)"
    )
    args = ap.parse_args()

    rows = []
    total = len(OPS) * len(SHAPES) * len(DTYPES) * len(REDS) * 2
    done = 0
    for op in OPS:
        for shape in SHAPES:
            for dt in DTYPES:
                for red in REDS:
                    r_pr = run_cell(args.pr_py, op, shape, dt, red)
                    done += 1
                    r_bl = run_cell(args.baseline_py, op, shape, dt, red)
                    done += 1
                    if "error" in r_pr or "error" in r_bl:
                        print(
                            f"[{done}/{total}] {op} {shape} {dt} {red} — ERROR",
                            flush=True,
                        )
                        continue
                    speedup = r_bl["fwdbwd_us"] / r_pr["fwdbwd_us"]
                    rows.append(
                        {
                            "op": op,
                            "shape": shape,
                            "dtype": dt,
                            "red": red,
                            "pr_us": r_pr["fwdbwd_us"],
                            "bl_us": r_bl["fwdbwd_us"],
                            "speedup": speedup,
                        }
                    )
                    print(
                        f"[{done}/{total}] {op:<10} {str(shape):<20} {dt:<5} {red:<5}  "
                        f"baseline={r_bl['fwdbwd_us']:>9.1f}µs  pr={r_pr['fwdbwd_us']:>9.1f}µs  "
                        f"speedup={speedup:.2f}x",
                        flush=True,
                    )

    print("\n=== SUMMARY ===")
    print(
        f"{'op':<10} {'shape':<20} {'dt':<5} {'red':<5} {'bl_µs':>9} {'pr_µs':>9} {'speedup':>9}"
    )
    for r in rows:
        print(
            f"{r['op']:<10} {str(r['shape']):<20} {r['dtype']:<5} {r['red']:<5} "
            f"{r['bl_us']:>9.1f} {r['pr_us']:>9.1f} {r['speedup']:>8.2f}x"
        )
    print(f"\ntotal cases: {len(rows)}")
    print(f"PR wins (>1.05x):     {sum(1 for r in rows if r['speedup'] > 1.05)}")
    print(
        f"matched (0.95-1.05x): {sum(1 for r in rows if 0.95 <= r['speedup'] <= 1.05)}"
    )
    print(f"PR regressions (<0.95x): {sum(1 for r in rows if r['speedup'] < 0.95)}")
    if rows:
        print(f"max speedup:     {max(r['speedup'] for r in rows):.2f}x")
        print(f"max regression:  {min(r['speedup'] for r in rows):.2f}x")


if __name__ == "__main__":
    main()
