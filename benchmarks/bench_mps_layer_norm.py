"""Benchmark Metal vs MPSGraph LayerNorm backward on Apple Silicon.

Usage:
    python benchmarks/bench_mps_layer_norm.py

Measures torch.nn.functional.layer_norm forward+backward using
torch.utils.benchmark.Timer.blocked_autorange (median reported).

Device: mps (Apple Silicon). Run with PYTHONPATH pointing to a Metal-enabled
PyTorch build for the 'metal' numbers; run with stock nightly for 'mpsgraph'.
"""
import argparse
import torch
from torch.utils.benchmark import Timer

parser = argparse.ArgumentParser()
parser.add_argument("--label", default="run", help="Label for this run")
parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
args = parser.parse_args()

dtype = getattr(torch, args.dtype)
device = "mps"

configs = [
    # (M, N, description)
    (1,    64,   "(1,1,64)    batch-1 small"),
    (1,   256,   "(1,1,256)   batch-1 mid"),
    (1,  1024,   "(1,1,1024)  batch-1 large"),
    (1,  4096,   "(1,1,4096)  batch-1 xlarge"),
    (512,   64,  "(512,1,64)  BERT-like small-N"),
    (512,  256,  "(512,1,256) BERT-like"),
    (512,  768,  "(512,1,768) BERT hidden"),
    (512, 1024,  "(512,1,1024)"),
    (512, 4096,  "(512,1,4096)"),
    (4096, 256,  "(4096,1,256) large-M"),
]

print(f"label={args.label}  dtype={args.dtype}  device={device}  torch={torch.__version__}")
print(f"Methodology: torch.utils.benchmark.Timer.blocked_autorange (median)")
print()
print(f"{'Config':<32} {'fwd+bwd (ms)':>14}")
print("-" * 48)

for M, N, label in configs:
    setup = f"""
import torch
x = torch.randn({M}, {N}, device='{device}', dtype=torch.{args.dtype}, requires_grad=True)
w = torch.randn({N}, device='{device}', dtype=torch.{args.dtype}, requires_grad=True)
b = torch.randn({N}, device='{device}', dtype=torch.{args.dtype}, requires_grad=True)
dy = torch.randn({M}, {N}, device='{device}', dtype=torch.{args.dtype})
"""
    stmt = f"""
xc = x.detach().requires_grad_(True)
wc = w.detach().requires_grad_(True)
bc = b.detach().requires_grad_(True)
y = torch.nn.functional.layer_norm(xc, [{N}], wc, bc)
y.backward(dy)
torch.mps.synchronize()
"""
    t = Timer(stmt=stmt, setup=setup).blocked_autorange(min_run_time=1.0)
    print(f"{label:<32} {t.median * 1e3:>14.3f}")
