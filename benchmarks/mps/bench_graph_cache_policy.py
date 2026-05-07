#!/usr/bin/env python3
"""Benchmark: MPSGraphCache policy impact on memory and throughput.

MPSGraph compiles a new graph for every unique tensor shape combination.
Unlike CUDA (pre-compiled, shape-agnostic kernels) or native Metal kernels,
MPSGraph bakes dimensions into the compiled graph. The cache has no eviction
and no max size -- it grows without bound.

Workload: Molecular GNN (ZINC, PyG graph batching).
  Shape key = (sum_nodes, sum_edges) -- a 2-D combinatorial space.
  With shuffled DataLoaders, batch compositions never repeat:
  P(repeat) ~= 1/C(N, batch_size) ~= 0.  Unbounded growth.

Strategy semantics:
  always              -- cache grows without bound (baseline; will OOM)
  freeze_after_warmup -- cache read-only after freeze_at; hits free, misses compile+discard
  clear_per_iter      -- graph cache flushed every iteration
  never               -- no caching; recompile every op on every call

Each strategy runs in an isolated subprocess for clean RSS measurement.

Related issues: #77753, #164299, #181213, transformers#33717

Usage:
    PYTHONPATH=~/pytorch python3 bench_graph_cache_policy.py
"""
import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.backends.mps.is_available(), "MPS not available"
assert hasattr(torch.mps, 'freeze_graph_cache'), \
    "Requires PyTorch >= 2.13 (pytorch/pytorch#182648)"

DEVICE = torch.device("mps")
WARMUP = 5
ITERS = 200


# -- helpers ------------------------------------------------------------------

def _cur_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


def _eff_2d(node_counts, edge_counts, bs):
    sn = statistics.stdev(node_counts)
    se = statistics.stdev(edge_counts)
    return math.sqrt(bs) * sn * math.sqrt(bs) * se * 2 * math.pi, sn, se


# -- workload -----------------------------------------------------------------

def make_molecular_gnn(quiet=False):
    """ZINC molecular GNN -- PyG graph batching, variable (sum_nodes, sum_edges)."""
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Batch
    from torch_geometric.datasets import ZINC

    root = os.path.join(tempfile.gettempdir(), "pyg_data")
    dataset = ZINC(root=root, subset=True, split="train")
    in_dim = dataset[0].x.shape[1]

    class MolGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(in_dim, 64)
            self.conv2 = GCNConv(64, 64)
            self.lin = nn.Linear(64, 1)

        def forward(self, x, edge_index, batch):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            return self.lin(global_mean_pool(x, batch))

    model = MolGNN().to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    bs = 32

    node_counts = [dataset[i].num_nodes for i in range(len(dataset))]
    edge_counts = [dataset[i].num_edges for i in range(len(dataset))]
    eff, sn, se = _eff_2d(node_counts, edge_counts, bs)

    if not quiet:
        print(f"  ZINC: {len(dataset)} molecules  "
              f"std_nodes={sn:.1f}  std_edges={se:.1f}  eff_2d={eff:.0f}")

    def fn(i):
        rng = random.Random(i)
        idx = list(range(len(dataset)))
        rng.shuffle(idx)
        batch = Batch.from_data_list(
            [dataset[idx[j]] for j in range(bs)]
        ).to(DEVICE)
        target = torch.randn(bs, 1, device=DEVICE)
        optimizer.zero_grad()
        out = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    return fn, eff, sn, se


# -- single-strategy runner (called in subprocess) ----------------------------

def run_strategy(strat_name, freeze_at, iters=ITERS, warmup=WARMUP):
    """Run one strategy from a clean process and return metrics as dict."""
    if strat_name == "never":
        torch.mps.set_graph_cache_policy("never")
    else:
        torch.mps.set_graph_cache_policy("always")

    fn, eff, sn, se = make_molecular_gnn(quiet=True)
    torch.mps.empty_cache()
    torch.mps.clear_graph_cache()
    torch.mps.synchronize()

    rss_pre = _cur_rss_mb()

    for i in range(warmup):
        fn(i)
    torch.mps.synchronize()

    rss_post_warmup_loop = _cur_rss_mb()

    times = []
    drvs = []
    iter_peak_rss = rss_pre
    rss_at_freeze = None

    for i in range(iters):
        step = warmup + i
        t0 = time.perf_counter()
        fn(step)

        if strat_name == "clear_per_iter":
            iter_peak_rss = max(iter_peak_rss, _cur_rss_mb())
            torch.mps.clear_graph_cache()
        elif strat_name == "freeze_after_warmup" and i == freeze_at:
            rss_at_freeze = _cur_rss_mb()
            torch.mps.freeze_graph_cache()

        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        drvs.append(torch.mps.driver_allocated_memory())

    torch.mps.synchronize()
    rss_final = _cur_rss_mb()

    n = len(drvs)
    x_mean = (n - 1) / 2.0
    denom = max(1, sum((j - x_mean)**2 for j in range(n)))
    slope_kbpi = sum(
        (j - x_mean) * (drvs[j] - drvs[0]) for j in range(n)
    ) / denom / 1024

    return {
        "strat":        strat_name,
        "rss_delta":    rss_final - rss_pre,
        "warmup_swell": rss_post_warmup_loop - rss_pre,
        "freeze_swell": (rss_at_freeze - rss_pre) if rss_at_freeze is not None else None,
        "iter_peak":    (iter_peak_rss - rss_pre) if strat_name == "clear_per_iter" else None,
        "mean_ms":      statistics.mean(times),
        "std_ms":       statistics.stdev(times) if len(times) > 1 else 0.0,
        "slope_kbpi":   slope_kbpi,
        "eff":          eff,
    }


# -- orchestrator (spawns one subprocess per strategy) ------------------------

def compare_strategies(label, iters=ITERS, warmup=WARMUP):
    fn, eff, sn, se = make_molecular_gnn(quiet=True)
    freeze_at = min(iters // 4, max(5, int(math.sqrt(eff))))

    strategies = ["always", "freeze_after_warmup", "clear_per_iter", "never"]

    print()
    print("=" * 78)
    print("  " + label)
    print("  eff_space=%.0f  std_nodes=%.1f  std_edges=%.1f" % (eff, sn, se))
    print("  warmup_burn=%d  freeze_at=%d  iters=%d  (training)" % (warmup, freeze_at, iters))
    print()
    print("  %-24s %9s %9s %6s  %s" % ("Strategy", "CurRSSΔ", "ms/iter", "+-", "note"))
    print("  " + "-"*24 + " " + "-"*9 + " " + "-"*9 + " " + "-"*6)

    always_ms = None
    always_slope_kbpi = None

    for strat_name in strategies:
        sys.stdout.write(f"  {strat_name:<24} running...\r")
        sys.stdout.flush()

        result = subprocess.run(
            [sys.executable, __file__,
             "--strategy", strat_name,
             "--freeze-at", str(freeze_at),
             "--iters", str(iters),
             "--warmup", str(warmup)],
            capture_output=True, text=True,
            env=os.environ.copy()
        )

        if result.returncode != 0:
            print(f"  {strat_name:<24} FAILED: {result.stderr[-300:]}")
            continue

        metrics = None
        for line in result.stdout.splitlines():
            if line.startswith("RESULT:"):
                metrics = json.loads(line[7:])
                break

        if metrics is None:
            print(f"  {strat_name:<24} no result (stdout: {result.stdout[-200:]})")
            continue

        if always_ms is None:
            always_ms = metrics["mean_ms"]
            always_slope_kbpi = metrics["slope_kbpi"]
            note = "(baseline)"
        else:
            note = "(%.2fx)" % (metrics["mean_ms"] / always_ms)

        print("  %-24s %+9.1f %9.1f %6.1f  %s" % (
            strat_name, metrics["rss_delta"], metrics["mean_ms"], metrics["std_ms"], note))

        if strat_name == "freeze_after_warmup":
            fs = metrics["freeze_swell"]
            print("    warmup loop: %+.1f MB  |  at freeze point: %+.1f MB" % (
                metrics["warmup_swell"], fs if fs is not None else float("nan")))

        if strat_name == "clear_per_iter":
            print("    peak/iter before clear: %+.1f MB  (net: %+.1f MB)" % (
                metrics["iter_peak"], metrics["rss_delta"]))

    if always_slope_kbpi and always_slope_kbpi > 0:
        iters_per_epoch = 312
        epochs = 100
        extrap_gb = always_slope_kbpi * iters_per_epoch * epochs / 1024 / 1024
        print()
        print("  always: %.1f KB/iter x %d iters/epoch x %d epochs = %.1f GB extrapolated" % (
            always_slope_kbpi, iters_per_epoch, epochs, extrap_gb))
    print()


# -- entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None,
                        choices=["always", "freeze_after_warmup", "clear_per_iter", "never"])
    parser.add_argument("--freeze-at", type=int, default=50)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()

    if args.strategy:
        metrics = run_strategy(args.strategy, args.freeze_at,
                               iters=args.iters, warmup=args.warmup)
        print("RESULT:" + json.dumps(metrics))
    else:
        print("PyTorch %s  |  MPS graph cache policy benchmark" % torch.__version__)
        compare_strategies(
            "Molecular GNN on ZINC (PyG graph batching -- unbounded 2-D shape space)",
            iters=args.iters, warmup=args.warmup)


if __name__ == "__main__":
    main()
