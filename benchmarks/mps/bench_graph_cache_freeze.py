#!/usr/bin/env python3
"""
Comprehensive benchmark: always vs clear_per_iter vs freeze_after_warmup vs never
across a broad range of synthetic GNN-like datasets with varying shape distributions.

Each (strategy, run) pair is executed in a fresh subprocess (multiprocessing spawn)
to guarantee a clean graph cache and no RSS leakage between runs.

Memory tracking uses psutil current RSS, which captures compiled MPSGraph objects
in IOKit/Metal shader cache memory (invisible to driver_allocated_memory() and
monotonically non-decreasing in ru_maxrss).

Run: PYTHONPATH=~/pytorch python3 bench_graph_cache_freeze.py
"""

import gc
import math
import os
import statistics
import sys
import time
import multiprocessing as mp

# ── Constants ─────────────────────────────────────────────────────────────────

BS = 32
ITERS = 200
WARMUP_BURN = 30
N_RUNS = 15

DATASETS = [
    # name,               n_mean, n_std,  epn
    ("Tiny-uniform", 8, 1, 2),
    ("Small-moderate", 20, 5, 3),
    ("Medium-moderate", 50, 10, 4),
    ("Medium-high", 50, 20, 4),
    ("Large-moderate", 100, 15, 5),
    ("Large-high", 100, 40, 5),
    ("XLarge-extreme", 200, 80, 6),
]

STRATEGIES = ["always", "clear_per_iter", "freeze_after_warmup", "never"]


# ── Worker (runs in a fresh subprocess) ───────────────────────────────────────

def _worker(args, result_queue):
    """Runs one (strategy, dataset) trial in a fresh process."""
    import gc
    import math
    import os
    import statistics
    import time
    import psutil
    import torch

    strategy, n_mean, n_std, epn, freeze_after, iters, warmup = args

    assert torch.backends.mps.is_available()
    device = torch.device("mps")
    proc = psutil.Process(os.getpid())

    def rss_mb():
        return proc.memory_info().rss / 1024 / 1024

    # ── Model ────────────────────────────────────────────────────────────────
    class SimpleGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = torch.nn.Linear(16, 64)
            self.lin2 = torch.nn.Linear(64, 1)

        def forward(self, x, edge_index, batch):
            row, col = edge_index
            agg = torch.zeros(x.size(0), x.size(1), device=x.device)
            agg.scatter_add_(0, col.unsqueeze(1).expand_as(x[row]), x[row])
            x = torch.relu(self.lin1(agg))
            num_graphs = int(batch.max().item()) + 1
            out = torch.zeros(num_graphs, x.size(1), device=x.device)
            out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
            counts = torch.bincount(batch, minlength=num_graphs).float().unsqueeze(1)
            out = out / counts.clamp(min=1)
            return self.lin2(out).squeeze(1)

    def make_batch():
        node_counts = torch.clamp(
            torch.normal(n_mean, n_std, (BS,)).long(), min=2, max=int(n_mean * 3)
        ).tolist()
        total_nodes = sum(node_counts)
        x = torch.randn(total_nodes, 16, device=device)
        batch_vec = torch.cat([
            torch.full((n,), i, dtype=torch.long) for i, n in enumerate(node_counts)
        ]).to(device)
        edges, offset = [], 0
        for n in node_counts:
            e_per_node = epn
            n_edges = max(1, int(n * e_per_node + torch.randn(1).item() * n * 0.3))
            src = torch.randint(0, n, (n_edges,)) + offset
            dst = torch.randint(0, n, (n_edges,)) + offset
            edges.append(torch.stack([src, dst], dim=0))
            offset += n
        edge_index = torch.cat(edges, dim=1).to(device)
        y = torch.randn(BS, device=device)
        return x, edge_index, batch_vec, y

    # ── Run ──────────────────────────────────────────────────────────────────
    model = SimpleGCN().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.mps.set_graph_cache_policy("never" if strategy == "never" else "always")
    torch.mps.empty_cache()
    torch.mps.clear_graph_cache()
    torch.mps.synchronize()

    times = []
    rss_samples = []
    pre_clear_rss = []
    rss_at_freeze = None

    for i in range(iters + warmup):
        x, ei, bv, y = make_batch()

        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(x, ei, bv)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if strategy == "clear_per_iter":
            if i >= warmup:
                pre_clear_rss.append(rss_mb())   # BEFORE clear
            torch.mps.clear_graph_cache()
        elif strategy == "freeze_after_warmup" and i == freeze_after:
            rss_at_freeze = rss_mb()              # AT freeze
            torch.mps.freeze_graph_cache()

        torch.mps.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000)
            rss_samples.append(rss_mb())

    torch.mps.synchronize()

    baseline = rss_samples[0] if rss_samples else 0.0
    rss_delta = rss_samples[-1] - baseline if rss_samples else 0.0
    rss_peak = max(rss_samples) - baseline if rss_samples else 0.0

    n = len(rss_samples)
    x_mean = (n - 1) / 2.0
    denom = max(1.0, sum((j - x_mean) ** 2 for j in range(n)))
    slope = sum((j - x_mean) * (rss_samples[j] - rss_samples[0])
                 for j in range(n)) / denom * 1024

    pre_clr_mb = (statistics.mean(pre_clear_rss) - baseline) if pre_clear_rss else 0.0
    at_freeze_mb = (rss_at_freeze - baseline) if rss_at_freeze is not None else 0.0

    result_queue.put({
        "ms":       statistics.mean(times),
        "sd":       statistics.stdev(times) if len(times) > 1 else 0.0,
        "delta":    rss_delta,
        "peak":     rss_peak,
        "slope":    slope,
        "pre_clr":  pre_clr_mb,
        "at_freeze":at_freeze_mb,
    })


def run_one(strategy, n_mean, n_std, epn, freeze_after):
    """Spawn a fresh process for one trial and return its result dict."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    args = (strategy, n_mean, n_std, epn, freeze_after, ITERS, WARMUP_BURN)
    p = ctx.Process(target=_worker, args=(args, queue))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Worker exited with code {p.exitcode}")
    return queue.get()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"PyTorch benchmark  |  MPS synthetic GNN  |  each run in fresh subprocess")
    print(f"BS={BS}  ITERS={ITERS}  WARMUP_BURN={WARMUP_BURN}  N_RUNS={N_RUNS}")
    print(f"Memory: psutil current RSS per subprocess (clean cache, no leakage)")
    print()

    print(f"{'Dataset':<22} {'eff_2d':>9} {'fz':>4}  "
          f"{'always':>38}  "
          f"{'clear_per_iter':>44}  "
          f"{'freeze_after_warmup':>44}  "
          f"{'never':>30}")
    print(f"{'':>38}  (ms+-sd | slope KB/it | peak MB)  "
          f"(ms+-sd | slope KB/it | pre_clr MB)  "
          f"(ms+-sd | slope KB/it | at_frz MB)  "
          f"(ms+-sd | slope KB/it)")
    print("-" * 185)

    def effective_2d(n_std, e_std):
        return math.sqrt(BS) * n_std * math.sqrt(BS) * e_std * 2 * math.pi

    for name, n_mean, n_std, epn in DATASETS:
        e_std = epn * n_std
        eff = effective_2d(n_std, e_std)
        freeze_at = min(ITERS // 4, max(5, int(math.sqrt(eff))))
        freeze_after = WARMUP_BURN + freeze_at

        results = {}
        for strategy in STRATEGIES:
            agg = {k: [] for k in ("ms", "sd", "delta", "peak", "slope", "pre_clr", "at_freeze")}
            for run_idx in range(N_RUNS):
                r = run_one(strategy, n_mean, n_std, epn, freeze_after)
                for k in agg:
                    agg[k].append(r[k])
                # flush so we can see progress mid-dataset
                sys.stdout.flush()

            results[strategy] = {k: statistics.mean(v) for k, v in agg.items()}

        always_ms = results["always"]["ms"]

        def fmt(strat):
            r = results[strat]
            rel = f"({r['ms']/always_ms:.2f}x)" if strat != "always" else "       "
            sd = r["sd"]
            base = f"{r['ms']:6.1f}ms+-{sd:4.1f} {rel} slope={r['slope']:+7.1f}KB/it"
            if strat == "always":
                return base + f"  peak={r['peak']:+7.1f}MB"
            elif strat == "clear_per_iter":
                return base + f"  pre_clr={r['pre_clr']:+7.1f}MB"
            elif strat == "freeze_after_warmup":
                return base + f"  at_frz={r['at_freeze']:+7.1f}MB"
            return base

        print(f"  {name:<20} {eff:>9.0f} {freeze_at:>4}  "
              f"{fmt('always')}  ||  "
              f"{fmt('clear_per_iter')}  ||  "
              f"{fmt('freeze_after_warmup')}  ||  "
              f"{fmt('never')}")
        sys.stdout.flush()

    print("\nDone.")


if __name__ == "__main__":
    main()
