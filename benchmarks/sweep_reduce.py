#!/usr/bin/env python3
"""Full reduce-op perf sweep for MPS — isolated-subprocess per cell, dumps JSON.

Lives in the worktree (benchmarks/mps/sweep_reduce.py) so the PR's perf numbers
stay reproducible. Run on each build (MPSGraph baseline vs Metal PR), then compare.

Usage:
  # one cell (used internally per-subprocess):
  python sweep_reduce.py --cell '{"op":"var","shape":[1024,4096],"dim":-1,"dtype":"float32"}'
  # full sweep (spawns one fresh subprocess per cell -> avoids cross-cell cache/thermal bleed):
  python sweep_reduce.py --out /tmp/sweep_<tag>.json
  # compare two result files:
  python sweep_reduce.py --compare /tmp/sweep_base.json /tmp/sweep_pr.json
"""
import argparse, itertools, json, subprocess, sys

OPS = ["sum", "mean", "nansum", "prod", "var", "std", "var_mean", "std_mean",
       "argmax", "argmin", "amax", "amin", "max", "min"]
SHAPES = [[128, 256], [1024, 4096], [4096, 1024], [1048576], [256, 256, 256],
          [8, 2048, 4096], [16, 512, 4096]]
DTYPES = ["float32", "float16", "bfloat16"]
DIMS = [0, -1, None]


def _valid(op, shape, dim):
    if dim is None and op in ("max", "min"):       # max(dim)/min(dim) need a dim
        return False
    if dim is not None and dim >= len(shape):
        return False
    if len(shape) == 1 and dim == 0:
        dim_ok = True  # 1D dim=0 == all-reduce, allow
    return True


def run_cell(cell):
    import torch
    from torch.utils.benchmark import Timer
    dt = getattr(torch, cell["dtype"])
    shape, dim, op = cell["shape"], cell["dim"], cell["op"]
    torch.manual_seed(0)
    if op in ("argmax", "argmin", "amax", "amin", "max", "min") and dt in (torch.float16, torch.bfloat16):
        x = torch.randn(*shape).to(dt).to("mps")
    else:
        x = torch.randn(*shape, dtype=torch.float32).to(dt).to("mps") if dt.is_floating_point else torch.randint(0, 8, shape, dtype=dt).to("mps")
    fn = getattr(torch, op)
    stmt = "fn(x) if d is None else fn(x, dim=d); torch.mps.synchronize()"
    try:
        Timer(stmt=stmt, globals={"fn": fn, "x": x, "d": dim}).blocked_autorange(min_run_time=0.3)  # warm
        m = Timer(stmt=stmt, globals={"fn": fn, "x": x, "d": dim}).blocked_autorange(min_run_time=0.5)
        return round(m.median * 1e6, 3)
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def sweep(out):
    cells = [dict(op=o, shape=s, dim=d, dtype=t)
             for o, s, t, d in itertools.product(OPS, SHAPES, DTYPES, DIMS) if _valid(o, s, d)]
    results = {}
    for i, c in enumerate(cells):
        key = f"{c['op']}|{c['shape']}|{c['dim']}|{c['dtype']}"
        r = subprocess.run([sys.executable, __file__, "--cell", json.dumps(c)],
                           capture_output=True, text=True)
        val = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"ERR:nostdout"
        try:
            results[key] = float(val)
        except ValueError:
            results[key] = val
        print(f"[{i+1}/{len(cells)}] {key} -> {results[key]}", flush=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out} ({len(results)} cells)")


def compare(base_f, pr_f):
    base = json.load(open(base_f)); pr = json.load(open(pr_f))
    wins = matched = regr = 0
    regr_list = []
    for k in sorted(set(base) & set(pr)):
        b, p = base[k], pr[k]
        if not (isinstance(b, float) and isinstance(p, float)):
            continue
        sp = b / p  # speedup of PR over baseline (>1 = faster)
        if sp > 1.05:
            wins += 1
        elif sp < 0.95:
            regr += 1
            regr_list.append((k, round(sp, 3), b, p))
        else:
            matched += 1
    print(f"wins(>1.05x): {wins}  matched: {matched}  REGRESSIONS(<0.95x): {regr}")
    print("worst regressions:")
    for k, sp, b, p in sorted(regr_list, key=lambda r: r[1])[:20]:
        print(f"  {sp}x  {k}  base={b} pr={p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.cell:
        print(run_cell(json.loads(a.cell)))
    elif a.out:
        sweep(a.out)
    elif a.compare:
        compare(a.compare[0], a.compare[1])
