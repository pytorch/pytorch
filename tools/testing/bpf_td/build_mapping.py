#!/usr/bin/env python3
"""Join bpftrace output with run_test.py pid records into a test-file -> touched-files map.

The tracer (trace_opens.bt) emits fork/exec/open events tagged with monotonic
nanosecond timestamps. run_test.py records, per test file, the pool-worker pid
that spawned the test subprocess plus the [start, end] monotonic window. Here we
reconstruct each test's pid subtree (descendants of the worker forked inside the
window) and attribute the files those pids opened, filtered to the repo checkout
and the installed torch package.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict


def parse_trace(path):
    """Return (forks, events) where forks is a list of (ns, ppid, cpid) sorted by
    ns, and events is a dict pid -> list of (ns, abspath) for open/exec events."""
    forks = []
    events = defaultdict(list)
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.split(" ", 3)
            if len(parts) < 3:
                continue
            tag = parts[0]
            try:
                ns = int(parts[1])
                pid = int(parts[2])
            except ValueError:
                continue
            if tag == "F":
                if len(parts) < 4:
                    continue
                try:
                    cpid = int(parts[3].strip())
                except ValueError:
                    continue
                forks.append((ns, pid, cpid))
            elif tag in ("O", "E"):
                if len(parts) < 4:
                    continue
                p = parts[3].rstrip("\n")
                if p.startswith("/"):
                    events[pid].append((ns, p))
    forks.sort()
    return forks, events


def subtree_pids(forks, owner, start_ns, end_ns):
    """Pids forked (transitively) from owner within [start_ns, end_ns].

    forks is sorted by ns, so a single ascending pass reaches fixpoint: a child
    is always forked after its parent joined the tree."""
    included = set()
    for ns, ppid, cpid in forks:
        if ns < start_ns or ns > end_ns:
            continue
        if ppid == owner or ppid in included:
            included.add(cpid)
    return included


# Runtime writes under the repo that are not source dependencies.
_NOISE = ("/.git/", "/.pytest_cache/", "/test/test-reports/", "/.additional_ci_files/")


def normalize(path, repo_root, torch_root):
    """Map an absolute path to a repo-relative path, or None if outside scope."""
    if any(n in path for n in _NOISE):
        return None
    if repo_root and path.startswith(repo_root + os.sep):
        rel = os.path.relpath(path, repo_root)
        if not rel.startswith(".."):
            return rel
    # Tests import the INSTALLED torch from site/dist-packages; map it back to
    # the repo-relative "torch/..." so a changed repo file joins to the runtime
    # open. This pattern is independent of torch_root env detection (which has
    # proven unreliable in CI).
    for marker in ("/site-packages/", "/dist-packages/"):
        i = path.find(marker + "torch/")
        if i != -1:
            return path[i + len(marker):]
    if torch_root and path.startswith(torch_root + os.sep):
        rel = os.path.relpath(path, os.path.dirname(torch_root))
        if not rel.startswith(".."):
            return rel
    return None


def build(forks, events, records, repo_root, torch_root, diag):
    mapping = defaultdict(set)
    for rec in records:
        owner = rec["pid"]
        start_ns, end_ns = rec["start_ns"], rec["end_ns"]
        pids = subtree_pids(forks, owner, start_ns, end_ns)
        diag["subtree_pids"] += len(pids)
        touched = mapping[rec["test_file"]]
        for pid in pids:
            for ns, path in events.get(pid, ()):
                if start_ns <= ns <= end_ns:
                    diag["paths_pre_filter"] += 1
                    rel = normalize(path, repo_root, torch_root)
                    if rel is not None:
                        touched.add(rel)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="bpftrace output file")
    ap.add_argument("--pids", required=True, help="test_pids.jsonl from run_test.py")
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--torch-root", default="", help="installed torch package dir")
    ap.add_argument("--config", default=os.environ.get("TEST_CONFIG", "unknown"))
    ap.add_argument("--shard", default=os.environ.get("SHARD_NUMBER", "0"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    repo_root = os.path.realpath(args.repo_root)
    torch_root = os.path.realpath(args.torch_root) if args.torch_root else ""

    forks, events = parse_trace(args.trace)
    records = []
    with open(args.pids, errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    diag = {"subtree_pids": 0, "paths_pre_filter": 0}
    mapping = build(forks, events, records, repo_root, torch_root, diag)

    # Diagnostics to distinguish failure modes without shipping the raw trace:
    # pid_overlap == 0 means the recorded owner pids never appear as fork
    # parents -> pid-namespace mismatch (need --pid=host). paths_pre_filter > 0
    # but total_paths == 0 -> the repo/torch normalize filter dropped everything.
    fork_ppids = {ppid for _, ppid, _ in forks}
    record_pids = {r["pid"] for r in records}
    sample_paths = sorted({p for evs in events.values() for _, p in evs})[:10]

    out = {tf: sorted(paths) for tf, paths in sorted(mapping.items())}
    out["meta"] = {
        "config": args.config,
        "shard": args.shard,
        "num_test_files": len(mapping),
        "num_fork_events": len(forks),
        "num_open_events": sum(len(v) for v in events.values()),
        "num_pid_records": len(records),
        "pid_overlap": len(record_pids & fork_ppids),
        "subtree_pids": diag["subtree_pids"],
        "paths_pre_filter": diag["paths_pre_filter"],
        "total_paths": sum(len(v) for v in mapping.values()),
        "repo_root": repo_root,
        "torch_root": torch_root,
        "sample_open_paths": sample_paths,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"wrote {args.out}: {out['meta']}")


if __name__ == "__main__":
    main()
