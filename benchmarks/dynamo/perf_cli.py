#!/usr/bin/env python3
"""
CLI tool for launching, summarizing, and reproducing inductor perf regression runs.

Usage:
    python benchmarks/dynamo/perf_cli.py launch [--device a100 h100 ...] [--ref BRANCH] [--wait]
    python benchmarks/dynamo/perf_cli.py summary <run-id|branch> [--top 5] [--config PATTERN]
    python benchmarks/dynamo/perf_cli.py repro <run-id> [--model MODEL] [--suite SUITE] [--print-only]

Requires: gh CLI (authenticated), internet access to S3 (gha-artifacts bucket).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from math import exp, log
from pathlib import Path


WORKFLOWS = {
    "a100": {
        "name": "inductor-A100-perf-nightly",
        "id": 42513231,
    },
    "a100-compare": {
        "name": "inductor-A100-perf-compare",
        "id": 50531883,
    },
    "h100": {
        "name": "inductor-perf-nightly-h100",
        "id": 144201955,
    },
    "b200": {
        "name": "inductor-perf-b200",
        "id": 173716622,
    },
    "rocm-mi300": {
        "name": "inductor-perf-nightly-rocm-mi300",
        "id": 197925166,
    },
    "rocm-mi355": {
        "name": "inductor-perf-nightly-rocm-mi355",
        "id": 197925165,
    },
    "x86": {
        "name": "inductor-perf-nightly-x86",
        "id": 108782874,
    },
    "x86-zen": {
        "name": "inductor-perf-nightly-x86-zen",
        "id": 167573808,
    },
    "aarch64": {
        "name": "inductor-perf-nightly-aarch64",
        "id": 109196799,
    },
    "macos": {
        "name": "inductor-perf-nightly-macos",
        "id": 117199085,
    },
    "xpu": {
        "name": "inductor-perf-nightly-xpu",
        "id": 201149053,
    },
}

DEVICE_CHOICES = sorted(k for k in WORKFLOWS if k != "a100-compare")

S3_BUCKET = "gha-artifacts"
S3_URL = f"https://{S3_BUCKET}.s3.amazonaws.com"
REPO = "pytorch/pytorch"

# Regex to parse test job names like:
# "cuda13.0-py3.10-gcc11-sm80 / test (inductor_huggingface_perf, 1, 5, linux.aws.a100)"
JOB_RE = re.compile(
    r"test \((?P<config>[^,]+),\s*(?P<shard>\d+),\s*(?P<num_shards>\d+),\s*(?P<runner>[^)]+)\)"
)

PERF_CONFIGS = re.compile(r"inductor_(huggingface|timm|torchbench)_perf")

SUITE_ALIASES = {
    "hf": "huggingface",
    "huggingface": "huggingface",
    "timm": "timm_models",
    "timm_models": "timm_models",
    "tb": "torchbench",
    "torchbench": "torchbench",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gh(*args: str, json_output: bool = False) -> str | dict | list:
    cmd = ["gh"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"gh error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    if json_output:
        return json.loads(result.stdout)
    return result.stdout.strip()


def git(*args: str) -> str:
    result = subprocess.run(["git"] + list(args), capture_output=True, text=True)
    return result.stdout.strip()


def gmean(values: list[float]) -> float:
    if not values:
        return 0.0
    return exp(sum(log(v) for v in values if v > 0) / max(len(values), 1))


@dataclass
class Metric:
    name: str
    field: str  # attribute on ModelResult
    unit: str  # display suffix
    higher_is_better: bool
    aggregate: str  # "gmean" or "mean"


METRICS = {
    "speedup": Metric("speedup", "speedup", "x", True, "gmean"),
    "compilation_latency": Metric(
        "compilation latency", "compilation_latency", "s", False, "mean"
    ),
    "compression_ratio": Metric(
        "memory compression", "compression_ratio", "x", True, "gmean"
    ),
    "abs_latency": Metric("absolute latency", "abs_latency", "ms", False, "mean"),
}

METRIC_CHOICES = list(METRICS.keys())

# HUD uses 5% relative threshold for flagging regressions
RELATIVE_THRESHOLD = 0.05

WORKFLOW_NAME_TO_NIGHTLY_ID = {
    v["name"]: v["id"] for k, v in WORKFLOWS.items() if k != "a100-compare"
}
# compare's baseline is A100 nightly
WORKFLOW_NAME_TO_NIGHTLY_ID["inductor-A100-perf-compare"] = WORKFLOWS["a100"]["id"]


def _short_config(config: str, device: str = "") -> str:
    """Compact label: e.g. 'a100 cudagraphs huggingface training'."""
    c = config
    backend = c
    for s in ("_huggingface_", "_timm_models_", "_torchbench_"):
        if s in c:
            backend = c.split(s)[0]
            break
    backend = backend.removeprefix("inductor_")
    mode = "training" if "training" in c else "inference" if "inference" in c else ""
    suite = ""
    for s in ("huggingface", "timm_models", "torchbench"):
        if s in c:
            suite = s
            break
    parts = [p for p in (device, backend, suite, mode) if p]
    return " ".join(parts)


@dataclass
class ModelResult:
    name: str
    speedup: float
    abs_latency: float = 0.0
    compilation_latency: float = 0.0
    compression_ratio: float = 0.0
    eager_peak_mem: float = 0.0
    dynamo_peak_mem: float = 0.0
    config: str = ""
    device: str = ""

    @property
    def short_config(self) -> str:
        return _short_config(self.config, self.device)


@dataclass
class PerfData:
    config: str  # e.g. "inductor_with_cudagraphs_huggingface_amp_training_cuda"
    models: list[ModelResult] = field(default_factory=list)
    device: str = ""

    @property
    def suite(self) -> str:
        for s in ("huggingface", "timm_models", "torchbench"):
            if s in self.config:
                return s
        return "unknown"

    @property
    def mode(self) -> str:
        if "training" in self.config:
            return "training"
        if "inference" in self.config:
            return "inference"
        return "unknown"

    @property
    def dtype(self) -> str:
        # Config format: {backend}_{suite}_{dtype}_{mode}_{device}
        # e.g. inductor_with_cudagraphs_huggingface_amp_training_cuda
        for s in ("_huggingface_", "_timm_models_", "_torchbench_"):
            if s in self.config:
                tail = self.config.split(s, 1)[1]
                # tail is e.g. "amp_training_cuda"
                parts = tail.split("_")
                if parts:
                    return parts[0]
        return "unknown"

    @property
    def short_name(self) -> str:
        name = self.config
        for s in ("_huggingface_", "_timm_models_", "_torchbench_"):
            if s in name:
                name = name.split(s)[0]
                break
        return name

    @property
    def runtime(self) -> str:
        # Last token in config: e.g. "..._training_cuda" → "cuda"
        parts = self.config.rsplit("_", 1)
        if len(parts) == 2:
            return parts[1]
        return "unknown"

    @property
    def qualified_config(self) -> str:
        if self.device:
            return f"{self.device}/{self.config}"
        return self.config

    def gmean_speedup(self) -> float:
        vals = [m.speedup for m in self.models if m.speedup > 0]
        return gmean(vals)

    def aggregate_metric(self, metric: Metric) -> float:
        vals = [
            getattr(m, metric.field)
            for m in self.models
            if getattr(m, metric.field) > 0
        ]
        if not vals:
            return 0.0
        if metric.aggregate == "gmean":
            return gmean(vals)
        return sum(vals) / len(vals)


# ---------------------------------------------------------------------------
# Artifact downloading
# ---------------------------------------------------------------------------


def get_run_jobs(run_id: int) -> list[dict]:
    data = gh(
        "run",
        "view",
        str(run_id),
        "--repo",
        REPO,
        "--json",
        "jobs",
        json_output=True,
    )
    return data["jobs"]


def get_perf_jobs(jobs: list[dict]) -> list[dict]:
    perf_jobs = []
    for job in jobs:
        m = JOB_RE.search(job["name"])
        if not m:
            continue
        config = m.group("config")
        if not PERF_CONFIGS.match(config):
            continue
        if job.get("conclusion") != "success":
            continue
        perf_jobs.append(
            {
                "config": config,
                "shard": m.group("shard"),
                "num_shards": m.group("num_shards"),
                "runner": m.group("runner"),
                "job_id": job["databaseId"],
                "name": job["name"],
            }
        )
    return perf_jobs


def s3_artifact_url(run_id: int, attempt: int, job: dict) -> str:
    config = job["config"]
    shard = job["shard"]
    num_shards = job["num_shards"]
    runner = job["runner"]
    job_id = job["job_id"]
    filename = f"test-reports-test-{config}-{shard}-{num_shards}-{runner}_{job_id}.zip"
    return f"{S3_URL}/{REPO}/{run_id}/{attempt}/artifact/{filename}"


CACHE_DIR = Path.home() / ".cache" / "perf_cli"


def get_cache_dir(run_id: int, attempt: int) -> Path:
    d = CACHE_DIR / f"{run_id}" / f"{attempt}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_and_extract_csvs(
    run_id: int,
    jobs: list[dict],
    attempt: int = 1,
    no_cache: bool = False,
) -> list[tuple[str, str]]:
    """Download artifacts and return list of (csv_filename, csv_content) pairs."""
    cache = get_cache_dir(run_id, attempt)
    results = []
    fetched = 0

    for job in jobs:
        # Check cache first
        cache_key = (
            f"{job['config']}-{job['shard']}-{job['num_shards']}-{job['job_id']}"
        )
        cache_marker = cache / f"{cache_key}.done"

        if not no_cache and cache_marker.exists():
            # Read cached CSVs
            for csv_file in cache.glob(f"{cache_key}__*.csv"):
                csv_name = csv_file.name.split("__", 1)[1]
                results.append((csv_name, csv_file.read_text()))
            continue

        url = s3_artifact_url(run_id, attempt, job)
        zip_path = cache / f"{cache_key}.zip"
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            fetched += 1
        except urllib.error.HTTPError as e:
            print(
                f"  warning: failed to download shard {job['config']} "
                f"shard {job['shard']}: {e}",
                file=sys.stderr,
            )
            continue

        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith("_performance.csv"):
                    csv_name = os.path.basename(name)
                    with zf.open(name) as f:
                        content = f.read().decode("utf-8")
                        results.append((csv_name, content))
                        # Write to cache
                        (cache / f"{cache_key}__{csv_name}").write_text(content)

        # Mark this shard as cached and remove the zip
        cache_marker.touch()
        zip_path.unlink(missing_ok=True)

    if fetched > 0:
        print(f"  downloaded {fetched} shards (cached at {cache})")
    elif results:
        print(f"  using cached data from {cache}")

    return results


def parse_csvs(csv_pairs: list[tuple[str, str]], device: str = "") -> list[PerfData]:
    grouped: dict[str, list[ModelResult]] = defaultdict(list)

    for csv_name, content in csv_pairs:
        # csv_name like: inductor_with_cudagraphs_huggingface_amp_training_cuda_performance.csv
        config = csv_name.replace("_performance.csv", "")
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            try:
                speedup = float(row.get("speedup", 0))
            except (ValueError, TypeError):
                continue
            grouped[config].append(
                ModelResult(
                    name=row.get("name", "?"),
                    speedup=speedup,
                    abs_latency=float(row.get("abs_latency", 0) or 0),
                    compilation_latency=float(row.get("compilation_latency", 0) or 0),
                    compression_ratio=float(row.get("compression_ratio", 0) or 0),
                    eager_peak_mem=float(row.get("eager_peak_mem", 0) or 0),
                    dynamo_peak_mem=float(row.get("dynamo_peak_mem", 0) or 0),
                    config=config,
                    device=device,
                )
            )

    return [
        PerfData(config=k, models=v, device=device) for k, v in sorted(grouped.items())
    ]


# ---------------------------------------------------------------------------
# S-curve rendering
# ---------------------------------------------------------------------------


def subsample(items: list, max_rows: int) -> list:
    """Evenly subsample a sorted list, always keeping first and last."""
    n = len(items)
    if n <= max_rows:
        return items
    # Always include first and last; evenly space the rest
    indices = {0, n - 1}
    for i in range(1, max_rows - 1):
        indices.add(round(i * (n - 1) / (max_rows - 1)))
    return [items[i] for i in sorted(indices)]


def render_scurve(
    perf: PerfData,
    metric: Metric,
    top_n: int = 5,
    term_width: int | None = None,
    term_height: int | None = None,
):
    if not perf.models:
        return

    if term_width is None or term_height is None:
        sz = shutil.get_terminal_size((100, 50))
        term_width = term_width or sz.columns
        term_height = term_height or sz.lines

    live = [m for m in perf.models if getattr(m, metric.field) > 0]
    if not live:
        return

    sorted_models = sorted(live, key=lambda m: getattr(m, metric.field))
    agg = perf.aggregate_metric(metric)
    n = len(sorted_models)

    # Reserve lines for header (2) + axis label (1) + padding (2)
    max_rows = max(term_height - 5, 15)
    display = subsample(sorted_models, max_rows)
    skipped = n - len(display)

    agg_label = metric.aggregate
    header = f"{perf.config} ({n} data points, {agg_label}={agg:.2f}{metric.unit})"
    if skipped > 0:
        header += f" [showing {len(display)}/{n}]"
    print(f"\n  {header}")
    print(f"  {'─' * min(len(header), term_width - 4)}")

    def fmt_val(v: float) -> str:
        if metric.unit == "s" or metric.unit == "ms":
            return f"{v:7.1f}{metric.unit}"
        return f"{v:5.2f}{metric.unit}"

    # Layout: "  {name:<max_name}  {val:>8}  {dots}"
    sample_val = fmt_val(display[0] and getattr(display[0], metric.field))
    max_name = min(max(len(m.name) for m in display), 30)
    val_width = len(sample_val)
    prefix_len = 2 + max_name + 2 + val_width + 2
    plot_width = max(term_width - prefix_len - 1, 20)

    def get_val(m):
        return getattr(m, metric.field)

    min_val = get_val(sorted_models[0])
    p95_idx = max(0, int(n * 0.95) - 1)
    p95_val = get_val(sorted_models[p95_idx])

    # For ratio metrics (speedup, compression_ratio), anchor at 1.0
    # For absolute metrics (latency), anchor at 0
    if metric.unit == "x":
        plot_min = min(min_val, 0.5)
        plot_max = max(p95_val * 1.1, 1.5)
        marker_val = 1.0
        marker_label = "1.0x"
    else:
        plot_min = 0
        plot_max = p95_val * 1.1
        marker_val = None
        marker_label = None

    span = plot_max - plot_min
    if span == 0:
        span = 1

    def val_to_col(v: float) -> int:
        return max(
            0, min(plot_width - 1, int((v - plot_min) / span * (plot_width - 1)))
        )

    marker_col = val_to_col(marker_val) if marker_val is not None else None

    for m in display:
        name = m.name[:max_name].ljust(max_name)
        v = get_val(m)
        col = val_to_col(v)
        bar = [" "] * plot_width
        if marker_col is not None:
            bar[marker_col] = "|"
        for i in range(col + 1):
            if marker_col is not None and i == marker_col:
                bar[i] = "|"
            else:
                bar[i] = "·"
        print(f"  {name}  {fmt_val(v)}  {''.join(bar)}")

    pad = " " * prefix_len
    if marker_label and marker_col is not None:
        print(f"{pad}{' ' * marker_col}{marker_label}")
    else:
        print()


def print_worst_offenders(perf: PerfData, metric: Metric, top_n: int = 5):
    def get_val(m):
        return getattr(m, metric.field)

    live = [m for m in perf.models if get_val(m) > 0]
    if not live:
        return
    # "worst" depends on metric direction
    if metric.higher_is_better:
        worst = sorted(live, key=get_val)[:top_n]
    else:
        worst = sorted(live, key=get_val, reverse=True)[:top_n]
    print(f"\n  Worst offenders ({metric.name}):")
    for i, m in enumerate(worst, 1):
        v = get_val(m)
        parts = [f"{v:.3f}{metric.unit}"]
        if m.config:
            parts.append(m.short_config)
        detail = "  ".join(parts)
        print(f"    {i}. {m.name:<30}  {detail}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def build_dispatch_inputs(args) -> list[str]:
    """Build -f flags for workflow dispatch inputs from CLI args."""
    flags = []
    bool_inputs = [
        "training",
        "inference",
        "default",
        "dynamic",
        "cppwrapper",
        "cudagraphs",
        "freezing_cudagraphs",
        "aotinductor",
        "maxautotune",
    ]
    for name in bool_inputs:
        val = getattr(args, name, None)
        if val is not None:
            flags.extend(["-f", f"{name}={'true' if val else 'false'}"])
    if args.benchmark_configs:
        flags.extend(["-f", f"benchmark_configs={args.benchmark_configs}"])
    return flags


def dispatch_one(device: str, ref: str, extra_flags: list[str]) -> int | None:
    wf = WORKFLOWS[device]
    print(f"\nLaunching {wf['name']} on ref: {ref}")

    dispatch_args = [
        "workflow",
        "run",
        str(wf["id"]),
        "--repo",
        REPO,
        "--ref",
        ref,
    ] + extra_flags

    gh(*dispatch_args)
    print("Dispatched. Waiting a few seconds for the run to appear...")
    time.sleep(5)

    runs = gh(
        "run",
        "list",
        "--repo",
        REPO,
        "--workflow",
        str(wf["id"]),
        "--branch",
        ref,
        "--limit",
        "1",
        "--json",
        "databaseId,status,url,createdAt",
        json_output=True,
    )
    if not runs:
        print("Could not find the dispatched run. Check the Actions tab manually.")
        return None

    run = runs[0]
    run_id = run["databaseId"]
    url = f"https://github.com/{REPO}/actions/runs/{run_id}"
    print(f"Run ID:  {run_id}")
    print(f"URL:     {url}")
    print(f"Status:  {run['status']}")
    return run_id


def wait_for_runs(pending: dict[str, int]) -> dict[str, int]:
    """Poll all runs until they complete. Returns dict of successful runs."""
    print(f"\nWaiting for {len(pending)} run(s)...", flush=True)
    remaining = dict(pending)
    succeeded: dict[str, int] = {}
    while remaining:
        time.sleep(30)
        done = []
        for device, run_id in remaining.items():
            data = gh(
                "run",
                "view",
                str(run_id),
                "--repo",
                REPO,
                "--json",
                "status,conclusion",
                json_output=True,
            )
            status = data.get("status", "unknown")
            if status == "completed":
                conclusion = data.get("conclusion", "unknown")
                print(f"  {device} (run {run_id}): {conclusion}")
                if conclusion == "success":
                    succeeded[device] = run_id
                done.append(device)
        for d in done:
            del remaining[d]
        if remaining:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"  [{ts}] still waiting: {', '.join(remaining)}...", flush=True)
    return succeeded


def cmd_launch(args):
    ref = args.ref or git("rev-parse", "--abbrev-ref", "HEAD")
    if ref == "HEAD":
        ref = git("rev-parse", "HEAD")

    extra_flags = build_dispatch_inputs(args)
    launched: dict[str, int] = {}
    for device in args.device:
        if device not in WORKFLOWS:
            print(f"Unknown device: {device}", file=sys.stderr)
            sys.exit(1)
        run_id = dispatch_one(device, ref, extra_flags)
        if run_id:
            launched[device] = run_id

    if (args.wait or args.wait_and_summarize) and launched:
        succeeded = wait_for_runs(launched)
        if args.wait_and_summarize and succeeded:
            # Use the first device's run ID as the positional arg; pass all
            # device→run_id pairs via _run_ids so cmd_summary skips resolution.
            first_run = next(iter(succeeded.values()))
            summary_args = argparse.Namespace(
                run_id=str(first_run),
                device=list(succeeded.keys()),
                _run_ids=succeeded,
                baseline="latest",
                metric="speedup",
                top=5,
                config=None,
                suite=None,
                mode=None,
                group_by=None,
                attempt=1,
                no_cache=False,
            )
            print(f"\n{'=' * 70}")
            print("Summary")
            print(f"{'=' * 70}")
            cmd_summary(summary_args)


def filter_perf(all_perf: list[PerfData], args) -> list[PerfData]:
    result = all_perf
    if getattr(args, "config", None):
        pattern = re.compile(args.config, re.IGNORECASE)
        result = [p for p in result if pattern.search(p.config)]
    if getattr(args, "suite", None):
        suite = SUITE_ALIASES.get(args.suite, args.suite)
        result = [p for p in result if p.suite == suite]
    if getattr(args, "mode", None):
        result = [p for p in result if p.mode == args.mode]
    if getattr(args, "backend", None):
        pattern = re.compile(args.backend, re.IGNORECASE)
        result = [p for p in result if pattern.search(p.short_name)]
    if getattr(args, "dtype", None):
        result = [p for p in result if p.dtype == args.dtype]
    if getattr(args, "runtime", None):
        result = [p for p in result if p.runtime == args.runtime]
    return result


GROUP_KEY_FNS: dict[str, callable] = {
    "config": lambda p: p.qualified_config,
    "suite": lambda p: p.suite,
    "mode": lambda p: p.mode,
    "backend": lambda p: p.short_name,
    "device": lambda p: p.device or "unknown",
    "dtype": lambda p: p.dtype,
    "runtime": lambda p: p.runtime,
}

GROUP_CHOICES = sorted(GROUP_KEY_FNS.keys())


def group_perf(all_perf: list[PerfData], group_by: list[str] | None) -> list[PerfData]:
    if not group_by:
        all_models = []
        for p in all_perf:
            all_models.extend(p.models)
        return [PerfData(config="all", models=all_models)]

    fns = []
    for key in group_by:
        fn = GROUP_KEY_FNS.get(key)
        if fn is None:
            print(f"Unknown group-by: {key}", file=sys.stderr)
            sys.exit(1)
        fns.append(fn)

    def composite_key(p: PerfData) -> str:
        return " / ".join(fn(p) for fn in fns)

    groups: dict[str, list[ModelResult]] = defaultdict(list)
    for p in all_perf:
        groups[composite_key(p)].extend(p.models)

    return [PerfData(config=k, models=v) for k, v in sorted(groups.items())]


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


NIGHTLY_WORKFLOW_IDS = {k: v["id"] for k, v in WORKFLOWS.items() if k != "a100-compare"}


def _find_latest_run(
    branch: str,
    device: str,
) -> dict | None:
    """Find the latest successful perf nightly run for a branch + device.

    Returns {databaseId, createdAt, headSha} or None.
    """
    wf_id = NIGHTLY_WORKFLOW_IDS.get(device)
    if wf_id is None:
        return None

    runs = gh(
        "run",
        "list",
        "--repo",
        REPO,
        "--workflow",
        str(wf_id),
        "--branch",
        branch,
        "--status",
        "success",
        "--limit",
        "1",
        "--json",
        "databaseId,createdAt,headSha",
        json_output=True,
    )
    if not runs:
        return None
    return runs[0]


def resolve_run(branch: str, device: str) -> int:
    """Find the latest successful perf nightly run for a branch + device."""
    run = _find_latest_run(branch, device)
    if not run:
        print(
            f"No successful perf run found for branch '{branch}' on {device}.",
            file=sys.stderr,
        )
        sys.exit(1)
    run_id = run["databaseId"]
    created = run["createdAt"][:10]
    print(f"Resolved '{branch}' → run {run_id} ({device}, {created})")
    return run_id


def discover_runs(branch: str) -> dict[str, int]:
    """Auto-discover all devices with successful runs for a branch.

    Finds the latest commit SHA that has runs, then returns all runs matching
    that commit.
    """
    candidates: list[tuple[str, dict]] = []
    for device in DEVICE_CHOICES:
        run = _find_latest_run(branch, device)
        if run:
            candidates.append((device, run))

    if not candidates:
        print(
            f"No successful perf runs found for branch '{branch}' on any device.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Pick the most recent commit (by createdAt) and collect all runs on it
    candidates.sort(key=lambda x: x[1]["createdAt"], reverse=True)
    target_sha = candidates[0][1]["headSha"]

    result = {}
    for device, run in candidates:
        if run["headSha"] == target_sha:
            run_id = run["databaseId"]
            created = run["createdAt"][:10]
            print(
                f"Discovered '{branch}' → run {run_id} ({device}, {created}, {target_sha[:10]})"
            )
            result[device] = run_id

    return result


def resolve_runs(branch: str, devices: list[str]) -> dict[str, int]:
    """Resolve the latest successful run for each device. Returns {device: run_id}."""
    result = {}
    for device in devices:
        result[device] = resolve_run(branch, device)
    return result


def resolve_baseline(head_run_id: int) -> int:
    """Find the latest successful nightly on main, skipping the head run itself."""
    run_data = gh(
        "run",
        "view",
        str(head_run_id),
        "--repo",
        REPO,
        "--json",
        "workflowName",
        json_output=True,
    )
    wf_name = run_data.get("workflowName", "")
    nightly_id = WORKFLOW_NAME_TO_NIGHTLY_ID.get(wf_name)
    if nightly_id is None:
        print(
            f"Don't know which nightly corresponds to workflow '{wf_name}'",
            file=sys.stderr,
        )
        sys.exit(1)

    runs = gh(
        "run",
        "list",
        "--repo",
        REPO,
        "--workflow",
        str(nightly_id),
        "--branch",
        "main",
        "--status",
        "success",
        "--limit",
        "5",
        "--json",
        "databaseId,createdAt,headBranch",
        json_output=True,
    )
    # Skip the head run itself to avoid comparing a run to itself
    for run in runs:
        if run["databaseId"] != head_run_id:
            baseline_id = run["databaseId"]
            created = run["createdAt"][:10]
            print(f"Baseline: run {baseline_id} (main, {created})")
            return baseline_id

    print("No suitable baseline nightly found on main.", file=sys.stderr)
    sys.exit(1)


def device_for_workflow(workflow_name: str) -> str:
    """Reverse-lookup device key from workflow name."""
    for k, v in WORKFLOWS.items():
        if v["name"] == workflow_name:
            return k
    return ""


@dataclass
class RunMeta:
    run_id: int
    head_sha: str
    head_branch: str
    workflow_name: str
    created_at: str
    event: str

    @property
    def short_sha(self) -> str:
        return self.head_sha[:10]

    @property
    def date(self) -> str:
        return self.created_at[:10]


def fetch_run_meta(run_id: int) -> RunMeta:
    data = gh(
        "api",
        f"repos/{REPO}/actions/runs/{run_id}",
        "-q",
        "{headSha: .head_sha, headBranch: .head_branch, workflowName: .name, createdAt: .created_at, event: .event}",
        json_output=True,
    )
    return RunMeta(
        run_id=run_id,
        head_sha=data.get("headSha", "unknown"),
        head_branch=data.get("headBranch", "unknown"),
        workflow_name=data.get("workflowName", "unknown"),
        created_at=data.get("createdAt", "unknown"),
        event=data.get("event", "unknown"),
    )


def print_run_header(
    label: str,
    metas: list[RunMeta],
    configs: list[str] | None = None,
):
    print(f"\n  {label}")
    print(f"  {'─' * len(label)}")
    if len(metas) == 1:
        m = metas[0]
        print(f"  Run:      {m.run_id}  ({m.workflow_name})")
        print(f"  Commit:   {m.short_sha}  ({m.head_branch}, {m.date})")
    else:
        # Show commit from first (should all match for multi-device)
        print(
            f"  Commit:   {metas[0].short_sha}  ({metas[0].head_branch}, {metas[0].date})"
        )
        print("  Runs:")
        for m in metas:
            print(f"    {m.run_id}  ({m.workflow_name})")
    if configs:
        print(f"  Configs:  {len(configs)} — {', '.join(sorted(configs))}")


def fetch_run_perf(
    run_id: int,
    attempt: int,
    no_cache: bool,
    device: str = "",
    allow_empty: bool = False,
) -> list[PerfData]:
    """Fetch and parse perf data for a run."""
    jobs = get_run_jobs(run_id)
    perf_jobs = get_perf_jobs(jobs)
    if not perf_jobs:
        if allow_empty:
            print(f"  no perf jobs in run {run_id}, skipping", file=sys.stderr)
            return []
        print(f"No successful perf jobs in run {run_id}.", file=sys.stderr)
        sys.exit(1)
    csv_pairs = download_and_extract_csvs(run_id, perf_jobs, attempt, no_cache=no_cache)
    if not csv_pairs:
        if allow_empty:
            print(f"  no CSVs in run {run_id}, skipping", file=sys.stderr)
            return []
        print(f"No performance CSVs in run {run_id}.", file=sys.stderr)
        sys.exit(1)
    return parse_csvs(csv_pairs, device=device)


@dataclass
class ModelDelta:
    name: str
    base_val: float
    head_val: float
    config: str = ""
    device: str = ""

    @property
    def delta_pct(self) -> float:
        if self.base_val == 0:
            return 0.0
        return (self.head_val - self.base_val) / self.base_val * 100

    @property
    def short_config(self) -> str:
        return _short_config(self.config, self.device)


@dataclass
class ConfigAgg:
    base_agg: float
    base_count: int
    head_agg: float
    head_count: int
    paired_ratio: float  # gmean(head_val / base_val) over paired models
    paired_count: int


def compute_deltas(
    head_perf: list[PerfData], base_perf: list[PerfData], metric: Metric
) -> tuple[list[ModelDelta], dict[str, ConfigAgg]]:
    """Join head and base on (device, config, model_name) and compute deltas.

    Returns (per_model_deltas, per_config_aggregates).
    per_config_aggregates maps qualified_config -> ConfigAgg.
    """
    # Build base lookup: (device, config, model_name) -> metric value
    base_lookup: dict[tuple[str, str, str], float] = {}
    for perf in base_perf:
        for m in perf.models:
            v = getattr(m, metric.field)
            if v > 0:
                base_lookup[(perf.device, perf.config, m.name)] = v

    deltas = []
    for perf in head_perf:
        for m in perf.models:
            head_val = getattr(m, metric.field)
            if head_val <= 0:
                continue
            key = (perf.device, perf.config, m.name)
            if key not in base_lookup:
                continue
            base_val = base_lookup[key]
            deltas.append(
                ModelDelta(
                    name=m.name,
                    base_val=base_val,
                    head_val=head_val,
                    config=perf.config,
                    device=perf.device,
                )
            )

    # Group deltas by qualified_config for paired aggregates
    deltas_by_qconfig: dict[str, list[ModelDelta]] = defaultdict(list)
    for d in deltas:
        qc = f"{d.device}/{d.config}" if d.device else d.config
        deltas_by_qconfig[qc].append(d)

    # Per-config aggregates (keyed by qualified_config for display)
    config_aggs: dict[str, ConfigAgg] = {}
    base_by_qconfig: dict[str, PerfData] = {p.qualified_config: p for p in base_perf}
    for perf in head_perf:
        qc = perf.qualified_config
        if qc not in base_by_qconfig:
            continue
        base_perf_data = base_by_qconfig[qc]
        head_agg = perf.aggregate_metric(metric)
        base_agg = base_perf_data.aggregate_metric(metric)
        head_count = len([m for m in perf.models if getattr(m, metric.field) > 0])
        base_count = len(
            [m for m in base_perf_data.models if getattr(m, metric.field) > 0]
        )

        paired = deltas_by_qconfig.get(qc, [])
        ratios = [d.head_val / d.base_val for d in paired if d.base_val > 0]
        config_aggs[qc] = ConfigAgg(
            base_agg=base_agg,
            base_count=base_count,
            head_agg=head_agg,
            head_count=head_count,
            paired_ratio=gmean(ratios) if ratios else 0.0,
            paired_count=len(ratios),
        )

    return deltas, config_aggs


def print_comparison_table(config_aggs: dict[str, ConfigAgg], metric: Metric):
    u = metric.unit
    print(f"\n{'Config':<55} {'base':>16} {'new':>16} {'head/base':>18}")
    print("─" * 108)
    for config in sorted(config_aggs):
        agg = config_aggs[config]
        flag = ""
        if agg.paired_ratio > 0:
            delta_pct = (agg.paired_ratio - 1.0) * 100
            if abs(delta_pct) > RELATIVE_THRESHOLD * 100:
                if metric.higher_is_better:
                    flag = " !!" if delta_pct < 0 else " ++"
                else:
                    flag = " !!" if delta_pct > 0 else " ++"
        print(
            f"  {config:<53} "
            f"{agg.base_agg:>5.2f}{u} (n={agg.base_count}) "
            f"{agg.head_agg:>5.2f}{u} (n={agg.head_count}) "
            f"{agg.paired_ratio:>5.3f}x (n={agg.paired_count}){flag}"
        )


def print_regressions(deltas: list[ModelDelta], metric: Metric, top_n: int):
    # For higher_is_better metrics, regression = negative delta
    # For lower_is_better metrics, regression = positive delta
    if metric.higher_is_better:
        bad = [d for d in deltas if d.delta_pct < -RELATIVE_THRESHOLD * 100]
        bad.sort(key=lambda d: d.delta_pct)
    else:
        bad = [d for d in deltas if d.delta_pct > RELATIVE_THRESHOLD * 100]
        bad.sort(key=lambda d: d.delta_pct, reverse=True)

    if not bad:
        print(f"\n  No regressions (>{RELATIVE_THRESHOLD * 100:.0f}% change).")
        return

    print(f"\n  Regressions ({len(bad)} models, showing top {min(top_n, len(bad))}):")
    for i, d in enumerate(bad[:top_n], 1):
        print(
            f"    {i}. {d.name:<30}  "
            f"{d.base_val:.2f}{metric.unit} → {d.head_val:.2f}{metric.unit}  "
            f"{d.delta_pct:>+6.1f}%  {d.short_config}"
        )


def render_delta_scurve(
    deltas: list[ModelDelta],
    metric: Metric,
    term_width: int | None = None,
    term_height: int | None = None,
):
    if not deltas:
        return

    if term_width is None or term_height is None:
        sz = shutil.get_terminal_size((100, 50))
        term_width = term_width or sz.columns
        term_height = term_height or sz.lines

    sorted_deltas = sorted(deltas, key=lambda d: d.delta_pct)
    n = len(sorted_deltas)
    max_rows = max(term_height - 5, 15)
    display = subsample(sorted_deltas, max_rows)
    skipped = n - len(display)

    header = f"Delta S-curve ({n} models)"
    if skipped > 0:
        header += f" [showing {len(display)}/{n}]"
    print(f"\n  {header}")
    print(f"  {'─' * min(len(header), term_width - 4)}")

    max_name = min(max(len(d.name) for d in display), 28)
    # "  name  +12.3%  {bar}"
    prefix_len = 2 + max_name + 2 + 7 + 2
    plot_width = max(term_width - prefix_len - 1, 20)

    # Range: cap at p5/p95 to avoid outlier squishing
    p5_idx = max(0, int(n * 0.05))
    p95_idx = min(n - 1, int(n * 0.95))
    range_lo = min(sorted_deltas[p5_idx].delta_pct, -10)
    range_hi = max(sorted_deltas[p95_idx].delta_pct, 10)
    # Ensure symmetric-ish around 0
    abs_max = max(abs(range_lo), abs(range_hi))
    range_lo = -abs_max
    range_hi = abs_max
    span = range_hi - range_lo
    if span == 0:
        span = 1

    def pct_to_col(pct: float) -> int:
        return max(
            0, min(plot_width - 1, int((pct - range_lo) / span * (plot_width - 1)))
        )

    zero_col = pct_to_col(0)

    for d in display:
        name = d.name[:max_name].ljust(max_name)
        col = pct_to_col(d.delta_pct)
        bar = [" "] * plot_width
        bar[zero_col] = "|"
        if col <= zero_col:
            for i in range(col, zero_col):
                bar[i] = "·"
            bar[zero_col] = "|"
        else:
            bar[zero_col] = "|"
            for i in range(zero_col + 1, col + 1):
                bar[i] = "·"
        print(f"  {name}  {d.delta_pct:>+6.1f}%  {''.join(bar)}")

    pad = " " * prefix_len
    print(f"{pad}{' ' * zero_col}0%")


def print_summary_table(all_perf: list[PerfData], metric: Metric):
    agg_label = metric.aggregate
    print(f"\n{'Config':<65} {'Models':>6} {agg_label:>10}")
    print("─" * 85)
    for perf in all_perf:
        n = len([m for m in perf.models if getattr(m, metric.field) > 0])
        agg = perf.aggregate_metric(metric)
        print(f"  {perf.qualified_config:<63} {n:>6} {agg:>8.2f}{metric.unit}")


def _resolve_head_runs(args) -> dict[str, int]:
    """Parse run_id arg into {device: run_id} mapping.

    run_id can be:
      - A single numeric run ID (device inferred from workflow)
      - A branch name (resolved across all --device values, or auto-discovered)
    Pre-resolved IDs can be passed via args._run_ids.
    """
    # Pre-resolved (from --wait-and-summarize)
    if hasattr(args, "_run_ids") and args._run_ids:
        return args._run_ids

    raw = args.run_id
    try:
        run_id = int(raw)
        # Single run ID — infer device from the workflow
        meta = fetch_run_meta(run_id)
        device = device_for_workflow(meta.workflow_name)
        return {device: run_id}
    except ValueError:
        # Branch name
        if args.device:
            return resolve_runs(raw, args.device)
        return discover_runs(raw)


def cmd_summary(args):
    attempt = args.attempt
    metric = METRICS[args.metric]

    head_run_ids = _resolve_head_runs(args)

    # Fetch head runs
    auto_discovered = not hasattr(args, "_run_ids") and not args.device
    head_metas: list[RunMeta] = []
    head_perf: list[PerfData] = []
    for device, run_id in list(head_run_ids.items()):
        print(f"Fetching head run {run_id} ({device})...")
        perf = fetch_run_perf(
            run_id,
            attempt,
            args.no_cache,
            device=device,
            allow_empty=auto_discovered,
        )
        if not perf:
            del head_run_ids[device]
            continue
        head_metas.append(fetch_run_meta(run_id))
        head_perf.extend(perf)

    head_perf = filter_perf(head_perf, args)
    if not head_perf:
        print("No configs matched filters.")
        sys.exit(1)

    head_configs = [p.qualified_config for p in head_perf]

    # Baseline comparison mode
    if args.baseline and args.baseline.lower() != "none":
        base_metas: list[RunMeta] = []
        base_perf: list[PerfData] = []

        for device, head_run_id in head_run_ids.items():
            if args.baseline == "latest":
                baseline_id = resolve_baseline(head_run_id)
            else:
                try:
                    baseline_id = int(args.baseline)
                except ValueError:
                    baseline_id = resolve_run(args.baseline, device)

            print(f"Fetching baseline run {baseline_id} ({device})...")
            base_data = fetch_run_perf(
                baseline_id,
                attempt,
                args.no_cache,
                device=device,
                allow_empty=auto_discovered,
            )
            if not base_data:
                continue
            base_metas.append(fetch_run_meta(baseline_id))
            base_perf.extend(base_data)

        base_perf = filter_perf(base_perf, args)
        if not base_perf:
            print("No baseline configs matched filters.")
            sys.exit(1)

        print_run_header("HEAD", head_metas, head_configs)
        print_run_header("BASE", base_metas, [p.qualified_config for p in base_perf])
        print()

        deltas, config_aggs = compute_deltas(head_perf, base_perf, metric)
        if not deltas:
            print("No matching models between head and baseline.")
            sys.exit(1)

        print_comparison_table(config_aggs, metric)
        print_regressions(deltas, metric, args.top)
        render_delta_scurve(deltas, metric)
        return

    # Absolute mode (no baseline)
    print_run_header("RUN", head_metas, head_configs)
    print()

    print_summary_table(head_perf, metric)
    grouped = group_perf(head_perf, args.group_by)
    for perf in grouped:
        print_worst_offenders(perf, metric, args.top)
        render_scurve(perf, metric, args.top)


CONFIG_RE = re.compile(
    r"(?P<backend>inductor_[a-z_]+?)_"
    r"(?P<suite>huggingface|timm_models|torchbench)_"
    r"(?P<dtype>\w+)_"
    r"(?P<mode>training|inference)_"
    r"(?P<device>\w+)"
)


def config_to_command(
    config: str,
    suite: str,
    model: str | None = None,
) -> str | None:
    """Turn a config name into a runnable benchmark command."""
    m = CONFIG_RE.match(config)
    if not m:
        return None

    backend_variant = m.group("backend")
    dtype = m.group("dtype")
    mode = m.group("mode")
    runtime = m.group("device")

    # Runtime → --device flag (strip platform suffix like _x86_zen)
    device_flag = runtime.split("_")[0]  # "cpu_x86_zen" → "cpu"

    cmd_parts = [
        "python",
        f"benchmarks/dynamo/{suite}.py",
        f"--{mode}",
        f"--{dtype}",
        "--backend",
        "inductor",
        "--device",
        device_flag,
    ]

    if "no_cudagraphs" in backend_variant:
        cmd_parts.append("--disable-cudagraphs")
    if "dynamic" in backend_variant:
        cmd_parts.extend(["--dynamic-shapes", "--dynamic-batch-only"])
    if "cpp_wrapper" in backend_variant:
        cmd_parts.insert(0, "TORCHINDUCTOR_CPP_WRAPPER=1")
        cmd_parts.append("--disable-cudagraphs")
    if "freezing" in backend_variant:
        cmd_parts.append("--freezing")
    if "max_autotune" in backend_variant:
        cmd_parts.insert(0, "TORCHINDUCTOR_MAX_AUTOTUNE=1")
    if "aot_inductor" in backend_variant:
        cmd_parts.append("--export-aot-inductor")
        cmd_parts.append("--disable-cudagraphs")

    cmd_parts.extend(["--performance", "--cold-start-latency"])

    if model:
        cmd_parts.extend(["--only", model])

    cmd_parts.extend(["--output", f"{config}_performance.csv"])

    return " ".join(cmd_parts)


def cmd_repro(args):
    run_ids = _resolve_head_runs(args)

    # Fetch perf data to discover configs
    auto_discovered = not hasattr(args, "_run_ids") and not args.device
    all_perf: list[PerfData] = []
    metas: list[RunMeta] = []
    for device, run_id in list(run_ids.items()):
        print(f"Fetching run {run_id} ({device})...")
        metas.append(fetch_run_meta(run_id))
        perf = fetch_run_perf(
            run_id,
            args.attempt,
            no_cache=False,
            device=device,
            allow_empty=auto_discovered,
        )
        all_perf.extend(perf)

    all_perf = filter_perf(all_perf, args)
    if not all_perf:
        print("No configs matched filters.")
        sys.exit(1)

    print_run_header("REPRO", metas)

    configs_seen: dict[str, str] = {}  # config_name → suite
    for perf in all_perf:
        configs_seen[perf.config] = perf.suite

    count = 0
    commands: list[str] = []
    for config in sorted(configs_seen):
        suite = configs_seen[config]
        cmd = config_to_command(config, suite, args.model)
        if not cmd:
            continue
        commands.append(f"# {config}\n{cmd}")
        count += 1

    print(f"\nReproducible commands ({count} configs):\n")
    for cmd in commands:
        print(f"{cmd}\n")


PIN_DIR = Path(".ci/docker/ci_commit_pins")


def read_pin(name: str) -> str:
    """Read a pinned commit or requirements file."""
    path = PIN_DIR / name
    if not path.exists():
        return f"<{name} not found>"
    return path.read_text().strip()


def cmd_prepare_repro(args):
    suites = set()
    if args.suite:
        suite = SUITE_ALIASES.get(args.suite, args.suite)
        suites = {suite}
    else:
        suites = {"huggingface", "timm_models", "torchbench"}

    torchbench_pin = read_pin("torchbench.txt")
    timm_pin = read_pin("timm.txt")
    hf_reqs = read_pin("huggingface-requirements.txt")

    print("# Setup commands for inductor perf benchmark suites")
    print("# These mirror what CI does in the inductor-benchmarks Docker image.")
    print("#")
    print("# Pinned versions (commits, package versions) are read live from")
    print("#   .ci/docker/ci_commit_pins/")
    print("# Install steps are based on:")
    print("#   .ci/docker/common/install_inductor_benchmark_deps.sh  (build-time)")
    print("#   .ci/pytorch/test.sh                                   (runtime)")
    print("# If the setup process changes, check those files.")
    print()

    if "huggingface" in suites:
        print("# ── HuggingFace ──")
        for line in hf_reqs.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                print(f"pip install {line}")
        print()

    if "timm_models" in suites:
        print("# ── Timm ──")
        print(
            f"pip install git+https://github.com/huggingface/pytorch-image-models@{timm_pin}"
        )
        print()

    if "torchbench" in suites:
        print("# ── TorchBench ──")
        print("git clone https://github.com/pytorch/benchmark torchbench")
        print(f"cd torchbench && git checkout {torchbench_pin}")
        print("python install.py --continue_on_fail")
        print("cd ..")
        print()
        print("# Set PYTHONPATH so benchmark scripts find torchbench")
        print("export PYTHONPATH=$(pwd)/torchbench")
        print()

    print("# ── Runtime dependencies ──")
    print("pip install torchvision torchaudio")
    if "torchbench" in suites:
        print("pip install opencv-python==4.8.0.74")
    print()

    print("# ── Environment variables ──")
    print("export TORCHINDUCTOR_FX_GRAPH_CACHE=True")
    print("export TORCHINDUCTOR_AUTOGRAD_CACHE=True")
    print()

    if not args.no_repro:
        # Also print repro commands if we have a run_id
        if args.run_id:
            # Reuse the repro logic
            repro_args = argparse.Namespace(
                run_id=args.run_id,
                device=args.device,
                model=args.model,
                suite=args.suite,
                mode=args.mode,
                backend=None,
                dtype=None,
                runtime=None,
                attempt=args.attempt,
            )
            print("# ── Benchmark commands ──")
            cmd_repro(repro_args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CLI tool for inductor perf regression runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch a perf run on your current branch (A100)
  python benchmarks/dynamo/perf_cli.py launch

  # Launch on both A100 and H100
  python benchmarks/dynamo/perf_cli.py launch --device a100 h100

  # Launch on H100 and wait for completion
  python benchmarks/dynamo/perf_cli.py launch --device h100 --wait

  # Launch with only inference, no training
  python benchmarks/dynamo/perf_cli.py launch --no-training --inference

  # Launch with dynamic shapes enabled
  python benchmarks/dynamo/perf_cli.py launch --dynamic

  # Launch on ROCm MI300
  python benchmarks/dynamo/perf_cli.py launch --device rocm-mi300

  # Check your branch against latest main nightly
  python benchmarks/dynamo/perf_cli.py summary my-feature-branch

  # Same but on H100
  python benchmarks/dynamo/perf_cli.py summary my-feature-branch --device h100

  # Compare across A100 and H100 in one summary
  python benchmarks/dynamo/perf_cli.py summary my-feature-branch --device a100 h100

  # Use a specific run ID instead of branch name
  python benchmarks/dynamo/perf_cli.py summary 22842783236

  # Compare against a specific baseline (run ID or branch)
  python benchmarks/dynamo/perf_cli.py summary 22842783236 --baseline 22816292132

  # Absolute metrics only (no comparison)
  python benchmarks/dynamo/perf_cli.py summary 22842783236 --baseline none

  # Filter to cudagraphs training
  python benchmarks/dynamo/perf_cli.py summary main --config cudagraphs --mode training

  # Show commands to reproduce a run locally for a single model
  python benchmarks/dynamo/perf_cli.py repro 22842783236 --model BERT_pytorch --suite tb
""",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- launch --
    p_launch = sub.add_parser(
        "launch", help="Launch a perf regression run on your branch"
    )
    p_launch.add_argument(
        "--device",
        nargs="+",
        default=["a100"],
        choices=DEVICE_CHOICES,
        metavar="DEVICE",
        help=f"Devices to launch (default: a100). Choices: {', '.join(DEVICE_CHOICES)}",
    )
    p_launch.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Git ref to benchmark (default: current branch)",
    )
    p_launch.add_argument(
        "--wait", action="store_true", help="Wait for all runs to complete"
    )
    p_launch.add_argument(
        "--wait-and-summarize",
        dest="wait_and_summarize",
        action="store_true",
        help="Wait for all runs, then print summary vs. latest main nightly",
    )
    # Workflow dispatch inputs
    launch_opts = p_launch.add_argument_group("workflow options")
    launch_opts.add_argument(
        "--training",
        action="store_true",
        default=None,
        help="Enable training benchmarks",
    )
    launch_opts.add_argument(
        "--no-training",
        dest="training",
        action="store_false",
        help="Disable training benchmarks",
    )
    launch_opts.add_argument(
        "--inference",
        action="store_true",
        default=None,
        help="Enable inference benchmarks",
    )
    launch_opts.add_argument(
        "--no-inference",
        dest="inference",
        action="store_false",
        help="Disable inference benchmarks",
    )
    launch_opts.add_argument(
        "--cudagraphs", action="store_true", default=None, help="Enable cudagraphs"
    )
    launch_opts.add_argument(
        "--no-cudagraphs",
        dest="cudagraphs",
        action="store_false",
        help="Disable cudagraphs",
    )
    launch_opts.add_argument(
        "--dynamic", action="store_true", default=None, help="Enable dynamic shapes"
    )
    launch_opts.add_argument(
        "--no-dynamic",
        dest="dynamic",
        action="store_false",
        help="Disable dynamic shapes",
    )
    launch_opts.add_argument(
        "--cppwrapper", action="store_true", default=None, help="Enable cpp wrapper"
    )
    launch_opts.add_argument(
        "--freezing-cudagraphs",
        dest="freezing_cudagraphs",
        action="store_true",
        default=None,
    )
    launch_opts.add_argument("--aotinductor", action="store_true", default=None)
    launch_opts.add_argument("--maxautotune", action="store_true", default=None)
    launch_opts.add_argument(
        "--default", dest="default", action="store_true", default=None
    )
    launch_opts.add_argument(
        "--benchmark-configs",
        dest="benchmark_configs",
        type=str,
        default=None,
        help="Override benchmark_configs input",
    )

    # -- summary --
    p_summary = sub.add_parser("summary", help="Summarize results of a perf run")
    p_summary.add_argument(
        "run_id", type=str, help="GitHub Actions run ID or branch name"
    )
    p_summary.add_argument(
        "--device",
        nargs="+",
        default=None,
        choices=DEVICE_CHOICES,
        metavar="DEVICE",
        help=f"Device(s) to summarize (default: auto-discover from branch). Choices: {', '.join(DEVICE_CHOICES)}",
    )
    p_summary.add_argument(
        "--baseline",
        type=str,
        default="latest",
        help="Baseline run ID, 'latest' for most recent main nightly (default), or 'none' to disable",
    )
    p_summary.add_argument(
        "--metric",
        type=str,
        default="speedup",
        choices=METRIC_CHOICES,
        help="Metric to display (default: speedup)",
    )
    p_summary.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of worst offenders to show (default: 5)",
    )
    # Filters
    filters = p_summary.add_argument_group("filters")
    filters.add_argument(
        "--config",
        type=str,
        default=None,
        help="Regex to filter config names (e.g. 'cudagraphs', 'dynamic')",
    )
    filters.add_argument(
        "--suite", type=str, default=None, help="Filter to suite: hf, timm, tb"
    )
    filters.add_argument(
        "--mode",
        type=str,
        choices=["training", "inference"],
        default=None,
        help="Filter to training or inference",
    )
    filters.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Regex to filter backend (e.g. 'cudagraphs', 'dynamic')",
    )
    filters.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Filter to dtype (e.g. amp, float16, bfloat16)",
    )
    filters.add_argument(
        "--runtime",
        type=str,
        default=None,
        help="Filter to runtime (e.g. cuda, cpu, xpu)",
    )
    p_summary.add_argument(
        "--group-by",
        dest="group_by",
        nargs="+",
        default=None,
        choices=GROUP_CHOICES,
        metavar="KEY",
        help=f"Group S-curves by key(s) (default: single combined). Choices: {', '.join(GROUP_CHOICES)}",
    )
    p_summary.add_argument(
        "--attempt", type=int, default=1, help="Run attempt number (default: 1)"
    )
    p_summary.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        default=False,
        help="Re-download artifacts even if cached",
    )

    # -- repro --
    p_repro = sub.add_parser("repro", help="Reproduce a remote perf run locally")
    p_repro.add_argument(
        "run_id", type=str, help="GitHub Actions run ID or branch name"
    )
    p_repro.add_argument(
        "--device",
        nargs="+",
        default=None,
        choices=DEVICE_CHOICES,
        metavar="DEVICE",
        help="Device(s) (default: auto-discover from branch)",
    )
    p_repro.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this model (e.g. BERT_pytorch)",
    )
    p_repro.add_argument(
        "--suite", type=str, default=None, help="Filter to suite (hf, timm, tb)"
    )
    p_repro.add_argument(
        "--mode",
        type=str,
        choices=["training", "inference"],
        default=None,
        help="Filter to mode",
    )
    p_repro.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Regex to filter backend (e.g. 'cudagraphs', 'dynamic')",
    )
    p_repro.add_argument(
        "--dtype", type=str, default=None, help="Filter to dtype (e.g. amp, bfloat16)"
    )
    p_repro.add_argument(
        "--runtime", type=str, default=None, help="Filter to runtime (e.g. cuda, cpu)"
    )
    p_repro.add_argument(
        "--attempt", type=int, default=1, help="Run attempt number (default: 1)"
    )

    # -- prepare-repro --
    p_prep = sub.add_parser(
        "prepare-repro", help="Show setup commands to prepare benchmark suites locally"
    )
    p_prep.add_argument(
        "run_id",
        nargs="?",
        type=str,
        default=None,
        help="Optional: run ID or branch to also show benchmark commands",
    )
    p_prep.add_argument(
        "--device",
        nargs="+",
        default=None,
        choices=DEVICE_CHOICES,
        metavar="DEVICE",
        help="Device(s) for benchmark commands",
    )
    p_prep.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Only show setup for this suite (hf, timm, tb)",
    )
    p_prep.add_argument(
        "--mode",
        type=str,
        choices=["training", "inference"],
        default=None,
        help="Filter benchmark commands to mode",
    )
    p_prep.add_argument(
        "--model", type=str, default=None, help="Filter benchmark commands to model"
    )
    p_prep.add_argument(
        "--no-repro",
        dest="no_repro",
        action="store_true",
        help="Only show setup, skip benchmark commands",
    )
    p_prep.add_argument(
        "--attempt", type=int, default=1, help="Run attempt number (default: 1)"
    )

    args = parser.parse_args()

    if args.command == "launch":
        cmd_launch(args)
    elif args.command == "summary":
        cmd_summary(args)
    elif args.command == "repro":
        cmd_repro(args)
    elif args.command == "prepare-repro":
        cmd_prepare_repro(args)


if __name__ == "__main__":
    main()
