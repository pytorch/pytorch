"""Compare operator microbenchmark results between a branch and a baseline.

Reads two directories containing benchmark JSON files (dashboard format) and
outputs a Markdown-formatted comparison table to stdout.

Usage:
    python compare_with_baseline.py --branch-dir /path/to/branch --baseline-dir /path/to/baseline
"""

import argparse
import json
import math
import os
import sys


def load_benchmark_records(directory):
    """Load all benchmark JSON records from a directory."""
    records = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(directory, fname)) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"warning: skipping invalid JSON: {fname}", file=sys.stderr)
                continue
            if isinstance(data, list):
                records.extend(data)
    return records


def extract_latency_map(records):
    """Extract a mapping of (test_name, use_compile) -> latency_us from records."""
    result = {}
    for rec in records:
        metric = rec.get("metric", {})
        if metric.get("name") != "latency":
            continue
        benchmark = rec.get("benchmark", {})
        model = rec.get("model", {})
        test_name = model.get("name", "unknown")
        use_compile = benchmark.get("extra_info", {}).get("use_compile", False)
        values = metric.get("benchmark_values", [])
        if values:
            result[(test_name, bool(use_compile))] = values[0]
    return result


def gmean(values):
    if not values:
        return 0.0
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    log_sum = sum(math.log(v) for v in positive)
    return math.exp(log_sum / len(positive))


def format_comparison(branch_map, baseline_map):
    """Format a Markdown comparison table. Returns the Markdown string."""
    all_keys = sorted(set(branch_map.keys()) | set(baseline_map.keys()))
    if not all_keys:
        return "No benchmark results found to compare.\n"

    # Group by compile mode
    eager_rows = []
    compile_rows = []
    for key in all_keys:
        test_name, use_compile = key
        branch_val = branch_map.get(key)
        baseline_val = baseline_map.get(key)
        if branch_val is None or baseline_val is None:
            continue
        if baseline_val > 0:
            delta_pct = (branch_val - baseline_val) / baseline_val * 100
        else:
            delta_pct = 0.0

        if delta_pct < -2:
            status = ":rocket:"
        elif delta_pct > 2:
            status = ":warning:"
        else:
            status = ":white_check_mark:"

        row = (test_name, branch_val, baseline_val, delta_pct, status)
        if use_compile:
            compile_rows.append(row)
        else:
            eager_rows.append(row)

    lines = []

    def render_table(rows, mode_label):
        if not rows:
            return
        lines.append(f"### {mode_label}")
        lines.append("")
        lines.append("| Case Name | Branch (us) | Baseline (us) | Delta | |")
        lines.append("|-----------|------------|--------------|-------|---|")
        speedups = []
        for name, bval, mval, delta, status in rows:
            lines.append(
                f"| {name} | {bval:.2f} | {mval:.2f} | {delta:+.1f}% | {status} |"
            )
            if mval > 0:
                speedups.append(mval / bval)

        faster = sum(1 for _, _, _, d, _ in rows if d < -2)
        slower = sum(1 for _, _, _, d, _ in rows if d > 2)
        neutral = len(rows) - faster - slower
        geo = gmean(speedups) if speedups else 1.0
        lines.append("")
        lines.append(
            f"**Summary ({mode_label}):** {faster} faster, {slower} slower, "
            f"{neutral} neutral out of {len(rows)}. "
            f"Geometric mean speedup: {geo:.3f}x"
        )
        lines.append("")

    render_table(eager_rows, "Eager Mode")
    render_table(compile_rows, "Compile Mode")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch-dir", required=True, help="Directory with branch JSON results"
    )
    parser.add_argument(
        "--baseline-dir", required=True, help="Directory with baseline JSON results"
    )
    args = parser.parse_args()

    branch_records = load_benchmark_records(args.branch_dir)
    baseline_records = load_benchmark_records(args.baseline_dir)

    if not branch_records:
        print("Error: no branch benchmark records found.", file=sys.stderr)
        sys.exit(1)
    if not baseline_records:
        print("Error: no baseline benchmark records found.", file=sys.stderr)
        sys.exit(1)

    branch_map = extract_latency_map(branch_records)
    baseline_map = extract_latency_map(baseline_records)

    md = format_comparison(branch_map, baseline_map)
    print(md)


if __name__ == "__main__":
    main()
