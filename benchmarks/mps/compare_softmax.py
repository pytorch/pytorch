"""Compare complete latency runs without dropping shapes or choosing favorable trials."""

import argparse
import json
import math
import statistics
from pathlib import Path

from softmax_common import CASES


def load_runs(paths):
    expected = {
        (op, case) for op in ("softmax", "log_softmax") for case in range(len(CASES))
    }
    samples = {key: [] for key in expected}
    signature = None
    for path in paths:
        data = json.loads(path.read_text())
        current = (data["options"]["dtype"], data["kernel_sha256"])
        if signature is not None and signature != current:
            raise ValueError("Repeated runs must use the same dtype and kernel source")
        signature = current
        rows = data["measurements"]
        keys = {(row["op"], row["case"]) for row in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(
                f"{path} must contain both operators and every model shape exactly once"
            )
        for row in rows:
            shape, dim = CASES[row["case"]]
            if tuple(row["shape"]) != shape or row["dim"] != dim:
                raise ValueError(
                    f"{path}: case {row['case']} does not match the model shape"
                )
            trials = row["trials"][1:]
            if not trials:
                raise ValueError(
                    f"{path}: need at least one trial after the discarded first trial"
                )
            values = [sample["event_us"] for sample in trials]
            if any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{path}: expected positive, finite timings")
            samples[row["op"], row["case"]].extend(values)
    return samples, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--limit-percent", type=float, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline, baseline_signature = load_runs(args.baseline)
    candidate, candidate_signature = load_runs(args.candidate)
    if baseline_signature[0] != candidate_signature[0]:
        raise ValueError("Baseline and candidate must use the same dtype")
    results = []
    for op, case in sorted(baseline):
        old = statistics.median(baseline[op, case])
        new = statistics.median(candidate[op, case])
        change = 100 * (new / old - 1)
        passed = change <= args.limit_percent
        results.append(
            {
                "op": op,
                "case": case,
                "shape": CASES[case][0],
                "dim": CASES[case][1],
                "baseline_us": old,
                "candidate_us": new,
                "change_percent": change,
                "passed": passed,
            }
        )
        print(
            f"{op:11s} {case:2d}: {old:11.3f} -> {new:11.3f} us ({change:+6.2f}%) {'PASS' if passed else 'FAIL'}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "baseline": list(map(str, args.baseline)),
                    "candidate": list(map(str, args.candidate)),
                    "limit_percent": args.limit_percent,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    failed = sum(not row["passed"] for row in results)
    print(
        f"{len(results) - failed}/{len(results)} shapes within the {args.limit_percent:g}% limit."
    )
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
