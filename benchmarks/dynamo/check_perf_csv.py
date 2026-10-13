import argparse
import math
import sys
import textwrap

import pandas as pd


def _optional_measurement(row, metric):
    value = row.get(metric)
    if pd.api.types.is_bool(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def check_perf_csv(filename, threshold, threshold_scale):
    """
    Basic performance checking.
    """
    try:
        df = pd.read_csv(filename, dtype={"name": str})
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: {filename} contains no benchmark results")
        sys.exit(1)

    if df.empty:
        print(f"Error: {filename} contains no benchmark results")
        sys.exit(1)

    missing_columns = {"name", "speedup"} - set(df.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        print(f"Error: {filename} is missing required column(s): {columns}")
        sys.exit(1)

    speedups = []
    invalid = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=2):
        model_name = row["name"]
        value = row["speedup"]
        if pd.isna(model_name) or not model_name.strip():
            invalid.append(f"row {row_number} has a missing model name")
        try:
            speedup = math.nan if pd.api.types.is_bool(value) else float(value)
        except (TypeError, ValueError):
            speedup = math.nan
        if not math.isfinite(speedup):
            name = model_name if pd.notna(model_name) else f"row {row_number}"
            invalid.append(f"{name} (speedup={value!r})")
        speedups.append(speedup)

    if invalid:
        print(f"Error: {filename} has invalid required result field(s):")
        for measurement in invalid:
            print(f"  - {measurement}")
        sys.exit(1)

    effective_threshold = threshold * threshold_scale
    print(f"Checking {filename} (speedup threshold >= {effective_threshold:.2f}x)\n")

    failed = []
    for (_, row), speedup in zip(df.iterrows(), speedups):
        model_name = row["name"]
        abs_latency = _optional_measurement(row, "abs_latency")
        compilation_latency = _optional_measurement(row, "compilation_latency")
        compression_ratio = _optional_measurement(row, "compression_ratio")
        eager_peak_mem = _optional_measurement(row, "eager_peak_mem")
        dynamo_peak_mem = _optional_measurement(row, "dynamo_peak_mem")

        perf_summary = f"{model_name:34} speedup={speedup:.3f}x"
        if abs_latency is not None:
            perf_summary += f", latency={abs_latency:.1f} ms/iter"
        if compilation_latency is not None:
            perf_summary += f", compile={compilation_latency:.3f}s"
        if compression_ratio is not None and compression_ratio > 0:
            perf_summary += f", mem_ratio={1 / compression_ratio:.2f}x"
            if eager_peak_mem is not None and dynamo_peak_mem is not None:
                perf_summary += (
                    f" (eager={eager_peak_mem:.1f} GB, dynamo={dynamo_peak_mem:.1f} GB)"
                )

        if speedup < effective_threshold:
            failed.append((model_name, speedup))

        print(perf_summary)

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} model(s) performance regressed
                    {" ".join([name for name, _ in failed])}
                """
            )
        )
        for name, sp in sorted(failed, key=lambda x: x[1]):
            pct_from_target = (sp / effective_threshold - 1.0) * 100.0
            print(
                f"  - {name}: {sp:.3f}x (< {effective_threshold:.2f}x; {pct_from_target:.1f}% from target)"
            )
        sys.exit(1)
    else:
        print(
            f"\nAll {len(df)} model(s) passed threshold check (>= {effective_threshold:.2f}x)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="csv file name")
    parser.add_argument(
        "--threshold", "-t", type=float, help="threshold speedup value to check against"
    )
    parser.add_argument(
        "--threshold-scale",
        "-s",
        type=float,
        default=1.0,
        help="multiply threshold by this value to relax the check",
    )
    args = parser.parse_args()
    check_perf_csv(args.file, args.threshold, args.threshold_scale)
