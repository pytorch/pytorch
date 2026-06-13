import argparse
import sys
import textwrap

import pandas as pd


SUPPORTED_METRICS = ("abs_latency", "speedup")


def _format_metric_value(metric, value):
    if metric == "speedup":
        return f"{value:.3f}x"
    if metric == "abs_latency":
        return f"{value:.1f} ms/iter"
    return f"{value:.3f}"


def _target_bounds(threshold, threshold_scale):
    if threshold_scale <= 0:
        raise ValueError("threshold_scale must be positive")

    lower_bound = threshold * threshold_scale
    upper_bound = threshold / threshold_scale
    return min(lower_bound, upper_bound), max(lower_bound, upper_bound)


def _read_optional_float(row, column):
    if column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def _read_required_float(row, column):
    if column not in row or pd.isna(row[column]):
        raise ValueError(f"Missing required column or value: {column}")
    return float(row[column])


def _print_failures(metric, failures, bound, comparator, label):
    is_improvement = (metric == "speedup" and comparator == ">") or (
        metric == "abs_latency" and comparator == "<"
    )
    status = "improved" if is_improvement else "regressed"
    model_names = " ".join([name for name, _ in failures])
    print(
        textwrap.dedent(
            f"""
            {label} {len(failures)} model(s) performance {status}
                {model_names}
            """
        )
    )
    for name, value in sorted(failures, key=lambda x: x[1]):
        print(
            f"  - {name}: {metric}={_format_metric_value(metric, value)} "
            f"({comparator} {_format_metric_value(metric, bound)})"
        )


def check_perf_csv(
    filename,
    threshold,
    threshold_scale,
    metric="speedup",
    fail_on_improvement=False,
):
    """
    Basic performance checking.
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)

    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")

    lower_bound, upper_bound = _target_bounds(threshold, threshold_scale)
    if metric == "speedup":
        regression_bound = (
            lower_bound if fail_on_improvement else threshold * threshold_scale
        )
    else:
        regression_bound = upper_bound

    if fail_on_improvement:
        print(
            f"Checking {filename} ({metric} target "
            f"{_format_metric_value(metric, threshold)}, allowed range "
            f"{_format_metric_value(metric, lower_bound)} - "
            f"{_format_metric_value(metric, upper_bound)})\n"
        )
    elif metric == "speedup":
        print(
            f"Checking {filename} (speedup threshold >= "
            f"{_format_metric_value(metric, regression_bound)})\n"
        )
    else:
        print(
            f"Checking {filename} ({metric} threshold <= "
            f"{_format_metric_value(metric, regression_bound)})\n"
        )

    failed = []
    improved = []
    for _, row in df.iterrows():
        model_name = row["name"]
        metric_value = _read_required_float(row, metric)
        speedup = _read_optional_float(row, "speedup")
        abs_latency = _read_optional_float(row, "abs_latency")
        compilation_latency = _read_optional_float(row, "compilation_latency")
        compression_ratio = _read_optional_float(row, "compression_ratio")
        eager_peak_mem = _read_optional_float(row, "eager_peak_mem")
        dynamo_peak_mem = _read_optional_float(row, "dynamo_peak_mem")

        perf_details = []
        if speedup is not None:
            perf_details.append(f"speedup={speedup:.3f}x")
        if abs_latency is not None:
            perf_details.append(f"latency={abs_latency:.1f} ms/iter")
        if compilation_latency is not None:
            perf_details.append(f"compile={compilation_latency:.3f}s")
        if compression_ratio is not None and compression_ratio != 0:
            memory_summary = f"mem_ratio={1 / compression_ratio:.2f}x"
            if eager_peak_mem is not None and dynamo_peak_mem is not None:
                memory_summary += (
                    f" (eager={eager_peak_mem:.1f} GB, dynamo={dynamo_peak_mem:.1f} GB)"
                )
            perf_details.append(memory_summary)

        perf_summary = f"{model_name:34}"
        if perf_details:
            perf_summary += f" {', '.join(perf_details)}"

        if metric == "speedup":
            if metric_value < regression_bound:
                failed.append((model_name, metric_value))
            elif fail_on_improvement and metric_value > upper_bound:
                improved.append((model_name, metric_value))
        elif metric == "abs_latency":
            if metric_value > regression_bound:
                failed.append((model_name, metric_value))
            elif fail_on_improvement and metric_value < lower_bound:
                improved.append((model_name, metric_value))

        print(perf_summary)

    if failed or improved:
        if failed:
            _print_failures(
                metric,
                failed,
                regression_bound,
                "<" if metric == "speedup" else ">",
                "Error:",
            )
        if improved:
            _print_failures(
                metric,
                improved,
                upper_bound if metric == "speedup" else lower_bound,
                ">" if metric == "speedup" else "<",
                "Improvement:",
            )
        print("\nIf this change is expected, update the performance baseline target.")
        sys.exit(1)

    if fail_on_improvement:
        print(
            f"\nAll {len(df)} model(s) passed threshold check "
            f"({_format_metric_value(metric, lower_bound)} <= {metric} <= "
            f"{_format_metric_value(metric, upper_bound)})"
        )
    else:
        comparator = ">=" if metric == "speedup" else "<="
        print(
            f"\nAll {len(df)} model(s) passed threshold check "
            f"({metric} {comparator} {_format_metric_value(metric, regression_bound)})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="csv file name")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        help="target metric value to check against",
    )
    parser.add_argument(
        "--threshold-scale",
        "-s",
        type=float,
        default=1.0,
        help="multiply threshold by this value to relax the check",
    )
    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="speedup",
        help="performance metric to validate",
    )
    parser.add_argument(
        "--fail-on-improvement",
        action="store_true",
        help="also fail when the metric improves beyond the tolerated target range",
    )
    args = parser.parse_args()
    check_perf_csv(
        args.file,
        args.threshold,
        args.threshold_scale,
        metric=args.metric,
        fail_on_improvement=args.fail_on_improvement,
    )
