"""Run from the repo root with ./venv/bin/python benchmarks/mps/log_softmax.py."""

import argparse
import gc
import math
import statistics
import time
from pathlib import Path

from softmax_common import add_arguments, operators, save_results, selected_cases

import torch


def summarize(values):
    percentiles = (
        statistics.quantiles(values, n=100, method="inclusive")
        if len(values) > 1
        else values * 99
    )
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        **{f"p{p}_us": percentiles[p - 1] for p in (25, 50, 75, 90, 99)},
    }


def plot_results(args, records):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    args.plot.mkdir(parents=True, exist_ok=True)
    for record in records:
        trials = record["trials"]
        count = len(trials)
        fig = Figure(figsize=(12, 8), layout="constrained")
        FigureCanvasAgg(fig)
        axes = fig.subplot_mosaic(
            [["trace", "cdf"], ["stats", "stats"], ["notes", "notes"]],
            height_ratios=[3.5, 1.1, 1.2],
        )
        fig.suptitle(
            f"MPS {record['op']} | case {record['case']} | "
            f"shape={record['shape']}, dim={record['dim']}\n"
            f"{args.dtype} | {count} trials | {record['steps']} calls averaged per trial",
            fontsize=14,
        )
        stats = ("min", "p25", "p50", "p75", "p90", "p99", "max")
        table_rows = []
        labels = []
        retained = [
            trial[key] for trial in trials[1:] for key in ("event_us", "wall_us")
        ]
        low, high = min(retained), max(retained)
        padding = 0.05 * (high - low or high or 1)
        limits = (max(0, low - padding), high + padding)
        for key, label, color, marker, linestyle in (
            ("event_us", "MPS event", "#0072B2", "o", "-"),
            ("wall_us", "Wall", "#D55E00", "s", "--"),
        ):
            values = [trial[key] for trial in trials]
            axes["trace"].plot(
                range(2, count + 1),
                values[1:],
                label=label,
                color=color,
                marker=marker,
                markersize=3,
                linewidth=1,
                linestyle=linestyle,
            )
            axes["trace"].scatter([1], values[:1], color=color, marker="x", s=60)
            ordered = sorted(values[1:])
            axes["cdf"].step(
                [limits[0], *ordered, limits[1]],
                [0, *(i / (count - 1) for i in range(1, count)), 1],
                where="post",
                label=label,
                color=color,
                linewidth=1.5,
                linestyle=linestyle,
            )
            summary = summarize(values[1:])
            table_rows.append([f"{summary[f'{stat}_us']:.3f}" for stat in stats])
            labels.append(label)

        axes["trace"].set(
            title="Trial averages in measurement order",
            xlabel="Trial number (trial 1 marked x)",
            ylabel="Average latency (us/call)",
            xlim=(0.5, count + 0.5),
        )
        axes["trace"].xaxis.set_major_locator(MaxNLocator(integer=True))
        axes["trace"].set_ylim(bottom=0)
        axes["cdf"].set(
            title=f"Empirical CDF of retained trial averages (n={count - 1})",
            xlabel="Average latency (us/call)",
            ylabel="Retained trials at or below latency",
            xlim=limits,
            ylim=(0, 1.02),
            yticks=[0, 0.25, 0.5, 0.75, 1],
        )
        axes["cdf"].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        for name in ("trace", "cdf"):
            axes[name].grid(alpha=0.2)
            axes[name].set_axisbelow(True)
            axes[name].legend(
                loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False
            )
            axes[name].spines[["top", "right"]].set_visible(False)

        axes["stats"].set_axis_off()
        axes["stats"].set_title("Retained trial averages (us/call; trial 1 excluded)")
        table = axes["stats"].table(
            cellText=table_rows,
            rowLabels=labels,
            colLabels=stats,
            cellLoc="center",
            bbox=[0.12, 0, 0.88, 0.85],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#DDDDDD")
            if row == 0 or col == -1:
                cell.set_facecolor("#F2F2F2")

        axes["notes"].set_axis_off()
        axes["notes"].text(
            0,
            1,
            "Each point is a batch duration divided by its call count; individual call latencies are not measured.\n"
            f"Trial 1 is shown only in the time series. CDF and statistics use trials 2-{count}; percentiles use linear interpolation.\n"
            "MPS event intervals include queue gaps. Wall time includes host submission and synchronization.\n"
            f"Inputs/outputs preallocated; {args.warmup} warmups and the pilot are omitted. Build: {torch.version.git_version}",
            va="top",
            fontsize=9,
            linespacing=1.6,
        )
        path = args.plot / f"case_{record['case']:02d}_{record['op']}_{args.dtype}.png"
        fig.savefig(path, dpi=160)
    print(f"Plots saved to {args.plot.resolve()}", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(
        description="Measure MPS softmax and log_softmax latency."
    )
    add_arguments(parser)
    parser.add_argument(
        "--plot",
        type=Path,
        nargs="?",
        const=Path("plots"),
        metavar="DIR",
        help="Save per-case/operator timing plots to DIR (default: plots); no GUI.",
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Calls per trial; 0 chooses 1-300 calls from a pilot.",
    )
    parser.add_argument(
        "--target-ms",
        type=float,
        default=200,
        help="Target duration for an adaptive trial.",
    )
    args = parser.parse_args()
    if args.trials < 2 or args.warmup < 1 or args.steps < 0 or args.target_ms <= 0:
        parser.error("Need trials >= 2, warmup >= 1, steps >= 0, and target-ms > 0")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    print(f"Build: {torch.version.git_version}; dtype: {args.dtype}", flush=True)
    print(
        "MPS event intervals include queue gaps; wall time includes batch submission and synchronization."
    )
    print("Inputs and outputs are preallocated. Trial 1 is excluded from summaries.")
    records = []
    for case, shape, dim in selected_cases(args):
        torch.manual_seed(case)
        x = torch.randn(shape, device="mps", dtype=getattr(torch, args.dtype))
        out = torch.empty_like(x)
        steps = {}
        samples = {name: [] for name in operators(args)}
        for name in samples:
            op = getattr(torch, name)
            for _ in range(args.warmup):
                op(x, dim=dim, out=out)
            start = torch.mps.Event(enable_timing=True)
            end = torch.mps.Event(enable_timing=True)
            start.record()
            op(x, dim=dim, out=out)
            end.record()
            torch.mps.synchronize()
            pilot_ms = max(start.elapsed_time(end), 0.001)
            steps[name] = args.steps or max(
                1, min(300, math.ceil(args.target_ms / pilot_ms))
            )

        for trial in range(args.trials):
            names = list(samples)
            if trial % 2:
                names.reverse()
            for name in names:
                op = getattr(torch, name)
                start = torch.mps.Event(enable_timing=True)
                end = torch.mps.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for _ in range(steps[name]):
                    op(x, dim=dim, out=out)
                end.record()
                torch.mps.synchronize()
                wall_us = (time.perf_counter() - wall_start) * 1e6 / steps[name]
                event_us = start.elapsed_time(end) * 1000 / steps[name]
                samples[name].append({"event_us": event_us, "wall_us": wall_us})

        for name, trials in samples.items():
            event_us = [sample["event_us"] for sample in trials[1:]]
            record = {
                "case": case,
                "op": name,
                "shape": shape,
                "dim": dim,
                "steps": steps[name],
                "trials": trials,
                **summarize(event_us),
            }
            records.append(record)
            summary = " ".join(
                f"{stat}={record[f'{stat}_us']:.3f}"
                for stat in ("min", "p25", "p50", "p75", "p90", "p99", "max")
            )
            print(
                f"{case:2d} {name:11s} {str(shape):24s} dim={dim} {summary} us",
                flush=True,
            )
        del x, out
        gc.collect()
        torch.mps.empty_cache()
        save_results(args, records)
    if args.plot is not None:
        plot_results(args, records)


if __name__ == "__main__":
    main()
