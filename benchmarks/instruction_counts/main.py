import argparse
import json
import sys
from typing import Callable, Dict, List, Optional, Tuple

from core.types import Label
from frontend.display import render_ab
from frontend.run import collect, patch_benchmark_utils


def single(source_cmd: Optional[str], output_file: Optional[str]) -> None:
    results = collect(
        (source_cmd,),
        ad_hoc=False,
        no_cpp=False,
        backtesting=False,
    )[0]

    simple_results: List[Tuple[Label, int, int, float]] = []
    label_width = 0

    for label, num_threads, auto_labels, (stats, times) in results:
        label = label + (
            auto_labels.runtime.value,
            auto_labels.autograd.value,
            auto_labels.language.value,
        )
        label_width = max(label_width, len(str(label)))
        simple_results.append((
            label, num_threads,
            int(stats.counts(denoise=True) / stats.number_per_run),
            times.median
        ))

    if output_file:
        # Placeholder format to start experimenting with CI
        with open(output_file, "wt") as f:
            json.dump(simple_results, f)

    else:
        # Mostly for debugging.
        for label, num_threads, count, t in simple_results:
            label_str = str(label).ljust(label_width)
            print(
                f"{label_str} {num_threads:>3}{'':>8}"
                f"{count:>9}{'':>8}{t * 1e6:>6.2f}"
            )


def single_main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_cmd", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args(argv)
    single(args.source_cmd, args.output_file)


def ab_test(
    source_cmd_a: str,
    source_cmd_b: str,
    patch_a: bool = False,
    patch_b: bool = False,
    ad_hoc: bool = False,
    no_cpp: bool = False,
    display_time: bool = False,
    colorize: bool = False,
    backtesting: bool = False,
) -> None:
    patch_benchmark_utils(source_cmd_a, clean_only=not patch_a)
    patch_benchmark_utils(source_cmd_b, clean_only=not patch_b)

    results = collect(
        (source_cmd_a, source_cmd_b),
        ad_hoc=ad_hoc,
        no_cpp=no_cpp,
        backtesting=backtesting,
    )
    render_ab(results[0], results[1], display_time=display_time, colorize=colorize)


def ab_main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_command_a", "--A", type=str, required=True)
    parser.add_argument("--source_command_b", "--B", type=str, required=True)
    parser.add_argument("--patch_a", action="store_true")
    parser.add_argument("--patch_b", action="store_true")
    parser.add_argument("--ad_hoc", action="store_true")
    parser.add_argument("--no_cpp", action="store_true")
    parser.add_argument("--display_time", action="store_true")
    parser.add_argument("--colorize", action="store_true")
    parser.add_argument("--backtesting", action="store_true")

    args = parser.parse_args(argv)
    ab_test(
        source_cmd_a=args.source_command_a,
        source_cmd_b=args.source_command_b,
        patch_a=args.patch_a,
        patch_b=args.patch_b,
        ad_hoc=args.ad_hoc,
        no_cpp=args.no_cpp,
        display_time=args.display_time,
        colorize=args.colorize,
        backtesting=args.backtesting,
    )


if __name__ == "__main__":
    modes: Dict[str, Callable[[List[str]], None]] = {
        "single": single_main,
        "A/B": ab_main,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(modes.keys()),
        required=True
    )
    args, remaining_argv = parser.parse_known_args(sys.argv[1:])
    modes[args.mode](remaining_argv)
