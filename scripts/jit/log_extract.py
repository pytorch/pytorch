import argparse
import functools
import traceback
from typing import Callable, List, Optional, Tuple

from torch.utils.jit.log_extract import (
    extract_ir,
    load_graph_and_inputs,
    run_baseline_no_fusion,
    run_nnc,
    run_nvfuser,
)

"""
Usage:
1. Run your script and pipe into a log file
  PYTORCH_JIT_LOG_LEVEL=">>graph_fuser" python3 my_test.py &> log.txt
2. Run log_extract:
  log_extract.py log.txt --nvfuser --nnc-dynamic --nnc-static

You can also extract the list of extracted IR:
  log_extract.py log.txt --output

Passing in --graphs 0 2 will only run graphs 0 and 2
"""


def test_runners(
    graphs: List[str],
    runners: List[Tuple[str, Callable]],
    graph_set: Optional[List[int]],
):
    for i, ir in enumerate(graphs):
        _, inputs = load_graph_and_inputs(ir)
        if graph_set and i not in graph_set:
            continue

        print(f"Running Graph {i}")
        prev_result = None
        prev_runner_name = None
        for runner in runners:
            runner_name, runner_fn = runner
            try:
                result = runner_fn(ir, inputs)
                if prev_result:
                    improvement = (prev_result / result - 1) * 100
                    print(
                        f"{runner_name} : {result:.6f} ms improvement over {prev_runner_name}: improvement: {improvement:.2f}%"
                    )
                else:
                    print(f"{runner_name} : {result:.6f} ms")
                prev_result = result
                prev_runner_name = runner_name
            except RuntimeError:
                print(f"  Graph {i} failed for {runner_name} :", traceback.format_exc())


def run():
    parser = argparse.ArgumentParser(
        description="Extracts torchscript IR from log files and, optionally, benchmarks it or outputs the IR"
    )
    parser.add_argument("filename", help="Filename of log file")
    parser.add_argument(
        "--nvfuser", dest="nvfuser", action="store_true", help="benchmark nvfuser"
    )
    parser.add_argument(
        "--no-nvfuser",
        dest="nvfuser",
        action="store_false",
        help="DON'T benchmark nvfuser",
    )
    parser.set_defaults(nvfuser=False)
    parser.add_argument(
        "--nnc-static",
        dest="nnc_static",
        action="store_true",
        help="benchmark nnc static",
    )
    parser.add_argument(
        "--no-nnc-static",
        dest="nnc_static",
        action="store_false",
        help="DON'T benchmark nnc static",
    )
    parser.set_defaults(nnc_static=False)

    parser.add_argument(
        "--nnc-dynamic",
        dest="nnc_dynamic",
        action="store_true",
        help="nnc with dynamic shapes",
    )
    parser.add_argument(
        "--no-nnc-dynamic",
        dest="nnc_dynamic",
        action="store_false",
        help="DONT't benchmark nnc with dynamic shapes",
    )
    parser.set_defaults(nnc_dynamic=False)

    parser.add_argument(
        "--baseline", dest="baseline", action="store_true", help="benchmark baseline"
    )
    parser.add_argument(
        "--no-baseline",
        dest="baseline",
        action="store_false",
        help="DON'T benchmark baseline",
    )
    parser.set_defaults(baseline=False)

    parser.add_argument(
        "--output", dest="output", action="store_true", help="Output graph IR"
    )
    parser.add_argument(
        "--no-output", dest="output", action="store_false", help="DON'T output graph IR"
    )
    parser.set_defaults(output=False)

    parser.add_argument(
        "--graphs", nargs="+", type=int, help="Run only specified graph indices"
    )

    args = parser.parse_args()
    graphs = extract_ir(args.filename)

    graph_set = args.graphs
    graph_set = graph_set if graph_set else None

    options = []
    if args.baseline:
        options.append(("Baseline no fusion", run_baseline_no_fusion))
    if args.nnc_dynamic:
        options.append(("NNC Dynamic", functools.partial(run_nnc, dynamic=True)))
    if args.nnc_static:
        options.append(("NNC Static", functools.partial(run_nnc, dynamic=False)))
    if args.nvfuser:
        options.append(("NVFuser", run_nvfuser))

    test_runners(graphs, options, graph_set)

    if args.output:
        quoted = []
        for i, ir in enumerate(graphs):
            if graph_set and i not in graph_set:
                continue
            quoted.append('"""' + ir + '"""')
        print("[" + ", ".join(quoted) + "]")


if __name__ == "__main__":
    run()
