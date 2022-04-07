from contextlib import contextmanager
from typing import Any, List, Tuple, Callable, Optional
import argparse
import random
import torch
import traceback
import time
import functools
from torch.utils.benchmark import Timer

'''
Usage:
1. Run your script and pipe into a log file
  PYTORCH_JIT_LOG_LEVEL=">>graph_fuser" python3 my_test.py &> log.txt
2. Run log_extract:
  log_extract.py log.txt --nvfuser --nnc-dynamic --nnc-static

You can also extract the list of extracted IR:
  log_extract.py log.txt --output

Passing in --graphs 0 2 will only run graphs 0 and 2
'''

def extract_ir(filename: str) -> List[str]:
    BEGIN = "<GRAPH_EXPORT>"
    END = "</GRAPH_EXPORT>"
    pfx = None
    current = ""
    graphs = []
    with open(filename, "r") as f:
        split_strs = f.read().split(BEGIN)
        for i, split_str in enumerate(split_strs):
            if i == 0:
                continue
            end_loc = split_str.find(END)
            if end_loc == -1:
                continue
            s = split_str[:end_loc]
            pfx = split_strs[i - 1].splitlines()[-1]
            lines = [x[len(pfx):] for x in s.splitlines(keepends=True)]
            graphs.append(''.join(lines))

    return graphs


def make_tensor_from_type(inp_type: torch._C.TensorType):
    size = inp_type.sizes()
    stride = inp_type.strides()
    device = inp_type.device()
    dtype = inp_type.dtype()
    return torch.empty_strided(size=size, stride=stride, device=device, dtype=dtype)

def load_graph_and_inputs(ir: str) -> Tuple[Any, List[Any]]:
    graph = torch._C.parse_ir(ir)
    graph.makeMultiOutputIntoTuple()
    inputs = []
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(random.uniform(.1, 100))
        elif isinstance(inp.type(), torch._C.IntType):
            inputs.append(random.randint(1, 100))
        elif isinstance(inp.type(), torch._C.TensorType):
            inputs.append(make_tensor_from_type(inp.type()))
        else:
            raise NotImplementedError(f"A default value is not implemented for type {inp.type()}")

    func = torch._C._create_function_from_graph("forward", graph)
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)

def time_cuda(fn, inputs, test_runs):
    t = Timer(stmt="fn(*inputs)", globals={"fn": fn, "inputs" : inputs})
    times = t.blocked_autorange()
    return times.median * 1000 # time in ms

def time_cpu(fn, inputs, test_runs):
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs


# TODO add support for timing on CPU
def run_test(ir, inputs, *, warmup_runs=10, test_runs=20) -> float:
    graph, _ = load_graph_and_inputs(ir)
    for _ in range(warmup_runs):
        graph(*inputs)

    is_cpu = None
    for input in inputs:
        if isinstance(input, torch.Tensor):
            is_cpu = input.device.type == "cpu"
            break
    assert is_cpu is not None

    out = time_cpu(graph, inputs, test_runs) if is_cpu else time_cuda(graph, inputs, test_runs)
    return out

@contextmanager
def no_fuser(*args, **kwargs):
    old_optimize = torch._C._get_graph_executor_optimize(False)
    try:
        yield
    finally:
        torch._C._get_graph_executor_optimize(old_optimize)

def run_baseline_no_fusion(ir, inputs) -> float:
    with no_fuser():
        return run_test(ir, inputs)


def run_nnc(ir, inputs, dynamic) -> float:
    try:
        strat = [("DYNAMIC", 10)] if dynamic else [("STATIC", 10)]
        old_strat = torch.jit.set_fusion_strategy(strat)
        with torch.jit.fuser("fuser1"):
            return run_test(ir, inputs)
    finally:
        torch.jit.set_fusion_strategy(old_strat)

def run_nvfuser(ir, inputs) -> float:
    with torch.jit.fuser("fuser2"):
        return run_test(ir, inputs)


def test_runners(graphs: List[str], runners: List[Tuple[str, Callable]], graph_set: Optional[List[int]]):
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
                    print(f"{runner_name} : {result:.6f} ms improvement over {prev_runner_name}: improvement: {improvement:.2f}%")
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
    parser.add_argument("--nvfuser", dest="nvfuser", action="store_true", help="benchmark nvfuser")
    parser.add_argument("--no-nvfuser", dest="nvfuser", action="store_false", help="DON'T benchmark nvfuser")
    parser.set_defaults(nvfuser=False)
    parser.add_argument("--nnc-static", dest="nnc_static", action="store_true", help="benchmark nnc static")
    parser.add_argument("--no-nnc-static", dest="nnc_static", action="store_false", help="DON'T benchmark nnc static")
    parser.set_defaults(nnc_static=False)

    parser.add_argument("--nnc-dynamic", dest="nnc_dynamic", action="store_true", help="nnc with dynamic shapes")
    parser.add_argument("--no-nnc-dynamic", dest="nnc_dynamic", action="store_false", help="DONT't benchmark nnc with dynamic shapes")
    parser.set_defaults(nnc_dynamic=False)


    parser.add_argument("--baseline", dest="baseline", action="store_true", help="benchmark baseline")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false", help="DON'T benchmark baseline")
    parser.set_defaults(baseline=False)

    parser.add_argument("--output", dest="output", action="store_true", help="Output graph IR")
    parser.add_argument("--no-output", dest="output", action="store_false", help="DON'T output graph IR")
    parser.set_defaults(output=False)

    parser.add_argument('--graphs', nargs="+", type=int, help="Run only specified graph indices")


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
            quoted.append("\"\"\"" + ir + "\"\"\"")
        print("[" + ", ".join(quoted) + "]")

if __name__ == "__main__":
    run()
