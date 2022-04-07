from contextlib import contextmanager
from torch.testing import make_tensor
from typing import Any, List, Tuple, Callable
import argparse
import random
import torch
import traceback
import time

'''
Usage:
1. Run your script and pipe into a log file
  PYTORCH_JIT_LOG_LEVEL=">>graph_fuser" python3 my_test.py &> log.txt
2. Run log_extract:
  log_extract.py log.txt --nvfuser

You can also extract the list of extracted IR:
  log_extract.py log.txt --output
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
    if inp_type.requires_grad() is not False:
        raise NotImplementedError("Tensors with requires_grad are not implemented")
    return make_tensor(
        inp_type.sizes(),
        dtype=inp_type.dtype(),
        device=inp_type.device())


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
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    torch.cuda.synchronize()
    for i in range(test_runs):
        fn(*inputs)
        torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / test_runs

def time_cpu(fn, inputs, test_runs):
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs

def run_test(ir, inputs, *, warmup_runs=10, test_runs=20) -> float:
    torch.jit._state._python_cu.drop_all_functions()
    graph, _ = load_graph_and_inputs(ir)
    for _ in range(warmup_runs):
        graph(*inputs)

    is_cpu = None
    for input in inputs:
        if isinstance(input, torch.Tensor):
            is_cpu = input.device.type == "cpu"
            break
    assert is_cpu != None

    out = time_cpu(graph, inputs, test_runs) if is_cpu else time_cuda(graph, inputs, test_runs)
    return out

@contextmanager
def no_fuser(*args, **kwargs):
    old_cpu_fuse = torch._C._jit_can_fuse_on_cpu()
    old_gpu_fuse = torch._C._jit_can_fuse_on_gpu()
    old_texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
    old_nvfuser_state = torch._C._jit_nvfuser_enabled()

    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(False)

    try:
        yield
    finally:
        torch._C._jit_override_can_fuse_on_cpu(old_cpu_fuse)
        torch._C._jit_override_can_fuse_on_gpu(old_gpu_fuse)
        torch._C._jit_set_texpr_fuser_enabled(old_texpr_fuser_state)
        torch._C._jit_set_nvfuser_enabled(old_nvfuser_state)


def run_baseline_no_fusion(ir, inputs) -> float:
    with no_fuser():
        return run_test(ir, inputs)


def run_nnc(ir, inputs) -> float:
    with torch.jit.fuser("fuser1"):
        return run_test(ir, inputs)


def run_nvfuser(ir, inputs) -> float:
    with torch.jit.fuser("fuser2"):
        return run_test(ir, inputs)


def test_runners(graphs: List[str], runners: List[Tuple[str, Callable]]):
    for i, ir in enumerate(graphs):
        _, inputs = load_graph_and_inputs(ir)
        print(f"Running Graph {ir}")
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
    parser.add_argument("--nnc", dest="nnc", action="store_true", help="benchmark nnc")
    parser.add_argument("--no-nnc", dest="nnc", action="store_false", help="DON'T benchmark nnc")
    parser.set_defaults(nnc=False)

    parser.add_argument("--baseline", dest="baseline", action="store_true", help="benchmark baseline")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false", help="DON'T benchmark baseline")
    parser.set_defaults(baseline=False)

    parser.add_argument("--output", dest="output", action="store_true", help="Output graph IR")
    parser.add_argument("--no-output", dest="output", action="store_false", help="DON'T output graph IR")
    parser.set_defaults(output=False)

    args = parser.parse_args()
    graphs = extract_ir(args.filename)

    options = []
    if args.baseline:
        options.append(("Baseline no fusion", run_baseline_no_fusion))
    if args.nnc:
        options.append(("NNC", run_nnc))
    if args.nvfuser:
        options.append(("NVFuser", run_nvfuser))

    test_runners(graphs, options)

    if args.output:
        quoted = []
        for ir in graphs:
            quoted.append("\"\"\"" + ir + "\"\"\"")
        print("[" + ", ".join(quoted) + "]")

if __name__ == "__main__":
    run()
