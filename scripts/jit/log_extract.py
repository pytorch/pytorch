from contextlib import contextmanager
from typing import Any, List, Tuple
import argparse
import torch


def extract_ir(filename: str) -> List[str]:
    BEGIN = "<GRAPH_EXPORT>"
    END = "</GRAPH_EXPORT>"
    pfx = None
    current = ""
    graphs = []
    with open(filename, "r") as f:
        for line in f.readlines():
            begin_loc = line.find(BEGIN)
            if begin_loc != -1:
                pfx = line[:begin_loc]
                continue

            end_loc = line.find(END)
            if end_loc != -1:
                graphs.append(current)
                current=""
                pfx = None
                continue

            if not pfx:
                continue

            if len(line) < len(pfx) or line[:len(pfx)] != pfx:
                raise RuntimeError("Expected prefix of '" + pfx + "'")

            current += line[len(pfx):]

    return graphs


def load_graph_and_inputs(ir: str) -> Tuple[Any, List[Any]]:
    graph = torch._C.parse_ir(ir)
    graph.makeMultiOutputIntoTuple()
    inputs = []
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(.5)
        else:
            inputs.append(torch._C._representative_tensor(inp.type()))

    func = torch._C._create_function_from_graph("forward", graph)
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)


def run_test(ir, inputs, *, warmup_runs=20, test_runs=20) -> float:
    graph, _ = load_graph_and_inputs(ir)
    for _ in range(warmup_runs):
        graph(*inputs)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_event.record()
    torch.cuda.synchronize()
    for i in range(test_runs):
        graph(*inputs)
        torch.cuda.synchronize()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / test_runs


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


def test_nvfuser(graphs: List[str], baseline_fn, nvfuser_fn):
    for i, ir in enumerate(graphs):
        _, inputs = load_graph_and_inputs(ir)
        baseline = baseline_fn(ir, inputs)
        nvfuser = nvfuser_fn(ir, inputs)
        improvement = (baseline / nvfuser - 1) * 100
        print(f"  Graph {i}; baseline: {baseline:.2f} ms; nvfuser: {nvfuser:.2f} ms; improvement: {improvement:.2f}%")


def run():
    parser = argparse.ArgumentParser(description="Extracts torchscript IR from log files and, optionally, benchmarks it or outputs the IR")
    parser.add_argument("filename", help="Filename of log file")
    parser.add_argument("--nvfuser", dest="nvfuser", action="store_true", help="benchmark nvfuser against no fusion")
    parser.add_argument("--no-nvfuser", dest="nvfuser", action="store_false", help="DON'T benchmark nvfuser against no fusion")
    parser.set_defaults(nvfuser=False)
    parser.add_argument("--nvfuser-nnc", dest="nvfuser_nnc", action="store_true", help="benchmark nvfuser against nnc")
    parser.add_argument("--no-nvfuser-nnc", dest="nvfuser_nnc", action="store_false", help="DON'T benchmark nvfuser against nnc")
    parser.set_defaults(nvfuser_nnc=False)
    parser.add_argument("--output", dest="output", action="store_true", help="Output graph IR")
    parser.add_argument("--no-output", dest="output", action="store_false", help="DON'T output graph IR")
    parser.set_defaults(output=False)

    args = parser.parse_args()
    graphs = extract_ir(args.filename)

    if args.nvfuser:
        print("NVFuser vs no fusion:")
        test_nvfuser(graphs, run_baseline_no_fusion, run_nvfuser)

    if args.nvfuser_nnc:
        print("NVFuser vs NNC:")
        test_nvfuser(graphs, run_nnc, run_nvfuser)
        #test_nvfuser(graphs, run_baseline_no_fusion, run_nnc)

    if args.output:
        quoted = []
        for ir in graphs:
            quoted.append("\"\"\"" + ir + "\"\"\"")
        print("[" + ", ".join(quoted) + "]")

if __name__ == "__main__":
    run()
