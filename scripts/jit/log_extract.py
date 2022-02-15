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


def run_test(graph, inputs, *, warmup_runs=20, test_runs=20) -> float:
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


def run_baseline(graph, inputs) -> float:
    with torch.jit.fuser("fuser0"):
        return run_test(graph, inputs)


def run_nvfuser(graph, inputs) -> float:
    with torch.jit.fuser("fuser2"):
        return run_test(graph, inputs)


def test_nvfuser(graphs: List[str]):
    for i, ir in enumerate(graphs):
        graph, inputs = load_graph_and_inputs(ir)
        baseline = run_baseline(graph, inputs)
        nvfuser = run_nvfuser(graph, inputs)
        improvement = (baseline / nvfuser - 1) * 100
        print(f"Graph {i}; baseline: {baseline:.2f} ms; nvfuser: {nvfuser:.2f} ms; improvement: {improvement:.2f}%")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--nvfuser", dest="nvfuser", action="store_true")
    parser.add_argument("--no-nvfuser", dest="nvfuser", action="store_false")
    parser.set_defaults(nvfuser=False)
    parser.add_argument("--output", dest="output", action="store_true")
    parser.add_argument("--no-output", dest="output", action="store_false")
    parser.set_defaults(output=True)

    args = parser.parse_args()
    graphs = extract_ir(args.filename)

    if args.nvfuser:
        test_nvfuser(graphs)

    if args.output:
        quoted = []
        for ir in graphs:
            quoted.append("\"\"\"" + ir + "\"\"\"")
        print("[" + ", ".join(quoted) + "]")

if __name__ == "__main__":
    run()
