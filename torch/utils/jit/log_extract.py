# mypy: allow-untyped-defs
from contextlib import contextmanager
from typing import Any, cast
import random
import torch
import time
from torch.utils.benchmark import Timer

def extract_ir(filename: str) -> list[str]:
    BEGIN = "<GRAPH_EXPORT>"
    END = "</GRAPH_EXPORT>"
    pfx = None
    graphs = []
    with open(filename) as f:
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
    assert size is not None
    assert stride is not None
    assert device is not None
    assert dtype is not None
    return torch.empty_strided(size=size, stride=stride, device=device, dtype=dtype)

def load_graph_and_inputs(ir: str) -> tuple[Any, list[Any]]:
    graph = torch._C.parse_ir(ir, parse_tensor_constants=True)
    graph.makeMultiOutputIntoTuple()
    inputs = []
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(random.uniform(.1, 100))
        elif isinstance(inp.type(), torch._C.IntType):
            inputs.append(random.randint(1, 100))
        elif isinstance(inp.type(), torch._C.TensorType):
            tensorType = cast(torch._C.TensorType, inp.type())
            inputs.append(make_tensor_from_type(tensorType))
        elif isinstance(inp.type(), torch._C.BoolType):
            inputs.append(random.randint(0, 1) == 1)
        else:
            raise NotImplementedError(f"A default value is not implemented for type {inp.type()}")

    func = torch._C._create_function_from_graph("forward", graph)
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)

def time_cuda(fn, inputs, test_runs):
    t = Timer(stmt="fn(*inputs)", globals={"fn": fn, "inputs" : inputs})
    times = t.blocked_autorange()
    return times.median * 1000  # time in ms

def time_cpu(fn, inputs, test_runs):
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs * 1000  # time in ms

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
