from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt

import torch
from torch.utils._python_dispatch import TorchDispatchMode


aten = torch.ops.aten

MB = 1024 * 1024.0

operator_names: Dict[str, int] = defaultdict(int)
mem_usage: Dict[str, float] = defaultdict(float)
markers: Dict[str, int] = defaultdict(int)
series: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def reduce_to_scalar_loss(inp):
    return inp.sum()


class MemoryProfileDispatchMode(TorchDispatchMode):
    def __init__(self, verbose=False):
        self.verbose: bool = verbose

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)
        if func == torch.ops.aten.detach.default:
            return rs
        mem: float = torch.cuda.memory_allocated() / MB
        func_name: str = func.__name__ + "_" + str(operator_names[func.__name__])
        operator_names[func.__name__] = operator_names[func.__name__] + 1
        mem_usage[func_name] = mem
        if self.verbose:
            print("Mem Usage (" + func_name + "): ", mem)
        return rs


def clear_state():
    operator_names.clear()
    mem_usage.clear()


def add_series(series_name):
    global mem_usage
    fin_usage = torch.cuda.memory_allocated() / MB
    mem_usage["fin_usage"] = fin_usage
    series[series_name] = mem_usage
    mem_usage = defaultdict(float)


def save_graph(filename: str):
    for series_name, mem_usage in series.items():
        y = mem_usage.values()
        min_val = min(y)
        max_val = max(y)
        x = list(i for i in range(len(y)))
        plt.plot(x, y, label=series_name)
    plt.xlabel("# Operator Calls")
    plt.ylabel("Allocated Memory (MB)")
    plt.title(filename)
    plt.show()
    for marker_name, marker in markers.items():
        plt.plot([marker, marker], [min_val, max_val], "k-", lw=2, label=marker_name)
    plt.legend()
    print(f"Saving Graph to {filename}")
    plt.savefig(filename)
    plt.close()
    markers.clear()
    series.clear()


def add_marker(marker_name):
    k = len(series.keys())
    last_val_num = len(mem_usage.values())
    markers[marker_name + str(k)] = last_val_num


def mem_profile_model(mod: torch.nn.Module, inp: torch.Tensor):
    with MemoryProfileDispatchMode(True):
        pred = mod(inp)
        loss = reduce_to_scalar_loss(pred)
        loss.backward()
        mod.zero_grad(True)
        torch.cuda.synchronize()
        clear_state()
        pred = mod(inp)
        loss = reduce_to_scalar_loss(pred)
        add_marker("fw_bw_boundary")
        loss.backward()
