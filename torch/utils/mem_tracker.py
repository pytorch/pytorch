from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict
import torch
import math
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map_only
import torchvision
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref
from enum import Enum

MB = 2**20
KB = 2**10
MEMORY_MAX = 0
PYTORCH_MIN_ALLOCATE = 2**9
MEMORY_USE = WeakIdKeyDictionary()
FIRST_OPT_ITER = True
parents = []
memory_tracking = defaultdict(lambda: defaultdict(lambda: defaultdict()))


class RefType(str, Enum):
    parameter = "parameter"
    buffer = "buffer"
    gradient = "gradient"
    activation = "activation"
    optstate = "optstate"


@dataclass
class WeakRefInfo:
    def __init__(self, size: int, element_size: int, reftype: RefType) -> None:
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.mem_consumed = (
            math.ceil((self.size * self.element_size) / PYTORCH_MIN_ALLOCATE)
            * PYTORCH_MIN_ALLOCATE
        )

    def get_mem_consumed(self) -> int:
        return self.mem_consumed


WINFO: Dict[weakref.ref, WeakRefInfo] = WeakIdKeyDictionary()


def update_stats():
    global MEMORY_MAX, WINFO
    curr_use = 0
    for k, v in MEMORY_USE.items():
        curr_use += WINFO[k].get_mem_consumed()

    if MEMORY_MAX < curr_use:
        MEMORY_MAX = curr_use


def track(t):
    def cb(_):
        update_stats()

    reftype = RefType.activation
    if isinstance(t, nn.Parameter):
        reftype = RefType.parameter
    st = t.untyped_storage()
    wt = weakref.ref(st, cb)
    winfo = WeakRefInfo(st.size(), st.element_size(), reftype)
    WINFO[st] = winfo
    MEMORY_USE[st] = wt
    update_stats()


class MemoryTrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        res = func(*args, **kwargs or {})
        tree_map_only(torch.Tensor, track, res)
        return res


def get_current_memory_allocated() -> Dict[str, float]:
    global MEMORY_USE, WINFO
    mem_stats = defaultdict(float)
    mem_stats[RefType.parameter] = 0
    mem_stats[RefType.gradient] = 0
    mem_stats[RefType.optstate] = 0
    mem_stats[RefType.activation] = 0
    mem_stats[RefType.buffer] = 0
    for k, v in MEMORY_USE.items():
        winfo = WINFO[k]
        mem_stats[winfo.reftype] += winfo.get_mem_consumed()
    mem_stats["total"] = sum([m for m in mem_stats.values()])
    return mem_stats


def get_fqn() -> str:
    fqn = ".".join(parents)
    return fqn


def enter_module_forward(name: str):
    @torch.no_grad
    def f(module: nn.Module, args: Any):
        global parents
        parents.append(name)
        fqn = get_fqn()
        memory_tracking[fqn]["Before Forward"] = get_current_memory_allocated()

    return f


def exit_module_forward(name: str):
    @torch.no_grad
    def f(module: nn.Module, args: Any, outputs: Any):
        global parents
        assert parents[-1] == name
        fqn = get_fqn()
        memory_tracking[fqn]["After Forward"] = get_current_memory_allocated()
        parents.pop()

    return f


def enter_module_backward(name: str):
    @torch.no_grad
    def f(module: nn.Module, grad_output: Any):
        global parents
        parents.append(name)
        fqn = get_fqn()
        memory_tracking[fqn]["Before Backward"] = get_current_memory_allocated()

    return f


def exit_module_backward(name: str):
    @torch.no_grad
    def f(module: nn.Module, grad_input: Any, grad_output: Any):
        global parents
        assert parents[-1] == name
        fqn = get_fqn()
        memory_tracking[fqn]["After Backward"] = get_current_memory_allocated()
        parents.pop()

    return f


def final_call():
    fqn = get_fqn()
    memory_tracking[fqn]["After Backward"] = get_current_memory_allocated()
    parents.pop()


def register_hooks(name: str, module: nn.Module):
    module.register_forward_pre_hook(enter_module_forward(name))
    module.register_forward_hook(exit_module_forward(name))
    module.register_full_backward_pre_hook(enter_module_backward(name))
    if name == "Root":
        return
    module.register_full_backward_hook(exit_module_backward(name))


def instrument_module(mod: nn.Module):
    register_hooks("Root", mod)

    for name, module in mod.named_children():
        if hasattr(module, "inplace") and getattr(module, "inplace"):
            setattr(module, "inplace", False)
        register_hooks(name, module)

    def grad_hook(param: nn.Parameter):
        global WINFO
        winfo = WINFO[param.grad.untyped_storage()]
        assert winfo is not None, "grad tensor not found in WINFO"
        winfo.reftype = RefType.gradient

    for param in mod.parameters():
        param.register_post_accumulate_grad_hook(grad_hook)
        winfo = WINFO[param.untyped_storage()]
        assert winfo is not None, "param tensor not found in WINFO"
        winfo.reftype = RefType.parameter

    for buffer in mod.buffers():
        winfo = WINFO[buffer.untyped_storage()]
        assert winfo is not None, "buffer not found in WINFO"
        winfo.reftype = RefType.buffer


def instrument_optimizer(optim: torch.optim.Optimizer):
    def outopt(optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
        global FIRST_OPT_ITER, WINFO, MEMORY_USE
        if FIRST_OPT_ITER:
            for param, states in optimizer.state.items():
                for val in states.values():
                    if isinstance(val, torch.Tensor):
                        winfo = WINFO[val.untyped_storage()]
                        assert winfo is not None, "opt state tensor not found in WINFO"
                        winfo.reftype = RefType.optstate
            FIRST_OPT_ITER = False

    optim.register_step_post_hook(outopt)


def print_mem_stats(stats: Dict[str, int]):
    for type, mem in stats.items():
        print(f"\t{type}: {round(mem/MB, 2)} MBs")


def display_mem_stats():
    for mod in memory_tracking.keys():
        print(f"Module: ", mod)
        for k, stats in memory_tracking[mod].items():
            print(f"{k}")
            print_mem_stats(stats)
        print()


def experiment():
    fake_mode = FakeTensorMode()
    mem_tracker = MemoryTrackingMode()
    torch.set_default_device("cuda")
    torch.cuda.reset_peak_memory_stats()
    with mem_tracker:
        model = torchvision.models.resnet18()
        optim = torch.optim.Adam(model.parameters())
        instrument_module(model)
        instrument_optimizer(optim)
        input = torch.randn(256, 3, 224, 224)
        output = model(input)
        output.sum().backward()
        final_call()
        optim.step()
        optim.zero_grad()

    display_mem_stats()
    print(f"Tracker measured: {MEMORY_MAX}")
    print(f"CUDA measured: {torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    experiment()
