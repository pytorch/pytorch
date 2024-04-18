import math
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Optional, Union

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.hooks import RemovableHandle
from torch.utils.weak import WeakIdKeyDictionary

_PYTORCH_MIN_ALLOCATE = 2**9

__all__ = ["MemoryTrackingMode"]


class _RefType(str, Enum):
    parameter = "parameter"
    unsharded_parameter = "unsharded_parameter"
    buffer = "buffer"
    gradient = "gradient"
    unsharded_gradient = "unsharded_gradient"
    activation = "activation"
    optstate = "optstate"


@dataclass
class _WeakRefInfo:
    def __init__(self, size: int, element_size: int, reftype: _RefType) -> None:
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self)->int:
        return (math.ceil((self.size * self.element_size) / _PYTORCH_MIN_ALLOCATE)
            * _PYTORCH_MIN_ALLOCATE)

    def get_mem_consumed(self, st: torch.UntypedStorage) -> int:
        if st.size() != self.size:
            self.size = st.size()
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed


class _ModuleHookHandles(NamedTuple):
    forward_pre_hook_handle: RemovableHandle
    forward_hook_handle: RemovableHandle
    backward_pre_hook_handle: RemovableHandle
    backward_hook_handle: RemovableHandle


class MemoryTrackingMode(TorchDispatchMode):
    def __init__(
        self,
        mod: Optional[torch.nn.Module] = None,
        optm: Optional[torch.optim.Optimizer] = None,
        depth: int = 2,
        units: str = "B",
        display_modulewise_stats: bool = True,
    ):
        self.mod = mod
        self.optm = optm
        self.depth = depth
        self.units = units
        self.display_modulewise_stats = display_modulewise_stats
        self.memory_tracking: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(defaultdict)
        )
        self.parents: List[str] = []
        self._MEMORY_MAX: int = 0
        self.FIRST_OPT_ITER: bool = True
        self._module_to_hook_handles: Dict[nn.Module, _ModuleHookHandles] = {}
        self._param_to_grad_hook_handles: Dict[nn.Parameter, RemovableHandle] = {}
        self._optimizer_hook_handle: Union[RemovableHandle, None] = None
        self.WINFO = WeakIdKeyDictionary()

    def _update_stats(self):
        curr_use: int = 0
        for st, winfo in self.WINFO.items():
            curr_use += winfo.get_mem_consumed(st)

        if self._MEMORY_MAX < curr_use:
            self._MEMORY_MAX = curr_use

    def _track(self, t: torch.Tensor):
        st = t.untyped_storage()
        if self.WINFO.get(st, None) is not None:
            return
        winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.activation)
        self.WINFO[st] = winfo
        self._update_stats()

    def _get_current_memory_allocated(self) -> Dict[str, int]:
        mem_stats = defaultdict(int)
        mem_stats[_RefType.parameter.name] = 0
        mem_stats[_RefType.unsharded_parameter.name] = 0
        mem_stats[_RefType.gradient.name] = 0
        mem_stats[_RefType.unsharded_gradient.name] = 0
        mem_stats[_RefType.optstate.name] = 0
        mem_stats[_RefType.activation.name] = 0
        mem_stats[_RefType.buffer.name] = 0
        for st, winfo in self.WINFO.items():
            mem_stats[winfo.reftype.name] += winfo.get_mem_consumed(st)
        mem_stats["TRACKER_total"] = sum([m for m in mem_stats.values()])
        if torch.cuda.is_available():
            mem_stats["CUDA_total"] = torch.cuda.memory_allocated()
        return mem_stats

    def print_mem_stats(self, stats: Optional[Dict[str, int]] = None):
        if stats is None:
            stats = self._get_current_memory_allocated()
        rounding_fn = lambda x, y, z: round(x / y, z)
        if self.units == "MB":
            divisor = 2**20
        elif self.units == "KB":
            divisor = 2**10
        else:
            divisor = 1
            rounding_fn = lambda x, y, z: x
        for mem_type, mem_val in stats.items():
            print(f"\t{mem_type}: {rounding_fn(mem_val, divisor, 2)} {self.units}")

    def get_max_memory(self) -> int:
        return self._MEMORY_MAX

    def _display_mem_stats(self, depth=None):
        if depth is None:
            depth = self.depth
        for mod in self.memory_tracking.keys():
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module: ", mod)
            for state, stats in self.memory_tracking[mod].items():
                print(f"{state}")
                self.print_mem_stats(stats)
            print()

    def _enter_module(self, name: str, state: str):
        def f(module: nn.Module, args: Any):
            self.parents.append(name)
            self.memory_tracking[name][state] = self._get_current_memory_allocated()

        return f

    def _exit_module(self, name: str, state: str):
        def f(module: nn.Module, args: Any, outputs: Any):
            assert self.parents[-1] == name, f"{self.parents[-1]} is not {name}"
            self.memory_tracking[name][state] = self._get_current_memory_allocated()
            self.parents.pop()

        return f

    def _register_module_hooks(self, name: str, module: nn.Module):
        fwd_pre_hook_handle = module.register_forward_pre_hook(
            self._enter_module(name, "Before Forward")
        )
        fwd_hook_handle = module.register_forward_hook(
            self._exit_module(name, "After Forward")
        )
        bwd_pre_hook_handle = module.register_full_backward_pre_hook(
            self._enter_module(name, "Before Backward")
        )
        bwd_hook_handle = module.register_full_backward_hook(
            self._exit_module(name, "After Backward")
        )
        self._module_to_hook_handles[module] = _ModuleHookHandles(
            fwd_pre_hook_handle, fwd_hook_handle, bwd_pre_hook_handle, bwd_hook_handle
        )

    def _instrument_module(self, mod: nn.Module):
        prefix = type(mod).__name__
        for name, module in mod.named_modules():
            if name == "":
                name = prefix
            else:
                name = ".".join([prefix, name])
            if hasattr(module, "inplace") and getattr(module, "inplace"):
                setattr(module, "inplace", False)
            self._register_module_hooks(name, module)

        def _grad_hook(param: nn.Parameter):
            if param.grad is not None:
                st = param.grad.untyped_storage()
                winfo = self.WINFO.get(st, None)
                assert winfo is not None, "grad tensor not found in WINFO"
                winfo.reftype = _RefType.gradient

        for param in mod.parameters():
            st = param.untyped_storage()
            winfo = self.WINFO.get(st, None)
            if winfo is None:
                winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.parameter)
                self.WINFO[st] = winfo
            grad_hook_handle = param.register_post_accumulate_grad_hook(_grad_hook)
            self._param_to_grad_hook_handles[param] = grad_hook_handle

        for buffer in mod.buffers():
            st = buffer.untyped_storage()
            winfo = self.WINFO.get(st, None)
            if winfo is None:
                winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.buffer)
                self.WINFO[st] = winfo

    def _instrument_optimizer(self, optim: torch.optim.Optimizer):
        def _opt_state(
            optimizer: torch.optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            if self.FIRST_OPT_ITER:
                for states in optimizer.state.values():
                    for val in states.values():
                        if isinstance(val, torch.Tensor):
                            winfo = self.WINFO[val.untyped_storage()]
                            assert (
                                winfo is not None
                            ), "opt state tensor not found in WINFO"
                            winfo.reftype = _RefType.optstate
                self.FIRST_OPT_ITER = False

        opt_hook_handle = optim.register_step_post_hook(_opt_state)
        self._optimizer_hook_handle = opt_hook_handle

    def _register_module_and_optimizer_hooks(self):
        if self.mod is not None:
            self._instrument_module(self.mod)
        if self.optm is not None:
            self._instrument_optimizer(self.optm)

    def _deregister_module_and_optimizer_hooks(self):
        for module_hook_handles in self._module_to_hook_handles.values():
            module_hook_handles.forward_pre_hook_handle.remove()
            module_hook_handles.forward_hook_handle.remove()
            module_hook_handles.backward_pre_hook_handle.remove()
            module_hook_handles.backward_hook_handle.remove()

        for grad_hook_handle in self._param_to_grad_hook_handles.values():
            grad_hook_handle.remove()

        if self._optimizer_hook_handle is not None:
            self._optimizer_hook_handle.remove()

    def __enter__(self):
        self._register_module_and_optimizer_hooks()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if self.display_modulewise_stats:
            self._display_mem_stats()
        self._deregister_module_and_optimizer_hooks()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        res = func(*args, **kwargs or {})
        tree_map_only(torch.Tensor, self._track, res)
        return res


def test():
    class DummyModel(nn.Module):
        def __init__(self, layers: int, dim: int):
            super(DummyModel, self).__init__()
            self._module_list = []
            for _ in range(layers):
                self._module_list.extend([nn.Linear(dim, dim), nn.ReLU()])
            self.module = nn.Sequential(*self._module_list)

        def forward(self, x):
            return self.module(x)

    batch_size = 100
    layers = 5
    dim = 10000
    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        torch.cuda.reset_peak_memory_stats()

    model = DummyModel(layers, dim)
    optim = torch.optim.Adam(model.parameters(), fused=True)
    mem_tracker = MemoryTrackingMode(model, optim, display_modulewise_stats=True)
    with mem_tracker as mt:
        input_batch = torch.randn(batch_size, dim)
        print(f"After Model and mini-batch init:")
        mt.print_mem_stats()
        output = model(input_batch)
        print(f"After Forward:")
        mt.print_mem_stats()
        output.sum().backward()
        output = None
        print(f"After Backward:")
        mt.print_mem_stats()
        optim.step()
        print(f"After Opt Step:")
        mt.print_mem_stats()
        optim.zero_grad()
        print(f"After Zero Grad:")
        mt.print_mem_stats()
        MAX_MEMORY = mt.get_max_memory()

    print(f"Tracker measured: {MAX_MEMORY}")
    if torch.cuda.is_available():
        CUDA_MEMORY_MAX = torch.cuda.max_memory_allocated()
        print(f"Cuda measured: {CUDA_MEMORY_MAX}")
        print(f"Peak comparison ratio: {MAX_MEMORY/CUDA_MEMORY_MAX}")


if __name__ == "__main__":
    test()
