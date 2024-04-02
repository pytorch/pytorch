from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, List
import torch
import math
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_map_only
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakIdKeyDictionary
from enum import Enum
from torch.utils.hooks import RemovableHandle
_MB = 2**20
_KB = 2**10
_PYTORCH_MIN_ALLOCATE = 2**9

class _RefType(str, Enum):
    parameter = "parameter"
    buffer = "buffer"
    gradient = "gradient"
    activation = "activation"
    optstate = "optstate"


@dataclass
class _WeakRefInfo:
    def __init__(self, size: int, element_size: int, reftype: _RefType) -> None:
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.mem_consumed = (
            math.ceil((self.size * self.element_size) / _PYTORCH_MIN_ALLOCATE)
            * _PYTORCH_MIN_ALLOCATE
        )

    def get_mem_consumed(self) -> int:
        return self.mem_consumed


class MemoryTrackingMode(TorchDispatchMode):

    def __init__(self, mod: Optional[torch.nn.Module] = None, 
            optm: Optional[torch.optim.Optimizer] = None,
            depth: int = 2,
            units: str = "B",
            display_modulewise_stats: bool = True):
        self.mod = mod
        self.optm = optm
        self.depth = depth
        self.units = units
        self.display_modulewise_stats = display_modulewise_stats
        self.memory_tracking = defaultdict(lambda: defaultdict(defaultdict))
        self.parents = []
        self.MEMORY_MAX: int = 0
        self.FIRST_OPT_ITER: bool = True
        self.WINFO: Dict[torch.storage.UntypedStorage, _WeakRefInfo] = WeakIdKeyDictionary()
        
    def _update_stats(self):
        curr_use = 0
        for winfo in self.WINFO.values():
            curr_use += winfo.get_mem_consumed()

        if self.MEMORY_MAX < curr_use:
            self.MEMORY_MAX = curr_use


    def _track(self, t: torch.Tensor):
        st = t.untyped_storage()
        if self.WINFO.get(st, None) is not None:
            return
        winfo = _WeakRefInfo(st.size(), st.element_size(), _RefType.activation)
        self.WINFO[st] = winfo
        self._update_stats()  
        

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        res = func(*args, **kwargs or {})
        tree_map_only(torch.Tensor, self._track, res)
        return res


    def _get_current_memory_allocated(self) -> Dict[str, float]:
        mem_stats = defaultdict(float)
        mem_stats[_RefType.parameter] = 0
        mem_stats[_RefType.gradient] = 0
        mem_stats[_RefType.optstate] = 0
        mem_stats[_RefType.activation] = 0
        mem_stats[_RefType.buffer] = 0
        for winfo in self.WINFO.values():
            mem_stats[winfo.reftype] += winfo.get_mem_consumed()
        mem_stats["total"] = sum([m for m in mem_stats.values()])
        return mem_stats


    def _get_fqn(self) -> str:
        fqn = ".".join(self.parents)
        return fqn

    def _enter_module(self, name: str, state:str):
        def f(module: nn.Module, args: Any):
            self.parents.append(name)
            fqn = self._get_fqn()
            self.memory_tracking[fqn][state] = self._get_current_memory_allocated()
        return f


    def _exit_module(self, name: str, state: str):
        def f(module: nn.Module, args: Any, outputs: Any):
            assert self.parents[-1] == name, f"{self.parents[-1]} is not {name}"
            fqn = self._get_fqn()
            self.memory_tracking[fqn][state] = self._get_current_memory_allocated()
            self.parents.pop()
        return f


    def _register_hooks(self, name: str, module: nn.Module):
        module.register_forward_pre_hook(self._enter_module(name, "Before Forward"))
        module.register_forward_hook(self._exit_module(name, "After Forward"))
        module.register_full_backward_pre_hook(self._enter_module(name, "Before Backward"))
        module.register_full_backward_hook(self._exit_module(name, "After Backward"))


    def _instrument_module(self, mod: nn.Module):
        prefix = type(mod).__name__
        for name, module in mod.named_modules():
            
            if name == "":
                name = prefix
            else:
                name = ".".join([prefix, name])
            if hasattr(module, "inplace") and getattr(module, "inplace"):
                setattr(module, "inplace", False)
            self._register_hooks(name, module)

        def _grad_hook(param: nn.Parameter):
            winfo = self.WINFO[param.grad.untyped_storage()]
            assert winfo is not None, "grad tensor not found in WINFO"
            winfo.reftype = _RefType.gradient

        for param_name, param in mod.named_parameters():
            param.register_post_accumulate_grad_hook(_grad_hook)
            winfo = self.WINFO[param.untyped_storage()]
            assert winfo is not None, f"param {param_name} not found in WINFO"
            winfo.reftype = _RefType.parameter

        for buffer_name, buffer in mod.named_buffers():
            winfo = self.WINFO[buffer.untyped_storage()]
            assert winfo is not None, f"buffer {buffer_name} not found in WINFO"
            winfo.reftype = _RefType.buffer


    def _instrument_optimizer(self, optim: torch.optim.Optimizer):
        def _opt_state(optimizer: torch.optim.Optimizer, args: Any, kwargs: Any) -> None:
            if self.FIRST_OPT_ITER:
                for states in optimizer.state.values():
                    for val in states.values():
                        if isinstance(val, torch.Tensor):
                            winfo = self.WINFO[val.untyped_storage()]
                            assert winfo is not None, "opt state tensor not found in WINFO"
                            winfo.reftype = _RefType.optstate
                self.FIRST_OPT_ITER = False

        optim.register_step_post_hook(_opt_state)

    def _register_module_and_optimizer_hooks(self):
        self._instrument_module(self.mod)
        self._instrument_optimizer(self.optm)

    def _deregister_module_and_optimizer_hooks(self):
        pass

    def __enter__(self):
        self._register_module_and_optimizer_hooks()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if self.display_modulewise_stats:
            self._display_mem_stats()
        self._deregister_module_and_optimizer_hooks()
        super().__exit__(*args)


    def print_mem_stats(self, stats: Optional[Dict[str, int]] = None):
        if stats is None:
            stats = self._get_current_memory_allocated()
        if self.units == "MB":
            divisor = _MB
        elif self.units == "KB":
            divisor = _KB
        elif self.units == "B":
            divisor = 1
        for type, mem in stats.items():
            print(f"\t{type}: {round(mem/divisor, 2)} {self.units}")

    def _display_mem_stats(self):

        for mod in self.memory_tracking.keys():
            print(f"Module: ", mod)
            for k, stats in self.memory_tracking[mod].items():
                print(f"{k}")
                self.print_mem_stats(stats)
            print()


def experiment():
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
    layers = 20
    dim = 1000        
    torch.set_default_device("cuda")
    torch.cuda.reset_peak_memory_stats()
    fake_mode = FakeTensorMode()
    mem_tracker = MemoryTrackingMode()
    

    with  mem_tracker:
        model = DummyModel(layers, dim)
        optim = torch.optim.Adam(model.parameters(), fused = True)
        input_batch = torch.randn(batch_size, dim)
        print(f"After Model and mini-batch init:")
        print(torch.cuda.memory_allocated())
        print_mem_stats()
        output = model(input_batch)
        print(f"After Forward:")
        print(torch.cuda.memory_allocated())
        print_mem_stats()
        output.sum().backward()
        output = None
        print(f"After Backward:")
        print(torch.cuda.memory_allocated())
        print_mem_stats()
        optim.step()
        print(f"After Opt Step:")
        print(torch.cuda.memory_allocated())
        print_mem_stats()
        optim.zero_grad()
        print(f"After Zero Grad:")
        print(torch.cuda.memory_allocated())
        print_mem_stats()

    
    print(f"Tracker measured: {MEMORY_MAX}")
    CUDA_MEMORY_MAX = torch.cuda.max_memory_allocated()
    print(f"Cuda measured: {CUDA_MEMORY_MAX}")
    print(f"Peak ratio: {MEMORY_MAX/CUDA_MEMORY_MAX}")
    # display_mem_stats()


if __name__ == "__main__":
    experiment()
