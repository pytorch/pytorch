from typing import Callable

import torch
from torch.ao.nn.intrinsic import _FusedModule
from torch.fx._symbolic_trace import Tracer
from torch.fx.proxy import Scope


__all__ = [
    "QuantizationTracer",
]


class ScopeContextManager(torch.fx.proxy.ScopeContextManager):
    def __init__(
        self, scope: Scope, current_module: torch.nn.Module, current_module_path: str
    ):
        super().__init__(scope, Scope(current_module_path, type(current_module)))


class QuantizationTracer(Tracer):
    def __init__(
        self, skipped_module_names: list[str], skipped_module_classes: list[Callable]
    ):
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type of top level
        # module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.record_stack_traces = True

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            (
                (
                    m.__module__.startswith("torch.nn")
                    or m.__module__.startswith("torch.ao.nn")
                )
                and not isinstance(m, torch.nn.Sequential)
            )
            or module_qualified_name in self.skipped_module_names
            or type(m) in self.skipped_module_classes
            or isinstance(m, _FusedModule)
        )
