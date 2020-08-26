"""Backend API
This module contains helper functions and utilities for the JIT backend
API.
"""

from torch.jit._script import (
    RecursiveScriptModule,
)
from typing import Callable, List


# Type for functions that lower RecursiveScriptModules to JIT backends.
ToBackendFnTy = Callable[
    [RecursiveScriptModule], RecursiveScriptModule
]


def selective_to_jit_backend(
    module: RecursiveScriptModule,
    to_backend_fn: ToBackendFnTy,
    modules_to_lower: List[str],
):
    """
    Selectively lower modules in a module hierarchy to a JIT backend.


    Arguments:
        module: The RecursiveScriptModule at the root of the module hierachy that
                    is to be selectively lowered.
        to_backend_fn:  Function that will be called on each submodule in modules_to_lower
                            to lower it to a JIT backend.
        modules_to_lower: List of fully qualified names of submodules to selectively
                            lower.
    """
    base = ""
    selective_to_jit_backend_impl(module, to_backend_fn, modules_to_lower, base)


def selective_to_jit_backend_impl(
    module: RecursiveScriptModule,
    to_backend_fn: ToBackendFnTy,
    modules_to_lower: List[str],
    base: str,
) -> RecursiveScriptModule:
    # For each submodule:
    for name, submodule in module._modules.items():
        # This is the fully qualified "path" to the attribute from the root module.
        sep = "." if base else ""
        qual_name = f"{base}{sep}{name}"
        if qual_name in modules_to_lower:
            # Lower the submodule.
            lowered_submodule = to_backend_fn(submodule)
            module_jit_type = module._c._type()
            # Modify the corresponding attribute on the owning module's type.
            module_jit_type.unsafe_remove_attribute(name)
            module_jit_type.add_attribute(name, lowered_submodule._c._type())
            # Remap the type in the graph for the Module.
            module.graph.remapTypes([submodule._c._type()], [lowered_submodule._c._type()])
            # Set the attribute of module to point to the lowered module.
            setattr(module, name, lowered_submodule)
        else:
            # Call selective_to_jit_backend_impl recursively because there might be submodules that need to be lowered.
            lowered_submodule = selective_to_jit_backend_impl(module, to_backend_fn, modules_to_lower, qual_name)
