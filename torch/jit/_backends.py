"""Backend API
This module contains helper functions and utilities for the JIT backend
API.
"""

from torch.jit._script import (
    RecursiveScriptModule,
)
from typing import Callable


# Type for functions that lower RecursiveScriptModules to JIT backends.
ToBackendFnTy = Callable[
    [RecursiveScriptModule], RecursiveScriptModule
]

def selective_to_jit_backend(
    module: RecursiveScriptModule, to_backend_fn: ToBackendFnTy
) -> RecursiveScriptModule:
    """
    Selectively lower modules in a module hierarchy to a JIT backend.


    Arguments:
        module: The RecursiveScriptModule at the root of the module hierachy that
                    is to be lowered.
        to_backend_fn:  Function that should will be called on each submodule to
                            lower it to a JIT backend. Logic to include/exclude
                            submodules from lowering should be implemented inside
                            this function.
    """
    # Get the JIT type of module. This will need to be adjusted as its submodules
    # are lowered.
    module_jit_type = module._c._type()

    # For each submodule:
    for name, submodule in module._modules.items():
        # Recursively process the submodule.
        lowered_submodule = selective_to_jit_backend(submodule, to_backend_fn)
        # Modify the corresponding attribute on the owning module's type.
        module_jit_type.unsafe_remove_attribute(name)
        module_jit_type.add_attribute(name, lowered_submodule._c._type())
        # Set the attribute of module to point to the lowered module.
        setattr(module, name, lowered_submodule)

    # Now, lower module and return it.
    return to_backend_fn(module)
