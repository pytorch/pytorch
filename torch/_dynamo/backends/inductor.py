# mypy: ignore-errors

"""
This module provides the TorchInductor backend integration for TorchDynamo.

TorchInductor is a compiler backend that generates optimized code for both CPU and GPU.
This module lazily imports and registers the TorchInductor compiler to avoid loading it
into memory when it is not being used. This helps reduce memory overhead when using
other backends.

The inductor backend can be used with torch.compile():
    model = torch.compile(model, backend="inductor")
"""

from torch._dynamo import register_backend
from torch._dynamo.utils import dynamo_timed


@register_backend
def inductor(*args, **kwargs):
    with dynamo_timed("inductor_import", log_pt2_compile_event=True):
        # do import here to avoid loading inductor into memory when it is not used
        from torch._inductor.compile_fx import compile_fx

    return compile_fx(*args, **kwargs)
