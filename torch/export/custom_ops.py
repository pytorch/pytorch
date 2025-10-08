# mypy: allow-untyped-defs
import importlib

import torch


lib = torch.library.Library("export", "FRAGMENT")  # noqa: TOR901

lib.define(
    "access_subclass_inner_tensor(Tensor src_subclass_tensor, str attr) -> Tensor"
)


@torch.library.impl(lib, "access_subclass_inner_tensor", "Autograd")
# When running under torch.inference_mode(), we seem to skip AUtograd key
# so we should desugar this op as soon as we start tracing to post-dispatch.
@torch.library.impl(lib, "access_subclass_inner_tensor", "Python")
def _access_subclass_inner_tensor(
    src_subclass_tensor: torch.Tensor, attr: str
) -> torch.Tensor:
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    assert is_traceable_wrapper_subclass(src_subclass_tensor)
    val = getattr(src_subclass_tensor, attr, None)
    if val is None or not isinstance(val, torch.Tensor):
        raise RuntimeError(
            f"Attribute {attr} is not a tensor or doesn't exist in {src_subclass_tensor}"
        )
    return val


def _call_custom_autograd_function_in_pre_dispatch(function_cls_name, *args, **kwargs):
    """
    Import a custom autograd function by string name and call it. This is pretty bad
    because:
    1) There is no schema

    Ideally we should automatically wrap custom autograd functions with a custom op, but
    that is too much work because we need to schematize custom autograd functions. For now,
    we just hackily put it in the IR.
    """
    # Parse module and class name
    module_name, class_name = function_cls_name.rsplit(".", 1)

    # Import the module and get the class
    module = importlib.import_module(module_name)
    function_cls = getattr(module, class_name)
    assert hasattr(function_cls, "apply")
    return function_cls.apply(*args, **kwargs)

def make_redistribute_closure(device_mesh, placements, async_op=False, forward_dtype=None, backward_dtype=None):
    """
    Create a closure function for DTensor.redistribute() that captures all arguments.
    This mimics dynamo's closure-based approach in variables/tensor.py:method_redistribute.
    The returned closure only takes the dtensor as input.
    """
    def redistribute_closure(dtensor):
        # All arguments are captured in the closure
        return dtensor.redistribute(
            device_mesh=device_mesh,
            placements=placements,
            async_op=async_op,
            forward_dtype=forward_dtype,
            backward_dtype=backward_dtype,
        )

    redistribute_closure.__name__ = f"redistribute_closure_{id(redistribute_closure)}"
    return redistribute_closure

def make_to_local_closure(grad_placements=None):
    """
    Create a closure function for DTensor.to_local() that captures grad_placements.
    """
    def to_local_closure(dtensor):
        return dtensor.to_local(grad_placements=grad_placements)

    to_local_closure.__name__ = f"to_local_closure_{id(to_local_closure)}"
    return to_local_closure
