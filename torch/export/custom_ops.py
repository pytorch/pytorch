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

    if not is_traceable_wrapper_subclass(src_subclass_tensor):
        raise AssertionError(
            f"Expected src_subclass_tensor to be a traceable wrapper subclass, "
            f"but got {type(src_subclass_tensor)}"
        )
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
    if not hasattr(function_cls, "apply"):
        raise AssertionError(
            f"Expected function class {function_cls_name} to have 'apply' method"
        )
    return function_cls.apply(*args, **kwargs)
