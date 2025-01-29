import torch 
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

__all__ = ["access_subclass_inner_tensor"]


def access_subclass_inner_tensor(src_subclass_tensor: torch.Tensor, attr: str) -> torch.Tensor:
    """
    This is custom python function that is used to access the inner tensor of a subclass tensor.
    This is particularly useful when we want to manipulate inner plain tensors of a subclass at 
    training IR level because subclasses are opaque to training IR. 
    """
    if torch.overrides.has_torch_function_unary(src_subclass_tensor):
        return torch.overrides.handle_torch_function(
            access_subclass_inner_tensor,
            (src_subclass_tensor,),
            src_subclass_tensor,
            attr,
        )
    assert is_traceable_wrapper_subclass(src_subclass_tensor)
    val = getattr(src_subclass_tensor, attr, None)
    if val is None or not isinstance(val, torch.Tensor):
        raise RuntimeError(f"Attribute {attr} is not a tensor or doesn't exist in {src_subclass_tensor}")
    return val

    
