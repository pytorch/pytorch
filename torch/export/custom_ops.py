import torch


lib = torch.library.Library("export", "FRAGMENT")  # noqa: TOR901

lib.define(
    "access_subclass_inner_tensor(Tensor src_subclass_tensor, str attr) -> Tensor"
)


@torch.library.impl(lib, "access_subclass_inner_tensor", "Autograd")
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
