import torch


# LITERALINCLUDE START: CUSTOM OPERATOR META
lib = torch.library.Library("openreg", "IMPL", "Meta")  # noqa: SCOPED_LIBRARY


@torch.library.impl(lib, "custom_abs")
def custom_abs(self):
    return torch.empty_like(self)


# LITERALINCLUDE END: CUSTOM OPERATOR META
