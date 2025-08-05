import torch


# LITERALINCLUDE START: CUSTOM OPERATOR META
lib = torch.library.Library("openreg", "IMPL", "Meta")  # noqa: TOR901


@torch.library.impl(lib, "custom_abs")
def custom_abs(self):
    return torch.empty_like(self)


# LITERALINCLUDE END: CUSTOM OPERATOR META
