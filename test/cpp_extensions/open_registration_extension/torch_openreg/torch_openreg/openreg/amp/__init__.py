import torch


class autocast(torch.amp.autocast_mode.autocast):
    def __init__(self, enabled=True, dtype=torch.float16, cache_enabled=True):
        super().__init__(
            "openreg", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
        )


# LITERALINCLUDE START: AMP GET_SUPPORTED_DTYPE
def get_amp_supported_dtype():
    return [torch.float16, torch.bfloat16]


# LITERALINCLUDE END: AMP GET_SUPPORTED_DTYPE
