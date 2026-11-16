import torch


# LITERALINCLUDE START: AMP GET_SUPPORTED_DTYPE
def get_amp_supported_dtype():
    return [torch.float16, torch.bfloat16]


# LITERALINCLUDE END: AMP GET_SUPPORTED_DTYPE
