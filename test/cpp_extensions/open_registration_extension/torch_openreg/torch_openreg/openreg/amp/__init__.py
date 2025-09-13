import torch

# LITERALINCLUDE START: GET_AMP_SUPPORTED_DTYPE

def get_amp_supported_dtype():
    return [torch.float16]

# LITERALINCLUDE END: GET_AMP_SUPPORTED_DTYPE