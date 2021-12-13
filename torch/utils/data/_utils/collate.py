import torch.utils.data.collate as collate

"""
These functions have been relocated to torch.utils.data.collate, but for backward-compatibility,
we preserve references to them here.
"""
default_collate = collate.default_collate
default_convert = collate.default_convert
