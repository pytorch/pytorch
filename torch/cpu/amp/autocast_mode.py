import torch

class autocast(torch.autocast_mode.autocast):
    r"""
    See :class:`torch.autocast`.
    ``torch.cpu.amp.autocast(args...)`` is equivalent to ``torch.autocast("cpu", args...)``
    """
    def __init__(self, enabled=True, fast_dtype=torch.bfloat16):
        super().__init__("cpu", enabled=enabled, fast_dtype=fast_dtype)
