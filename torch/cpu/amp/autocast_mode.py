import torch

class autocast(torch.autocast):
    def __init__(self, enabled=True, fast_dtype=torch.float16):
        super().__init__("cpu", enabled=enabled, fast_dtype=fast_dtype)
