import os
import torch._C._lazy_ts_backend


def init():
    """Initializes the lazy Torchscript backend"""
    if os.environ.get("PARSH_AUTORELOAD_CONTEXT") != "1":
        torch._C._lazy_ts_backend._init()
