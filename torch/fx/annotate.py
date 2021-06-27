import torch
from torch.fx.proxy import Proxy

def annotate(val, type):
    # val could be either a regular value (not tracing)
    # or fx.Proxy (tracing)
    if isinstance(val, Proxy):
        if val.node.type:
            raise RuntimeError("type already exists")
        else:
            val.node.type = type
        return val
    else:
        return val
