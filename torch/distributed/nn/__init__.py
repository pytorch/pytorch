import torch
if torch.distributed.rpc.is_available():
    from .api.remote_module import RemoteModule
from .functional import *  # noqa: F403
