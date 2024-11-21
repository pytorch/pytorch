import torch

from .functional import *  # noqa: F403


if torch.distributed.rpc.is_available():
    from .api.remote_module import RemoteModule
