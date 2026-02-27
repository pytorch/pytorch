import torch


__version__ = torch.version.cuda

from .cuda import *  # noqa: F403
