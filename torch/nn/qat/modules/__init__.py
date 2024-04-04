# flake8: noqa: F401
r"""QAT Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.modules` instead.
"""
from torch.ao.nn.qat.modules.linear import Linear
from torch.ao.nn.qat.modules.conv import Conv1d
from torch.ao.nn.qat.modules.conv import Conv2d
from torch.ao.nn.qat.modules.conv import Conv3d
from torch.ao.nn.qat.modules.embedding_ops import EmbeddingBag, Embedding

from . import conv
from . import embedding_ops
from . import linear

__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
