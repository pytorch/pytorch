import torch
from torch import Tensor

import torch._prims.utils as utils
from torch._prims.utils import (
    TensorLike,
    TensorLikeType,
    TensorMeta,
    ShapeType,
    getnvFuserDtype,
    DimsSequenceType,
    StrideType,
)

from typing import Sequence, Optional, Union, Callable, List, Tuple, Any
from numbers import Number
from functools import reduce, partial
from enum import Enum
import operator

__all__ = [
    "_slice_meta",
    "_slice_aten",
    "_slice_doc",
]
