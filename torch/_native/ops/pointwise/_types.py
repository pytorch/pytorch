from collections.abc import Hashable
from typing import Literal

import torch


Shape = tuple[int, ...]
DTypes = tuple[torch.dtype, ...]
RowGap = tuple[int, int, int]
TransposeSpec = tuple[int, int, int, int, int, int]
PromotionOperands = tuple[tuple[torch.dtype, bool], ...]
CacheKey = tuple[Hashable, ...]
Number = bool | int | float | complex
Mode = str | None
Load = Literal["direct", "tma"]
Path = Literal["vec", "strided", "tma"]
OperatorResult = torch.Tensor | tuple[torch.Tensor, ...]
