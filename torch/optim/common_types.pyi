from typing import Any, Dict, Iterable, Union

from torch import Tensor

_params_t = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
