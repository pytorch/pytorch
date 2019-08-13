from typing import Union, Iterable
from ... import Tensor

_tensor_or_tensors = Union[Tensor, Iterable[Tensor]]


def clip_grad_norm_(parameters: _tensor_or_tensors, max_norm: float, norm_type: float = ...): ...


def clip_grad_value_(parameters: _tensor_or_tensors, clip_value: float): ...
