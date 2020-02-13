from typing import Iterable
from ... import Tensor


def parameters_to_vector(parameters: Iterable[Tensor]) -> Tensor: ...


def vector_to_parameters(vec: Tensor, parameters: Iterable[Tensor]) -> None: ...
