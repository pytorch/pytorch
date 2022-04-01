from typing import Dict, List, Tuple, Union, Any, Callable, Set

import torch


"""Useful functions to deal with tensor types with other python container types."""


def _apply_to_tensors(
    fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]
) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x: Union[torch.Tensor, Dict, List, Tuple, Set]) -> Any:
        if torch.is_tensor(x):
            return fn(x)
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)
