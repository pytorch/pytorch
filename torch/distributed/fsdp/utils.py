from typing import Dict, List, Tuple, Union, Any, Callable, Set

import torch

from collections import OrderedDict

"""Useful functions to deal with tensor types with other python container types."""


def _apply_to_tensors(
    fn: Callable, container: Union[torch.Tensor, Dict, List, Tuple, Set]
) -> Any:
    """Recursively apply to all tensor in different kinds of container types."""

    def apply(x: Union[torch.Tensor, Dict, List, Tuple, Set, OrderedDict]) -> Any:
        if torch.is_tensor(x):
            return fn(x)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)


def _replace_by_prefix(
    state_dict: Union[Dict[str, torch.Tensor], "OrderedDict[str, torch.Tensor]"],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]
