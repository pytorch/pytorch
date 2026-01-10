# mypy: allow-untyped-defs
import collections
from itertools import repeat
from typing import Any


__all__ = ["consume_prefix_in_state_dict_if_present"]


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_single = _ntuple(1, "_single")
_pair = _ntuple(2, "_pair")
_triple = _ntuple(3, "_triple")
_quadruple = _ntuple(4, "_quadruple")


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _list_with_default(out_size: list[int], defaults: list[int]) -> list[int]:
    import torch

    if isinstance(out_size, (int, torch.SymInt)):
        # pyrefly: ignore [bad-return]
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(f"Input dimension should be at least {len(out_size) + 1}")
    return [
        v if v is not None else d
        for v, d in zip(out_size, defaults[-len(out_size) :], strict=False)
    ]


def consume_prefix_in_state_dict_if_present(
    state_dict: dict[str, Any],
    prefix: str,
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    .. note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = list(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if hasattr(state_dict, "_metadata"):
        keys = list(state_dict._metadata.keys())
        for key in keys:
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.
            if len(key) == 0:
                continue
            # handling both, 'module' case and  'module.' cases
            if key == prefix.replace(".", "") or key.startswith(prefix):
                newkey = key[len(prefix) :]
                state_dict._metadata[newkey] = state_dict._metadata.pop(key)
