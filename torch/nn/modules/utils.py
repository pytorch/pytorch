import collections
from itertools import repeat
from typing import List, Dict, Any

__all__ = ['consume_prefix_in_state_dict_if_present']


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


def _list_with_default(out_size: List[int], defaults: List[int]) -> List[int]:
    import torch
    if isinstance(out_size, (int, torch.SymInt)):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(
            f"Input dimension should be at least {len(out_size) + 1}"
        )
    return [
        v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size) :])
    ]


def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys_to_move = [key for key in state_dict.keys()]
    for key in keys_to_move:
        # Key starts with the prefix
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            state_dict[new_key] = state_dict.pop(key)
           
        # Key corresponds to the metadata
        elif key == "_metadata":
            metadata = state_dict["_metadata"]
            metadata_keys = [key for key in state_dict["_metadata"].keys()]
            for metadata_key in metadata_keys:
                if len(metadata_key) == 0:
                    continue
                new_key = metadata_key
                if metadata_key.startswith(prefix):
                    new_key = metadata_key[len(prefix):]
                metadata[new_key] = metadata.pop(metadata_key)
               
            # while the order is kept within _metadata, we need to reinsert it
            state_dict["_metadata"] = state_dict.pop("_metadata")
           
        # if any of the previous options did not work, we reinsert it as it is        
        else:
            state_dict[key] = state_dict.pop(key)