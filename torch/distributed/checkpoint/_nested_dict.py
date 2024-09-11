# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, Tuple

from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

from . import _version
from ._traverse import (
    OBJ_PATH,
    set_element,
    STATE_DICT_ITEM,
    traverse_state_dict,
    traverse_state_dict_v_2_3,
)


"""
TODO:
Need to add ability to handle tuple, OrderedDict, NamedTuple.
Update mappings from dict to a class.
Change set_element to recreate the right type for tuple, OrderedDict, and NamedTuple.
"""


FLATTEN_MAPPING = Dict[str, OBJ_PATH]


# TODO: Update Docstring for nested_dict.py
def flatten_state_dict(
    state_dict: STATE_DICT_TYPE,
) -> Tuple[STATE_DICT_TYPE, FLATTEN_MAPPING]:
    """
    Flatten ``state_dict`` made of nested dicts and lists into a top level dictionary.

    Use ``unflatten_state_dict`` to revert this process.
    Returns:
        A tuple with the flatten state_dict and a mapping from original to new state_dict.
    N.B. The new keys are derived from the object paths, joined by dot.
        For example: ``{ 'a': {'b':...}}`` results in the key `a.b`.
    """
    flattened: STATE_DICT_TYPE = {}
    mappings: FLATTEN_MAPPING = {}

    def flat_copy(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    # We started to flatten dictionary since v2.4. But in order to not break
    # the checkpoints that were saved before v2.4, we need to keep the old
    # traversal so that we can reconstruct those checkpoints.
    use_v_2_3 = (
        _version._derived_version is not None and _version._derived_version == "2_3"
    )
    if use_v_2_3:
        traverse_state_dict_v_2_3(state_dict, flat_copy)
    else:
        traverse_state_dict(state_dict, flat_copy)
    return flattened, mappings


def unflatten_state_dict(
    state_dict: STATE_DICT_TYPE, mapping: FLATTEN_MAPPING
) -> STATE_DICT_TYPE:
    """Restore the original nested state_dict according to ``mapping`` and the flattened ``state_dict``."""
    nested: STATE_DICT_TYPE = {}
    for key, value in state_dict.items():
        set_element(nested, mapping[key], value)
    return nested
