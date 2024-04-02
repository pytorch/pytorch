# Copyright (c) Meta Platforms, Inc. and affiliates
import torch

from typing import (
    Callable,
    Collection,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
)
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DTensor

PATH_ITEM = Union[str, int]
OBJ_PATH = Tuple[PATH_ITEM, ...]
T = TypeVar("T")

STATE_DICT_ITEM = object
CONTAINER_TYPE = MutableMapping[PATH_ITEM, STATE_DICT_ITEM]

__all__ = ["traverse_state_dict", "set_element", "get_element", "print_tensor"]


def _keep_visiting_tensors(value: STATE_DICT_ITEM) -> bool:
    return isinstance(value, torch.Tensor)


# TODO: update docstring for traverse.py
def traverse_state_dict(
    state_dict: STATE_DICT_TYPE,
    visitor: Callable[[OBJ_PATH, STATE_DICT_ITEM], None],
    keep_traversing: Callable[[STATE_DICT_ITEM], bool] = _keep_visiting_tensors,
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.

    Traversal is short-circuited when if finds a collection for which ``keep_visiting_tensors`` evaluates
    to false for all elements.
    By default, all collections with at least one ``torch.Tensor`` element are traversed.
    Visitor takes a path argument that is a tuple of the keys used to reach it.
    """
    # a value is terminal if it has no other containers values inside it
    def _is_terminal(value: STATE_DICT_ITEM) -> bool:
        values: Collection[STATE_DICT_ITEM]
        if isinstance(value, Mapping):
            values = value.values()
        elif isinstance(value, list):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list)) and not _is_terminal(entry):
                return False
            if keep_traversing is not None and keep_traversing(entry):
                return False
        return True

    def _traverse_obj(path: OBJ_PATH, value: STATE_DICT_ITEM) -> None:
        if _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def set_element(
    root_dict: STATE_DICT_TYPE, path: OBJ_PATH, value: STATE_DICT_ITEM
) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    cur_container = cast(CONTAINER_TYPE, root_dict)

    def extend_list(lst: List[STATE_DICT_ITEM], idx: int) -> None:
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        key = path[i]
        def_val = cast(STATE_DICT_ITEM, {} if type(key) == str else [])

        if isinstance(cur_container, Mapping):
            cur_container = cast(
                CONTAINER_TYPE, cur_container.setdefault(prev_key, def_val)
            )
        else:
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    key = path[-1]
    if type(key) == int:
        extend_list(cast(List[STATE_DICT_ITEM], cur_container), key)

    cur_container[key] = value


def get_element(
    root_dict: STATE_DICT_TYPE,
    path: OBJ_PATH,
    default_value: Optional[T] = None,
) -> Optional[T]:
    """Retrieve the value at ``path``from ``root_dict``, returning ``default_value`` if not found."""
    cur_value = cast(CONTAINER_TYPE, root_dict)
    for part in path:
        if type(part) is int:
            if not isinstance(cur_value, list) or len(cur_value) < part:
                return default_value
        elif not isinstance(cur_value, Mapping) or part not in cur_value:
            return default_value

        cur_value = cast(CONTAINER_TYPE, cur_value[part])
    return cast(Optional[T], cur_value)


def _print_nested(
    value: STATE_DICT_ITEM,
    prefix: str = "",
    print_fun: Callable[[str], None] = print,
) -> None:
    if type(value) is ShardedTensor:
        print_fun(f"{prefix} ShardedTensor size: {value.size()}")
        for shard in value.local_shards():
            _print_nested(
                shard.tensor,
                f"{shard.metadata.shard_offsets} ",
                print_fun=print_fun,
            )
    elif type(value) is (DTensor):
        print_fun(f"{prefix} DistributedTensor size: {value.size()}")
        # TODO: add local offset for _local_tensor in print_nested.
        _print_nested(
            value._local_tensor,
            print_fun=print_fun,
        )
    elif isinstance(value, torch.Tensor):
        print_fun(f"{prefix} Tensor size: {value.size()}")
    else:
        print_fun(f"{prefix} Type: {type(value)}")


def print_tensor(
    path: OBJ_PATH,
    value: STATE_DICT_ITEM,
    print_fun: Callable[[str], None] = print,
) -> None:
    """
    Use this callback with traverse_state_dict to print its content.

    By default the content is printed using the builtin ``print`` but this can
    be change by passing a different ``print_fun` callable.
    """
    _print_nested(value, prefix=str(path), print_fun=print_fun)
