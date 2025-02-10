# mypy: allow-untyped-defs
import io
import pickle
import warnings
from collections.abc import Collection
from typing import Optional, Union

from torch.utils._import_utils import dill_available
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe


__all__ = ["traverse", "traverse_dps"]

DataPipe = Union[IterDataPipe, MapDataPipe]
DataPipeGraph = dict[int, tuple[DataPipe, "DataPipeGraph"]]


def _stub_unpickler():
    return "STUB"


# TODO(VitalyFedyunin): Make sure it works without dill module installed
def _list_connected_datapipes(
    scan_obj: DataPipe, only_datapipe: bool, cache: set[int]
) -> list[DataPipe]:
    f = io.BytesIO()
    p = pickle.Pickler(
        f
    )  # Not going to work for lambdas, but dill infinite loops on typing and can't be used as is
    if dill_available():
        from dill import Pickler as dill_Pickler

        d = dill_Pickler(f)
    else:
        d = None

    captured_connections = []

    def getstate_hook(ori_state):
        state = None
        if isinstance(ori_state, dict):
            state = {}
            for k, v in ori_state.items():
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state[k] = v
        elif isinstance(ori_state, (tuple, list)):
            state = []  # type: ignore[assignment]
            for v in ori_state:
                if isinstance(v, (IterDataPipe, MapDataPipe, Collection)):
                    state.append(v)  # type: ignore[attr-defined]
        elif isinstance(ori_state, (IterDataPipe, MapDataPipe, Collection)):
            state = ori_state  # type: ignore[assignment]
        return state

    def reduce_hook(obj):
        if obj == scan_obj or id(obj) in cache:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            # Adding id to remove duplicate DataPipe serialized at the same level
            cache.add(id(obj))
            return _stub_unpickler, ()

    datapipe_classes: tuple[type[DataPipe]] = (IterDataPipe, MapDataPipe)  # type: ignore[assignment]

    try:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(reduce_hook)
            if only_datapipe:
                cls.set_getstate_hook(getstate_hook)
        try:
            p.dump(scan_obj)
        except (pickle.PickleError, AttributeError, TypeError):
            if dill_available():
                d.dump(scan_obj)
            else:
                raise
    finally:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(None)
            if only_datapipe:
                cls.set_getstate_hook(None)
        if dill_available():
            from dill import extend as dill_extend

            dill_extend(False)  # Undo change to dispatch table
    return captured_connections


def traverse_dps(datapipe: DataPipe) -> DataPipeGraph:
    r"""
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    This only looks into the attribute from each DataPipe that is either a
    DataPipe and a Python collection object such as ``list``, ``tuple``,
    ``set`` and ``dict``.

    Args:
        datapipe: the end DataPipe of the graph
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
    cache: set[int] = set()
    return _traverse_helper(datapipe, only_datapipe=True, cache=cache)


def traverse(datapipe: DataPipe, only_datapipe: Optional[bool] = None) -> DataPipeGraph:
    r"""
    Traverse the DataPipes and their attributes to extract the DataPipe graph.

    [Deprecated]
    When ``only_dataPipe`` is specified as ``True``, it would only look into the
    attribute from each DataPipe that is either a DataPipe and a Python collection object
    such as ``list``, ``tuple``, ``set`` and ``dict``.

    Note:
        This function is deprecated. Please use `traverse_dps` instead.

    Args:
        datapipe: the end DataPipe of the graph
        only_datapipe: If ``False`` (default), all attributes of each DataPipe are traversed.
          This argument is deprecating and will be removed after the next release.
    Returns:
        A graph represented as a nested dictionary, where keys are ids of DataPipe instances
        and values are tuples of DataPipe instance and the sub-graph
    """
    msg = (
        "`traverse` function and will be removed after 1.13. "
        "Please use `traverse_dps` instead."
    )
    if not only_datapipe:
        msg += " And, the behavior will be changed to the equivalent of `only_datapipe=True`."
    warnings.warn(msg, FutureWarning)
    if only_datapipe is None:
        only_datapipe = False
    cache: set[int] = set()
    return _traverse_helper(datapipe, only_datapipe, cache)


# Add cache here to prevent infinite recursion on DataPipe
def _traverse_helper(
    datapipe: DataPipe, only_datapipe: bool, cache: set[int]
) -> DataPipeGraph:
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise RuntimeError(
            f"Expected `IterDataPipe` or `MapDataPipe`, but {type(datapipe)} is found"
        )

    dp_id = id(datapipe)
    if dp_id in cache:
        return {}
    cache.add(dp_id)
    # Using cache.copy() here is to prevent the same DataPipe pollutes the cache on different paths
    items = _list_connected_datapipes(datapipe, only_datapipe, cache.copy())
    d: DataPipeGraph = {dp_id: (datapipe, {})}
    for item in items:
        # Using cache.copy() here is to prevent recursion on a single path rather than global graph
        # Single DataPipe can present multiple times in different paths in graph
        d[dp_id][1].update(_traverse_helper(item, only_datapipe, cache.copy()))
    return d
