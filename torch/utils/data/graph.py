import io
import pickle

from torch.utils.data import IterDataPipe, MapDataPipe
from torch.utils.data._utils.serialization import DILL_AVAILABLE

from typing import Any, Dict, Set, Tuple, Type, Union

__all__ = ["traverse", ]

DataPipe = Union[IterDataPipe, MapDataPipe]
reduce_ex_hook = None


def _stub_unpickler():
    return "STUB"


# TODO(VitalyFedyunin): Make sure it works without dill module installed
def _list_connected_datapipes(scan_obj, only_datapipe, cache):
    f = io.BytesIO()
    p = pickle.Pickler(f)  # Not going to work for lambdas, but dill infinite loops on typing and can't be used as is
    if DILL_AVAILABLE:
        from dill import Pickler as dill_Pickler
        d = dill_Pickler(f)
    else:
        d = None

    def stub_pickler(obj):
        return _stub_unpickler, ()

    captured_connections = []

    def getstate_hook(obj):
        state = {}
        for k, v in obj.__dict__.items():
            if isinstance(v, (IterDataPipe, MapDataPipe, tuple)):
                state[k] = v
        return state

    def reduce_hook(obj):
        if obj == scan_obj or obj in cache:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            return _stub_unpickler, ()

    datapipe_classes: Tuple[Type[DataPipe]] = (IterDataPipe, MapDataPipe)  # type: ignore[assignment]

    try:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(reduce_hook)
            if only_datapipe:
                cls.set_getstate_hook(getstate_hook)
        try:
            p.dump(scan_obj)
        except (pickle.PickleError, AttributeError, TypeError):
            if DILL_AVAILABLE:
                d.dump(scan_obj)
            else:
                raise
    finally:
        for cls in datapipe_classes:
            cls.set_reduce_ex_hook(None)
            if only_datapipe:
                cls.set_getstate_hook(None)
        if DILL_AVAILABLE:
            from dill import extend as dill_extend
            dill_extend(False)  # Undo change to dispatch table
    return captured_connections


def traverse(datapipe, only_datapipe=False):
    cache: Set[DataPipe] = set()
    return _traverse_helper(datapipe, only_datapipe, cache)


# Add cache here to prevent infinite recursion on DataPipe
def _traverse_helper(datapipe, only_datapipe, cache):
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise RuntimeError("Expected `IterDataPipe` or `MapDataPipe`, but {} is found".format(type(datapipe)))

    cache.add(datapipe)
    items = _list_connected_datapipes(datapipe, only_datapipe, cache)
    d: Dict[DataPipe, Any] = {datapipe: {}}
    for item in items:
        # Using cache.copy() here is to prevent recursion on a single path rather than global graph
        # Single DataPipe can present multiple times in different paths in graph
        d[datapipe].update(_traverse_helper(item, only_datapipe, cache.copy()))
    return d
