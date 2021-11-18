import io
import pickle

from torch.utils.data import IterableDataset

from typing import Any, Dict, Generator

reduce_ex_hook = None

PRIMITIVE = (int, float, complex, str, bytes, bytearray, Generator)


def stub_unpickler():
    return "STUB"


# TODO(VitalyFedyunin): Make sure it works without dill module installed
def list_connected_datapipes(scan_obj, exclude_primitive):

    f = io.BytesIO()
    p = pickle.Pickler(f)  # Not going to work for lambdas, but dill infinite loops on typing and can't be used as is

    def stub_pickler(obj):
        return stub_unpickler, ()

    captured_connections = []

    def getstate_hook(obj):
        state = {}
        for k, v in obj.__dict__.items():
            if callable(v) or isinstance(v, PRIMITIVE):
                continue
            state[k] = v
        return state

    def reduce_hook(obj):
        if obj == scan_obj:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            return stub_unpickler, ()

    # TODO(VitalyFedyunin):  Better do it as `with` context for safety
    IterableDataset.set_reduce_ex_hook(reduce_hook)
    if exclude_primitive:
        IterableDataset.set_getstate_hook(getstate_hook)
    p.dump(scan_obj)
    IterableDataset.set_reduce_ex_hook(None)
    if exclude_primitive:
        IterableDataset.set_getstate_hook(None)
    return captured_connections


def traverse(datapipe, exclude_primitive=False):
    if not isinstance(datapipe, IterableDataset):
        raise RuntimeError("Expected `IterDataPipe`, but {} is found".format(type(datapipe)))

    items = list_connected_datapipes(datapipe, exclude_primitive)
    d: Dict[Any, Any] = {datapipe: {}}
    for item in items:
        d[datapipe].update(traverse(item, exclude_primitive))
    return d
