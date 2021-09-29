import io
import pickle

from torch.utils.data import IterableDataset

from typing import Any, Dict

reduce_ex_hook = None


def stub_unpickler():
    return "STUB"

# TODO(VitalyFedyunin): Make sure it works without dill module installed
def list_connected_datapipes(scan_obj):

    f = io.BytesIO()
    p = pickle.Pickler(f)  # Not going to work for lambdas, but dill infinite loops on typing and can't be used as is

    def stub_pickler(obj):
        return stub_unpickler, ()

    captured_connections = []

    def reduce_hook(obj):
        if obj == scan_obj:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            return stub_unpickler, ()

    # TODO(VitalyFedyunin):  Better do it as `with` context for safety
    IterableDataset.set_reduce_ex_hook(reduce_hook)
    p.dump(scan_obj)
    IterableDataset.set_reduce_ex_hook(None)
    return captured_connections


def traverse(datapipe):
    items = list_connected_datapipes(datapipe)
    d: Dict[Any, Any] = {datapipe: {}}
    for item in items:
        d[datapipe].update(traverse(item))
    return d
