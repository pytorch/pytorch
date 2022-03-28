import io
import pickle

from torch.utils.data import IterDataPipe, MapDataPipe

from typing import Any, Dict, Union, Type

DataPipe = Union[IterDataPipe, MapDataPipe]
reduce_ex_hook = None


def stub_unpickler():
    return "STUB"


# TODO(VitalyFedyunin): Make sure it works without dill module installed
def list_connected_datapipes(scan_obj, only_datapipe):

    f = io.BytesIO()
    p = pickle.Pickler(f)  # Not going to work for lambdas, but dill infinite loops on typing and can't be used as is

    def stub_pickler(obj):
        return stub_unpickler, ()

    captured_connections = []

    def getstate_hook(obj):
        state = {}
        for k, v in obj.__dict__.items():
            if isinstance(v, (IterDataPipe, MapDataPipe, tuple)):
                state[k] = v
        return state

    def reduce_hook(obj):
        if obj == scan_obj:
            raise NotImplementedError
        else:
            captured_connections.append(obj)
            return stub_unpickler, ()

    if isinstance(scan_obj, MapDataPipe):
        cls: Type[DataPipe] = MapDataPipe
    else:
        cls = IterDataPipe

    try:
        cls.set_reduce_ex_hook(reduce_hook)
        if only_datapipe:
            cls.set_getstate_hook(getstate_hook)
        p.dump(scan_obj)
    except AttributeError:  # unpickable DataPipesGraph
        pass  # TODO(VitalyFedyunin): We need to tight this requirement after migrating from old DataLoader
    finally:
        cls.set_reduce_ex_hook(None)
        if only_datapipe:
            cls.set_getstate_hook(None)
    return captured_connections


def traverse(datapipe, only_datapipe=False):
    if not isinstance(datapipe, (IterDataPipe, MapDataPipe)):
        raise RuntimeError("Expected `IterDataPipe` or `MapDataPipe`, but {} is found".format(type(datapipe)))

    items = list_connected_datapipes(datapipe, only_datapipe)
    d: Dict[DataPipe, Any] = {datapipe: {}}
    for item in items:
        d[datapipe].update(traverse(item, only_datapipe))
    return d
