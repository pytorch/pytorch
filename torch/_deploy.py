import io
import torch
from torch.package._package_pickler import create_pickler
from torch.package._package_unpickler import PackageUnpickler
from torch.package import sys_importer, OrderedImporter, PackageImporter, Importer
from torch.serialization import _maybe_decode_ascii

def _save_storages(importer, obj):
    serialized_storages = []
    serialized_dtypes = []

    importer = importer if isinstance(importer, torch.package.PackageImporter) else None
    importers: Importer
    if importer is not None:
        importers = OrderedImporter(importer, sys_importer)
    else:
        importers = sys_importer

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if torch.is_storage(obj):
            serialized_storages.append(obj)
            serialized_dtypes.append(obj.dtype)
            return ('storage', len(serialized_storages) - 1)

        if hasattr(obj, "__reduce_deploy__"):
            if _serialized_reduces.get(id(obj)) is None:
                _serialized_reduces[id(obj)] = (
                    "reduce_deploy",
                    id(obj),
                    *obj.__reduce_deploy__(importers),
                )
            return _serialized_reduces[id(obj)]

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    pickler = create_pickler(data_buf, importers)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    return data_value, serialized_storages, serialized_dtypes, importer.zip_reader if importer else None

def _load_storages(id, zip_reader, obj_bytes, serialized_storages):

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == 'storage':
            return serialized_storages[data[0]]

        if typename == 'reduce_deploy':
            reduce_id, func, args = data
            if reduce_id not in _loaded_reduces:
                _loaded_reduces[reduce_id] = func(_raw_packages[zip_reader], *args)
            return _loaded_reduces[reduce_id]

        return None


    importer: Importer
    if zip_reader is not None:
        importer = OrderedImporter(_get_package(zip_reader), sys_importer)
    else:
        importer = sys_importer

    unpickler = PackageUnpickler(importer, io.BytesIO(obj_bytes))
    unpickler.persistent_load = persistent_load
    result = _deploy_objects[id] = unpickler.load()
    return result

def _get_package(zip_reader):
    if zip_reader not in _raw_packages:
        _raw_packages[zip_reader] = PackageImporter(zip_reader)
    return _raw_packages[zip_reader]


_raw_packages: dict = {}
_deploy_objects: dict = {}
_serialized_reduces: dict = {}
_loaded_reduces: dict = {}
