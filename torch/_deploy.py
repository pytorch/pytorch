import io
import torch
from torch.package._package_pickler import create_pickler
from torch.package._package_unpickler import PackageUnpickler
from torch.package import sys_importer, OrderedImporter, PackageImporter, Importer
from torch.serialization import _maybe_decode_ascii

def _save_storages(importer, obj):
    serialized_storages = []
    serialized_dtypes = []

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
        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    importer = importer if isinstance(importer, torch.package.PackageImporter) else None
    importers: Importer
    if importer is not None:
        importers = OrderedImporter(importer, sys_importer)
    else:
        importers = sys_importer
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

        assert typename == 'storage', \
            f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        return serialized_storages[data[0]]


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
