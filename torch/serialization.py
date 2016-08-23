import os
import sys
import tempfile
import tarfile
import pickle
import shutil
import struct
from contextlib import closing, contextmanager
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch

LONG_SIZE = struct.Struct('=l').size
INT_SIZE = struct.Struct('=i').size
SHORT_SIZE = struct.Struct('=h').size

def _add_to_tar(fn, tar_file, name):
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    fn(tmp_file)
    tmp_file.close()

    tar_file.add(tmp_file.name, arcname=name)
    if os.path.isfile(tmp_file.name):
        os.remove(tmp_file.name)


@contextmanager
def mkdtemp():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


# TODO: choose pickle protocol
def save(obj, f, pickle_module=pickle, pickle_protocol=pickle.DEFAULT_PROTOCOL):
    serialized_tensors = {}
    serialized_storages = {}

    def persistent_id(obj):
        if torch.isTensor(obj):
            serialized_tensors[obj._cdata] = obj
            return str(obj._cdata)
        elif torch.isStorage(obj):
            serialized_storages[obj._cdata] = obj
            return str(obj._cdata)
        return None

    def save_tensors(f):
        pickle_module.dump(len(serialized_tensors), f, protocol=pickle_protocol)
        for key, tensor in serialized_tensors.items():
            storage = tensor.storage()
            serialized_storages[storage._cdata] = storage

            pickle_module.dump((key, type(tensor), storage._cdata), f, protocol=pickle_protocol)
            f.flush()
            tensor._write_metadata(f)

    def save_storages(f):
        pickle_module.dump(len(serialized_storages), f, protocol=pickle_protocol)
        for key, storage in serialized_storages.items():
            pickle_module.dump((key, type(storage)), f, protocol=pickle_protocol)
            f.flush()
            storage._write_file(f)

    def pickle_objects(f):
        pickler = pickle_module.Pickler(f, protocol=pickle_protocol)
        pickler.persistent_id = persistent_id
        pickler.dump(obj)

    def save_sys_info(f):
        sys_info = dict(
            protocol_version=1000,
            little_endian=sys.byteorder == 'little',
            type_sizes = dict(
                short=SHORT_SIZE,
                int=INT_SIZE,
                long=LONG_SIZE,
            ),
        )
        pickle_module.dump(sys_info, f, protocol=pickle_protocol)

    with closing(tarfile.open(fileobj=f, mode='w:', format=tarfile.PAX_FORMAT)) as tar:
        _add_to_tar(save_sys_info, tar, 'sys_info')
        _add_to_tar(pickle_objects, tar, 'pickle')
        _add_to_tar(save_tensors, tar, 'tensors')
        _add_to_tar(save_storages, tar, 'storages')


def load(f, pickle_module=pickle):
    deserialized_objects = {}

    def persistent_load(saved_id):
        return deserialized_objects[int(saved_id)]

    with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, \
         mkdtemp() as tmpdir:

        def extract(name, init):
            tar.extract(name, path=tmpdir)
            with open(os.path.join(tmpdir, name), 'rb', 0) as f:
                num_storages = pickle_module.load(f)
                for i in range(num_storages):
                    args = pickle_module.load(f)
                    key, args = args[0], args[1:]
                    obj = init(f, *args)
                    deserialized_objects[key] = obj

        extract('storages', lambda f, storage_type: storage_type._new_with_file(f))
        extract('tensors', lambda f, tensor_type, storage_id: \
                tensor_type._new_with_metadata_file(f, deserialized_objects[storage_id]))

        pickle_file = tar.extractfile('pickle')
        unpickler = pickle_module.Unpickler(pickle_file)
        unpickler.persistent_load = persistent_load
        result = unpickler.load()
        return result

