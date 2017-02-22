import difflib
import inspect
import os
import shutil
import struct
import sys
import torch
import tarfile
import tempfile
import warnings
from contextlib import closing, contextmanager
from ._utils import _import_dotted_name
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

DEFAULT_PROTOCOL = 2

LONG_SIZE = struct.Struct('=l').size
INT_SIZE = struct.Struct('=i').size
SHORT_SIZE = struct.Struct('=h').size


class SourceChangeWarning(Warning):
    pass


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


_package_registry = []


def register_package(priority, tagger, deserializer):
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()


def _cpu_tag(obj):
    if type(obj).__module__ == 'torch':
        return 'cpu'


def _cuda_tag(obj):
    if type(obj).__module__ == 'torch.cuda':
        return 'cuda:' + str(obj.get_device())


def _cpu_deserialize(obj, location):
    if location == 'cpu':
        return obj


def _cuda_deserialize(obj, location):
    if location.startswith('cuda'):
        device_id = max(int(location[5:]), 0)
        return obj.cuda(device_id)


register_package(10, _cpu_tag, _cpu_deserialize)
register_package(20, _cuda_tag, _cuda_deserialize)


def location_tag(storage):
    for _, tagger, _ in _package_registry:
        location = tagger(storage)
        if location:
            return location
    raise RuntimeError("don't know how to determine data location of " +
                       torch.typename(storage))


def default_restore_location(storage, location):
    for _, _, fn in _package_registry:
        result = fn(storage, location)
        if result is not None:
            return result
    raise RuntimeError("don't know how to restore data location of " +
                       torch.typename(storage) + " (tagged with " + location + ")")


def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def storage_to_tensor_type(storage):
    storage_type = type(storage)
    module = _import_dotted_name(storage_type.__module__)
    return getattr(module, storage_type.__name__.replace('Storage', 'Tensor'))


def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
    """Saves an object to a disk file.

    Args:
        obj: saved object
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol
    """
    new_fd = False
    if isinstance(f, str) or isinstance(f, unicode):
        new_fd = True
        f = open(f, "wb")
    try:
        return _save(obj, f, pickle_module, pickle_protocol)
    finally:
        if new_fd:
            f.close()


def _save(obj, f, pickle_module, pickle_protocol):
    import torch.nn as nn
    serialized_tensors = {}
    serialized_storages = {}
    serialized_container_types = {}

    def persistent_id(obj):
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if obj in serialized_container_types:
                return None
            serialized_container_types[obj] = True
            source_file = source = None
            try:
                source_file = inspect.getsourcefile(obj)
                source = inspect.getsource(obj)
            except (TypeError, IOError):
                warnings.warn("Couldn't retrieve source code for container of "
                              "type " + obj.__name__ + ". It won't be checked "
                              "for correctness upon loading.")
            return (obj, source_file, source)
        if torch.is_tensor(obj):
            serialized_tensors[obj._cdata] = obj
            return str(obj._cdata)
        elif torch.is_storage(obj):
            serialized_storages[obj._cdata] = obj
            return str(obj._cdata)
        return None

    def save_tensors(f):
        pickle_module.dump(len(serialized_tensors), f, protocol=pickle_protocol)
        for key, tensor in serialized_tensors.items():
            storage = tensor.storage()
            if storage is not None:
                storage_id = storage._cdata
                serialized_storages[storage_id] = storage
            else:
                storage_id = None

            pickle_module.dump((key, storage_id, type(tensor)), f,
                               protocol=pickle_protocol)
            f.flush()
            tensor._write_metadata(f)

    def save_storages(f):
        storage_views = []
        storage_views_roots = {}

        for key, storage in serialized_storages.items():
            root, offset = storage._root_storage()
            if root is not storage:
                storage_views_roots[root._cdata] = root
                storage_views.append((storage._cdata, root._cdata, offset,
                                      storage.size()))
        for view_info in storage_views:
            del serialized_storages[view_info[0]]
        serialized_storages.update(storage_views_roots)

        pickle_module.dump(len(serialized_storages), f, protocol=pickle_protocol)
        for key, storage in serialized_storages.items():
            location = location_tag(storage)
            storage_type = normalize_storage_type(type(storage))
            pickle_module.dump((key, location, storage_type), f,
                               protocol=pickle_protocol)
            f.flush()
            storage._write_file(f)

        pickle_module.dump(storage_views, f, protocol=pickle_protocol)

    def pickle_objects(f):
        pickler = pickle_module.Pickler(f, protocol=pickle_protocol)
        pickler.persistent_id = persistent_id
        pickler.dump(obj)

    def save_sys_info(f):
        sys_info = dict(
            protocol_version=1000,
            little_endian=sys.byteorder == 'little',
            type_sizes=dict(
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


def load(f, map_location=None, pickle_module=pickle):
    """Loads an object saved with torch.save from a disk file.

    torch.load can dynamically remap storages to be loaded on a different device
    using the map_location argument. If it's a callable, it will be called with
    two arguments: storage and location tag. It's expected to either return a
    storage that's been moved to a different location, or None (and the location
    will be resolved using the default method). If this argument is a dict it's
    expected to be a mapping from location tags used in a file, to location
    tags of the current system.

    By default the location tags are 'cpu' for host tensors and 'cuda:device_id'
    (e.g. 'cuda:2') for cuda tensors. User extensions can register their own
    tagging and deserialization methods using register_package.

    Args:
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name
        map_location: a function or a dict specifying how to remap storage locations
        pickle_module: module used for unpickling metadata and objects (has to match
            the pickle_module used to serialize file)
    """
    new_fd = False
    if isinstance(f, str) or isinstance(f, unicode):
        new_fd = True
        f = open(f, 'rb')
    try:
        return _load(f, map_location, pickle_module)
    finally:
        if new_fd:
            f.close()


def _load(f, map_location, pickle_module):
    deserialized_objects = {}

    if map_location is None:
        restore_location = default_restore_location
    elif isinstance(map_location, dict):
        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)
    else:
        def restore_location(storage, location):
            result = map_location(storage, location)
            if not result:
                result = default_restore_location(storage, location)
            return result

    def _check_container_source(container_type, source_file, original_source):
        current_source = inspect.getsource(container_type)
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + '.patch'
                diff = difflib.unified_diff(
                    current_source.split('\n'),
                    original_source.split('\n'),
                    source_file,
                    source_file, lineterm="")
                lines = '\n'.join(diff)
                try:
                    with open(file_name, 'a+') as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise IOError
                    msg = ("Saved a reverse patch to " + file_name + ". "
                           "Run `patch -p0 < " + file_name + "` to revert your "
                           "changes.")
                except IOError:
                    msg = ("Tried to save a patch, but couldn't create a "
                           "writable file " + file_name + ". Make sure it "
                           "doesn't exist and your working directory is "
                           "writable.")
            else:
                msg = ("you can retrieve the original source code by "
                       "accessing the object's source attribute or set "
                       "`torch.nn.Module.dump_patches = True` and use the "
                       "patch tool to revert the changes.")
            msg = ("source code of class '{}' has changed. {}"
                   .format(torch.typename(container_type), msg))
            warnings.warn(msg, SourceChangeWarning)

    def persistent_load(saved_id):
        if isinstance(saved_id, tuple):
            # Ignore containers that don't have any sources saved
            if all(saved_id[1:]):
                _check_container_source(*saved_id)
            return saved_id[0]
        return deserialized_objects[int(saved_id)]

    with closing(tarfile.open(fileobj=f, mode='r:', format=tarfile.PAX_FORMAT)) as tar, \
            mkdtemp() as tmpdir:

        tar.extract('storages', path=tmpdir)
        with open(os.path.join(tmpdir, 'storages'), 'rb', 0) as f:
            num_storages = pickle_module.load(f)
            for i in range(num_storages):
                args = pickle_module.load(f)
                key, location, storage_type = args
                obj = storage_type._new_with_file(f)
                obj = restore_location(obj, location)
                deserialized_objects[key] = obj

            storage_views = pickle_module.load(f)
            for target_cdata, root_cdata, offset, size in storage_views:
                root = deserialized_objects[root_cdata]
                deserialized_objects[target_cdata] = root[offset:offset + size]

        tar.extract('tensors', path=tmpdir)
        with open(os.path.join(tmpdir, 'tensors'), 'rb', 0) as f:
            num_tensors = pickle_module.load(f)
            for i in range(num_tensors):
                args = pickle_module.load(f)
                key, storage_id, original_tensor_type = args
                storage = deserialized_objects[storage_id]
                tensor_type = storage_to_tensor_type(storage)
                tensor = tensor_type._new_with_metadata_file(f, storage)
                deserialized_objects[key] = tensor

        pickle_file = tar.extractfile('pickle')
        unpickler = pickle_module.Unpickler(pickle_file)
        unpickler.persistent_load = persistent_load
        result = unpickler.load()
        return result
