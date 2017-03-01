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

MAGIC_NUMBER = 0x1950a86a20f9469cfc6c
PROTOCOL_VERSION = 1001
STORAGE_KEY_SEPARATOR = ','


class SourceChangeWarning(Warning):
    pass


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
                       torch.typename(storage) + " (tagged with " +
                       location + ")")


def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def storage_to_tensor_type(storage):
    storage_type = type(storage)
    module = _import_dotted_name(storage_type.__module__)
    return getattr(module, storage_type.__name__.replace('Storage', 'Tensor'))


def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
    """Saves an object to a disk file.

    See also: :ref:`recommend-saving-models`

    Args:
        obj: saved object
        f: a file-like object (has to implement fileno that returns a file descriptor)
            or a string containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol
    """
    new_fd = False
    if isinstance(f, str) or (sys.version_info[0] == 2 and isinstance(f, unicode)):
        new_fd = True
        f = open(f, "wb")
    try:
        return _save(obj, f, pickle_module, pickle_protocol)
    finally:
        if new_fd:
            f.close()


def _save(obj, f, pickle_module, pickle_protocol):
    import torch.nn as nn
    serialized_container_types = {}
    serialized_storages = {}

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
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
            return ('module', obj, source_file, source)
        elif torch.is_storage(obj):
            storage_type = normalize_storage_type(type(obj))
            root, offset = obj._root_storage()
            root_key = str(root._cdata)
            location = location_tag(obj)
            serialized_storages[root_key] = root
            is_view = obj._cdata != root._cdata
            if is_view:
                view_metadata = (str(obj._cdata), offset, obj.size())
            else:
                view_metadata = None

            return ('storage',
                    storage_type,
                    root_key,
                    location,
                    root.size(),
                    view_metadata)

        return None

    sys_info = dict(
        protocol_version=PROTOCOL_VERSION,
        little_endian=sys.byteorder == 'little',
        type_sizes=dict(
            short=SHORT_SIZE,
            int=INT_SIZE,
            long=LONG_SIZE,
        ),
    )

    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)
    pickler = pickle_module.Pickler(f, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)

    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        serialized_storages[key]._write_file(f)


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
        f: a file-like object (has to implement fileno that returns a file descriptor,
            and must implement seek), or a string containing a file name
        map_location: a function or a dict specifying how to remap storage locations
        pickle_module: module used for unpickling metadata and objects (has to match
            the pickle_module used to serialize file)
    """
    new_fd = False
    if isinstance(f, str) or (sys.version_info[0] == 2 and isinstance(f, unicode)):
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
                diff = difflib.unified_diff(current_source.split('\n'),
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

    def legacy_load(f):
        deserialized_objects = {}

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

    deserialized_objects = {}

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = saved_id[0]
        data = saved_id[1:]

        if typename == 'module':
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == 'storage':
            data_type, root_key, location, size, view_metadata = data
            if root_key not in deserialized_objects:
                deserialized_objects[root_key] = restore_location(
                    data_type(size), location)
            storage = deserialized_objects[root_key]
            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                if view_key not in deserialized_objects:
                    deserialized_objects[view_key] = storage[offset:offset + view_size]
                return deserialized_objects[view_key]
            else:
                return storage
        else:
            raise RuntimeError("Unknown saved id type: %s" % saved_id[0])

    # try the legacy loader first, which only works if f is a tarfile
    try:
        return legacy_load(f)
    except tarfile.TarError:
        pass

    f.seek(0)
    magic_number = pickle_module.load(f)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError("Invalid protocol version: %s" % protocol_version)

    _sys_info = pickle_module.load(f)
    unpickler = pickle_module.Unpickler(f)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle_module.load(f)

    offset = f.tell()
    for key in deserialized_storage_keys:
        assert key in deserialized_objects
        deserialized_objects[key]._set_from_file(f, offset)
        offset = None

    return result
