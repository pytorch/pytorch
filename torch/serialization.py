import difflib
import inspect
import os
import io
import shutil
import struct
import sys
import torch
import tarfile
import zipfile
import tempfile
import warnings
from contextlib import closing, contextmanager
from ._utils import _import_dotted_name
from ._six import string_classes as _string_classes
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
    import pathlib

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


def validate_cuda_device(location):
    if isinstance(location, torch.device):
        location = str(location)
    if not isinstance(location, _string_classes):
        raise ValueError("location should be a string or torch.device")
    if location[5:] == '':
        device = 0
    else:
        device = max(int(location[5:]), 0)

    if not torch.cuda.is_available():
        raise RuntimeError('Attempting to deserialize object on a CUDA '
                           'device but torch.cuda.is_available() is False. '
                           'If you are running on a CPU-only machine, '
                           'please use torch.load with map_location=\'cpu\' '
                           'to map your storages to the CPU.')
    if device >= torch.cuda.device_count():
        raise RuntimeError('Attempting to deserialize object on CUDA device '
                           '{} but torch.cuda.device_count() is {}. Please use '
                           'torch.load with map_location to map your storages '
                           'to an existing device.'.format(
                               device, torch.cuda.device_count()))
    return device


def _cuda_deserialize(obj, location):
    if location.startswith('cuda'):
        device = validate_cuda_device(location)
        if getattr(obj, "_torch_load_uninitialized", False):
            storage_type = getattr(torch.cuda, type(obj).__name__)
            with torch.cuda.device(device):
                return storage_type(obj.size())
        else:
            return obj.cuda(device)


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


def _with_file_like(f, mode, body):
    """
    Executes a body function with a file object for f, opening
    it in 'mode' if it is a string filename.
    """
    new_fd = False
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)) or \
            (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        f = open(f, mode)
    try:
        return body(f)
    finally:
        if new_fd:
            f.close()


def _is_compressed_file(f):
    compress_modules = ['gzip']
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False


def _should_read_directly(f):
    """
    Checks if f is a file that should be read directly. It should be read
    directly if it is backed by a real file (has a fileno) and is not a
    a compressed file (e.g. gzip)
    """
    if _is_compressed_file(f):
        return False
    try:
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        return False
    except AttributeError:
        return False


def _check_seekable(f):

    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (str(e) + ". You can only torch.load from a file that is seekable." +
                                " Please pre-load the data into a buffer like io.BytesIO and" +
                                " try to load from it instead.")
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)


def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL):
    """Saves an object to a disk file.

    See also: :ref:`recommend-saving-models`

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string
           containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. warning::
        If you are using Python 2, torch.save does NOT support StringIO.StringIO
        as a valid file-like object. This is because the write method should return
        the number of bytes written; StringIO.write() does not do this.

        Please use something like io.BytesIO instead.

    Example:
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, 'tensor.pt')
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    """
    return _with_file_like(f, "wb", lambda f: _save(obj, f, pickle_module, pickle_protocol))


def _save(obj, f, pickle_module, pickle_protocol):
    if sys.version_info[0] == 2:
        import StringIO
        if isinstance(f, StringIO.StringIO):
            msg = ('torch.save received unsupported StringIO.StringIO file object, whose '
                   'write method does not return the number of bytes written. '
                   'Please use something like io.BytesIO for torch.save instead.')
            raise RuntimeError(msg)

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
            except Exception:  # saving the source is optional, so we can ignore any errors
                warnings.warn("Couldn't retrieve source code for container of "
                              "type " + obj.__name__ + ". It won't be checked "
                              "for correctness upon loading.")
            return ('module', obj, source_file, source)
        elif torch.is_storage(obj):
            storage_type = normalize_storage_type(type(obj))
            # Offset is always 0, but we keep it for backwards compatibility
            # with the old serialization format (which supported storage views)
            offset = 0
            obj_key = str(obj._cdata)
            location = location_tag(obj)
            serialized_storages[obj_key] = obj
            is_view = obj._cdata != obj._cdata
            if is_view:
                view_metadata = (str(obj._cdata), offset, obj.size())
            else:
                view_metadata = None

            return ('storage',
                    storage_type,
                    obj_key,
                    location,
                    obj.size(),
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
        serialized_storages[key]._write_file(f, _should_read_directly(f))


def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Loads an object saved with :func:`torch.save` from a file.

    :meth:`torch.load` uses Python's unpickling facilities but treats storages,
    which underlie tensors, specially. They are first deserialized on the
    CPU and are then moved to the device they were saved from. If this fails
    (e.g. because the run time system doesn't have certain devices), an exception
    is raised. However, storages can be dynamically remapped to an alternative
    set of devices using the `map_location` argument.

    If `map_location` is a callable, it will be called once for each serialized
    storage with two arguments: storage and location. The storage argument
    will be the initial deserialization of the storage, residing on the CPU.
    Each serialized storage has a location tag associated with it which
    identifies the device it was saved from, and this tag is the second
    argument passed to map_location. The builtin location tags are `'cpu'` for
    CPU tensors and `'cuda:device_id'` (e.g. `'cuda:2'`) for CUDA tensors.
    `map_location` should return either None or a storage. If `map_location` returns
    a storage, it will be used as the final deserialized object, already moved to
    the right device. Otherwise, :math:`torch.load` will fall back to the default
    behavior, as if `map_location` wasn't specified.

    If `map_location` is a string, it should be a device tag, where all tensors
    should be loaded.

    Otherwise, if `map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using `register_package`.

    Args:
        f: a file-like object (has to implement read, readline, tell, and seek),
            or a string containing a file name
        map_location: a function, torch.device, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the pickle_module used to serialize file)
        pickle_load_args: optional keyword arguments passed over to
            ``pickle_module.load`` and ``pickle_module.Unpickler``, e.g.,
            ``encoding=...``.

    .. note::
        When you call :meth:`torch.load()` on a file which contains GPU tensors, those tensors
        will be loaded to GPU by default. You can call `torch.load(.., map_location='cpu')`
        and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.

    .. note::
        In Python 3, when loading files saved by Python 2, you may encounter
        ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x...``. This is
        caused by the difference of handling in byte strings in Python2 and
        Python 3. You may use extra ``encoding`` keyword argument to specify how
        these objects should be loaded, e.g., ``encoding='latin1'`` decodes them
        to strings using ``latin1`` encoding, and ``encoding='bytes'`` keeps them
        as byte arrays which can be decoded later with ``byte_array.decode(...)``.

    Example:
        >>> torch.load('tensors.pt')
        # Load all tensors onto the CPU
        >>> torch.load('tensors.pt', map_location=torch.device('cpu'))
        # Load all tensors onto the CPU, using a function
        >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage)
        # Load all tensors onto GPU 1
        >>> torch.load('tensors.pt', map_location=lambda storage, loc: storage.cuda(1))
        # Map tensors from GPU 1 to GPU 0
        >>> torch.load('tensors.pt', map_location={'cuda:1':'cuda:0'})
        # Load tensor from io.BytesIO object
        >>> with open('tensor.pt', 'rb') as f:
                buffer = io.BytesIO(f.read())
        >>> torch.load(buffer)
    """
    new_fd = False
    if isinstance(f, str) or \
            (sys.version_info[0] == 2 and isinstance(f, unicode)):
        new_fd = True
        f = open(f, 'rb')
    elif (sys.version_info[0] == 3 and isinstance(f, pathlib.Path)):
        new_fd = True
        f = f.open('rb')
    try:
        return _load(f, map_location, pickle_module, **pickle_load_args)
    finally:
        if new_fd:
            f.close()


def _load(f, map_location, pickle_module, **pickle_load_args):
    deserialized_objects = {}

    if map_location is None:
        restore_location = default_restore_location
    elif isinstance(map_location, dict):
        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)
    elif isinstance(map_location, _string_classes):
        def restore_location(storage, location):
            return default_restore_location(storage, map_location)
    elif isinstance(map_location, torch.device):
        def restore_location(storage, location):
            return default_restore_location(storage, str(map_location))
    else:
        def restore_location(storage, location):
            result = map_location(storage, location)
            if result is None:
                result = default_restore_location(storage, location)
            return result

    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = inspect.getsource(container_type)
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn("Couldn't retrieve source code for container of "
                          "type " + container_type.__name__ + ". It won't be checked "
                          "for correctness upon loading.")
            return
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
                num_storages = pickle_module.load(f, **pickle_load_args)
                for i in range(num_storages):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, location, storage_type = args
                    obj = storage_type._new_with_file(f)
                    obj = restore_location(obj, location)
                    deserialized_objects[key] = obj

                storage_views = pickle_module.load(f, **pickle_load_args)
                for target_cdata, root_cdata, offset, size in storage_views:
                    root = deserialized_objects[root_cdata]
                    deserialized_objects[target_cdata] = root[offset:offset + size]

            tar.extract('tensors', path=tmpdir)
            with open(os.path.join(tmpdir, 'tensors'), 'rb', 0) as f:
                num_tensors = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_tensors):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, storage_id, original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    tensor_type = storage_to_tensor_type(storage)
                    ndim, = struct.unpack('<i', f.read(4))
                    # skip next 4 bytes; legacy encoding treated ndim as 8 bytes
                    f.read(4)
                    size = struct.unpack('<{}q'.format(ndim), f.read(8 * ndim))
                    stride = struct.unpack('<{}q'.format(ndim), f.read(8 * ndim))
                    storage_offset, = struct.unpack('<q', f.read(8))
                    tensor = tensor_type().set_(storage, storage_offset, size, stride)
                    deserialized_objects[key] = tensor

            pickle_file = tar.extractfile('pickle')
            unpickler = pickle_module.Unpickler(pickle_file, **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            return result

    deserialized_objects = {}

    def maybe_decode_ascii(bytes_str):
        # When using encoding='bytes' in Py3, some **internal** keys stored as
        # strings in Py2 are loaded as bytes. This function decodes them with
        # ascii encoding, one that Py3 uses by default.
        #
        # NOTE: This should only be used on internal keys (e.g., `typename` and
        #       `location` in `persistent_load` below!
        if isinstance(bytes_str, bytes):
            return bytes_str.decode('ascii')
        return bytes_str

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == 'module':
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == 'storage':
            data_type, root_key, location, size, view_metadata = data
            location = maybe_decode_ascii(location)
            if root_key not in deserialized_objects:
                obj = data_type(size)
                obj._torch_load_uninitialized = True
                deserialized_objects[root_key] = restore_location(obj, location)
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

    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)

    if f_should_read_directly and f.tell() == 0:
        # legacy_load requires that f has fileno()
        # only if offset is zero we can attempt the legacy tar file loader
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if zipfile.is_zipfile(f):
                # .zip is used for torch.jit.save and will throw an un-pickling error here
                raise RuntimeError("{} is a zip archive (did you mean to use torch.jit.load()?)".format(f.name))
            # if not a tarfile, reset file offset and proceed
            f.seek(0)

    magic_number = pickle_module.load(f, **pickle_load_args)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f, **pickle_load_args)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError("Invalid protocol version: %s" % protocol_version)

    _sys_info = pickle_module.load(f, **pickle_load_args)
    unpickler = pickle_module.Unpickler(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)

    offset = f.tell() if f_should_read_directly else None
    for key in deserialized_storage_keys:
        assert key in deserialized_objects
        deserialized_objects[key]._set_from_file(f, offset, f_should_read_directly)
        offset = None

    return result
