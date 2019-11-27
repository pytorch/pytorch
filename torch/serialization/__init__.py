import copyreg
import io
import os
import struct
import sys
import torch
import warnings
from torch._six import PY2
if PY2:
    import cPickle as pickle
else:
    import pickle


import torch.serialization._file_utils as _file_utils
import torch.serialization._legacy as _legacy
from torch.serialization._tensor_utils import location_tag, default_restore_location, \
    normalize_storage_type, storage_to_tensor_type, validate_cuda_device, get_restore_location

DEFAULT_PROTOCOL = 2


def save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=False):
    """Saves an object to a disk file.

    See also: :ref:`recommend-saving-models`

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string
           containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. warning::
        If you are using Python 2, :func:`torch.save` does NOT support :class:`StringIO.StringIO`
        as a valid file-like object. This is because the write method should return
        the number of bytes written; :meth:`StringIO.write()` does not do this.

        Please use something like :class:`io.BytesIO` instead.

    Example:
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, 'tensor.pt')
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    """
    if _use_new_zipfile_serialization:
        with _file_utils.open_zipfile_writer(f) as opened_file:
            _save(obj, opened_file, pickle_module, pickle_protocol)
            return

    with _file_utils.open_file_like(f, 'wb') as opened_file:
        _legacy.legacy_save(obj, opened_file, pickle_module, pickle_protocol)


def _save(obj, zip_file, pickle_module, pickle_protocol):
    serialized_storages = {}

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if torch.is_storage(obj):
            storage_type = normalize_storage_type(type(obj))
            obj_key = str(obj._cdata)
            location = location_tag(obj)
            serialized_storages[obj_key] = obj

            return ('storage',
                    storage_type,
                    obj_key,
                    location,
                    obj.size())
        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    zip_file.write_record('data.pkl', data_value, len(data_value))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in sorted(serialized_storages.keys()):
        name = 'data/{}'.format(key)
        storage = serialized_storages[key]
        num_bytes = storage.size() * storage.element_size()
        zip_file.write_record(name, storage.data_ptr(), num_bytes)


def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
    """Loads an object saved with :func:`torch.save` from a file.

    :func:`torch.load` uses Python's unpickling facilities but treats storages,
    which underlie tensors, specially. They are first deserialized on the
    CPU and are then moved to the device they were saved from. If this fails
    (e.g. because the run time system doesn't have certain devices), an exception
    is raised. However, storages can be dynamically remapped to an alternative
    set of devices using the :attr:`map_location` argument.

    If :attr:`map_location` is a callable, it will be called once for each serialized
    storage with two arguments: storage and location. The storage argument
    will be the initial deserialization of the storage, residing on the CPU.
    Each serialized storage has a location tag associated with it which
    identifies the device it was saved from, and this tag is the second
    argument passed to :attr:`map_location`. The builtin location tags are ``'cpu'``
    for CPU tensors and ``'cuda:device_id'`` (e.g. ``'cuda:2'``) for CUDA tensors.
    :attr:`map_location` should return either ``None`` or a storage. If
    :attr:`map_location` returns a storage, it will be used as the final deserialized
    object, already moved to the right device. Otherwise, :func:`torch.load` will
    fall back to the default behavior, as if :attr:`map_location` wasn't specified.

    If :attr:`map_location` is a :class:`torch.device` object or a string contraining
    a device tag, it indicates the location where all tensors should be loaded.

    Otherwise, if :attr:`map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using :func:`torch.serialization.register_package`.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth`readline`, :meth`tell`, and :meth`seek`),
            or a string containing a file name
        map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the :attr:`pickle_module` used to serialize file)
        pickle_load_args: (Python 3 only) optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`, e.g.,
            :attr:`errors=...`.

    .. note::
        When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
        will be loaded to GPU by default. You can call ``torch.load(.., map_location='cpu')``
        and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.

    .. note::
        By default, we decode byte strings as ``utf-8``.  This is to avoid a common error
        case ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x...``
        when loading files saved by Python 2 in Python 3.  If this default
        is incorrect, you may use an extra :attr:`encoding` keyword argument to specify how
        these objects should be loaded, e.g., :attr:`encoding='latin1'` decodes them
        to strings using ``latin1`` encoding, and :attr:`encoding='bytes'` keeps them
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
        # Load a module with 'ascii' encoding for unpickling
        >>> torch.load('module.pt', encoding='ascii')
    """
    if sys.version_info >= (3, 0) and 'encoding' not in pickle_load_args.keys():
        pickle_load_args['encoding'] = 'utf-8'

    with _file_utils.open_file_like(f, 'rb') as opened_file:
        if _file_utils.is_zipfile(opened_file):
            with _file_utils.open_zipfile_reader(f) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    warnings.warn("'torch.load' received a zip file that looks like a TorchScript archive"
                                  " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                                  " silence this warning)", UserWarning)
                    return torch.jit.load(f)
                return _load(opened_zipfile, map_location, pickle_module, **pickle_load_args)
        return _legacy.legacy_load(opened_file, map_location, pickle_module, **pickle_load_args)


# Register pickling support for layout instances such as
# torch.sparse_coo, etc
def _get_layout(name):
    """Get layout extension object from its string representation.
    """
    cache = _get_layout.cache
    if not cache:
        for v in torch.__dict__.values():
            if isinstance(v, torch.layout):
                cache[str(v)] = v
    return cache[name]


_get_layout.cache = {}
copyreg.pickle(torch.layout, lambda obj: (_get_layout, (str(obj),)))


def _load(zip_file, map_location, pickle_module, **pickle_load_args):
    restore_location = get_restore_location(map_location)

    loaded_storages = {}

    def load_tensor(obj, size, key, location):
        loaded_storages[key] = restore_location(obj, location)
        name = 'data/{}'.format(key)
        size_long = struct.pack("<Q", size)
        tensor_file = io.BytesIO(size_long + zip_file.get_record(name))
        offset = None
        is_real_file = False
        loaded_storages[key]._set_from_file(tensor_file, offset, is_real_file)

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _file_utils.maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert typename == 'storage', \
            "Unknown typename for persistent_load, expected 'storage' but got '{}'".format(typename)
        data_type, key, location, size = data
        if key not in loaded_storages:
            load_tensor(data_type(size), size, key, _file_utils.maybe_decode_ascii(location))
        storage = loaded_storages[key]
        return storage

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = io.BytesIO(zip_file.get_record('data.pkl'))
    unpickler = pickle_module.Unpickler(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    return result


def _is_torchscript_zip(zip_file):
    for file_name in zip_file.get_all_records():
        parts = file_name.split(os.sep)
        if len(parts) > 1 and parts[1] == 'constants.pkl':
            return True
    return False
