# mypy: allow-untyped-defs
import copyreg
import difflib
import functools
import io
import os
import pickle
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import threading
import warnings
from contextlib import closing, contextmanager
from enum import Enum
from typing import (
    Any,
    BinaryIO,
    Callable,
    cast,
    Dict,
    IO,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from typing_extensions import TypeAlias, TypeIs

import torch
import torch._weights_only_unpickler as _weights_only_unpickler
from torch._sources import get_source_lines_and_file
from torch._utils import _import_dotted_name
from torch.storage import _get_dtype_from_pickle_storage_type
from torch.types import Storage


__all__ = [
    "SourceChangeWarning",
    "mkdtemp",
    "register_package",
    "check_module_version_greater_or_equal",
    "validate_cuda_device",
    "validate_hpu_device",
    "location_tag",
    "default_restore_location",
    "normalize_storage_type",
    "storage_to_tensor_type",
    "save",
    "load",
    "StorageType",
    "LoadEndianness",
    "get_crc32_options",
    "set_crc32_options",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "get_default_mmap_options",
    "set_default_mmap_options",
    "clear_safe_globals",
    "get_safe_globals",
    "add_safe_globals",
    "safe_globals",
    "get_unsafe_globals_in_checkpoint",
    "skip_data",
]

DEFAULT_PROTOCOL = 2

LONG_SIZE = struct.Struct("=l").size
INT_SIZE = struct.Struct("=i").size
SHORT_SIZE = struct.Struct("=h").size

MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
PROTOCOL_VERSION = 1001
STORAGE_KEY_SEPARATOR = ","

FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
MAP_LOCATION: TypeAlias = Optional[
    Union[Callable[[Storage, str], Storage], torch.device, str, Dict[str, str]]
]
STORAGE: TypeAlias = Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage]

IS_WINDOWS = sys.platform == "win32"

if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None  # type: ignore[assignment]


def _default_to_weights_only(pickle_module):
    is_fbcode = not hasattr(torch.version, "git_version")
    return pickle_module is None and not is_fbcode


# _serialization_tls is used to store thread local state specific to serialization
# that needs to be propagated to other files, in particular we use this for
# (1) map_location (needed for wrapper subclasses/third party devices to torch._utils)
# (2) skip_data (needed for torch.Tensor.__reduce_ex__ for skip_data ctx)
# (3) materialize_fake_tensors (needed for torch.Tensor.__reduce_ex__ for skip_data ctx)
class _SerializationLocal(threading.local):
    def __init__(self):
        super().__init__()
        self.map_location: Optional[MAP_LOCATION] = None
        self.skip_data: bool = False
        self.materialize_fake_tensors: bool = False


_serialization_tls = _SerializationLocal()


class SourceChangeWarning(Warning):
    pass


@contextmanager
def mkdtemp():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)


_package_registry: List[
    Tuple[
        int,
        Callable[[STORAGE], Optional[str]],
        Callable[[STORAGE, str], Optional[STORAGE]],
    ]
] = []


class LoadEndianness(Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3


_default_load_endian: Optional[LoadEndianness] = None


def get_default_load_endianness() -> Optional[LoadEndianness]:
    """
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    """
    return _default_load_endian


def set_default_load_endianness(endianness):
    """
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    """
    global _default_load_endian
    if not isinstance(endianness, LoadEndianness) and endianness is not None:
        raise TypeError("Invalid argument type in function set_default_load_endianness")
    _default_load_endian = endianness


_compute_crc32: bool = True


def get_crc32_options() -> bool:
    """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """
    return _compute_crc32


def set_crc32_options(compute_crc32: bool):
    """
    Set whether :func:`torch.save` computes and writes crc32 for each record.

    .. note::
        Setting this to ``False`` may make unzipping of the ``torch.save`` output
        fail or warn due to corrupted CRC32. However ``torch.load`` will be
        able to load the file.

    Args:
        compute_crc32 (bool): set crc32 compuation flag
    """
    global _compute_crc32
    _compute_crc32 = compute_crc32


_default_mmap_options: int = MAP_PRIVATE


def get_default_mmap_options() -> int:
    """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """
    return _default_mmap_options


class set_default_mmap_options:
    """
    Context manager or function to set default mmap options for :func:`torch.load` with ``mmap=True`` to flags.

    For now, only either ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED`` are supported.
    Please open an issue if you need any other option to be added here.

    .. note::
        This feature is currently not supported for Windows.

    Args:
        flags: ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED``
    """

    def __init__(self, flags: int) -> None:
        if IS_WINDOWS:
            raise RuntimeError(
                "Changing the default mmap options is currently not supported for Windows"
            )
        if flags != MAP_PRIVATE and flags != MAP_SHARED:
            raise ValueError(
                "Invalid argument in function set_default_mmap_options, "
                f"expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got {flags}"
            )
        global _default_mmap_options
        self.prev = _default_mmap_options
        _default_mmap_options = flags

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        global _default_mmap_options
        _default_mmap_options = self.prev


def clear_safe_globals() -> None:
    """
    Clears the list of globals that are safe for ``weights_only`` load.
    """
    _weights_only_unpickler._clear_safe_globals()


def get_safe_globals() -> List[Any]:
    """
    Returns the list of user-added globals that are safe for ``weights_only`` load.
    """
    return _weights_only_unpickler._get_safe_globals()


def add_safe_globals(safe_globals: List[Any]) -> None:
    """
    Marks the given globals as safe for ``weights_only`` load. For example, functions
    added to this list can be called during unpickling, classes could be instantiated
    and have state set.

    Args:
        safe_globals (List[Any]): list of globals to mark as safe

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     torch.serialization.add_safe_globals([MyTensor])
        ...     torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
    """
    _weights_only_unpickler._add_safe_globals(safe_globals)


class safe_globals(_weights_only_unpickler._safe_globals):
    r"""Context-manager that adds certain globals as safe for ``weights_only`` load.

    Args:
        safe_globals: List of globals for weights_only load.

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     with torch.serialization.safe_globals([MyTensor]):
        ...         torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
        >>> assert torch.serialization.get_safe_globals() == []
    """


def get_unsafe_globals_in_checkpoint(f: FILE_LIKE) -> List[str]:
    """Returns a list of strings of functions/classes in a ``torch.save`` object that are not safe for ``weights_only``.

    For a given function or class ``f``, the corresponding string will be of the form
    ``{f.__module__}.{f.__name__}``.

    This function will return any GLOBALs in the checkpoint that are not in the set marked safe
    for ``weights_only`` (either via :func:`add_safe_globals` or :class:`safe_globals` context or
    allowlisted by ``torch`` by default).

    .. note::
        This function will statically disassemble the pickle file in the checkpoint.
        The implication is any classes dynamically pushed onto the stack during unpickling
        will not be included in the output.

    Args:
        f: File-like object or string containing the checkpoint object saved via ``torch.save``

    Returns:
        A list of strings of pickle GLOBALs in the checkpoint that are not allowlisted for ``weights_only``.
    """
    default_safe_globals_strings = set(
        _weights_only_unpickler._get_allowed_globals().keys()
    )
    user_safe_global_strings = set(
        _weights_only_unpickler._get_user_allowed_globals().keys()
    )
    safe_global_strings = default_safe_globals_strings.union(user_safe_global_strings)

    with _open_file_like(f, "rb") as opened_file:
        if not _is_zipfile(opened_file):
            raise ValueError("Expected input to be a checkpoint returned by torch.save")
        with _open_zipfile_reader(opened_file) as zip_file:
            if _is_torchscript_zip(zip_file):
                raise ValueError(
                    "Expected input to be a checkpoint returned by torch.save but got a torchscript checkpoint"
                )
            data_file = io.BytesIO(zip_file.get_record("data.pkl"))
            all_globals = _weights_only_unpickler.get_globals_in_pkl(data_file)
            return list(all_globals.difference(safe_global_strings))


class skip_data:
    """
    Context-manager that skips writing storage bytes for ``torch.save`` calls.

    Storages will still be saved, but the space that their bytes would usually be written to
    will be empty space. The storage bytes can then be populated in a separate pass.

    .. warning::
        The ``skip_data`` context manager is an early prototype and is subject to change.

    Args:
        materialize_fake_tensors: Whether to materialize FakeTensors.

    Example:
        >>> # xdoctest: +SKIP("NamedTemporaryFile on Windows")
        >>> import tempfile
        >>> t = torch.randn(2, 3)
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     with torch.serialization.skip_data():
        ...         torch.save(t, f.name)
        ...     torch.load(f.name, weights_only=True)
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
    """

    def __init__(self, materialize_fake_tensors: bool = False):
        self.materialize_fake_tensors = materialize_fake_tensors

    def __enter__(self):
        global _serialization_tls
        self._old_skip_data = _serialization_tls.skip_data
        self._old_materialize_fake_tensors = _serialization_tls.materialize_fake_tensors
        _serialization_tls.skip_data = True
        _serialization_tls.materialize_fake_tensors = self.materialize_fake_tensors

    def __exit__(self, type, value, tb):
        global _serialization_tls
        _serialization_tls.skip_data = self._old_skip_data
        _serialization_tls.materialize_fake_tensors = self._old_materialize_fake_tensors


def _is_zipfile(f) -> bool:
    # This is a stricter implementation than zipfile.is_zipfile().
    # zipfile.is_zipfile() is True if the magic number appears anywhere in the
    # binary. Since we expect the files here to be generated by torch.save or
    # torch.jit.save, it's safe to only check the start bytes and avoid
    # collisions and assume the zip has only 1 file.
    # See bugs.python.org/issue28494.

    start = f.tell()
    # Read the first few bytes and match against the ZIP file signature
    local_header_magic_number = b"PK\x03\x04"
    read_bytes = f.read(len(local_header_magic_number))
    f.seek(start)
    return read_bytes == local_header_magic_number


def register_package(
    priority: int,
    tagger: Callable[[STORAGE], Optional[str]],
    deserializer: Callable[[STORAGE, str], Optional[STORAGE]],
):
    """
    Registers callables for tagging and deserializing storage objects with an associated priority.
    Tagging associates a device with a storage object at save time while deserializing moves a
    storage object to an appropriate device at load time. :attr:`tagger` and :attr:`deserializer`
    are run in the order given by their :attr:`priority` until a tagger/deserializer returns a
    value that is not `None`.

    To override the deserialization behavior for a device in the global registry, one can register a
    tagger with a higher priority than the existing tagger.

    This function can also be used to register a tagger and deserializer for new devices.

    Args:
        priority: Indicates the priority associated with the tagger and deserializer, where a lower
            value indicates higher priority.
        tagger: Callable that takes in a storage object and returns its tagged device as a string
            or None.
        deserializer: Callable that takes in storage object and a device string and returns a storage
            object on the appropriate device or None.

    Returns:
        `None`

    Example:
        >>> def ipu_tag(obj):
        >>>     if obj.device.type == 'ipu':
        >>>         return 'ipu'
        >>> def ipu_deserialize(obj, location):
        >>>     if location.startswith('ipu'):
        >>>         ipu = getattr(torch, "ipu", None)
        >>>         assert ipu is not None, "IPU device module is not loaded"
        >>>         assert torch.ipu.is_available(), "ipu is not available"
        >>>         return obj.ipu(location)
        >>> torch.serialization.register_package(11, ipu_tag, ipu_deserialize)
    """
    queue_elem = (priority, tagger, deserializer)
    _package_registry.append(queue_elem)
    _package_registry.sort()


def check_module_version_greater_or_equal(
    module,
    req_version_tuple,
    error_if_malformed=True,
):
    """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """
    try:
        version_strs = module.__version__.split(".")
        # Cast module version fields to match the types of the required version
        module_version = tuple(
            type(req_field)(version_strs[idx])
            for idx, req_field in enumerate(req_version_tuple)
        )
        requirement_is_met = module_version >= req_version_tuple

    except Exception as e:
        message = (
            f"'{module.__name__}' module version string is malformed '{module.__version__}' and cannot be compared"
            f" with tuple {str(req_version_tuple)}"
        )
        if error_if_malformed:
            raise RuntimeError(message) from e
        else:
            warnings.warn(message + ", but continuing assuming that requirement is met")
            requirement_is_met = True

    return requirement_is_met


def _cpu_tag(obj):
    if obj.device.type == "cpu":
        return "cpu"


def _mps_tag(obj):
    if obj.device.type == "mps":
        return "mps"


def _meta_tag(obj):
    if obj.device.type == "meta":
        return "meta"


def _backend_tag(backend_name, obj):
    if backend_name == "privateuse1":
        backend_name = torch._C._get_privateuse1_backend_name()
    if obj.device.type == backend_name:
        if obj.device.index is None:
            return backend_name
        else:
            return backend_name + ":" + str(obj.device.index)


def _cpu_deserialize(obj, location):
    if location == "cpu":
        return obj


def _mps_deserialize(obj, location):
    if location.startswith("mps"):
        return obj.mps()


def _meta_deserialize(obj, location):
    if location == "meta":
        return torch.UntypedStorage(obj.nbytes(), device="meta")


def _validate_device(location, backend_name):
    """
    Check whether the device index of specified backend is valid

    In case of privateuse1 backend, your must first register a device_module for
    privateuse1 using torch._register_device_module. Implement the following
    methods in device_module like cuda: device_module._utils._get_device_index(location, True),
    device_module.device_count().

    Args:
        location: string of device
        backend_name: the backend name or the name of privateuse1, which can be renamed

    Returns:
        device_index: int
    """
    if not hasattr(torch, backend_name):
        raise RuntimeError(
            f"The {backend_name.upper()} device module is not registered. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    device_module = getattr(torch, backend_name)
    if hasattr(device_module, "_utils") and hasattr(
        device_module._utils, "_get_device_index"
    ):
        device_index = device_module._utils._get_device_index(location, True)
        device = torch.device(backend_name, device_index)
    else:
        device = torch.device(location)
        device_index = device.index if device.index else 0
    if hasattr(device_module, "is_available") and not device_module.is_available():
        raise RuntimeError(
            f"Attempting to deserialize object on a {backend_name.upper()} "
            f"device but torch.{backend_name}.is_available() is False. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    if hasattr(device_module, "device_count"):
        device_count = device_module.device_count()
        if device_index >= device_count:
            raise RuntimeError(
                f"Attempting to deserialize object on {backend_name.upper()} device "
                f"{device_index} but torch.{backend_name}.device_count() is {device_count}. "
                "Please use torch.load with map_location to map your storages "
                "to an existing device."
            )
    return device


def validate_cuda_device(location):
    return _validate_device(location, "cuda").index


def validate_hpu_device(location):
    return _validate_device(location, "hpu").index


def _deserialize(backend_name, obj, location):
    if backend_name == "privateuse1":
        backend_name = torch._C._get_privateuse1_backend_name()
    if location.startswith(backend_name):
        device = _validate_device(location, backend_name)
        return obj.to(device=device)


register_package(10, _cpu_tag, _cpu_deserialize)
register_package(
    20,
    functools.partial(_backend_tag, "cuda"),
    functools.partial(_deserialize, "cuda"),
)
register_package(21, _mps_tag, _mps_deserialize)
register_package(22, _meta_tag, _meta_deserialize)
register_package(
    23,
    functools.partial(_backend_tag, "privateuse1"),
    functools.partial(_deserialize, "privateuse1"),
)
register_package(
    24,
    functools.partial(_backend_tag, "hpu"),
    functools.partial(_deserialize, "hpu"),
)
register_package(
    25,
    functools.partial(_backend_tag, "xpu"),
    functools.partial(_deserialize, "xpu"),
)


def location_tag(
    storage: Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage],
):
    for _, tagger, _ in _package_registry:
        location = tagger(storage)
        if location:
            return location
    raise RuntimeError(
        "don't know how to determine data location of " + torch.typename(storage)
    )


def default_restore_location(storage, location):
    """
    Restores `storage` using a deserializer function registered for the `location`.

    This function looks in the registry for deserializer functions that match the `location`.
    If found, it attempts to use them, in priority order, to restore `storage` until one
    returns a not `None` result. If no deserializer can be found in the registry, or all found fail
    to bear a result, it raises a `RuntimeError`.

    Args:
        storage (STORAGE): the storage object to restore
        location (str): the location tag associated with the storage object

    Returns:
        storage: Optional[STORAGE]

    Raises:
        RuntimeError: If no deserializer matching `location` is found in the registry or if
           all matching ones return `None`.
    """
    for _, _, fn in _package_registry:
        result = fn(storage, location)
        if result is not None:
            return result
    raise RuntimeError(
        "don't know how to restore data location of "
        + torch.typename(storage)
        + " (tagged with "
        + location
        + ")"
    )


def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


def storage_to_tensor_type(storage):
    storage_type = type(storage)
    module = _import_dotted_name(storage_type.__module__)
    return getattr(module, storage_type.__name__.replace("Storage", "Tensor"))


def _is_path(name_or_buffer) -> TypeIs[Union[str, os.PathLike]]:
    return isinstance(name_or_buffer, (str, os.PathLike))


class _opener:
    def __init__(self, file_like):
        self.file_like = file_like

    def __enter__(self):
        return self.file_like

    def __exit__(self, *args):
        pass


class _open_file(_opener):
    def __init__(self, name, mode):
        super().__init__(open(name, mode))

    def __exit__(self, *args):
        self.file_like.close()


class _open_buffer_reader(_opener):
    def __init__(self, buffer):
        super().__init__(buffer)
        _check_seekable(buffer)


class _open_buffer_writer(_opener):
    def __exit__(self, *args):
        self.file_like.flush()


def _open_file_like(name_or_buffer, mode):
    if _is_path(name_or_buffer):
        return _open_file(name_or_buffer, mode)
    else:
        if "w" in mode:
            return _open_buffer_writer(name_or_buffer)
        elif "r" in mode:
            return _open_buffer_reader(name_or_buffer)
        else:
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")


class _open_zipfile_reader(_opener):
    def __init__(self, name_or_buffer) -> None:
        super().__init__(torch._C.PyTorchFileReader(name_or_buffer))


class _open_zipfile_writer_file(_opener):
    def __init__(self, name) -> None:
        self.file_stream = None
        self.name = str(name)
        try:
            self.name.encode("ascii")
        except UnicodeEncodeError:
            # PyTorchFileWriter only supports ascii filename.
            # For filenames with non-ascii characters, we rely on Python
            # for writing out the file.
            self.file_stream = io.FileIO(self.name, mode="w")
            super().__init__(
                torch._C.PyTorchFileWriter(self.file_stream, _compute_crc32)
            )
        else:
            super().__init__(torch._C.PyTorchFileWriter(self.name, _compute_crc32))

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        if self.file_stream is not None:
            self.file_stream.close()


class _open_zipfile_writer_buffer(_opener):
    def __init__(self, buffer) -> None:
        if not callable(getattr(buffer, "write", None)):
            msg = f"Buffer of {str(type(buffer)).strip('<>')} has no callable attribute 'write'"
            if not hasattr(buffer, "write"):
                raise AttributeError(msg)
            raise TypeError(msg)
        self.buffer = buffer
        super().__init__(torch._C.PyTorchFileWriter(buffer, _compute_crc32))

    def __exit__(self, *args) -> None:
        self.file_like.write_end_of_file()
        self.buffer.flush()


def _open_zipfile_writer(name_or_buffer):
    container: Type[_opener]
    if _is_path(name_or_buffer):
        container = _open_zipfile_writer_file
    else:
        container = _open_zipfile_writer_buffer
    return container(name_or_buffer)


def _is_compressed_file(f) -> bool:
    compress_modules = ["gzip"]
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


def _check_seekable(f) -> bool:
    def raise_err_msg(patterns, e):
        for p in patterns:
            if p in str(e):
                msg = (
                    str(e)
                    + ". You can only torch.load from a file that is seekable."
                    + " Please pre-load the data into a buffer like io.BytesIO and"
                    + " try to load from it instead."
                )
                raise type(e)(msg)
        raise e

    try:
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        raise_err_msg(["seek", "tell"], e)
    return False


def _check_dill_version(pickle_module) -> None:
    """Checks if using dill as the pickle module, and if so, checks if it is the correct version.
    If dill version is lower than 0.3.1, a ValueError is raised.

    Args:
        pickle_module: module used for pickling metadata and objects

    """
    if pickle_module is not None and pickle_module.__name__ == "dill":
        required_dill_version = (0, 3, 1)
        if not check_module_version_greater_or_equal(
            pickle_module, required_dill_version, False
        ):
            raise ValueError(
                (
                    "'torch' supports dill >= {}, but you have dill {}."
                    " Please upgrade dill or switch to 'pickle'"
                ).format(
                    ".".join([str(num) for num in required_dill_version]),
                    pickle_module.__version__,
                )
            )


def _check_save_filelike(f):
    if not _is_path(f) and not hasattr(f, "write"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with "
            "a 'write' attribute"
        )


def save(
    obj: object,
    f: FILE_LIKE,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)

    Saves an object to a disk file.

    See also: :ref:`saving-loading-tensors`

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. note::
        A common PyTorch convention is to save tensors using .pt file extension.

    .. note::
        PyTorch preserves storage sharing across serialization. See
        :ref:`preserve-storage-sharing` for more details.

    .. note::
        The 1.6 release of PyTorch switched ``torch.save`` to use a new
        zipfile-based file format. ``torch.load`` still retains the ability to
        load files in the old format. If for any reason you want ``torch.save``
        to use the old format, pass the kwarg ``_use_new_zipfile_serialization=False``.

    Example:
        >>> # xdoctest: +SKIP("makes cwd dirty")
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, "tensor.pt")
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    """
    torch._C._log_api_usage_once("torch.save")
    _check_dill_version(pickle_module)
    _check_save_filelike(f)

    if _use_new_zipfile_serialization:
        with _open_zipfile_writer(f) as opened_zipfile:
            _save(
                obj,
                opened_zipfile,
                pickle_module,
                pickle_protocol,
                _disable_byteorder_record,
            )
            return
    else:
        global _serialization_tls
        if _serialization_tls.skip_data:
            raise RuntimeError(
                "Cannot use skip_data=True with _use_new_zipfile_serialization=False"
            )
        with _open_file_like(f, "wb") as opened_file:
            _legacy_save(obj, opened_file, pickle_module, pickle_protocol)


def _legacy_save(obj, f, pickle_module, pickle_protocol) -> None:
    import torch.nn as nn

    serialized_container_types = {}
    serialized_storages: Dict[str, Tuple[torch.UntypedStorage, torch.dtype]] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj: Any) -> Optional[Tuple]:
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
                source_lines, _, source_file = get_source_lines_and_file(obj)
                source = "".join(source_lines)
            except (
                Exception
            ):  # saving the source is optional, so we can ignore any errors
                warnings.warn(
                    "Couldn't retrieve source code for container of "
                    "type " + obj.__name__ + ". It won't be checked "
                    "for correctness upon loading."
                )
            return ("module", obj, source_file, source)

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            storage: torch.UntypedStorage

            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                dtype = obj.dtype
                storage_numel = obj._size()

            elif isinstance(obj, torch.UntypedStorage):
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                dtype = torch.uint8
                storage_numel = storage.nbytes()
            else:
                raise TypeError(f"type not recognized: {type(obj)}")

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            view_metadata: Optional[Tuple[str, int, int]]

            # Offset is always 0, but we keep it for backwards compatibility
            # with the old serialization format (which supported storage views)
            offset = 0
            storage_key = str(storage._cdata)
            location = location_tag(storage)

            # TODO: There's an issue here with FC. It might be impossible to
            # solve, but it's worth noting. Imagine we save a list `[storage,
            # tensor]`, where `tensor.storage()` is the same as `storage`, and
            # `tensor.element_size() > 1`. Let's say that `tensor.dtype ==
            # torch.float`.  The storage will be serialized with element size
            # of 1, since we're choosing to serialize the first occurance of
            # a duplicate storage. Since this legacy serialization format saves
            # the numel of the storage, rather than nbytes directly, we'll be
            # effectively saving nbytes in this case.  We'll be able to load it
            # and the tensor back up with no problems in _this_ and future
            # versions of pytorch, but in older versions, here's the problem:
            # the storage will be loaded up as a UntypedStorage, and then the
            # FloatTensor will loaded and the UntypedStorage will be assigned to
            # it. Since the storage dtype does not match the tensor dtype, this
            # will cause an error.  If we reverse the list, like `[tensor,
            # storage]`, then we will save the `tensor.storage()` as a faked
            # `FloatStorage`, and the saved size will be the correct
            # dtype-specific numel count that old versions expect. `tensor`
            # will be able to load up properly in old versions, pointing to
            # a FloatStorage. However, `storage` is still being translated to
            # a UntypedStorage, and it will try to resolve to the same
            # FloatStorage that `tensor` contains. This will also cause an
            # error. It doesn't seem like there's any way around this.
            # Probably, we just cannot maintain FC for the legacy format if the
            # saved list contains both a tensor and a storage that point to the
            # same data.  We should still be able to maintain FC for lists of
            # just tensors, as long as all views share the same dtype as the
            # tensor they are viewing.

            if storage_key not in serialized_storages:
                serialized_storages[storage_key] = (storage, dtype)
            is_view = storage._cdata != storage._cdata
            if is_view:
                view_metadata = (str(storage._cdata), offset, storage.nbytes())
            else:
                view_metadata = None

            res = (
                "storage",
                storage_type,
                storage_key,
                location,
                storage_numel,
                view_metadata,
            )
            return res
        return None

    sys_info = dict(
        protocol_version=PROTOCOL_VERSION,
        little_endian=sys.byteorder == "little",
        type_sizes=dict(
            short=SHORT_SIZE,
            int=INT_SIZE,
            long=LONG_SIZE,
        ),
    )

    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)

    class PyTorchLegacyPickler(pickle_module.Pickler):
        def persistent_id(self, obj):
            return persistent_id(obj)

    pickler = PyTorchLegacyPickler(f, protocol=pickle_protocol)
    pickler.dump(obj)

    serialized_storage_keys = sorted(serialized_storages.keys())
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    f.flush()
    for key in serialized_storage_keys:
        storage, dtype = serialized_storages[key]
        storage._write_file(
            f, _should_read_directly(f), True, torch._utils._element_size(dtype)
        )


def _save(
    obj,
    zip_file,
    pickle_module,
    pickle_protocol,
    _disable_byteorder_record,
):
    serialized_storages = {}
    id_map: Dict[int, str] = {}

    # Since loading storages that view the same data with different dtypes is
    # not supported, we need to keep track of the dtype associated with each
    # storage data_ptr and throw an error if the dtype is ever different.
    # TODO: This feature could be added in the future
    storage_dtypes: Dict[int, torch.dtype] = {}

    def persistent_id(obj):
        # FIXME: the docs say that persistent_id should only return a string
        # but torch store returns tuples. This works only in the binary protocol
        # see
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: Once we decide to break serialization FC, this case
                # can be deleted
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # If storage is allocated, ensure that any other saved storages
            # pointing to the same data all have the same dtype. If storage is
            # not allocated, don't perform this check
            if str(storage.device) != "meta" and storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            if hasattr(obj, "_fake_device") and obj._fake_device is not None:
                location = str(obj._fake_device)
            else:
                location = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None

    # Write the pickle data for `obj`
    data_buf = io.BytesIO()

    class PyTorchPickler(pickle_module.Pickler):  # type: ignore[name-defined]
        def persistent_id(self, obj):
            return persistent_id(obj)

    pickler = PyTorchPickler(data_buf, protocol=pickle_protocol)
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    zip_file.write_record("data.pkl", data_value, len(data_value))

    # Write byte order marker
    if not _disable_byteorder_record:
        if sys.byteorder not in ["little", "big"]:
            raise ValueError("Unknown endianness type: " + sys.byteorder)

        zip_file.write_record("byteorder", sys.byteorder, len(sys.byteorder))

    # Write each tensor to a file named tensor/the_tensor_key in the zip archive
    for key in sorted(serialized_storages.keys()):
        name = f"data/{key}"
        storage = serialized_storages[key]
        num_bytes = storage.nbytes()
        global _serialization_tls
        if _serialization_tls.skip_data:
            zip_file.write_record_metadata(name, num_bytes)
        else:
            # given that we copy things around anyway, we might use storage.cpu()
            # this means to that to get tensors serialized, you need to implement
            # .cpu() on the underlying Storage
            if storage.device.type != "cpu":
                storage = storage.cpu()
            # Now that it is on the CPU we can directly copy it into the zip file
            zip_file.write_record(name, storage, num_bytes)


def load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: Optional[bool] = None,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)

    Loads an object saved with :func:`torch.save` from a file.

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

    If :attr:`map_location` is a :class:`torch.device` object or a string containing
    a device tag, it indicates the location where all tensors should be loaded.

    Otherwise, if :attr:`map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using :func:`torch.serialization.register_package`.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
        map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the :attr:`pickle_module` used to serialize file)
        weights_only: Indicates whether unpickler should be restricted to
            loading only tensors, primitive types, dictionaries
            and any types added via :func:`torch.serialization.add_safe_globals`.
            See :ref:`weights-only` for more details.
        mmap: Indicates whether the file should be mmaped rather than loading all the storages into memory.
            Typically, tensor storages in the file will first be moved from disk to CPU memory, after which they
            are moved to the location that they were tagged with when saving, or specified by ``map_location``. This
            second step is a no-op if the final location is CPU. When the ``mmap`` flag is set, instead of copying the
            tensor storages from disk to CPU memory in the first step, ``f`` is mmaped.
        pickle_load_args: (Python 3 only) optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`, e.g.,
            :attr:`errors=...`.

    .. warning::
        :func:`torch.load()` unless `weights_only` parameter is set to `True`,
        uses ``pickle`` module implicitly, which is known to be insecure.
        It is possible to construct malicious pickle data which will execute arbitrary code
        during unpickling. Never load data that could have come from an untrusted
        source in an unsafe mode, or that could have been tampered with. **Only load data you trust**.

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
        >>> # xdoctest: +SKIP("undefined filepaths")
        >>> torch.load("tensors.pt", weights_only=True)
        # Load all tensors onto the CPU
        >>> torch.load("tensors.pt", map_location=torch.device("cpu"), weights_only=True)
        # Load all tensors onto the CPU, using a function
        >>> torch.load(
        ...     "tensors.pt", map_location=lambda storage, loc: storage, weights_only=True
        ... )
        # Load all tensors onto GPU 1
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location=lambda storage, loc: storage.cuda(1),
        ...     weights_only=True,
        ... )  # type: ignore[attr-defined]
        # Map tensors from GPU 1 to GPU 0
        >>> torch.load("tensors.pt", map_location={"cuda:1": "cuda:0"}, weights_only=True)
        # Load tensor from io.BytesIO object
        # Loading from a buffer setting weights_only=False, warning this can be unsafe
        >>> with open("tensor.pt", "rb") as f:
        ...     buffer = io.BytesIO(f.read())
        >>> torch.load(buffer, weights_only=False)
        # Load a module with 'ascii' encoding for unpickling
        # Loading from a module setting weights_only=False, warning this can be unsafe
        >>> torch.load("module.pt", encoding="ascii", weights_only=False)
    """
    torch._C._log_api_usage_once("torch.load")
    UNSAFE_MESSAGE = (
        "Re-running `torch.load` with `weights_only` set to `False` will likely succeed, "
        "but it can result in arbitrary code execution. Do it only if you got the file from a "
        "trusted source."
    )
    DOCS_MESSAGE = (
        "\n\nCheck the documentation of torch.load to learn more about types accepted by default with "
        "weights_only https://pytorch.org/docs/stable/generated/torch.load.html."
    )

    def _get_wo_message(message: str) -> str:
        unsafe_global_pattern = r"GLOBAL (\S+) was not an allowed global by default."
        has_unsafe_global = re.search(unsafe_global_pattern, message) is not None
        blocklist_pattern = r"whose module (\S+) is blocked"
        has_blocklist = re.search(blocklist_pattern, message) is not None
        import_pattern = r"(\S+) must be (\S+) to load"
        has_import = re.search(import_pattern, message) is not None
        if has_unsafe_global:
            updated_message = (
                "Weights only load failed. This file can still be loaded, to do so you have two options, "
                "\033[1mdo those steps only if you trust the source of the checkpoint\033[0m. "
                f"\n\t(1) {UNSAFE_MESSAGE}\n\t(2) Alternatively, to load with `weights_only=True` please check "
                "the recommended steps in the following error message.\n\tWeightsUnpickler error: "
                + message
            )
        else:
            if has_import:
                return f"Weights only load failed. {message}"
            else:
                updated_message = f"Weights only load failed. {UNSAFE_MESSAGE}\n"
                if not has_blocklist:
                    updated_message += (
                        "Please file an issue with the following so that we can make "
                        "`weights_only=True` compatible with your use case: WeightsUnpickler error: "
                    )
            updated_message += message
        return updated_message + DOCS_MESSAGE

    global _serialization_tls
    skip_data = _serialization_tls.skip_data
    if skip_data:
        raise RuntimeError(
            "`torch.load` called within a torch.serialization.skip_data context manager "
            "is not supported yet. Please call torch.load outside the skip_data context manager."
        )

    weights_only_not_set = weights_only is None

    if weights_only_not_set:
        weights_only = _default_to_weights_only(pickle_module)

    true_values = ["1", "y", "yes", "true"]
    # Add ability to force safe only or non-safe weight loads via environment variables
    force_weights_only_load = (
        os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0") in true_values
    )
    force_no_weights_only_load = (
        os.getenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "0") in true_values
    )

    if force_weights_only_load and force_no_weights_only_load:
        raise RuntimeError(
            "Only one of `TORCH_FORCE_WEIGHTS_ONLY_LOAD` or `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` "
            "should be set, but both were set."
        )
    elif force_weights_only_load:
        weights_only = True
    elif force_no_weights_only_load:
        # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD can only override if callsite did not explicitly set weights_only
        if weights_only_not_set:
            warnings.warn(
                "Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the"
                "`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.",
                UserWarning,
                stacklevel=2,
            )
            weights_only = False

    if weights_only:
        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )
    else:
        if pickle_module is None:
            pickle_module = pickle

    # make flipping default BC-compatible
    if mmap is None:
        mmap = False

    _check_dill_version(pickle_module)

    if "encoding" not in pickle_load_args.keys():
        pickle_load_args["encoding"] = "utf-8"

    with _open_file_like(f, "rb") as opened_file:
        if _is_zipfile(opened_file):
            # The zipfile reader is going to advance the current file position.
            # If we want to actually tail call to torch.jit.load, we need to
            # reset back to the original position.
            orig_position = opened_file.tell()
            overall_storage = None
            with _open_zipfile_reader(opened_file) as opened_zipfile:
                if _is_torchscript_zip(opened_zipfile):
                    warnings.warn(
                        "'torch.load' received a zip file that looks like a TorchScript archive"
                        " dispatching to 'torch.jit.load' (call 'torch.jit.load' directly to"
                        " silence this warning)",
                        UserWarning,
                    )
                    opened_file.seek(orig_position)
                    return torch.jit.load(opened_file, map_location=map_location)
                if mmap:
                    if not _is_path(f):
                        raise ValueError(
                            "f must be a file path in order to use the mmap argument"
                        )
                    size = os.path.getsize(f)
                    if not IS_WINDOWS:
                        shared = get_default_mmap_options() == MAP_SHARED
                    else:
                        shared = False
                    overall_storage = torch.UntypedStorage.from_file(
                        os.fspath(f), shared, size
                    )
                if weights_only:
                    try:
                        return _load(
                            opened_zipfile,
                            map_location,
                            _weights_only_unpickler,
                            overall_storage=overall_storage,
                            **pickle_load_args,
                        )
                    except pickle.UnpicklingError as e:
                        raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
                return _load(
                    opened_zipfile,
                    map_location,
                    pickle_module,
                    overall_storage=overall_storage,
                    **pickle_load_args,
                )
        if mmap:
            f_name = "" if not isinstance(f, str) else f"{f}, "
            raise RuntimeError(
                "mmap can only be used with files saved with "
                f"`torch.save({f_name}_use_new_zipfile_serialization=True), "
                "please torch.save your checkpoint with this option in order to use mmap."
            )
        if weights_only:
            try:
                return _legacy_load(
                    opened_file,
                    map_location,
                    _weights_only_unpickler,
                    **pickle_load_args,
                )
            except pickle.UnpicklingError as e:
                raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
        return _legacy_load(
            opened_file, map_location, pickle_module, **pickle_load_args
        )


# Register pickling support for layout instances such as
# torch.sparse_coo, etc
def _get_layout(name):
    """Get layout extension object from its string representation."""
    cache = _get_layout.cache  # type: ignore[attr-defined]
    if not cache:
        for v in torch.__dict__.values():
            if isinstance(v, torch.layout):
                cache[str(v)] = v
    return cache[name]


# There are yet not good way to type annotate function attributes https://github.com/python/mypy/issues/2087
_get_layout.cache = {}  # type: ignore[attr-defined]
copyreg.pickle(torch.layout, lambda obj: (_get_layout, (str(obj),)))


def _legacy_load(f, map_location, pickle_module, **pickle_load_args):
    deserialized_objects: Dict[int, Any] = {}

    restore_location = _get_restore_location(map_location)

    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            return super().find_class(mod_name, name)

    def _check_container_source(container_type, source_file, original_source):
        try:
            current_source = "".join(get_source_lines_and_file(container_type)[0])
        except Exception:  # saving the source is optional, so we can ignore any errors
            warnings.warn(
                "Couldn't retrieve source code for container of "
                "type " + container_type.__name__ + ". It won't be checked "
                "for correctness upon loading."
            )
            return
        if original_source != current_source:
            if container_type.dump_patches:
                file_name = container_type.__name__ + ".patch"
                diff = difflib.unified_diff(
                    current_source.split("\n"),
                    original_source.split("\n"),
                    source_file,
                    source_file,
                    lineterm="",
                )
                lines = "\n".join(diff)
                try:
                    with open(file_name, "a+") as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise OSError
                    msg = (
                        "Saved a reverse patch to " + file_name + ". "
                        "Run `patch -p0 < " + file_name + "` to revert your "
                        "changes."
                    )
                except OSError:
                    msg = (
                        "Tried to save a patch, but couldn't create a "
                        "writable file " + file_name + ". Make sure it "
                        "doesn't exist and your working directory is "
                        "writable."
                    )
            else:
                msg = (
                    "you can retrieve the original source code by "
                    "accessing the object's source attribute or set "
                    "`torch.nn.Module.dump_patches = True` and use the "
                    "patch tool to revert the changes."
                )
            msg = f"source code of class '{torch.typename(container_type)}' has changed. {msg}"
            warnings.warn(msg, SourceChangeWarning)

    def legacy_load(f):
        deserialized_objects: Dict[int, Any] = {}

        def persistent_load(saved_id):
            if isinstance(saved_id, tuple):
                # Ignore containers that don't have any sources saved
                if all(saved_id[1:]):
                    _check_container_source(*saved_id)
                return saved_id[0]
            return deserialized_objects[int(saved_id)]

        with closing(
            tarfile.open(fileobj=f, mode="r:", format=tarfile.PAX_FORMAT)
        ) as tar, mkdtemp() as tmpdir:
            tar.extract("storages", path=tmpdir)
            with open(os.path.join(tmpdir, "storages"), "rb", 0) as f:
                num_storages = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_storages):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, location, storage_type = args
                    dtype = storage_type._dtype
                    obj = cast(Storage, torch.UntypedStorage)._new_with_file(
                        f, torch._utils._element_size(dtype)
                    )
                    obj = restore_location(obj, location)
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[key] = torch.storage.TypedStorage(
                        wrap_storage=obj, dtype=dtype, _internal=True
                    )

                storage_views = pickle_module.load(f, **pickle_load_args)
                for target_cdata, root_cdata, offset, numel in storage_views:
                    root = deserialized_objects[root_cdata]
                    element_size = torch._utils._element_size(root.dtype)
                    offset_bytes = offset * element_size
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[target_cdata] = torch.storage.TypedStorage(
                        wrap_storage=root._untyped_storage[
                            offset_bytes : offset_bytes + numel * element_size
                        ],
                        dtype=root.dtype,
                        _internal=True,
                    )

            tar.extract("tensors", path=tmpdir)
            with open(os.path.join(tmpdir, "tensors"), "rb", 0) as f:
                num_tensors = pickle_module.load(f, **pickle_load_args)
                for _ in range(num_tensors):
                    args = pickle_module.load(f, **pickle_load_args)
                    key, storage_id, _original_tensor_type = args
                    storage = deserialized_objects[storage_id]
                    (ndim,) = struct.unpack("<i", f.read(4))
                    # skip next 4 bytes; legacy encoding treated ndim as 8 bytes
                    f.read(4)
                    numel = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
                    stride = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
                    (storage_offset,) = struct.unpack("<q", f.read(8))
                    tensor = torch.empty((0,), dtype=storage.dtype).set_(
                        storage._untyped_storage, storage_offset, numel, stride
                    )
                    deserialized_objects[key] = tensor

            pickle_file = tar.extractfile("pickle")
            unpickler = UnpicklerWrapper(pickle_file, **pickle_load_args)
            unpickler.persistent_load = persistent_load
            result = unpickler.load()
            return result

    deserialized_objects = {}

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        if typename == "module":
            # Ignore containers that don't have any sources saved
            if all(data[1:]):
                _check_container_source(*data)
            return data[0]
        elif typename == "storage":
            storage_type, root_key, location, numel, view_metadata = data
            location = _maybe_decode_ascii(location)
            dtype = storage_type.dtype

            nbytes = numel * torch._utils._element_size(dtype)

            if root_key not in deserialized_objects:
                if torch._guards.active_fake_mode() is not None:
                    obj = cast(Storage, torch.UntypedStorage(nbytes, device="meta"))
                else:
                    obj = cast(Storage, torch.UntypedStorage(nbytes))
                    obj._torch_load_uninitialized = True
                    obj = restore_location(obj, location)
                # TODO: Once we decide to break serialization FC, we can
                # stop wrapping with TypedStorage
                typed_storage = torch.storage.TypedStorage(
                    wrap_storage=obj, dtype=dtype, _internal=True
                )
                deserialized_objects[root_key] = typed_storage
            else:
                typed_storage = deserialized_objects[root_key]
                if typed_storage._data_ptr() == 0:
                    typed_storage = torch.storage.TypedStorage(
                        device=typed_storage._untyped_storage.device,
                        dtype=dtype,
                        _internal=True,
                    )

            if view_metadata is not None:
                view_key, offset, view_size = view_metadata
                offset_bytes = offset * torch._utils._element_size(dtype)
                view_size_bytes = view_size * torch._utils._element_size(dtype)
                if view_key not in deserialized_objects:
                    # TODO: Once we decide to break serialization FC, we can
                    # stop wrapping with TypedStorage
                    deserialized_objects[view_key] = torch.storage.TypedStorage(
                        wrap_storage=typed_storage._untyped_storage[
                            offset_bytes : offset_bytes + view_size_bytes
                        ],
                        dtype=dtype,
                        _internal=True,
                    )
                res = deserialized_objects[view_key]

            else:
                res = typed_storage
            return res
        else:
            raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    _check_seekable(f)
    f_should_read_directly = _should_read_directly(f)

    if f_should_read_directly and f.tell() == 0:
        # legacy_load requires that f has fileno()
        # only if offset is zero we can attempt the legacy tar file loader
        try:
            return legacy_load(f)
        except tarfile.TarError:
            if _is_zipfile(f):
                # .zip is used for torch.jit.save and will throw an un-pickling error here
                raise RuntimeError(
                    f"{f.name} is a zip archive (did you mean to use torch.jit.load()?)"
                ) from None
            # if not a tarfile, reset file offset and proceed
            f.seek(0)

    if not hasattr(f, "readinto") and (3, 8, 0) <= sys.version_info < (3, 8, 2):
        raise RuntimeError(
            "torch.load does not work with file-like objects that do not implement readinto on Python 3.8.0 and 3.8.1. "
            f'Received object of type "{type(f)}". Please update to Python 3.8.2 or newer to restore this '
            "functionality."
        )

    magic_number = pickle_module.load(f, **pickle_load_args)
    if magic_number != MAGIC_NUMBER:
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f, **pickle_load_args)
    if protocol_version != PROTOCOL_VERSION:
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")

    _sys_info = pickle_module.load(f, **pickle_load_args)
    unpickler = UnpicklerWrapper(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()

    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)

    if torch._guards.active_fake_mode() is None:
        offset = f.tell() if f_should_read_directly else None
        for key in deserialized_storage_keys:
            assert key in deserialized_objects
            typed_storage = deserialized_objects[key]
            typed_storage._untyped_storage._set_from_file(
                f,
                offset,
                f_should_read_directly,
                torch._utils._element_size(typed_storage.dtype),
            )
            if offset is not None:
                offset = f.tell()

    torch._utils._validate_loaded_sparse_tensors()

    return result


def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    # When using encoding='bytes' in Py3, some **internal** keys stored as
    # strings in Py2 are loaded as bytes. This function decodes them with
    # ascii encoding, one that Py3 uses by default.
    #
    # NOTE: This should only be used on internal keys (e.g., `typename` and
    #       `location` in `persistent_load` below!
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


def _get_restore_location(map_location):
    if map_location is None:
        restore_location = default_restore_location
    elif isinstance(map_location, dict):

        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)

    elif isinstance(map_location, (str, bytes)):

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

    return restore_location


class StorageType:
    def __init__(self, name):
        self._dtype = _get_dtype_from_pickle_storage_type(name)

    @property
    def dtype(self):
        return self._dtype

    def __str__(self):
        return f"StorageType(dtype={self.dtype})"


def _load(
    zip_file,
    map_location,
    pickle_module,
    pickle_file="data.pkl",
    overall_storage=None,
    **pickle_load_args,
):
    restore_location = _get_restore_location(map_location)

    loaded_storages = {}

    # check if byteswapping is needed
    byteordername = "byteorder"
    byteorderdata = None
    if zip_file.has_record(byteordername):
        byteorderdata = zip_file.get_record(byteordername)
        if byteorderdata not in [b"little", b"big"]:
            raise ValueError("Unknown endianness type: " + byteorderdata.decode())
    elif (
        get_default_load_endianness() == LoadEndianness.LITTLE
        or get_default_load_endianness() is None
    ):
        byteorderdata = b"little"
    elif get_default_load_endianness() == LoadEndianness.BIG:
        byteorderdata = b"big"
    elif get_default_load_endianness() == LoadEndianness.NATIVE:
        pass
    else:
        raise ValueError("Invalid load endianness type")

    if (
        not zip_file.has_record(byteordername)
        and get_default_load_endianness() is None
        and sys.byteorder == "big"
    ):
        # Default behaviour was changed
        # See https://github.com/pytorch/pytorch/issues/101688
        warnings.warn(
            "The default load endianness for checkpoints without a byteorder mark "
            "on big endian machines was changed from 'native' to 'little' endian, "
            "to avoid this behavior please use "
            "torch.serialization.set_default_load_endianness to set "
            "the desired default load endianness",
            UserWarning,
        )

    def load_tensor(dtype, numel, key, location):
        name = f"data/{key}"
        if torch._guards.detect_fake_mode(None) is not None:
            nbytes = numel * torch._utils._element_size(dtype)
            storage = torch.UntypedStorage(nbytes, device="meta")
        elif overall_storage is not None:
            storage_offset = zip_file.get_record_offset(name)
            storage = overall_storage[storage_offset : storage_offset + numel]
        else:
            storage = (
                zip_file.get_storage_from_record(name, numel, torch.UntypedStorage)
                ._typed_storage()
                ._untyped_storage
            )
        # swap here if byteswapping is needed
        if byteorderdata is not None:
            if byteorderdata.decode() != sys.byteorder:
                storage.byteswap(dtype)

        # TODO: Once we decide to break serialization FC, we can
        # stop wrapping with TypedStorage
        typed_storage = torch.storage.TypedStorage(
            wrap_storage=restore_location(storage, location),
            dtype=dtype,
            _internal=True,
        )

        if typed_storage._data_ptr() != 0:
            loaded_storages[key] = typed_storage

        return typed_storage

    def persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = _maybe_decode_ascii(saved_id[0])
        data = saved_id[1:]

        assert (
            typename == "storage"
        ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        storage_type, key, location, numel = data
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            nbytes = numel * torch._utils._element_size(dtype)
            typed_storage = load_tensor(
                dtype, nbytes, key, _maybe_decode_ascii(location)
            )

        return typed_storage

    load_module_mapping: Dict[str, str] = {
        # See https://github.com/pytorch/pytorch/pull/51633
        "torch.tensor": "torch._tensor"
    }

    # Need to subclass Unpickler instead of directly monkey-patching the find_class method
    # because it's marked readonly in pickle.
    # The type: ignore is because mypy can't statically determine the type of this class.
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # from https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # Lets us override the imports that pickle uses when unpickling an object.
        # This is useful for maintaining BC if we change a module path that tensor instantiation relies on.
        def find_class(self, mod_name, name):
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            mod_name = load_module_mapping.get(mod_name, mod_name)
            return super().find_class(mod_name, name)

    # Load the data (which may in turn use `persistent_load` to load tensors)
    data_file = io.BytesIO(zip_file.get_record(pickle_file))

    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    # Needed for tensors where storage device and rebuild tensor device are
    # not connected (wrapper subclasses and tensors rebuilt using numpy)
    global _serialization_tls
    _serialization_tls.map_location = map_location
    result = unpickler.load()
    _serialization_tls.map_location = None

    torch._utils._validate_loaded_sparse_tensors()
    torch._C._log_api_usage_metadata(
        "torch.load.metadata", {"serialization_id": zip_file.serialization_id()}
    )
    return result


def _is_torchscript_zip(zip_file):
    return "constants.pkl" in zip_file.get_all_records()
