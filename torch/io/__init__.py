import io
import os
from typing import Union, Type
from typing_extensions import TypeIs

from .file import *
from .hdfs import HDFSOpener
from .http import HTTPOpener
from .opener import Opener

__all__ = [
    "is_path",
    "is_protocol",
    "open_file_like",
    "open_zipfile_writer",
    "open_zipfile_reader",
    "check_seekable"
]

supported_protocol = {"hdfs": HDFSOpener, "http": HTTPOpener}


def is_path(file_like) -> TypeIs[Union[str, os.PathLike]]:
    if not isinstance(file_like, (str, os.PathLike)):
        return False

    return not is_protocol(file_like)


def is_protocol(file_like) -> bool:
    if not isinstance(file_like, str):
        return False

    return '://' in file_like


def _open_protocol(url: str, mode) -> Opener:
    url_list = url.split('://')
    if len(url_list) != 2 or url_list[0] not in supported_protocol:
        raise RuntimeError("unsupported protocol: %s" % url)

    return supported_protocol[url_list[0]](url, mode)


def open_file_like(file_like, mode):
    if is_path(file_like):
        return OpenFile(file_like, mode)
    elif is_protocol(file_like):
        return _open_protocol(file_like, mode)
    else:
        if "w" in mode:
            return OpenBufferWriter(file_like)
        elif "r" in mode:
            return OpenBufferReader(file_like)
        else:
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")


def open_zipfile_writer(file_like):
    container: Type[Opener]
    if is_path(file_like):
        container = OpenZipfileWriterFile
    else:
        container = OpenZipfileWriterBufferOrProtocol
    return container(file_like)


def open_zipfile_reader(file_like) -> Type[Opener]:
    return OpenZipfileReader(file_like)
