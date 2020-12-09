import os
import sys
import fnmatch
import warnings
import tarfile
import zipfile
from typing import List, Union, Iterable, Any, Callable
from io import IOBase

class StreamWrapper:
    # this is a wrapper class which wraps file/stream handle
    def __init__(self, stream):
        self.stream = stream
    def read(self, *args, **kw):
        res = self.stream.read(*args, **kw)
        return res
    def close(self):
        self.stream.close()
    def __del__(self):
        self.close()


def match_masks(name : str, masks : Union[str, List[str]]) -> bool:
    # empty mask matches any input name
    if not masks:
        return True

    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)

    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False

def get_file_pathnames_from_root(
        root: str,
        masks: Union[str, List[str]],
        recursive: bool = False,
        abspath: bool = False) -> Iterable[str]:

    # print out an error message and raise the error out
    def onerror(err : OSError):
        warnings.warn(err.filename + " : " + err.strerror)
        raise err

    for path, dirs, files in os.walk(root, onerror=onerror):
        if abspath:
            path = os.path.abspath(path)
        files.sort()
        for f in files:
            if match_masks(f, masks):
                yield os.path.join(path, f)
        if not recursive:
            break


def get_file_binaries_from_pathnames(pathnames : Iterable):

    if not isinstance(pathnames, Iterable):
        warnings.warn("get_file_binaries_from_pathnames needs the input be an Iterable")
        raise TypeError

    for pathname in pathnames:
        if not isinstance(pathname, str):
            warnings.warn("file pathname must be string type, but got {}".format(type(pathname)))
            raise TypeError

        yield (pathname, StreamWrapper(open(pathname, 'rb')))


def validate_pathname_binary(rec):
    if not isinstance(rec, tuple):
        return "pathname_binary should be tuple type, but got {}".format(type(rec))
    if len(rec) != 2:
        return "pathname_binary tuple length should be 2, but got {}".format(str(len(rec)))
    if not isinstance(rec[0], str):
        return "pathname_binary should have string type pathname, but got {}".format(type(rec[0]))
    if not isinstance(rec[1], IOBase):
        return "pathname_binary should have IOBase based binary type, but got {}".format(type(rec[1]))
    return ""


def extract_files_from_pathname_binaries(pathname_binaries : Iterable, extract_fn : Callable):
    if not isinstance(pathname_binaries, Iterable):
        warnings.warn("pathname_binaries must be Iterable type got {}".format(type(pathname_binaries)))
        raise TypeError

    for rec in pathname_binaries:
        ret = validate_pathname_binary(rec)
        if ret:
            warnings.warn("encounter invalid pathname and binary record ({}), abort!".format(ret))
            raise TypeError
        yield from extract_fn(rec[0], rec[1])


def extract_files_from_single_tar_pathname_binary(
        pathname : str,
        binary_stream : Any):
    # test whether binary_stream is seekable (eg. PIPE stream from webdata is not seekable)
    seekable = hasattr(binary_stream, "seekable") and binary_stream.seekable()

    try:
        with tarfile.open(fileobj=binary_stream) as tar:
            for tarinfo in tar:
                if not tarinfo.isfile():
                    continue

                extract_fobj = tar.extractfile(tarinfo)
                if extract_fobj is None:
                    warnings.warn("failed to extract tar file {}".format(tarinfo.name))
                    raise tarfile.ExtractError

                inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                yield (inner_pathname, extract_fobj)
            return
    except tarfile.TarError as e:
        # Note: We have no way to verify whether a non-seekable stream (eg. PIPE stream) is tar without
        #       changing stream handle position, however, there is no way to move such stream's handle back.
        #       So the entire tar extraction process will be aborted if a non-seekable stream is not tar exactable.
        if not seekable:
            warnings.warn("Unable to reset the non-tarfile stream {}, abort!".format(pathname))
            raise e
        binary_stream.seek(0)

    yield (pathname, binary_stream)


def extract_files_from_single_zip_pathname_binary(
        pathname : str,
        binary_stream : Any):
    # test whether binary_stream is seekable (eg. PIPE stream from webdata is not seekable)
    seekable = hasattr(binary_stream, "seekable") and binary_stream.seekable()

    try:
        with zipfile.ZipFile(binary_stream) as zips:
            for zipinfo in zips.infolist():
                # major version should always be 3 here.
                if (sys.version_info[1] < 6 and zipinfo.filename.endswith('/')) or zipinfo.is_dir():
                    continue

                inner_pathname = os.path.normpath(os.path.join(pathname, zipinfo.filename))
                yield (inner_pathname, zips.open(zipinfo))
            return
    except zipfile.BadZipFile as e:
        # Note: We have no way to verify whether a non-seekable stream (eg. PIPE stream) is zip without
        #       changing stream handle position, however, there is no way to move such stream's handle back.
        #       So the entire zip extraction process will be aborted if a non-seekable stream is not tar exactable.
        if not seekable:
            warnings.warn("Unable to reset the non-zip stream {}, skip!".format(pathname))
            raise e
        binary_stream.seek(0)

    yield (pathname, binary_stream)
