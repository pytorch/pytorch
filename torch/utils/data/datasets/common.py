import os
import sys
import fnmatch
import warnings
import tarfile
import zipfile
from typing import List, Union, Iterable, Any, Callable
from io import BufferedIOBase


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

        yield (pathname, open(pathname, 'rb'))


def validate_pathname_binary(rec):
    if not isinstance(rec, tuple):
        return "pathname_binary should be tuple type, but got {}".format(type(rec))
    if len(rec) != 2:
        return "pathname_binary tuple length should be 2, but got {}".format(str(len(rec)))
    if not isinstance(rec[0], str):
        return "pathname_binary should have string type pathname, but got {}".format(type(rec[0]))
    if not isinstance(rec[1], BufferedIOBase):
        return "pathname_binary should have BufferedIOBase based binary type, but got {}".format(type(rec[1]))
    return ""


def extract_files_from_pathname_binaries(
        pathname_binaries : Iterable,
        src_file_handle_register : List[tuple],
        extract_fn : Callable):
    if not isinstance(pathname_binaries, Iterable):
        warnings.warn("pathname_binaries must be Iterable type got {}".format(type(pathname_binaries)))
        raise TypeError

    for rec in pathname_binaries:
        ret = validate_pathname_binary(rec)
        if ret:
            warnings.warn("encounter invalid pathname and binary record ({}), abort!".format(ret))
            raise TypeError
        yield from extract_fn(rec[0], rec[1], src_file_handle_register)


def extract_files_from_single_tar_pathname_binary(
        pathname : str,
        binary_stream : Any,
        tarfile_handle_register : List[tuple]):

    try:
        tar = tarfile.open(fileobj=binary_stream, mode="r:*")
        tarfile_handle_register.append((pathname, tar))

        for tarinfo in tar:
            if not tarinfo.isfile():
                continue

            extract_fobj = tar.extractfile(tarinfo)
            if extract_fobj is None:
                warnings.warn("failed to extract tar file {}".format(tarinfo.name))
                raise tarfile.ExtractError

            inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
            yield (inner_pathname, extract_fobj)
    except tarfile.TarError as e:
        warnings.warn("Unable to extract files from corrupted tarfile stream {}, abort!".format(pathname))
        raise e


def extract_files_from_single_zip_pathname_binary(
        pathname : str,
        binary_stream : Any,
        zipfile_handle_register : List[tuple]):

    try:
        zips = zipfile.ZipFile(binary_stream)
        zipfile_handle_register.append((pathname, zips))

        for zipinfo in zips.infolist():
            # major version should always be 3 here.
            if (sys.version_info[1] < 6 and zipinfo.filename.endswith('/')) or zipinfo.is_dir():
                continue

            inner_pathname = os.path.normpath(os.path.join(pathname, zipinfo.filename))
            yield (inner_pathname, zips.open(zipinfo))
    except zipfile.BadZipFile as e:
        warnings.warn("Unable to extract files from corrupted zipfile stream {}, abort!".format(pathname))
        raise e
