import sys
import os
import fnmatch
import warnings
import tarfile
import zipfile
from typing import List, Iterable, Union, Any
from io import IOBase

def match_masks(name : str, masks : Union[str, List[str]]):
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
        abspath: bool = False) :

    def onerror(err : OSError):
        warnings.warn(err.strerror + " : " + err.filename)

    for path, dirs, files in os.walk(root, onerror=onerror):
        if abspath:
            path = os.path.abspath(path)
        for f in files:
            if match_masks(f, masks):
                yield os.path.join(path, f)
        if not recursive:
            break


def extract_files_from_single_pathname_binary(
        pathname : str,
        binary_stream : Any,
        extract_tars : bool = True,
        extract_zips : bool = True,
        recursive : bool = True,
        mask_tars : Union[str, List[str]] = '',
        mask_zips : Union[str, List[str]] = ''):

    # test whether binary_stream is seekable (eg. PIPE stream from webdata is not seekable)
    seekable = hasattr(binary_stream, "seekable") and binary_stream.seekable()

    # Note: We have no way to verify whether a non-seekable stream (eg. PIPE stream) is tar/zip without
    #       changing stream handle position, however, there is no way to move such stream's handle back.
    #       For non-seekable stream, the entire stream will be discard if it can not be extracted

    # extract tar file
    if extract_tars and match_masks(pathname, mask_tars):
        try:
            with tarfile.open(fileobj=binary_stream) as tar:
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue

                    extract_fobj = tar.extractfile(tarinfo)
                    if extract_fobj is None:
                        warnings.warn("failed to extract tar file {}".format(tarinfo.name))
                        continue

                    inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                    if recursive:
                        yield from extract_files_from_single_pathname_binary(
                            inner_pathname, extract_fobj, extract_tars, extract_zips, recursive, mask_tars, mask_zips)
                    else:
                        yield (inner_pathname, extract_fobj)
            return
        except tarfile.TarError:
            if not seekable:
                warnings.warn("Unable to reset the non-tarfile stream {}, skip!".format(pathname))
                return
            binary_stream.seek(0)

    # extract zip file
    if extract_zips and match_masks(pathname, mask_zips):
        try:
            with zipfile.ZipFile(binary_stream) as zips:
                for zipinfo in zips.infolist():
                    # major version should always be 3 here.
                    if (sys.version_info[1] < 6 and zipinfo.filename.endswith('/')) or zipinfo.is_dir():
                        continue

                    inner_pathname = os.path.normpath(os.path.join(pathname, zipinfo.filename))
                    if recursive:
                        yield from extract_files_from_single_pathname_binary(
                            inner_pathname, zips.open(zipinfo), extract_tars, extract_zips, recursive, mask_tars, mask_zips)
                    else:
                        yield (inner_pathname, zips.open(zipinfo))
            return
        except zipfile.BadZipFile:
            if not seekable:
                warnings.warn("Unable to reset the non-zip stream {}, skip!".format(pathname))
                return
            binary_stream.seek(0)

    yield (pathname, binary_stream)


def get_file_binaries_from_pathnames(
        pathnames : Iterable,
        masks : Union[str, List[str]] = '',
        unzip_tars : bool = False,
        unzip_zips : bool = False,
        unzip_recursive : bool = False,
        unzip_mask_tars : Union[str, List[str]] = '',
        unzip_mask_zips : Union[str, List[str]] = ''):

    if not isinstance(pathnames, Iterable):
        warnings.warn("get_file_binaries_from_pathnames needs the input be an Iterable")
        return

    for pathname in pathnames:
        if not isinstance(pathname, str):
            warnings.warn("file pathname is skipped due to unsupported type {}".format(type(pathname)))
            continue

        try:
            with open(pathname, 'rb') as f:
                if not match_masks(pathname, masks):
                    continue
                if unzip_tars or unzip_zips:
                    yield from extract_files_from_single_pathname_binary(
                        pathname, f, unzip_tars, unzip_zips, unzip_recursive, unzip_mask_tars, unzip_mask_zips)
                else:
                    yield (pathname, f)
        except Exception as e:
            warnings.warn("skip file {} due to {}".format(pathname, e))


def extract_files_from_pathname_binaries(
        pathname_binaries : Iterable,
        masks : Union[str, List[str]] = '',
        unzip_tars : bool = False,
        unzip_zips : bool = False,
        unzip_recursive : bool = False,
        unzip_mask_tars : Union[str, List[str]] = '',
        unzip_mask_zips : Union[str, List[str]] = ''):

    if not isinstance(pathname_binaries, Iterable):
        warnings.warn("extract_files_from_pathname_binaries needs the input be an Iterable")
        return

    for rec in pathname_binaries:
        if not (isinstance(rec, tuple) and len(rec) == 2 and isinstance(rec[0], str) and isinstance(rec[1], IOBase)):
            warnings.warn("encounter invalid file name and binary record, skip!")
            continue

        if not match_masks(rec[0], masks):
            continue

        yield from extract_files_from_single_pathname_binary(
            rec[0], rec[1], unzip_tars, unzip_zips, unzip_recursive, unzip_mask_tars, unzip_mask_zips)
