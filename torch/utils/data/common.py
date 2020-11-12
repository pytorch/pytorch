import sys
import io
import os
import fnmatch
import warnings
import tarfile
import zipfile
from typing import List, Tuple, Iterable


def get_file_pathnames_from_root(
    root: str,
    mask: str,
    recursive: bool = False,
    abspath: bool = False) -> List:

    def onerror(err : OSError):
        warnings.warn(err.strerror + " : " +  err.filename)

    pathnames = []
    for path, dirs, files in os.walk(root, onerror=onerror):
        if abspath:
            path = os.path.abspath(path)
        for f in files:
            if not mask or fnmatch.fnmatch(f, mask):
                pathnames.append(os.path.join(path, f))
        if not recursive:
            break
    return pathnames


def extract_files_from_single_pathname_binary(
    pathname : str,
    binary : bytes,
    recursive : bool = True) -> List[Tuple[str, bytes]]:
    ret : List[Tuple[str, bytes]] = []

    # verify whether file obj is tar
    def is_tarfile(binary_bytes):
        fobj = io.BytesIO(binary_bytes)
        try:
            t = tarfile.open(fileobj=fobj)
            t.close()
        except tarfile.TarError:
            return False
        return True

    # verify whether file obj is zip
    def is_zipfile(binary_bytes):
        fobj = io.BytesIO(binary_bytes)
        ret = zipfile.is_zipfile(fobj)
        fobj.close()
        return ret


    # extract tar file
    if is_tarfile(binary):
        fobj = io.BytesIO(binary)
        with tarfile.open(fileobj=fobj) as tar:
            for tarinfo in tar:
                if not tarinfo.isfile():
                    continue

                extract_binary = tar.extractfile(tarinfo).read()
                if extract_binary is None:
                    warnings.warn("failed to extract tar file {}".format(tarinfo.name))
                    continue

                inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
                if recursive:
                    inner_ret = extract_files_from_single_pathname_binary(
                        inner_pathname, extract_binary, recursive)
                    ret.extend(inner_ret)
                else:
                    ret.append((inner_pathname, extract_binary))
        return ret

    # extract zip file
    if is_zipfile(binary):
        fobj = io.BytesIO(binary)
        with zipfile.ZipFile(fobj) as zips:
            for zipinfo in zips.infolist():
                # major version should always be 3 here.
                if (sys.version_info[1] < 6 and zipinfo.filename.endswith('/')) or zipinfo.is_dir():
                    continue

                inner_pathname = os.path.normpath(os.path.join(pathname, zipinfo.filename))
                if recursive:
                    inner_ret = extract_files_from_single_pathname_binary(
                        inner_pathname, zips.open(zipinfo).read(), recursive)
                    ret.extend(inner_ret)
                else:
                    ret.append((inner_pathname, zips.open(zipinfo).read()))
        return ret

    return [(pathname, binary)]

def get_file_binaries_from_pathnames(
    pathnames : Iterable,
    auto_extract : bool = False) -> List:

    if not isinstance(pathnames, Iterable):
        warnings.warn("get_file_binaries_from_pathnames needs the input be an Iterable")
        return None

    pathname_binaries = []
    for pathname in pathnames:
        if not isinstance(pathname, str):
            warnings.warn("file pathname is skipped due to unsupported type {}".format(type(pathname)))
            continue

        try:
            with open(pathname, 'rb') as f:
                if auto_extract:
                    pathname_binaries.extend(extract_files_from_single_pathname_binary(pathname, f.read()))
                else:
                    pathname_binaries.append((pathname, f.read()))
        except Exception as e:
            warnings.warn("skip file {} due to {}".format(pathname, e))

    return pathname_binaries

def extract_files_from_pathname_binaries(
    pathname_binaries : Iterable,
    recursive : bool = True) -> List:

    if not isinstance(pathname_binaries, Iterable):
        warnings.warn("extract_files_from_pathname_binaries needs the input be an Iterable")
        return None

    output_pathname_binaries = []
    for rec in pathname_binaries:
        if not (isinstance(rec, tuple) and len(rec) == 2 and isinstance(rec[0], str) and isinstance(rec[1], bytes)):
            warnings.warn("encounter invalid file name and binary record, skip!")

        ret = extract_files_from_single_pathname_binary(rec[0], rec[1], recursive)
        output_pathname_binaries.extend(ret)

    return output_pathname_binaries
