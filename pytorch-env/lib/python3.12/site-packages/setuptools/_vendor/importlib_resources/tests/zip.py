"""
Generate zip test data files.
"""

import contextlib
import os
import pathlib
import zipfile

import zipp


def make_zip_file(src, dst):
    """
    Zip the files in src into a new zipfile at dst.
    """
    with zipfile.ZipFile(dst, 'w') as zf:
        for src_path, rel in walk(src):
            dst_name = src.name / pathlib.PurePosixPath(rel.as_posix())
            zf.write(src_path, dst_name)
        zipp.CompleteDirs.inject(zf)
    return dst


def walk(datapath):
    for dirpath, dirnames, filenames in os.walk(datapath):
        with contextlib.suppress(ValueError):
            dirnames.remove('__pycache__')
        for filename in filenames:
            res = pathlib.Path(dirpath) / filename
            rel = res.relative_to(datapath)
            yield res, rel
