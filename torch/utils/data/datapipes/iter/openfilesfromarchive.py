import os
import sys
import tarfile
import warnings
import zipfile

from io import BufferedIOBase
from typing import Dict, Iterable, Iterator, Tuple, Optional, IO, cast

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import validate_pathname_binary_tuple


class OpenFilesFromArchiveIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r""" :class:`OpenFilesFromArchiveIterDataPipe`.

    Base Iterable DataPipe to open file binary streams from tuples of pathname and
    archive binary stream of source DataPipe. Yield pathname and opened binary stream
    in tuples.

    Register subclass by specifying the `archive_type` at subclass declaration.

    args:
        datapipe: Iterable DataPipe that provides pathname and archive binary stream in tuples
        length: a nominal length of the datapipe
    """
    _dispatch: Dict[str, IterDataPipe[Tuple[str, BufferedIOBase]]] = {}

    @classmethod
    def register(cls, archive_type: str):
        if archive_type in cls._dispatch:
            raise ValueError("Unable to add archive type {} as it has already been taken"
                             .format(archive_type))

        def _register(sub_cls):
            cls._dispatch[archive_type] = sub_cls
            return sub_cls

        return _register

    @classmethod
    def open_from(cls, archive_type: str, *args, **kwargs):
        if archive_type not in cls._dispatch:
            raise ValueError("Archive type {} has not been implemented or not been registered"
                             "to `OpenFilesFromArchiveIterDataPipe`"
                             .format(archive_type))
        return cls._dispatch[archive_type](*args, **kwargs)  # type: ignore[operator]

    def __init__(
            self,
            datapipe : Iterable[Tuple[str, BufferedIOBase]],
            length : int = -1):
        super().__init__()
        self.datapipe : Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.length : int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, archive_stream = data
            try:
                yield from self.open(pathname, archive_stream)
            except Exception as e:
                raise RuntimeError("Unable to open files from corrupted file stream {}"
                                   "due to: {}".format(pathname, e)) from e

    def open(self,
             pathname: str,
             archive_stream: BufferedIOBase) -> Iterator[Tuple[str, BufferedIOBase]]:
        raise NotImplementedError

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length


@OpenFilesFromArchiveIterDataPipe.register('tar')
class OpenFilesFromTarIterDataPipe(OpenFilesFromArchiveIterDataPipe):
    r""" :class:`OpenFilesFromTarIterDataPipe`.

    Iterable DataPipe to extract file binary streams from tuples of pathname and tar
    binary stream of source DataPipe. Yield pathname and extraced binary stream in tuples.
    args:
        datapipe: Iterable DataPipe that provides pathname and tar binary stream in tuples
        length: a nominal length of the datapipe
    """
    def open(self,
             pathname: str,
             archive_stream: BufferedIOBase) -> Iterator[Tuple[str, BufferedIOBase]]:
        # typing.cast is used here to silence mypy's type checker
        tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], archive_stream), mode="r:*")
        for tarinfo in tar:
            if not tarinfo.isfile():
                continue
            extracted_fobj = tar.extractfile(tarinfo)
            if extracted_fobj is None:
                warnings.warn("failed to extract file {} from source tarfile {}".format(tarinfo.name, pathname))
                raise tarfile.ExtractError
            inner_pathname = os.path.normpath(os.path.join(pathname, tarinfo.name))
            # Add a reference of the source tarfile into extracted_fobj, so the source
            # tarfile handle won't be released until all the extracted file objs are destroyed.
            extracted_fobj.source_ref = tar  # type: ignore[attr-defined]
            # typing.cast is used here to silence mypy's type checker
            yield (inner_pathname, cast(BufferedIOBase, extracted_fobj))


@OpenFilesFromArchiveIterDataPipe.register('zip')
class OpenFilesFromZipIterDataPipe(OpenFilesFromArchiveIterDataPipe):
    r""" :class:`OpenFilesFromZipIterDataPipe`.

    Iterable DataPipe to extract file binary streams from tuples of pathname and zip
    binary stream of source DataPipe. Yield pathname and extraced binary stream in tuples.
    args:
        datapipe: Iterable DataPipe that provides pathname and zip binary stream in tuples
        length: a nominal length of the datapipe
    """
    def __init__(
            self,
            datapipe : Iterable[Tuple[str, BufferedIOBase]],
            length : int = -1):
        warnings.warn("Reading from zip file is not efficient, please consider to decompress"
                      "your zip archive")
        super().__init__(datapipe, length)

    def open(self,
             pathname: str,
             archive_stream: BufferedIOBase) -> Iterator[Tuple[str, BufferedIOBase]]:
        # typing.cast is used here to silence mypy's type checker
        zips = zipfile.ZipFile(cast(IO[bytes], archive_stream))
        for zipinfo in zips.infolist():
            # major version should always be 3 here.
            if sys.version_info[1] >= 6:
                if zipinfo.is_dir():
                    continue
            elif zipinfo.filename.endswith('/'):
                continue

            extracted_fobj = zips.open(zipinfo)
            inner_pathname = os.path.normpath(os.path.join(pathname, zipinfo.filename))
            # Add a reference of the source zipfile into extracted_fobj, so the source
            # zipfile handle won't be released until all the extracted file objs are destroyed.
            extracted_fobj.source_ref = zips  # type: ignore[attr-defined]
            # typing.cast is used here to silence mypy's type checker
            yield (inner_pathname, cast(BufferedIOBase, extracted_fobj))
