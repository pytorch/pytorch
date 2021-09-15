from torch.utils.data import IterDataPipe
from typing import Iterable, Iterator, Tuple
from io import BufferedIOBase

import os
import sys
import zipfile
import warnings

class ZipArchiveReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r""" :class:`ZipArchiveReaderIterDataPipe`.

    Iterable data pipe to extract zip binary streams from input iterable which contains
    pathnames, yields a tuple of pathname and extracted binary stream.

    Args:
        datapipe: Iterable datapipe that provides pathnames
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed automatically if the default DecoderDataPipe
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodly.
    """
    def __init__(
            self,
            datapipe: Iterable[str],
            length: int = -1):
        super().__init__()
        self.datapipe: Iterable[str] = datapipe
        self.length: int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for pathname in self.datapipe:
            if not isinstance(pathname, str):
                raise TypeError(f"pathname should be of string type, but is type {type(pathname)}")
            try:
                zips = zipfile.ZipFile(pathname)
                for zipinfo in zips.infolist():
                    # major version should always be 3 here.
                    if sys.version_info[1] >= 6:
                        if zipinfo.is_dir():
                            continue
                    elif zipinfo.filename.endswith('/'):
                        continue
                    extracted_fobj = zips.open(zipinfo)
                    inner_pathname = os.path.normpath(os.path.join(pathname, zipinfo.filename))
                    yield (inner_pathname, extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted zipfile stream {pathname} due to: {e}, abort!")
                raise e

    def __len__(self):
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
