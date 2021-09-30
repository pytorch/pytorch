from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import validate_pathname_binary_tuple, deprecation_warning_torchdata
from typing import Iterable, Iterator, Tuple, IO, cast
from io import BufferedIOBase

import os
import sys
import zipfile
import warnings

class ZipArchiveReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r""" :class:`ZipArchiveReaderIterDataPipe`.

    Iterable data pipe to extract zip binary streams from input iterable which contains a tuple of pathname and
    zip binary stream. This yields a tuple of pathname and extracted binary stream.

    Args:
        datapipe: Iterable datapipe that provides tuples of pathname and zip binary stream
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed automatically if the default DecoderDataPipe
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically. Due to how zipfiles implements its open() method,
        the data_stream variable below cannot be closed within the scope of this function.
    """
    def __init__(
            self,
            datapipe: Iterable[Tuple[str, BufferedIOBase]],
            length: int = -1):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.length: int = length
        deprecation_warning_torchdata(type(self).__name__)

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            folder_name = os.path.dirname(pathname)
            try:
                # typing.cast is used here to silence mypy's type checker
                zips = zipfile.ZipFile(cast(IO[bytes], data_stream))
                for zipinfo in zips.infolist():
                    # major version should always be 3 here.
                    if sys.version_info[1] >= 6:
                        if zipinfo.is_dir():
                            continue
                    elif zipinfo.filename.endswith('/'):
                        continue
                    extracted_fobj = zips.open(zipinfo)
                    inner_pathname = os.path.normpath(os.path.join(folder_name, zipinfo.filename))
                    yield (inner_pathname, extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted zipfile stream {pathname} due to: {e}, abort!")
                raise e
            # We are unable to close 'data_stream' here, because it needs to be available to use later

    def __len__(self):
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
