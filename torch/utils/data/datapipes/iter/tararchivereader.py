from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import validate_pathname_binary_tuple, deprecation_warning_torchdata
from typing import Iterable, Iterator, Tuple, Optional, IO, cast
from io import BufferedIOBase

import os
import tarfile
import warnings

class TarArchiveReaderIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r""" :class:`TarArchiveReaderIterDataPipe`.

    Iterable datapipe to extract tar binary streams from input iterable which contains tuples of pathnames and
    tar binary stream. This yields a tuple of pathname and extracted binary stream.

    Args:
        datapipe: Iterable datapipe that provides tuples of pathname and tar binary stream
        mode: File mode used by `tarfile.open` to read file object.
            Mode has to be a string of the form 'filemode[:compression]'
        length: a nominal length of the datapipe

    Note:
        The opened file handles will be closed automatically if the default DecoderDataPipe
        is attached. Otherwise, user should be responsible to close file handles explicitly
        or let Python's GC close them periodically.
    """
    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        mode: str = "r:*",
        length: int = -1
    ):
        super().__init__()
        self.datapipe: Iterable[Tuple[str, BufferedIOBase]] = datapipe
        self.mode: str = mode
        self.length: int = length
        deprecation_warning_torchdata(type(self).__name__)

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            folder_name = os.path.dirname(pathname)
            try:
                # typing.cast is used here to silence mypy's type checker
                tar = tarfile.open(fileobj=cast(Optional[IO[bytes]], data_stream), mode=self.mode)
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn("failed to extract file {} from source tarfile {}".format(tarinfo.name, pathname))
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(os.path.join(folder_name, tarinfo.name))
                    yield (inner_pathname, extracted_fobj)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(
                    "Unable to extract files from corrupted tarfile stream {} due to: {}, abort!".format(pathname, e))
                raise e

    def __len__(self):
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length
