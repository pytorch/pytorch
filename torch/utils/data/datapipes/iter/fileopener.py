from collections.abc import Iterable, Iterator
from io import IOBase

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames


__all__ = [
    "FileOpenerIterDataPipe",
]


@functional_datapipe("open_files")
class FileOpenerIterDataPipe(IterDataPipe[tuple[str, IOBase]]):
    r"""
    Given pathnames, opens files and yield pathname and file stream in a tuple (functional name: ``open_files``).

    Args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by ``open()``. It defaults to ``r``, other options are
            ``b`` for reading in binary mode and ``t`` for text mode.
        encoding: An optional string that specifies the encoding of the
            underlying file. It defaults to ``None`` to match the default encoding of ``open``.
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed by Python's GC periodically. Users can choose
        to close them explicitly.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import (
        ...     FileLister,
        ...     FileOpener,
        ...     StreamReader,
        ... )
        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith(".txt"))
        >>> dp = FileOpener(dp)
        >>> dp = StreamReader(dp)
        >>> list(dp)
        [('./abc.txt', 'abc')]
    """

    def __init__(
        self,
        datapipe: Iterable[str],
        mode: str = "r",
        encoding: str | None = None,
        length: int = -1,
    ) -> None:
        super().__init__()
        self.datapipe: Iterable[str] = datapipe
        self.mode: str = mode
        self.encoding: str | None = encoding

        if self.mode not in ("b", "t", "rb", "rt", "r"):
            raise ValueError(f"Invalid mode {mode}")
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken

        if "b" in mode and encoding is not None:
            raise ValueError("binary mode doesn't take an encoding argument")

        self.length: int = length

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self) -> Iterator[tuple[str, IOBase]]:
        yield from get_file_binaries_from_pathnames(
            self.datapipe, self.mode, self.encoding
        )

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
        return self.length
