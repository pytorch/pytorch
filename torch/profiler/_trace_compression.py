import gzip
from collections.abc import Callable
from functools import partial
from typing import Any, IO

from torch._utils import try_import


def _get_trace_file_opener(
    path: str, *, compresslevel: int | None = None
) -> Callable[..., IO[Any]]:
    if path.endswith(".gz"):
        level = 9 if compresslevel is None else compresslevel
        return partial(gzip.open, compresslevel=level)
    if path.endswith(".zst"):
        level = 1 if compresslevel is None else compresslevel
        zstandard = try_import("zstandard")
        if zstandard is not None:
            return partial(zstandard.open, cctx=zstandard.ZstdCompressor(level=level))
        pyzstd = try_import("pyzstd")
        if pyzstd is not None:
            return partial(pyzstd.open, level_or_option=level)
        raise ImportError(
            "Zstd trace compression requires zstandard or pyzstd. "
            "Please install either with `pip install zstandard` or `pip install pyzstd`."
        )
    return open
