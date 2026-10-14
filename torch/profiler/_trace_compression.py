import gzip
from collections.abc import Callable
from functools import partial
from typing import Any, IO


def _get_trace_file_opener(
    path: str, *, compresslevel: int | None = None
) -> Callable[..., IO[Any]]:
    if path.endswith(".gz"):
        level = 9 if compresslevel is None else compresslevel
        return partial(gzip.open, compresslevel=level)
    if path.endswith(".zst"):
        level = 1 if compresslevel is None else compresslevel
        try:
            import zstandard  # pyrefly: ignore[missing-import]
        except ImportError:
            try:
                import pyzstd  # pyrefly: ignore[missing-import]
            except ImportError as exc:
                raise ImportError(
                    "Zstd trace compression requires zstandard or pyzstd. "
                    "Please install either with `pip install zstandard` or `pip install pyzstd`."
                ) from exc
            return partial(pyzstd.open, level_or_option=level)
        return partial(zstandard.open, cctx=zstandard.ZstdCompressor(level=level))
    return open
