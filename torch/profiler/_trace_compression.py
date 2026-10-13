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
        try:
            import zstandard  # pyrefly: ignore[missing-import]
        except ImportError as exc:
            raise ImportError(
                "Zstd trace compression requires zstandard. "
                "Please install it with `pip install zstandard`."
            ) from exc
        level = 1 if compresslevel is None else compresslevel
        return partial(zstandard.open, cctx=zstandard.ZstdCompressor(level=level))
    return open
