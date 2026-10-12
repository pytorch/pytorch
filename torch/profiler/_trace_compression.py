import gzip
from collections.abc import Callable
from functools import partial
from typing import Any, IO


def _get_trace_file_opener(
    path: str, *, gzip_compresslevel: int = 9
) -> Callable[..., IO[Any]]:
    if path.endswith(".gz"):
        return partial(gzip.open, compresslevel=gzip_compresslevel)
    if path.endswith(".zst"):
        try:
            import zstandard  # pyrefly: ignore[missing-import]
        except ImportError as exc:
            raise ImportError(
                "Zstd trace compression requires zstandard. "
                "Please install it with `pip install zstandard`."
            ) from exc
        return partial(zstandard.open, cctx=zstandard.ZstdCompressor(level=1))
    return open
