from __future__ import annotations

import hashlib
from typing import Any


def epilogue_source_digest(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def make_epilogue_cache_key(symbol_name: str, source: str) -> str:
    return f"epilogue:{symbol_name}:sha256:{epilogue_source_digest(source)}"


def set_epilogue_source_cache_key(epilogue_fn: Any, source: str) -> str:
    cache_key = make_epilogue_cache_key(epilogue_fn.__name__, source)
    setattr(epilogue_fn, "__quack_cache_key__", cache_key)
    return cache_key
