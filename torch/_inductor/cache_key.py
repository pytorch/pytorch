from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from typing import Any, Literal, Protocol
from typing_extensions import assert_never


CacheKeyComponent = str | bytes | bytearray | memoryview


class _HashLike(Protocol):
    def update(self, data: bytes, /) -> None: ...
    def digest(self) -> bytes: ...
    def hexdigest(self) -> str: ...


@dataclasses.dataclass(frozen=True)
class CacheKeyStrategy:
    """
    Describes how an Inductor cache turns stable components into a cache key.

    Different caches intentionally use different key formats for compatibility
    with existing on-disk and remote cache layouts. Keeping those choices in
    named strategies makes the composition explicit at the call site.
    """

    # Human-readable label for repr/debugging; it is not part of the cache key.
    name: str
    digest_format: Literal["base32", "hex"]
    prefix: str = ""
    separator: bytes | None = None
    base32_length: int = 51

    @staticmethod
    def _to_bytes(component: CacheKeyComponent) -> bytes:
        if isinstance(component, str):
            return component.encode("utf-8")
        if isinstance(component, bytes):
            return component
        if isinstance(component, bytearray):
            return bytes(component)
        if isinstance(component, memoryview):
            return component.tobytes()
        raise TypeError(f"Unsupported cache key component: {type(component)!r}")

    def _hasher(self, components: tuple[CacheKeyComponent, ...]) -> _HashLike:
        hasher = hashlib.sha256()
        for idx, component in enumerate(components):
            if idx > 0 and self.separator is not None:
                hasher.update(self.separator)
            hasher.update(self._to_bytes(component))
        return hasher

    def digest(self, *components: CacheKeyComponent) -> str:
        hasher = self._hasher(components)
        if self.digest_format == "hex":
            return hasher.hexdigest()
        if self.digest_format == "base32":
            # [:51] strips the "Q====" suffix common to every SHA256 base32 digest.
            return (
                base64.b32encode(hasher.digest())[: self.base32_length]
                .decode("utf-8")
                .lower()
            )
        assert_never(self.digest_format)

    def key(self, *components: CacheKeyComponent) -> str:
        return f"{self.prefix}{self.digest(*components)}"

    def key_from_json(self, value: Any, *, sort_keys: bool = True) -> str:
        return self.key(json.dumps(value, sort_keys=sort_keys))


COMPACT_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="compact",
    digest_format="base32",
)

CODE_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="code",
    digest_format="base32",
    prefix="c",
    separator=b"||",
)

FX_GRAPH_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="fx_graph",
    digest_format="base32",
    prefix="f",
)

SYSTEM_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="system",
    digest_format="hex",
)

AUTOTUNE_CACHE_KEY_STRATEGY = CacheKeyStrategy(
    name="autotune",
    digest_format="hex",
    # Preserve the existing autotune cache format, which concatenates components.
    separator=None,
)
