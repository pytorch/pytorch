from collections.abc import Iterator, MutableMapping
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


# Used for fast next key access (using the fact that the dict is ordered)
# Note: doesn't support deletion but we don't need it!
class IndexedDict(MutableMapping[K, V], Generic[K, V]):
    """A dict that maintains insertion order with O(1) index access."""

    __slots__ = ("_dict", "_keys", "_key_to_index")

    def __init__(self) -> None:
        self._dict: dict[K, V] = {}
        self._keys: list[K] = []  # typing: ignore[bad-override]
        self._key_to_index: dict[K, int] = {}

    def __setitem__(self, key: K, value: V) -> None:
        if key not in self._dict:
            self._key_to_index[key] = len(self._keys)
            self._keys.append(key)
        self._dict[key] = value

    def __getitem__(self, key: K) -> V:
        return self._dict[key]

    def __delitem__(self, key: K) -> None:
        raise NotImplementedError("Deletion not supported for IndexedDict")

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def next_key(self, key: K) -> K | None:
        """Get the next key in insertion order. O(1)."""
        idx = self._key_to_index.get(key)
        if idx is not None and idx + 1 < len(self._keys):
            return self._keys[idx + 1]
        return None

    def prev_key(self, key: K) -> K | None:
        """Get the previous key in insertion order. O(1)."""
        idx = self._key_to_index.get(key)
        if idx is not None and idx > 0:
            return self._keys[idx - 1]
        return None
