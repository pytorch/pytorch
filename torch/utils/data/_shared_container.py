import pickle
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Generic, TypeVar

import torch


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

_ONE_MB = 1 << 20


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    try:
        return t.numpy().tobytes()
    except RuntimeError:
        return bytes(t.tolist())


class SharedList(Sequence, Generic[T]):
    r"""A list-like container whose data is stored in a single :class:`torch.Tensor`.

    When used inside a :class:`~torch.utils.data.Dataset` with
    :class:`~torch.utils.data.DataLoader` workers (``num_workers > 0``),
    ``SharedList`` avoids the copy-on-write memory replication that occurs
    when worker processes read ordinary Python lists or dicts from the
    parent process.  The serialized data is stored in a ``torch.Tensor`` so
    that PyTorch's ``ForkingPickler`` automatically places it into shared
    memory (``/dev/shm``), where all workers can read it without triggering
    page copies.

    ``SharedList`` snapshots the input data at construction time.  Mutations
    to the original list after construction are not reflected.

    Args:
        data (Iterable): An iterable of items to store. Every item must be
            picklable.
    """

    def __init__(self, data: Sequence | Iterator) -> None:
        if isinstance(data, SharedList):
            self._index = data._index.clone()
            self._storage = data._storage.clone()
            self._length = data._length
            return
        serialized: list[bytes] = [pickle.dumps(item) for item in data]
        offsets = [0]
        for s in serialized:
            offsets.append(offsets[-1] + len(s))
        raw = bytearray(offsets[-1])
        for i, s in enumerate(serialized):
            raw[offsets[i] : offsets[i + 1]] = s
        self._index = torch.tensor(offsets, dtype=torch.int64)
        if offsets[-1] > 0:
            self._storage = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        else:
            self._storage = torch.empty(0, dtype=torch.uint8)
        self._length = len(serialized)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int | slice) -> Any:
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._length))]
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(
                f"index {idx} is out of bounds for SharedList of len {self._length}"
            )
        start = int(self._index[idx].item())
        end = int(self._index[idx + 1].item())
        return pickle.loads(_tensor_to_bytes(self._storage[start:end]))

    def __iter__(self) -> Iterator:
        for i in range(self._length):
            yield self[i]

    def __contains__(self, item: Any) -> bool:
        for x in self:
            if x == item:
                return True
        return False

    def __repr__(self) -> str:
        preview = ", ".join(repr(x) for x in self[:5])
        suffix = ", ..." if self._length > 5 else ""
        return f"SharedList([{preview}{suffix}])"

    def index(self, item: Any, start: int = 0, end: int | None = None) -> int:
        if end is None:
            end = self._length
        for i in range(start, end):
            if self[i] == item:
                return i
        raise ValueError(f"{item!r} is not in SharedList")

    def count(self, item: Any) -> int:
        return sum(1 for x in self if x == item)

    def copy(self) -> "SharedList":
        result = object.__new__(SharedList)
        result._index = self._index.clone()
        result._storage = self._storage.clone()
        result._length = self._length
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SharedList):
            return NotImplemented
        if self._length != other._length:
            return False
        return torch.equal(self._index, other._index) and torch.equal(
            self._storage, other._storage
        )


class SharedDict(Mapping, Generic[K, V]):
    r"""A dict-like container whose keys and values are stored in shared-memory
    tensors.

    Like :class:`SharedList`, this avoids copy-on-write memory replication
    when worker processes access the dictionary via a
    :class:`~torch.utils.data.DataLoader` with ``num_workers > 0``.

    ``SharedDict`` snapshots the input data at construction time.  Mutations
    to the original dict after construction are not reflected.

    .. warning::
        Key lookup is O(n) — each access deserializes every key until a
        match is found.  ``SharedDict`` is best suited for small mappings
        (hundreds of entries).  For larger key-value stores, prefer storing
        the data in a :class:`SharedList` with a separate index.

    Args:
        mapping (dict-like): A mapping whose keys and values are all picklable.
    """

    def __init__(self, mapping=None, /, **kwargs) -> None:
        source: dict[Any, Any] = {}
        if mapping is not None:
            source.update(mapping)
        source.update(kwargs)
        keys_serialized: list[bytes] = [pickle.dumps(k) for k in source]
        values_serialized: list[bytes] = [pickle.dumps(v) for v in source.values()]
        k_offsets = [0]
        v_offsets = [0]
        for k, v in zip(keys_serialized, values_serialized):
            k_offsets.append(k_offsets[-1] + len(k))
            v_offsets.append(v_offsets[-1] + len(v))
        k_raw = bytearray(k_offsets[-1])
        v_raw = bytearray(v_offsets[-1])
        for i, (k, v) in enumerate(zip(keys_serialized, values_serialized)):
            k_raw[k_offsets[i] : k_offsets[i + 1]] = k
            v_raw[v_offsets[i] : v_offsets[i + 1]] = v
        self._k_index = torch.tensor(k_offsets, dtype=torch.int64)
        if k_offsets[-1] > 0:
            self._k_storage = torch.frombuffer(bytearray(k_raw), dtype=torch.uint8)
        else:
            self._k_storage = torch.empty(0, dtype=torch.uint8)
        self._v_index = torch.tensor(v_offsets, dtype=torch.int64)
        if v_offsets[-1] > 0:
            self._v_storage = torch.frombuffer(bytearray(v_raw), dtype=torch.uint8)
        else:
            self._v_storage = torch.empty(0, dtype=torch.uint8)
        self._length = len(source)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, key: Any) -> Any:
        for i in range(self._length):
            k = self._deserialize_key(i)
            if k == key:
                return self._deserialize_value(i)
        raise KeyError(key)

    def __contains__(self, key: Any) -> bool:
        for i in range(self._length):
            if self._deserialize_key(i) == key:
                return True
        return False

    def __iter__(self) -> Iterator:
        for i in range(self._length):
            yield self._deserialize_key(i)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{repr(k)}: {repr(v)}" for k, v in self.items()[:5])
        suffix = ", ..." if self._length > 5 else ""
        return f"SharedDict({{{pairs}{suffix}}})"

    def _deserialize_key(self, idx: int) -> Any:
        start = int(self._k_index[idx].item())
        end = int(self._k_index[idx + 1].item())
        return pickle.loads(_tensor_to_bytes(self._k_storage[start:end]))

    def _deserialize_value(self, idx: int) -> Any:
        start = int(self._v_index[idx].item())
        end = int(self._v_index[idx + 1].item())
        return pickle.loads(_tensor_to_bytes(self._v_storage[start:end]))

    def keys(self) -> list[Any]:  # type: ignore[override]
        return [self._deserialize_key(i) for i in range(self._length)]

    def values(self) -> list[Any]:  # type: ignore[override]
        return [self._deserialize_value(i) for i in range(self._length)]

    def items(self) -> list[tuple[Any, Any]]:  # type: ignore[override]
        return [
            (self._deserialize_key(i), self._deserialize_value(i))
            for i in range(self._length)
        ]

    def get(self, key: Any, default: Any = None) -> Any:
        for i in range(self._length):
            if self._deserialize_key(i) == key:
                return self._deserialize_value(i)
        return default

    def copy(self) -> "SharedDict":
        result = object.__new__(SharedDict)
        result._k_index = self._k_index.clone()
        result._k_storage = self._k_storage.clone()
        result._v_index = self._v_index.clone()
        result._v_storage = self._v_storage.clone()
        result._length = self._length
        return result

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SharedDict):
            return NotImplemented
        if self._length != other._length:
            return False
        return (
            torch.equal(self._k_index, other._k_index)
            and torch.equal(self._k_storage, other._k_storage)
            and torch.equal(self._v_index, other._v_index)
            and torch.equal(self._v_storage, other._v_storage)
        )


def to_shared_dataset(dataset, threshold_bytes: int = _ONE_MB):
    """Convert list/dict attributes of a :class:`~torch.utils.data.Dataset`
    to :class:`SharedList` / :class:`SharedDict` automatically.

    Walks ``dataset.__dict__``, replacing any ``list`` or ``dict`` attribute
    whose serialized size exceeds *threshold_bytes* with the corresponding
    shared-memory container.  Smaller attributes and non-list/dict attributes
    are left unchanged.

    When used inside a :class:`~torch.utils.data.DataLoader` with
    ``num_workers > 0``, the converted attributes are stored in shared memory
    and shared across worker processes via :mod:`torch.multiprocessing`'s
    ``ForkingPickler`` — eliminating copy-on-write memory replication.

    Args:
        dataset: A :class:`~torch.utils.data.Dataset` instance to inspect.
        threshold_bytes (int): Minimum serialized size (in bytes) required
            before a list/dict is converted.  Default: 1 MiB.

    Returns:
        The same *dataset* instance (modified in-place).

    Example::

        class ImageDataset(Dataset):
            def __init__(self, paths, labels):
                self.paths = paths  # large list
                self.labels = labels  # large dict
                self.transform = ...  # non-data, skipped


        ds = ImageDataset(paths, labels)
        ds = to_shared_dataset(ds)
        # ds.paths is now a SharedList, ds.labels is now a SharedDict
    """

    import logging

    logger = logging.getLogger(__name__)

    for attr, value in list(dataset.__dict__.items()):
        if isinstance(value, (SharedList, SharedDict)):
            continue
        if isinstance(value, list):
            try:
                raw = pickle.dumps(value)
            except Exception:
                continue
            if len(raw) >= threshold_bytes:
                dataset.__dict__[attr] = SharedList(value)
                logger.info(
                    "to_shared_dataset: converted %s (list, %d bytes -> SharedList)",
                    attr,
                    len(raw),
                )
        elif isinstance(value, dict):
            try:
                raw = pickle.dumps(value)
            except Exception:
                continue
            if len(raw) >= threshold_bytes:
                dataset.__dict__[attr] = SharedDict(value)
                logger.info(
                    "to_shared_dataset: converted %s (dict, %d bytes -> SharedDict)",
                    attr,
                    len(raw),
                )
    return dataset


class SharedTensor:
    """A list-of-tensors backed by a single flat shared-memory buffer.

    Every element is a :class:`torch.Tensor` view into shared storage.
    Unlike :class:`SharedList`, this avoids pickle serialization — data is
    stored directly in a flat ``torch.Tensor`` and indexed by pre-computed
    offsets.  This is intended for datasets that are already tensor-based
    (e.g., preloaded image tensors).

    Args:
        tensors (Iterable[torch.Tensor]): Tensors to store. All must have the
            same ``dtype``.
    """

    def __init__(self, tensors: list[torch.Tensor]) -> None:
        tensors = list(tensors)
        if not tensors:
            self._length = 0
            self._storage = torch.empty(0, dtype=torch.uint8)
            self._index = torch.zeros(1, dtype=torch.int64)
            return
        dtype = tensors[0].dtype
        offsets = [0]
        for t in tensors:
            offsets.append(offsets[-1] + t.numel())
        total = offsets[-1]
        self._storage = torch.empty(total, dtype=dtype)
        for i, t in enumerate(tensors):
            self._storage[offsets[i] : offsets[i + 1]] = t.reshape(-1)
        self._index = torch.tensor(offsets, dtype=torch.int64)
        self._length = len(tensors)
        self._dtype = dtype

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(
                f"index {idx} out of range for SharedTensor of len {self._length}"
            )
        start = int(self._index[idx].item())
        end = int(self._index[idx + 1].item())
        return self._storage[start:end]

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(self._length):
            yield self[i]

    def __repr__(self) -> str:
        return f"SharedTensor(len={self._length}, dtype={self._dtype})"


__all__ = ["SharedList", "SharedDict", "SharedTensor", "to_shared_dataset"]
