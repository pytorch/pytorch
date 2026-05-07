"""Shard pytest test node IDs across a fixed number of bins.

Tests are flattened (in sorted-file order, preserving collect order
within each file) and split into evenly-sized contiguous chunks. With
N shards and T tests, sizes differ by at most 1. A few files straddle
shard boundaries, but contiguous file order keeps that to N-1 splits
in the worst case.

Input shape is `dict[str, list[Entry]]` keyed by file path, where each
`Entry` is `{"test": <node_id>, "markers": [<marker_name>, ...]}` as
emitted by `zippo collect`.
"""

from __future__ import annotations

from typing import Any


Entry = dict[str, Any]


def assign_shards(
    tests_by_file: dict[str, list[Entry]], n_shards: int
) -> list[list[str]]:
    """Distribute tests into `n_shards` contiguous chunks of node IDs.

    Sizes differ by at most 1: with `T` tests and `N` shards, the first
    `T % N` shards get `ceil(T / N)` tests; the rest get `floor(T / N)`.
    Files are flattened in sorted-path order (collect order within each
    file) so a file that crosses a boundary spans at most two shards.
    Returns a list of length `n_shards`; index 0 is shard 1.
    """
    if n_shards <= 0:
        raise ValueError(f"n_shards must be positive, got {n_shards}")
    flat = [
        entry["test"]
        for path in sorted(tests_by_file)
        for entry in tests_by_file[path]
    ]
    base, rem = divmod(len(flat), n_shards)
    shards: list[list[str]] = []
    pos = 0
    for i in range(n_shards):
        size = base + (1 if i < rem else 0)
        shards.append(flat[pos : pos + size])
        pos += size
    return shards


def filter_by_marker(
    tests_by_file: dict[str, list[Entry]], marker: str
) -> dict[str, list[Entry]]:
    """Return a new mapping with only entries whose markers contain `marker`.

    Files with zero matching entries are dropped from the result so empty
    keys do not clutter downstream shard counts.
    """
    out: dict[str, list[Entry]] = {}
    for path, entries in tests_by_file.items():
        kept = [e for e in entries if marker in e.get("markers", ())]
        if kept:
            out[path] = kept
    return out
