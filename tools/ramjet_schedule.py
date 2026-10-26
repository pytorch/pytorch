"""Prioritize Ramjet batches using previous runs on the same platform."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


def _seconds(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _test_file(value) -> bool:
    return (
        isinstance(value, str)
        and value.endswith(".py")
        and not PurePosixPath(value).is_absolute()
        and ".." not in PurePosixPath(value).parts
    )


def _batch_key(nodes: list[str]) -> str:
    data = json.dumps(sorted(nodes), separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def load_history(path: Path) -> dict:
    try:
        history = json.loads(path.read_text())
        if history.get("version") != 1 or not isinstance(history["platforms"], dict):
            raise ValueError("unsupported timing history format")
        for platform, tables in history["platforms"].items():
            if not isinstance(platform, str) or not platform.startswith(
                "tci.platform:"
            ):
                raise ValueError("invalid timing platform")
            for kind in ("batches", "files"):
                for key, entry in tables[kind].items():
                    valid_key = (
                        len(key) == 64 and all(c in "0123456789abcdef" for c in key)
                        if kind == "batches"
                        else _test_file(key)
                    )
                    if (
                        not valid_key
                        or not _seconds(entry["seconds"])
                        or not isinstance(entry["source"], str)
                        or not entry["source"]
                    ):
                        raise ValueError("invalid timing entry")
        return history
    except FileNotFoundError:
        pass
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
    ) as error:
        print(f"Ignoring Ramjet timing history {path}: {error}", flush=True)
    return {"version": 1, "platforms": {}}


def order_shards(
    batches: list[list[str]], platform: str, history: dict
) -> tuple[list[list[str]], dict]:
    tables = history["platforms"].get(platform, {"batches": {}, "files": {}})
    estimates = []
    sources = set()
    counts = {"exact_batches": 0, "estimated_batches": 0, "unknown_batches": 0}
    for batch in batches:
        exact = tables["batches"].get(_batch_key(batch))
        fallback = tables["files"].get(batch[0].split("::", 1)[0])
        entry = exact if exact is not None else fallback
        if entry is None:
            counts["unknown_batches"] += 1
            estimates.append(0.0)
        else:
            counts["exact_batches" if exact is not None else "estimated_batches"] += 1
            estimates.append(
                entry["seconds"] * (1 if exact is not None else len(batch))
            )
            sources.add(entry["source"])
    order = sorted(range(len(batches)), key=lambda i: estimates[i], reverse=True)
    return [batches[i] for i in order], {
        **counts,
        "sources": sorted(sources),
        "estimated_worker_seconds": sum(estimates),
    }


def update_history(path: Path, records: Iterable[dict], source: Path) -> int:
    """Learn from shard records after the runner has validated their reports."""
    history = load_history(path)
    totals: dict[tuple[str, str], tuple[float, int]] = {}
    accepted = 0
    for record in records:
        try:
            if type(record.get("returncode")) is not int or record["returncode"] != 0:
                continue
            nodes = record["nodes"]
            platform = record["platform"]
            start, end = record["started_at"], record["finished_at"]
            if (
                not isinstance(nodes, list)
                or not nodes
                or any(not isinstance(node, str) for node in nodes)
                or any(not node.partition("::")[2] for node in nodes)
                or len(set(nodes)) != len(nodes)
                or not isinstance(platform, str)
                or not platform.startswith("tci.platform:")
                or not _seconds(start)
                or not _seconds(end)
                or end < start
                or record.get("worker_errors")
            ):
                continue
            files = {node.split("::", 1)[0] for node in nodes}
            if len(files) != 1 or not _test_file(next(iter(files))):
                continue
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
            continue
        tables = history["platforms"].setdefault(platform, {"batches": {}, "files": {}})
        tables["batches"][_batch_key(nodes)] = {
            "seconds": end - start,
            "source": str(source),
        }
        key = platform, next(iter(files))
        elapsed, count = totals.get(key, (0.0, 0))
        totals[key] = elapsed + end - start, count + len(nodes)
        accepted += 1
    if not accepted:
        return 0
    for (platform, file), (elapsed, count) in totals.items():
        history["platforms"][platform]["files"][file] = {
            "seconds": elapsed / count,
            "source": str(source),
        }
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, delete=False
    ) as temporary:
        try:
            json.dump(history, temporary, separators=(",", ":"), allow_nan=False)
            temporary.close()
            os.replace(temporary.name, path)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
    return accepted
