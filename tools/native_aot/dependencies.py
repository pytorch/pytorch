"""Source dependencies of the native-AOT kernels embedded in a CI wheel."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


MANIFEST = "native_aot_dependencies.json"
CI_MANIFEST = PurePosixPath(".additional_ci_files") / MANIFEST


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def validate_sources(sources: object) -> dict[str, str]:
    if not isinstance(sources, dict):
        raise ValueError("native-AOT dependencies must contain a sources mapping")
    for path, digest in sources.items():
        if (
            not isinstance(path, str)
            or not path
            or "\\" in path
            or ":" in path
            or "\0" in path
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or PurePosixPath(path).as_posix() != path
            or path == "."
        ):
            raise ValueError(f"invalid native-AOT dependency path: {path!r}")
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{16}", digest):
            raise ValueError(f"invalid native-AOT dependency hash for {path}")
    return sources


def merge_sources(closures: Iterable[dict[str, str]]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for closure in closures:
        if not closure:
            raise ValueError("generated kernel has no recorded source dependencies")
        for path, digest in validate_sources(closure).items():
            if path in sources and sources[path] != digest:
                raise ValueError(f"conflicting native-AOT dependency hashes for {path}")
            sources[path] = digest
    return dict(sorted(sources.items()))


def read_manifest(contents: bytes | str) -> dict[str, str]:
    data = json.loads(contents)
    if (
        not isinstance(data, dict)
        or type(data.get("version")) is not int
        or data["version"] != 1
    ):
        raise ValueError(
            "missing or unsupported native-AOT dependency manifest version"
        )
    return validate_sources(data.get("sources"))


def write_manifest(path: Path, sources: dict[str, str]) -> None:
    contents = (
        json.dumps(
            {"version": 1, "sources": validate_sources(sources)},
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )
    if path.exists() and path.read_text() == contents:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        staged.write_text(contents)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def changed_source(sources: dict[str, str], repo: Path) -> str | None:
    for path, digest in sources.items():
        try:
            if file_hash(repo / path) == digest:
                continue
        except OSError:
            pass
        return path
    return None
