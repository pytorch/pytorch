"""Carry Triton's on-disk JIT cache between processes as one verified bundle.

The bundle holds the entries Triton's ``FileCacheManager`` commits under
``TRITON_CACHE_DIR``: compiled kernel groups, native launcher modules, and
autotuning results. Importing it rebuilds that layout in a fresh cache directory
before the first kernel launch, so ``@triton.jit`` and ``@triton.autotune``
kernels hit the cache instead of compiling or benchmarking.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


_MAGIC = b"TORCH_TRITON_RUNTIME_CACHE\0"
_VERSION = 1
# Written into a cache directory once it holds exactly the bundle with this
# digest, so a directory is either empty, fully hydrated, or rejected.
_READY = ".runtime_cache_bundle"
_BINARY_SUFFIXES = (".cubin", ".hsaco", ".zebin")
_AUTOTUNE_SUFFIX = ".autotune.json"


def runtime_cache_root(*, require_explicit: bool = False) -> Path:
    """Return Triton's cache directory if it can be exported or imported.

    Export needs an explicit absolute ``TRITON_CACHE_DIR`` so it snapshots only
    the capture's own entries. Import also accepts Triton's default directory
    once it holds an imported bundle.
    """
    from triton import knobs
    from triton.runtime.cache import FileCacheManager

    configured = os.environ.get("TRITON_CACHE_DIR")
    root = Path(configured or knobs.cache.dir)
    if (
        not root.is_absolute()
        or not configured
        and (require_explicit or not (root / _READY).is_file())
    ):
        raise RuntimeError(
            "Triton runtime-cache transport requires an explicit absolute "
            "TRITON_CACHE_DIR"
        )
    if configured and root != Path(knobs.cache.dir):
        raise RuntimeError(
            "Triton runtime-cache transport requires Triton to use TRITON_CACHE_DIR"
        )
    if knobs.cache.manager_class not in (None, FileCacheManager):
        raise RuntimeError(
            "Triton runtime-cache transport does not support a custom cache manager"
        )
    if root.is_symlink():
        raise RuntimeError(
            "Triton runtime-cache transport requires a private directory, not a symlink"
        )
    return root


def _compatibility(context: Any) -> dict[str, Any]:
    from torch._inductor.runtime.triton_compat import triton_key

    return {
        "triton_key": hashlib.sha256(triton_key().encode()).hexdigest(),
        "python": sys.version,
        "cache_tag": sys.implementation.cache_tag,
        "soabi": sysconfig.get_config_var("SOABI"),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "gil_disabled": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "machine": platform.machine(),
        "system": platform.system(),
        "libc": list(platform.libc_ver()),
        "context": context,
    }


def _member_name(name: object) -> str:
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or any(c in name for c in ("/", "\\", "\0"))
    ):
        raise RuntimeError(f"Invalid Triton runtime-cache member name: {name!r}")
    return name


def _cache_key(key: object) -> str:
    # FileCacheManager directories are base32 digests.
    if not isinstance(key, str) or re.fullmatch(r"[A-Z2-7]+", key) is None:
        raise RuntimeError(f"Invalid Triton runtime-cache key: {key!r}")
    return key


def export_runtime_cache(*, context: Any = None, exclude: Iterable[str] = ()) -> bytes:
    """Snapshot the committed entries of a capture-private Triton cache.

    The caller must have drained all compile work, and must have set this cache
    directory before the first kernel was imported or launched. ``exclude``
    names cache keys the caller ships by other means; their entries are left
    out. The source directory is marked as holding this bundle.
    """
    root = runtime_cache_root(require_explicit=True)
    excluded = {_cache_key(key) for key in exclude}
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError(
            "Triton runtime-cache transport requires a Python extension suffix"
        )
    records: list[dict[str, Any]] = []
    payloads: dict[tuple[str, str], bytes] = {}

    def add_file(key: str, name: str) -> None:
        path = root / key / _member_name(name)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Incomplete Triton runtime-cache entry: {path}")
        payloads[key, name] = path.read_bytes()

    for directory in sorted(root.iterdir()):
        # The marker, and its temporary copy if an earlier export failed.
        if directory.name.startswith(_READY):
            continue
        key = _cache_key(directory.name)
        if key in excluded:
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"Invalid Triton runtime-cache directory: {directory}")
        for path in sorted(directory.iterdir()):
            name = path.name
            # FileCacheManager.put writes through tmp.pid_* and holds no lock
            # file open once an entry is committed.
            if name == "lock" or path.is_dir() and name.startswith("tmp.pid_"):
                continue
            if name.startswith("__grp__"):
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"Invalid Triton runtime-cache group: {path}")
                group = _member_name(name.removeprefix("__grp__"))
                children = json.loads(path.read_text())["child_paths"]
                if not isinstance(children, dict) or group not in children:
                    raise RuntimeError(f"Incomplete Triton runtime-cache group: {path}")
                members = list(children)
                if not any(member.endswith(_BINARY_SUFFIXES) for member in members):
                    raise RuntimeError(
                        f"Triton runtime-cache group has no compiled binary: {path}"
                    )
                for member, original in children.items():
                    member = _member_name(member)
                    if Path(original) != directory / member:
                        raise RuntimeError(
                            f"Triton runtime-cache group points outside its entry: "
                            f"{path}"
                        )
                    add_file(key, member)
                records.append(
                    {"kind": "kernel", "key": key, "group": group, "members": members}
                )
            elif name.endswith(suffix):
                add_file(key, name)
                records.append({"kind": "native", "key": key, "name": name})
            elif name.endswith(_AUTOTUNE_SUFFIX):
                add_file(key, name)
                choice = json.loads(payloads[key, name])
                if not isinstance(choice, dict) or not isinstance(
                    choice.get("configs_timings"), list
                ):
                    raise RuntimeError(
                        f"Unrecognized Triton autotuning cache entry: {path}"
                    )
                records.append({"kind": "autotune", "key": key, "name": name})

    files = []
    contents = []
    offset = 0
    for (key, name), payload in sorted(payloads.items()):
        files.append(
            {
                "key": key,
                "name": name,
                "offset": offset,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        contents.append(payload)
        offset += len(payload)
    header = json.dumps(
        {
            "format": "torch-triton-runtime-cache",
            "version": _VERSION,
            "compatibility": _compatibility(context),
            "records": records,
            "files": files,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    bundle = _MAGIC + struct.pack("!Q", len(header)) + header + b"".join(contents)
    temporary = root / f"{_READY}.tmp"
    try:
        temporary.write_text(hashlib.sha256(bundle).hexdigest())
        os.replace(temporary, root / _READY)
    finally:
        temporary.unlink(missing_ok=True)
    return bundle


def _decode(
    bundle: bytes, context: Any
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], bytes]]:
    start = len(_MAGIC) + 8
    if (
        not isinstance(bundle, bytes)
        or not bundle.startswith(_MAGIC)
        or len(bundle) < start
    ):
        raise RuntimeError("Invalid Triton runtime-cache bundle")
    (header_size,) = struct.unpack("!Q", bundle[len(_MAGIC) : start])
    payload_start = start + header_size
    if payload_start > len(bundle):
        raise RuntimeError("Truncated Triton runtime-cache bundle")
    header = json.loads(bundle[start:payload_start])
    if (
        header.get("format") != "torch-triton-runtime-cache"
        or header.get("version") != _VERSION
    ):
        raise RuntimeError("Incompatible Triton runtime-cache bundle format")
    if header.get("compatibility") != _compatibility(context):
        raise RuntimeError("Incompatible Triton runtime-cache ABI or build identity")
    payloads: dict[tuple[str, str], bytes] = {}
    end = 0
    for file in header["files"]:
        key = _cache_key(file["key"])
        name = _member_name(file["name"])
        size = file["size"]
        offset = file["offset"]
        if (
            type(size) is not int
            or size < 0
            or type(offset) is not int
            or offset != end
        ):
            raise RuntimeError("Invalid Triton runtime-cache payload extent")
        end += size
        payload = bundle[payload_start + offset : payload_start + end]
        if (
            len(payload) != size
            or hashlib.sha256(payload).hexdigest() != file["sha256"]
        ):
            raise RuntimeError(f"Corrupt Triton runtime-cache member: {key}/{name}")
        if (key, name) in payloads:
            raise RuntimeError(f"Duplicate Triton runtime-cache member: {key}/{name}")
        payloads[key, name] = payload
    if payload_start + end != len(bundle):
        raise RuntimeError("Unexpected trailing Triton runtime-cache data")
    records = header["records"]
    references = set()
    record_names = set()
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    for record in records:
        key = _cache_key(record["key"])
        kind = record["kind"]
        if kind == "kernel":
            group = _member_name(record["group"])
            members = record["members"]
            if group not in members or len(members) != len(set(members)):
                raise RuntimeError("Incomplete Triton runtime-cache kernel group")
            if not any(name.endswith(_BINARY_SUFFIXES) for name in members):
                raise RuntimeError("Triton runtime-cache kernel group has no binary")
            name = "__grp__" + group
        elif kind in ("native", "autotune"):
            name = _member_name(record["name"])
            expected = suffix if kind == "native" else _AUTOTUNE_SUFFIX
            if not expected or not name.endswith(expected):
                raise RuntimeError(f"Invalid Triton runtime-cache {kind} member")
            members = [name]
        else:
            raise RuntimeError(f"Unsupported Triton runtime-cache record: {kind!r}")
        if (key, name) in record_names:
            raise RuntimeError(f"Duplicate Triton runtime-cache record: {key}/{name}")
        record_names.add((key, name))
        references.update((key, _member_name(member)) for member in members)
    if references != payloads.keys():
        raise RuntimeError("Incomplete or unreferenced Triton runtime-cache payload")
    return records, payloads


def _group_contents(root: Path, record: dict[str, Any]) -> str:
    # The layout FileCacheManager.put_group writes; get_group checks each path.
    key = record["key"]
    return json.dumps(
        {"child_paths": {name: str(root / key / name) for name in record["members"]}}
    )


def import_runtime_cache(bundle: bytes, *, context: Any = None) -> None:
    """Verify a bundle and hydrate Triton's cache directory with it atomically.

    Call before any Triton kernel is imported or launched. An empty (or
    missing) directory is filled in one rename. A nonempty one must already
    hold exactly this bundle, or this raises.
    """
    root = runtime_cache_root()
    try:
        records, payloads = _decode(bundle, context)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Malformed Triton runtime-cache bundle") from exc
    digest = hashlib.sha256(bundle).hexdigest()

    def validate_ready() -> None:
        marker = root / _READY
        if not marker.is_file() or marker.is_symlink() or marker.read_text() != digest:
            raise RuntimeError(
                "Triton cache directory is nonempty and does not hold this "
                f"runtime cache: {root}"
            )
        for (key, name), payload in payloads.items():
            path = root / key / name
            if (
                (root / key).is_symlink()
                or path.is_symlink()
                or not path.is_file()
                or path.read_bytes() != payload
            ):
                raise RuntimeError(
                    f"Incomplete or incompatible hydrated Triton cache member: {path}"
                )
        for record in records:
            if record["kind"] == "kernel":
                path = root / record["key"] / ("__grp__" + record["group"])
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or json.loads(path.read_text())
                    != json.loads(_group_contents(root, record))
                ):
                    raise RuntimeError(
                        "Incomplete or incompatible hydrated Triton cache group: "
                        f"{path}"
                    )

    if root.exists() and any(root.iterdir()):
        validate_ready()
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{root.name}.hydrate-", dir=root.parent))
    try:
        for (key, name), payload in payloads.items():
            (staging / key).mkdir(exist_ok=True)
            (staging / key / name).write_bytes(payload)
        for record in records:
            if record["kind"] == "kernel":
                group = staging / record["key"] / ("__grp__" + record["group"])
                group.write_text(_group_contents(root, record))
        (staging / _READY).write_text(digest)
        try:
            os.replace(staging, root)
        except OSError:
            # Windows cannot rename over a directory, even an empty one.
            with contextlib.suppress(OSError):
                root.rmdir()
                os.replace(staging, root)
            if staging.exists():
                # Another process hydrated the directory first.
                validate_ready()
    finally:
        if staging.exists():
            shutil.rmtree(staging)
