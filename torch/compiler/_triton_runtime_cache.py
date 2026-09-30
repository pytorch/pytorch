"""Carry Triton's on-disk JIT cache between processes as one verified bundle.

The bundle holds the entries Triton's ``FileCacheManager`` commits under
``TRITON_CACHE_DIR``: compiled kernel groups, native launcher modules, and
autotuning results. Importing it rebuilds that layout in a fresh cache directory
before the first kernel launch, so ``@triton.jit`` and ``@triton.autotune``
kernels hit the cache instead of compiling or benchmarking.

The header does not record the GPU target. Triton's kernel and autotuning keys
hash it, so on another target the entries are never read and those kernels
compile on first use. Checking the target on import would start the GPU runtime
before the application does.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import struct
import sys
import sysconfig
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable


_MAGIC = b"TORCH_TRITON_RUNTIME_CACHE\0"
_VERSION = 1
# Marks a cache directory that holds every entry of the bundle with this
# digest. Export marks its source too, so a process can import the bundle it
# produced; import still compares each entry byte for byte, so later compiles
# into the same directory do not invalidate the mark.
_READY = ".runtime_cache_bundle"
_BINARY_SUFFIXES = (".cubin", ".hsaco", ".zebin")
_NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")
_AUTOTUNE_SUFFIX = ".autotune.json"


def _create_root(root: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        raise RuntimeError(
            f"Triton cache directory {root} is not a directory"
        ) from None
    except OSError as exc:
        raise RuntimeError(f"Cannot create Triton cache directory {root}") from exc


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
    if not root.is_absolute() or (
        not configured and (require_explicit or not (root / _READY).is_file())
    ):
        raise RuntimeError(
            "Triton runtime-cache transport requires an explicit absolute "
            "TRITON_CACHE_DIR"
        )
    if configured and os.path.normpath(configured) != os.path.normpath(knobs.cache.dir):
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

    try:
        # The header holds context as JSON, so import compares it in that form.
        context = json.loads(json.dumps(context, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Triton runtime-cache context must be JSON-serializable"
        ) from exc
    return {
        "triton_key": hashlib.sha256(triton_key().encode()).hexdigest(),
        "python": list(sys.version_info[:3]),
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


def _read_json(path: Path, data: bytes) -> Any:
    try:
        return json.loads(data)
    except ValueError as exc:
        raise RuntimeError(f"Unreadable Triton runtime-cache entry: {path}") from exc


def export_runtime_cache(*, context: Any = None, exclude: Iterable[str] = ()) -> bytes:
    """Snapshot the committed entries of a capture-private Triton cache.

    The caller must have drained all compile work, and must have set this cache
    directory before the first kernel was imported or launched. ``exclude``
    names cache keys the caller ships by other means; their entries are left
    out. Every other file must belong to a committed entry, except the stage
    files of a compile that failed before Triton committed it. The source
    directory is marked as holding this bundle.
    """
    root = runtime_cache_root(require_explicit=True)
    # Triton creates the directory on first use, and a capture may launch no
    # Triton kernel.
    _create_root(root)
    excluded = {_cache_key(key) for key in exclude}
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not suffix:
        raise RuntimeError(
            "Triton runtime-cache transport requires a Python extension suffix"
        )
    records: list[dict[str, Any]] = []
    payloads: dict[tuple[str, str], bytes] = {}

    def add_file(key: str, name: str) -> bytes:
        path = root / key / _member_name(name)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Incomplete Triton runtime-cache entry: {path}")
        payloads[key, name] = path.read_bytes()
        return payloads[key, name]

    for directory in sorted(root.iterdir()):
        # The marker, and its temporary copy if an earlier export failed.
        if directory.name.startswith(_READY):
            continue
        key = _cache_key(directory.name)
        if key in excluded:
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"Invalid Triton runtime-cache directory: {directory}")
        consumed: set[str] = set()
        names: set[str] = set()
        for path in sorted(directory.iterdir()):
            name = path.name
            # FileCacheManager.put writes through tmp.pid_* and holds no lock
            # file open once an entry is committed.
            if name == "lock" or (path.is_dir() and name.startswith("tmp.pid_")):
                continue
            names.add(name)
            if name.startswith("__grp__"):
                if path.is_symlink() or not path.is_file():
                    raise RuntimeError(f"Invalid Triton runtime-cache group: {path}")
                group = _member_name(name.removeprefix("__grp__"))
                entry = _read_json(path, path.read_bytes())
                children = entry.get("child_paths") if isinstance(entry, dict) else None
                if not isinstance(children, dict) or group not in children:
                    raise RuntimeError(f"Incomplete Triton runtime-cache group: {path}")
                members = list(children)
                if not any(member.endswith(_BINARY_SUFFIXES) for member in members):
                    raise RuntimeError(
                        f"Triton runtime-cache group has no compiled binary: {path}"
                    )
                for member, original in children.items():
                    member = _member_name(member)
                    # Triton records paths under knobs.cache.dir, which may
                    # spell the same directory differently.
                    if not isinstance(original, str) or os.path.normpath(
                        original
                    ) != os.path.normpath(directory / member):
                        raise RuntimeError(
                            f"Triton runtime-cache group points outside its entry: "
                            f"{path}"
                        )
                    add_file(key, member)
                consumed.update(members)
                consumed.add(name)
                records.append(
                    {"kind": "kernel", "key": key, "group": group, "members": members}
                )
            elif name.endswith(suffix):
                add_file(key, name)
                consumed.add(name)
                records.append({"kind": "native", "key": key, "name": name})
            elif name.endswith(_AUTOTUNE_SUFFIX):
                choice = _read_json(path, add_file(key, name))
                if not isinstance(choice, dict) or not isinstance(
                    choice.get("configs_timings"), list
                ):
                    raise RuntimeError(
                        f"Unrecognized Triton autotuning cache entry: {path}"
                    )
                consumed.add(name)
                records.append({"kind": "autotune", "key": key, "name": name})
        leftover = sorted(names - consumed)
        # Triton commits a kernel's stage files one by one and its group last,
        # so a compile that failed, such as an autotuning config ptxas
        # rejected, leaves stage files that Triton never reads.
        if leftover and (
            consumed
            or any(
                name.endswith(_BINARY_SUFFIXES + _NATIVE_SUFFIXES) for name in leftover
            )
        ):
            raise RuntimeError(
                f"Unrecognized Triton runtime-cache files in {directory}: {leftover}"
            )

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
    bundle = b"".join([_MAGIC, struct.pack("!Q", len(header)), header, *contents])
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
        for member in members:
            # Import writes group files after members, so one would be clobbered.
            if _member_name(member).startswith("__grp__"):
                raise RuntimeError(
                    f"Invalid Triton runtime-cache member name: {member!r}"
                )
            references.add((key, member))
    if references != payloads.keys():
        raise RuntimeError("Incomplete or unreferenced Triton runtime-cache payload")
    return records, payloads


def _group_matches(path: Path, root: Path, record: dict[str, Any]) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    group = _read_json(path, path.read_bytes())
    children = group.get("child_paths") if isinstance(group, dict) else None
    expected = {name: root / record["key"] / name for name in record["members"]}
    # Triton records paths under knobs.cache.dir, which may spell the same
    # directory differently.
    return (
        isinstance(children, dict)
        and children.keys() == expected.keys()
        and all(
            isinstance(child, str)
            and os.path.normpath(child) == os.path.normpath(expected[name])
            for name, child in children.items()
        )
    )


def _group_contents(root: Path, record: dict[str, Any]) -> str:
    # The layout FileCacheManager.put_group writes; get_group checks each path.
    key = record["key"]
    return json.dumps(
        {"child_paths": {name: str(root / key / name) for name in record["members"]}}
    )


def import_runtime_cache(bundle: bytes, *, context: Any = None) -> None:
    """Verify a bundle and hydrate Triton's cache directory with it.

    Call before any Triton kernel is imported or launched. Each cache entry
    appears in one rename, and the directory is marked as holding the bundle
    only once every entry is in place. The directory itself is kept, so it may
    be a mount point and keeps its owner and permissions. It must be empty or
    hold only this bundle, so each bundle needs its own directory. An import
    killed partway leaves its staging directory, which is ignored and can be
    removed once no import is running.
    """
    root = runtime_cache_root()
    try:
        records, payloads = _decode(bundle, context)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Malformed Triton runtime-cache bundle") from exc
    digest = hashlib.sha256(bundle).hexdigest()
    keys = {key for key, _ in payloads}

    def foreign() -> RuntimeError:
        return RuntimeError(
            f"Triton cache directory {root} is nonempty and does not hold this "
            "runtime cache; import each runtime cache into its own empty "
            "TRITON_CACHE_DIR"
        )

    def reject_foreign_entries() -> None:
        if any(
            entry.name not in keys and not entry.name.startswith(_READY)
            for entry in root.iterdir()
        ):
            raise foreign()

    def validate_members() -> None:
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
                if not _group_matches(path, root, record):
                    raise RuntimeError(
                        "Incomplete or incompatible hydrated Triton cache group: "
                        f"{path}"
                    )

    marker = root / _READY
    if marker.exists():
        if marker.is_symlink() or not marker.is_file() or marker.read_text() != digest:
            raise foreign()
        validate_members()
        return
    _create_root(root)
    reject_foreign_entries()
    # The marker prefix keeps the staging directory out of exports.
    staging = root / f"{_READY}.hydrate-{os.getpid()}-{secrets.token_hex(8)}"
    staging.mkdir()
    try:
        for (key, name), payload in payloads.items():
            (staging / key).mkdir(exist_ok=True)
            (staging / key / name).write_bytes(payload)
        for record in records:
            if record["kind"] == "kernel":
                group = staging / record["key"] / ("__grp__" + record["group"])
                group.write_text(_group_contents(root, record))
        for key in sorted(keys):
            try:
                os.rename(staging / key, root / key)
            except OSError:
                # An interrupted or concurrent import placed this entry first.
                if not (root / key).is_dir():
                    raise
        # Another import may have hydrated a different bundle meanwhile.
        reject_foreign_entries()
        validate_members()
        (staging / _READY).write_text(digest)
        os.replace(staging / _READY, marker)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
