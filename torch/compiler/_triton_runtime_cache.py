"""Snapshot Triton's on-disk JIT cache as one self-describing bundle.

The bundle holds the entries Triton's ``FileCacheManager`` commits under
``TRITON_CACHE_DIR``: compiled kernel groups, native launcher modules, and
autotuning results. Each member carries its checksum, and the header records
the Triton build and Python ABI that produced them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import struct
import sys
import sysconfig
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
