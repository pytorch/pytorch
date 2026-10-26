"""Content manifests and incremental source transfers for Ramjet."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from pathlib import Path


BOOTSTRAP = {
    "tools/__init__.py",
    "tools/ramjet.py",
    "tools/ramjet_source.py",
    "tools/ramjet_build.py",
}
BUILD_EXCLUDED_ROOTS = {"test", "docs", ".github", ".agents", ".claude", ".spin"}
EXCLUDED_ROOTS = {"build", ".venv", "agent_space"}


def _signature(value: os.stat_result) -> list[int]:
    return [
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    ]


def _inspect(root: Path, relative: str, content: bool = False) -> tuple[dict, bytes]:
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Invalid source path: {relative}")
    path = root / relative
    before = path.lstat()
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Source path escapes the repository: {relative}")
    entry: dict = {"mode": stat.S_IMODE(before.st_mode)}
    data = b""
    if stat.S_ISLNK(before.st_mode):
        entry["symlink_target"] = os.readlink(path)
        if Path(entry["symlink_target"]).is_absolute():
            raise ValueError(f"Source symlink must be relative: {relative}")
    elif stat.S_ISREG(before.st_mode):
        with path.open("rb") as source:
            if content:
                data = source.read()
                digest = hashlib.sha256(data).hexdigest()
            else:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
        entry.update(sha256=digest, size=before.st_size)
    else:
        raise ValueError(f"Expected a regular file or symlink: {relative}")
    if _signature(before) != _signature(path.lstat()):
        raise RuntimeError(f"Source changed while reading {relative}; retry Ramjet")
    return entry, data


def _write_json(path: Path, value: dict) -> bytes:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        try:
            temporary.write(data)
            temporary.close()
            os.replace(temporary.name, path)
        finally:
            Path(temporary.name).unlink(missing_ok=True)
    return data


def _archive(
    root: Path,
    output: Path,
    entries: dict,
    metadata_name: str,
    metadata: dict,
    name: str,
    prefix: str = "",
) -> tuple[Path, str]:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output, prefix=f"{name}-") as raw:
        with (
            gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w") as archive,
        ):
            data = json.dumps(metadata, sort_keys=True).encode()
            info = tarfile.TarInfo(metadata_name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            for relative, expected in sorted(entries.items()):
                actual, data = _inspect(root, relative, content=True)
                if actual != expected:
                    raise RuntimeError(
                        f"Source changed after snapshot: {relative}; retry Ramjet"
                    )
                info = tarfile.TarInfo(prefix + relative)
                info.mode = expected["mode"]
                if "symlink_target" in expected:
                    info.type = tarfile.SYMTYPE
                    info.linkname = expected["symlink_target"]
                    archive.addfile(info)
                else:
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        raw.flush()
        with open(raw.name, "rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        path = output / f"{name}-{digest}.tar.gz"
        try:
            path.hardlink_to(raw.name)
        except FileExistsError:
            pass
    return path, digest


def _freeze_bootstrap(root: Path, output: Path, entries: dict) -> tuple[Path, str]:
    missing = BOOTSTRAP - entries.keys()
    if missing:
        raise RuntimeError(
            f"Missing Ramjet bootstrap files: {', '.join(sorted(missing))}"
        )
    serialized = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(serialized).hexdigest()
    directory = output / f"bootstrap-{digest}"
    for relative, expected in sorted(entries.items()):
        if "symlink_target" in expected:
            raise ValueError(f"Ramjet bootstrap must be a regular file: {relative}")
        path = directory / relative
        if path.is_file() and not path.is_symlink():
            continue
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Expected a frozen bootstrap file at {path}")
        actual, data = _inspect(root, relative, content=True)
        if actual != expected:
            raise RuntimeError(
                f"Source changed after snapshot: {relative}; retry Ramjet"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent) as temporary:
            temporary.write(data)
            temporary.flush()
            os.chmod(temporary.name, expected["mode"])
            try:
                path.hardlink_to(temporary.name)
            except FileExistsError:
                pass
    return directory, digest


def snapshot_source(root: Path, output: Path) -> dict[str, tuple[Path, str]]:
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    submodules = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=root, text=True
    )
    if any(line.startswith(("-", "U")) for line in submodules.splitlines()):
        raise RuntimeError("Initialize submodules and resolve conflicts first")
    tracked = subprocess.check_output(
        ["git", "ls-files", "--recurse-submodules", "-z"], cwd=root
    )
    paths = {os.fsdecode(path) for path in tracked.split(b"\0") if path}
    paths.update(path for path in BOOTSTRAP if (root / path).exists())
    metadata = {
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "submodules": submodules,
    }
    cache_path = output / "source-cache.json"
    try:
        cache = json.loads(cache_path.read_text())
        if cache.get("root") != str(root):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    cached_files = cache.get("files", {})
    files, build, runtime, bootstrap = {}, {}, {}, {}
    for relative in sorted(paths):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid source path: {relative}")
        if path.parts[0] in EXCLUDED_ROOTS or ".git" in path.parts:
            continue
        is_runtime = path.parts[0] in {"test", "tools"} or relative == "pytest.ini"
        is_build = (
            path.parts[0] not in BUILD_EXCLUDED_ROOTS
            and relative != "pytest.ini"
            and not relative.startswith("tools/test/")
            and relative not in BOOTSTRAP - {"tools/__init__.py"}
            and relative != "tools/ramjet.md"
        )
        if not (is_runtime or is_build):
            continue
        try:
            signature = _signature((root / path).lstat())
        except FileNotFoundError:
            continue
        cached = cached_files.get(relative, {})
        if cached.get("stat") == signature:
            entry = cached["entry"]
        else:
            entry, _ = _inspect(root, relative)
            if _signature((root / path).lstat()) != signature:
                raise RuntimeError(f"Source changed while snapshotting {relative}")
        files[relative] = {"stat": signature, "entry": entry}
        if is_build:
            build[relative] = entry
        if relative in BOOTSTRAP:
            bootstrap[relative] = entry
        elif is_runtime:
            runtime[relative] = entry
    frozen_bootstrap = _freeze_bootstrap(root, output, bootstrap)
    serialized = json.dumps(build, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(serialized).hexdigest()
    manifest = output / f"build-{digest}.json"
    if not manifest.exists():
        _write_json(manifest, build)
    metadata["build_sha256"] = digest
    runtime_content = json.dumps(
        {**runtime, **bootstrap}, sort_keys=True, separators=(",", ":")
    ).encode()
    metadata["runtime_sha256"] = hashlib.sha256(runtime_content).hexdigest()
    runtime_key = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    cached_runtime = cache.get("runtime", {})
    test_digest = cached_runtime.get("sha256", "")
    tests = output / f"tests-{test_digest}.tar.gz"
    if cached_runtime.get("key") != runtime_key or not tests.is_file():
        tests, test_digest = _archive(
            root, output, runtime, "ramjet-source.json", metadata, "tests"
        )
    _write_json(
        cache_path,
        {
            "root": str(root),
            "files": files,
            "runtime": {"key": runtime_key, "sha256": test_digest},
        },
    )
    return {
        "build": (manifest, digest),
        "tests": (tests, test_digest),
        "bootstrap": frozen_bootstrap,
    }


def create_source_delta(
    root: Path, manifest_path: Path, previous: dict, output: Path
) -> Path:
    serialized = manifest_path.read_bytes()
    current = json.loads(serialized)
    for relative in current.keys() | previous.keys():
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or ".git" in path.parts
        ):
            raise ValueError(f"Invalid source path: {relative}")
    changed = {
        path: entry for path, entry in current.items() if previous.get(path) != entry
    }
    metadata = {
        "build_sha256": hashlib.sha256(serialized).hexdigest(),
        "deleted": sorted(previous.keys() - current.keys()),
        "files": sorted(changed),
    }
    path, _ = _archive(
        root.resolve(), output, changed, "manifest.json", metadata, "delta", "files/"
    )
    return path
