"""Build Ramjet wheels while preserving the native build directory."""


def build_wheel(cache_volume: str, build_sha256: str, delta_path: str) -> None:
    import json
    import os
    import re
    import shutil
    import subprocess
    import sys
    import tarfile
    import time
    from pathlib import Path, PurePosixPath

    import modal  # pyrefly: ignore[missing-import]  # Optional runner dependency.

    source = Path("/opt/pytorch")
    source.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    restored = 0
    build_dir = source / "build"
    if build_dir.joinpath(".ninja_deps").is_file():
        dependencies = subprocess.check_output(
            ["ninja", "-t", "deps"], cwd=build_dir, text=True
        )
        pattern = r"^(.+): #deps \d+, deps mtime (\d+) \(STALE\)$"
        for name, mtime in re.findall(pattern, dependencies, flags=re.MULTILINE):
            target = build_dir / name
            if not target.resolve().is_relative_to(build_dir):
                raise ValueError(f"Ninja output escapes the build directory: {name!r}")
            try:
                state = target.stat()
            except FileNotFoundError:
                continue
            # Reuse Ninja's recorded output timestamps after restoring an image.
            os.utime(target, ns=(state.st_atime_ns, int(mtime)))
            restored += 1
    restore_seconds = time.monotonic() - started
    if restored:
        print(
            f"Ramjet restored {restored} cached build timestamps "
            f"in {restore_seconds:.2f} seconds",
            flush=True,
        )
    started = time.monotonic()

    def checked_path(value: str) -> Path:
        path = PurePosixPath(value)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
        ):
            raise ValueError(f"Invalid source path: {value!r}")
        target = source / value
        if not target.parent.resolve().is_relative_to(source):
            raise ValueError(f"Source path escapes the build directory: {value!r}")
        return target

    def remove(target: Path) -> None:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)

    with tarfile.open(delta_path) as archive:
        members = archive.getmembers()
        entries = {entry.name: entry for entry in members}
        if len(entries) != len(members):
            raise ValueError("Source delta contains duplicate archive members")
        manifest_entry = entries.get("manifest.json")
        if manifest_entry is None or not manifest_entry.isfile():
            raise ValueError("Source delta must contain a regular manifest.json")
        stream = archive.extractfile(manifest_entry)
        if stream is None:
            raise ValueError("Source delta manifest has no contents")
        with stream:
            manifest = json.load(stream)
        if (
            not isinstance(manifest, dict)
            or manifest.get("build_sha256") != build_sha256
        ):
            raise ValueError("Source delta does not match the requested build")
        changed = manifest.get("files")
        deleted = manifest.get("deleted")
        if not isinstance(changed, list) or not isinstance(deleted, list):
            raise ValueError("Source delta must list changed and deleted files")
        for value in [*changed, *deleted]:
            if not isinstance(value, str):
                raise ValueError(f"Invalid source path: {value!r}")
            checked_path(value)
        if len(set(changed)) != len(changed) or len(set(deleted)) != len(deleted):
            raise ValueError("Source delta contains duplicate source paths")
        if set(changed) & set(deleted):
            raise ValueError("Source delta both changes and deletes the same path")
        expected = {"manifest.json", *(f"files/{path}" for path in changed)}
        if set(entries) != expected:
            raise ValueError("Source delta archive does not match its manifest")
        for value in changed:
            entry = entries[f"files/{value}"]
            if not (entry.isfile() or entry.issym() or entry.islnk()):
                raise ValueError(f"Unsupported source archive member: {value!r}")
            if entry.issym():
                target = checked_path(value)
                link = PurePosixPath(entry.linkname)
                resolved = (target.parent / entry.linkname).resolve()
                if link.is_absolute() or not resolved.is_relative_to(source):
                    raise ValueError(
                        f"Source symlink escapes the build directory: {value!r}"
                    )
            elif entry.islnk() and entry.linkname not in expected - {"manifest.json"}:
                raise ValueError(f"Invalid source hard link: {value!r}")

        deleted.sort(key=lambda path: len(PurePosixPath(path).parts), reverse=True)
        for value in deleted:
            remove(checked_path(value))
        for value in sorted(changed):
            target = checked_path(value)
            entry = entries[f"files/{value}"]
            remove(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.issym():
                target.symlink_to(entry.linkname)
            else:
                stream = archive.extractfile(entry)
                if stream is None:
                    raise ValueError(f"Source file has no contents: {value!r}")
                with stream, target.open("wb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(entry.mode & 0o777)
            # Delta archives have deterministic timestamps; Ninja needs new mtimes.
            os.utime(target, None, follow_symlinks=False)

    Path(delta_path).unlink()
    version = source.joinpath("version.txt").read_text().strip()
    os.environ["PYTORCH_BUILD_VERSION"] = f"{version}+ramjet.{build_sha256[:12]}"
    os.environ["PYTORCH_BUILD_NUMBER"] = "1"
    sync_seconds = time.monotonic() - started
    print(
        f"Ramjet applied {len(changed)} changed and {len(deleted)} deleted files "
        f"in {sync_seconds:.2f} seconds",
        flush=True,
    )
    wheels = Path("/opt/wheels")
    wheels.mkdir(exist_ok=True)
    for wheel in wheels.glob("*.whl"):
        wheel.unlink()
    Path(os.environ["CCACHE_TEMPDIR"]).mkdir(parents=True, exist_ok=True)
    pip = [sys.executable, "-m", "pip"]
    before = subprocess.check_output(["ccache", "--print-stats"], text=True)
    print(f"ccache before build:\n{before}", flush=True)
    started = time.monotonic()
    info = {
        "build_sha256": build_sha256,
        "native_cache_restore_seconds": restore_seconds,
        "restored_build_outputs": restored,
        "source_sync_seconds": sync_seconds,
        "changed_files": len(changed),
        "deleted_files": len(deleted),
        "ccache_before": before,
    }
    try:
        subprocess.run(
            [
                *pip,
                "wheel",
                ".",
                "-v",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(wheels),
            ],
            cwd=source,
            check=True,
        )
    finally:
        elapsed = time.monotonic() - started
        try:
            after = subprocess.check_output(["ccache", "--print-stats"], text=True)
            info.update(wheel_build_seconds=elapsed, ccache_after=after)
            Path("/opt/ramjet-build.json").write_text(json.dumps(info, indent=2))
            print(f"ccache after build:\n{after}", flush=True)
            print(f"Ramjet wheel build took {elapsed:.2f} seconds", flush=True)
        finally:
            modal.Volume.from_name(cache_volume).commit()
    built = list(wheels.glob("torch-*.whl"))
    if len(built) != 1:
        raise RuntimeError(f"Expected one PyTorch wheel, found {built}")
    started = time.monotonic()
    subprocess.run(
        [*pip, "install", "--no-deps", "--force-reinstall", str(built[0])],
        check=True,
    )
    info["wheel_install_seconds"] = time.monotonic() - started
    Path("/opt/ramjet-build.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-volume", required=True)
    parser.add_argument("--build-sha256", required=True)
    parser.add_argument("--delta-path", required=True)
    args = parser.parse_args()
    build_wheel(args.cache_volume, args.build_sha256, args.delta_path)
