from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from tools import ramjet_source

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestRamjetSource(TestCase):
    root: Path
    output: Path
    tracked: set[str]
    git_sha: str
    submodules: str

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-source-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "repo"
        self.output = self.root.parent / "snapshots"
        self.tracked = {
            "README.md",
            "cmake/config.cmake",
            "docs/example.md",
            "pytest.ini",
            "scripts/example.sh",
            "test/test_example.py",
            "third_party/pkg/LICENSE",
            "tools/autograd/derivatives.yaml",
            "tools/ramjet.md",
            "tools/stats/helper.py",
            "tools/test/test_ramjet.py",
            "torch/native.cpp",
            "torch/python_module.py",
            "torch/testing/_internal/tci.py",
            *ramjet_source.BOOTSTRAP,
        }
        for name in self.tracked:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"original {name}")
        self.git_sha = "first-revision"
        self.submodules = " abc123 third_party/pkg (heads/main)\n"

        def git(command, **kwargs):
            self.assertEqual(kwargs["cwd"], self.root)
            if command[1:3] == ["submodule", "status"]:
                return self.submodules
            if command[1] == "ls-files":
                return b"\0".join(os.fsencode(path) for path in self.tracked) + b"\0"
            self.assertEqual(command, ["git", "rev-parse", "HEAD"])
            return self.git_sha + "\n"

        self.enterContext(
            mock.patch.object(ramjet_source.subprocess, "check_output", side_effect=git)
        )

    def metadata(self, path, name="ramjet-source.json"):
        with tarfile.open(path) as archive:
            stream = archive.extractfile(name)
            if stream is None:
                raise AssertionError(f"Expected archive file {name}")
            return json.load(stream)

    @parametrize(
        "path,runtime_changes",
        [
            ("test/test_example.py", True),
            ("tools/test/test_ramjet.py", True),
            ("tools/ramjet.md", True),
            ("docs/example.md", False),
        ],
    )
    def test_runtime_changes_preserve_build_key(self, path, runtime_changes):
        before = ramjet_source.snapshot_source(self.root, self.output)
        (self.root / path).write_text("changed")
        after = ramjet_source.snapshot_source(self.root, self.output)
        self.assertEqual(before["build"], after["build"])
        self.assertEqual(before["tests"] != after["tests"], runtime_changes)

    @parametrize(
        "path",
        [
            "torch/native.cpp",
            "torch/python_module.py",
            "torch/testing/_internal/tci.py",
            "tools/autograd/derivatives.yaml",
            "cmake/config.cmake",
            "third_party/pkg/LICENSE",
            "README.md",
        ],
    )
    def test_wheel_inputs_change_build_key(self, path):
        before = ramjet_source.snapshot_source(self.root, self.output)
        (self.root / path).write_text("changed")
        after = ramjet_source.snapshot_source(self.root, self.output)
        self.assertNotEqual(before["build"], after["build"])
        self.assertEqual(
            self.metadata(after["tests"][0])["build_sha256"], after["build"][1]
        )

    @parametrize(
        "path", ["tools/ramjet.py", "tools/ramjet_source.py", "tools/ramjet_build.py"]
    )
    def test_bootstrap_code_has_runtime_identity(self, path):
        before = ramjet_source.snapshot_source(self.root, self.output)
        (self.root / path).write_text("changed bootstrap")
        after = ramjet_source.snapshot_source(self.root, self.output)
        self.assertEqual(before["build"], after["build"])
        self.assertNotEqual(
            self.metadata(before["tests"][0])["runtime_sha256"],
            self.metadata(after["tests"][0])["runtime_sha256"],
        )
        with tarfile.open(after["tests"][0]) as archive:
            members = set(archive.getnames())
        self.assertTrue(ramjet_source.BOOTSTRAP.isdisjoint(members))
        self.assertTrue(
            {"tools/stats/helper.py", "test/test_example.py", "pytest.ini"} <= members
        )

    @parametrize("relative", sorted(ramjet_source.BOOTSTRAP))
    def test_bootstrap_files_are_frozen_copies(self, relative):
        live = self.root / relative
        live.chmod(0o751)
        original = live.read_bytes()
        before = ramjet_source.snapshot_source(self.root, self.output)
        directory, digest = before["bootstrap"]
        frozen = directory / relative
        self.assertEqual(directory.name, f"bootstrap-{digest}")
        self.assertEqual(frozen.stat().st_mode & 0o777, 0o751)
        live.write_bytes(b"changed live checkout")
        self.assertEqual(frozen.read_bytes(), original)
        after = ramjet_source.snapshot_source(self.root, self.output)
        self.assertNotEqual(before["bootstrap"], after["bootstrap"])
        self.assertEqual(
            (after["bootstrap"][0] / relative).read_bytes(), live.read_bytes()
        )
        self.assertEqual(frozen.read_bytes(), original)

    @parametrize("relative", sorted(ramjet_source.BOOTSTRAP))
    def test_snapshot_requires_every_bootstrap_file(self, relative):
        (self.root / relative).unlink()
        with self.assertRaisesRegex(RuntimeError, "Missing Ramjet bootstrap files"):
            ramjet_source.snapshot_source(self.root, self.output)

    def test_bootstrap_publication_recovers_from_failure(self):
        original_link = Path.hardlink_to

        def publish(path, target):
            if path.name == "ramjet_build.py":
                raise OSError("bootstrap publication failed")
            original_link(path, target)

        with (
            mock.patch.object(Path, "hardlink_to", publish),
            self.assertRaisesRegex(OSError, "bootstrap publication failed"),
        ):
            ramjet_source.snapshot_source(self.root, self.output)
        directories = list(self.output.glob("bootstrap-*"))
        self.assertEqual(len(directories), 1)
        directory = directories[0]
        self.assertFalse((directory / "tools/ramjet_build.py").exists())
        initial_stat = (directory / "tools/__init__.py").stat()
        snapshot = ramjet_source.snapshot_source(self.root, self.output)
        self.assertEqual(snapshot["bootstrap"][0], directory)
        self.assertEqual((directory / "tools/__init__.py").stat(), initial_stat)
        self.assertEqual(
            {path.name for path in (directory / "tools").iterdir()},
            {Path(path).name for path in ramjet_source.BOOTSTRAP},
        )

    def test_bootstrap_copy_rejects_change_after_hashing(self):
        original_inspect = ramjet_source._inspect

        def inspect(root, relative, content=False):
            if relative == "tools/ramjet_build.py" and content:
                (root / relative).write_text("changed before freezing")
            return original_inspect(root, relative, content)

        with (
            mock.patch.object(ramjet_source, "_inspect", side_effect=inspect),
            self.assertRaisesRegex(RuntimeError, "Source changed after snapshot"),
        ):
            ramjet_source.snapshot_source(self.root, self.output)
        self.assertFalse(any(self.output.glob("bootstrap-*/tools/ramjet_build.py")))

    def test_bootstrap_rejects_symlink_to_unfrozen_file(self):
        path = self.root / "tools/ramjet.py"
        path.unlink()
        path.symlink_to("stats/helper.py")
        with self.assertRaisesRegex(ValueError, "bootstrap must be a regular file"):
            ramjet_source.snapshot_source(self.root, self.output)

    def test_revision_is_provenance_not_build_identity(self):
        before = ramjet_source.snapshot_source(self.root, self.output)
        self.git_sha = "second-revision"
        self.submodules = " def456 third_party/pkg (heads/main)\n"
        after = ramjet_source.snapshot_source(self.root, self.output)
        self.assertEqual(before["build"], after["build"])
        metadata = self.metadata(after["tests"][0])
        self.assertEqual(metadata["git_sha"], self.git_sha)
        self.assertEqual(metadata["submodules"], self.submodules)

    def test_unchanged_source_reuses_hashes_and_archive(self):
        before = ramjet_source.snapshot_source(self.root, self.output)
        with mock.patch.object(ramjet_source, "_inspect") as inspect:
            after = ramjet_source.snapshot_source(self.root, self.output)
        inspect.assert_not_called()
        self.assertEqual(before, after)

    def test_changed_file_with_restored_mtime_is_rehashed(self):
        before = ramjet_source.snapshot_source(self.root, self.output)
        path = self.root / "torch/native.cpp"
        original = path.stat()
        path.write_bytes(b"x" * original.st_size)
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        after = ramjet_source.snapshot_source(self.root, self.output)
        self.assertNotEqual(before["build"], after["build"])

    def test_delta_contains_only_changed_new_and_deleted_inputs(self):
        before = ramjet_source.snapshot_source(self.root, self.output)
        previous = json.loads(before["build"][0].read_text())
        (self.root / "torch/native.cpp").write_text("changed native")
        (self.root / "torch/new.cpp").write_text("new source")
        (self.root / "torch/link").symlink_to("python_module.py")
        (self.root / "README.md").unlink()
        (self.root / "scripts/example.sh").chmod(0o755)
        self.tracked.update({"torch/new.cpp", "torch/link"})
        current = ramjet_source.snapshot_source(self.root, self.output)
        delta = ramjet_source.create_source_delta(
            self.root, current["build"][0], previous, self.output
        )
        changed = [
            "scripts/example.sh",
            "torch/link",
            "torch/native.cpp",
            "torch/new.cpp",
        ]
        self.assertEqual(
            self.metadata(delta, "manifest.json"),
            {
                "build_sha256": current["build"][1],
                "deleted": ["README.md"],
                "files": changed,
            },
        )
        with tarfile.open(delta) as archive:
            self.assertEqual(
                set(archive.getnames()),
                {"manifest.json", *(f"files/{path}" for path in changed)},
            )
            link = archive.getmember("files/torch/link")
            self.assertTrue(link.issym())
            self.assertEqual(link.linkname, "python_module.py")
            self.assertEqual(archive.getmember("files/scripts/example.sh").mode, 0o755)
            stream = archive.extractfile("files/torch/native.cpp")
            if stream is None:
                raise AssertionError("Expected changed native source")
            self.assertEqual(stream.read(), b"changed native")
            self.assertTrue(all(member.mtime == 0 for member in archive))

    def test_unchanged_delta_has_no_source_members(self):
        snapshot = ramjet_source.snapshot_source(self.root, self.output)
        previous = json.loads(snapshot["build"][0].read_text())
        delta = ramjet_source.create_source_delta(
            self.root, snapshot["build"][0], previous, self.output
        )
        with tarfile.open(delta) as archive:
            self.assertEqual(archive.getnames(), ["manifest.json"])
        self.assertEqual(self.metadata(delta, "manifest.json")["files"], [])

    def test_delta_rejects_source_changed_after_snapshot(self):
        snapshot = ramjet_source.snapshot_source(self.root, self.output)
        (self.root / "torch/native.cpp").write_text("changed after snapshot")
        with self.assertRaisesRegex(RuntimeError, "Source changed after snapshot"):
            ramjet_source.create_source_delta(
                self.root, snapshot["build"][0], {}, self.output
            )

    def test_symlink_target_change_is_transferred(self):
        link = self.root / "torch/link"
        link.symlink_to("python_module.py")
        self.tracked.add("torch/link")
        before = ramjet_source.snapshot_source(self.root, self.output)
        previous = json.loads(before["build"][0].read_text())
        link.unlink()
        link.symlink_to("testing", target_is_directory=True)
        after = ramjet_source.snapshot_source(self.root, self.output)
        delta = ramjet_source.create_source_delta(
            self.root, after["build"][0], previous, self.output
        )
        self.assertEqual(self.metadata(delta, "manifest.json")["files"], ["torch/link"])
        with tarfile.open(delta) as archive:
            member = archive.getmember("files/torch/link")
            self.assertTrue(member.issym())
            self.assertEqual(member.linkname, "testing")

    def test_snapshot_rejects_source_changed_during_hashing(self):
        original_digest = hashlib.file_digest

        def file_digest(source, algorithm):
            digest = original_digest(source, algorithm)
            if Path(source.name) == self.root / "torch/native.cpp":
                Path(source.name).write_text("changed during hashing")
            return digest

        with (
            mock.patch.object(hashlib, "file_digest", side_effect=file_digest),
            self.assertRaisesRegex(RuntimeError, "Source changed while reading"),
        ):
            ramjet_source.snapshot_source(self.root, self.output)

    @parametrize("target", ["../../private", "/absolute/path"])
    def test_snapshot_rejects_escaping_symlinks(self, target):
        (self.root / "torch/link").symlink_to(target)
        self.tracked.add("torch/link")
        with self.assertRaisesRegex(ValueError, "escapes the repository"):
            ramjet_source.snapshot_source(self.root, self.output)

    @parametrize("state", ["-", "U"])
    def test_snapshot_requires_initialized_submodules(self, state):
        self.submodules = f"{state}abc123 third_party/pkg\n"
        with self.assertRaisesRegex(RuntimeError, "Initialize submodules"):
            ramjet_source.snapshot_source(self.root, self.output)

    def test_concurrent_snapshot_publication(self):
        barrier = threading.Barrier(2)
        original_archive = ramjet_source._archive
        original_inspect = ramjet_source._inspect

        def inspect(root, relative, content=False):
            result = original_inspect(root, relative, content)
            if relative == "tools/__init__.py" and content:
                barrier.wait(timeout=10)
            return result

        def archive(*args, **kwargs):
            barrier.wait(timeout=10)
            return original_archive(*args, **kwargs)

        with (
            mock.patch.object(ramjet_source, "_archive", side_effect=archive),
            mock.patch.object(ramjet_source, "_inspect", side_effect=inspect),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures = [
                pool.submit(ramjet_source.snapshot_source, self.root, self.output)
                for _ in range(2)
            ]
            first, second = [future.result(timeout=20) for future in futures]
        self.assertEqual(first, second)
        for name in ("build", "tests"):
            path, digest = first[name]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        directory, digest = first["bootstrap"]
        entries = {
            path: original_inspect(directory, path)[0]
            for path in ramjet_source.BOOTSTRAP
        }
        serialized = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(serialized).hexdigest(), digest)
        self.assertEqual(ramjet_source.snapshot_source(self.root, self.output), first)


if __name__ == "__main__":
    run_tests()
