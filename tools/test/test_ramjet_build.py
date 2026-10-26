from __future__ import annotations

import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.ramjet_build import build_wheel

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


BUILD_SHA256 = "0123456789abcdef" * 4


@instantiate_parametrized_tests
class TestRamjetBuild(TestCase):
    root: Path
    source: Path
    cache: mock.Mock
    commands: list[list[str]]
    dependencies: str
    build_error: subprocess.CalledProcessError | None

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-build-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.source.joinpath("version.txt").write_text("2.17.0a0\n")
        self.cache = mock.Mock()
        self.commands = []
        self.dependencies = ""
        self.build_error = None

    def make_delta(self, changed=None, deleted=(), symlinks=None, digest=BUILD_SHA256):
        changed = changed or {}
        symlinks = symlinks or {}
        delta = self.root / "delta.tar.gz"
        manifest = {
            "build_sha256": digest,
            "files": [*changed, *symlinks],
            "deleted": list(deleted),
        }
        contents = {f"files/{path}": text for path, text in changed.items()}
        contents["manifest.json"] = json.dumps(manifest)
        with tarfile.open(delta, "w:gz") as archive:
            for path, text in contents.items():
                content = text.encode()
                entry = tarfile.TarInfo(path)
                entry.size = len(content)
                entry.mode = 0o644
                archive.addfile(entry, io.BytesIO(content))
            for path, target in symlinks.items():
                entry = tarfile.TarInfo(f"files/{path}")
                entry.type = tarfile.SYMTYPE
                entry.linkname = target
                archive.addfile(entry)
        return delta

    def run_build(self, delta):
        paths = {
            "/opt/pytorch": self.source,
            "/opt/wheels": self.root / "wheels",
            "/opt/ramjet-build.json": self.root / "build.json",
        }

        def remote_path(value):
            return paths.get(str(value), Path(value))

        def run(command, **kwargs):
            self.commands.append(command)
            version = f"2.17.0a0+ramjet.{BUILD_SHA256[:12]}"
            self.assertEqual(os.environ["PYTORCH_BUILD_VERSION"], version)
            self.assertEqual(os.environ["PYTORCH_BUILD_NUMBER"], "1")
            if "wheel" in command:
                self.assertEqual(kwargs["cwd"], self.source)
                if self.build_error is not None:
                    raise self.build_error
                paths["/opt/wheels"].joinpath("torch-new.whl").write_text("wheel")

        statistics = iter(["before", "after"])

        def check_output(command, **kwargs):
            if command == ["ninja", "-t", "deps"]:
                self.assertEqual(kwargs["cwd"], self.source / "build")
                self.assertEqual(self.commands, [])
                return self.dependencies
            self.assertEqual(command, ["ccache", "--print-stats"])
            return next(statistics)

        volume = SimpleNamespace(from_name=mock.Mock(return_value=self.cache))
        modules = {
            "pathlib": SimpleNamespace(**{**vars(pathlib), "Path": remote_path}),
            "modal": SimpleNamespace(Volume=volume),
        }
        environment = {"CCACHE_TEMPDIR": str(self.root / "ccache-tmp")}
        with (
            mock.patch.dict(sys.modules, modules),
            mock.patch.dict(os.environ, environment),
            mock.patch("subprocess.check_output", side_effect=check_output),
            mock.patch("subprocess.run", side_effect=run),
        ):
            build_wheel("ramjet-test-cache", BUILD_SHA256, str(delta))

    def test_incremental_build_preserves_untouched_files_and_native_outputs(self):
        changed = self.source / "changed.cpp"
        changed.write_text("old")
        untouched = self.source / "untouched.cpp"
        untouched.write_text("same")
        removed = self.source / "removed.cpp"
        removed.write_text("deleted")
        native = self.source / "build/object.o"
        native.parent.mkdir()
        native.write_text("compiled")
        for path in (changed, untouched, native):
            os.utime(path, ns=(1_000_000_000, 1_000_000_000))
        wheels = self.root / "wheels"
        wheels.mkdir()
        wheels.joinpath("torch-old.whl").write_text("old wheel")
        delta = self.make_delta(
            {"changed.cpp": "new", "nested/new.cpp": "created"},
            deleted=["removed.cpp"],
            symlinks={"nested/link.cpp": "../changed.cpp"},
        )

        self.run_build(delta)

        self.assertEqual(changed.read_text(), "new")
        self.assertGreater(changed.stat().st_mtime_ns, 1_000_000_000)
        self.assertEqual(untouched.read_text(), "same")
        self.assertEqual(untouched.stat().st_mtime_ns, 1_000_000_000)
        self.assertEqual(native.read_text(), "compiled")
        self.assertEqual(native.stat().st_mtime_ns, 1_000_000_000)
        self.assertFalse(removed.exists())
        self.assertEqual(self.source.joinpath("nested/new.cpp").read_text(), "created")
        self.assertTrue(self.source.joinpath("nested/link.cpp").is_symlink())
        self.assertEqual(self.source.joinpath("nested/link.cpp").read_text(), "new")
        self.assertFalse(delta.exists())
        self.assertEqual([path.name for path in wheels.iterdir()], ["torch-new.whl"])
        self.cache.commit.assert_called_once()
        self.assertEqual(len(self.commands), 2)
        self.assertIn("--no-build-isolation", self.commands[0])
        self.assertIn("--no-deps", self.commands[0])
        self.assertEqual(self.commands[1][-1], str(wheels / "torch-new.whl"))
        info = json.loads(self.root.joinpath("build.json").read_text())
        self.assertEqual(info["build_sha256"], BUILD_SHA256)
        self.assertEqual(info["changed_files"], 3)
        self.assertEqual(info["deleted_files"], 1)
        self.assertEqual(info["ccache_after"], "after")
        self.assertGreaterEqual(info["wheel_install_seconds"], 0)

    @parametrize("file_to_directory", [True, False])
    def test_file_directory_transitions(self, file_to_directory):
        parent = self.source / "transition"
        if file_to_directory:
            parent.write_text("old")
            delta = self.make_delta(
                {"transition/new.cpp": "new"}, deleted=["transition"]
            )
        else:
            parent.mkdir()
            parent.joinpath("old.cpp").write_text("old")
            delta = self.make_delta(
                {"transition": "new"}, deleted=["transition/old.cpp"]
            )

        self.run_build(delta)

        path = parent / "new.cpp" if file_to_directory else parent
        self.assertEqual(path.read_text(), "new")

    @parametrize("path", ["../escape", "/absolute", "nested/../../escape"])
    def test_rejects_unsafe_paths_before_building(self, path):
        delta = self.make_delta({path: "unsafe"})
        with self.assertRaisesRegex(ValueError, "Invalid source path"):
            self.run_build(delta)
        self.assertEqual(self.commands, [])
        self.cache.commit.assert_not_called()

    @parametrize("target", ["../../escape", "/absolute"])
    def test_rejects_symlinks_outside_source(self, target):
        delta = self.make_delta(symlinks={"nested/link": target})
        with self.assertRaisesRegex(ValueError, "Source symlink escapes"):
            self.run_build(delta)
        self.assertFalse(self.source.joinpath("nested/link").is_symlink())
        self.assertEqual(self.commands, [])

    def test_rejects_wrong_build_digest(self):
        delta = self.make_delta({"file.cpp": "new"}, digest="other")
        with self.assertRaisesRegex(ValueError, "does not match the requested build"):
            self.run_build(delta)
        self.assertFalse(self.source.joinpath("file.cpp").exists())
        self.assertEqual(self.commands, [])

    def test_commits_compiler_cache_after_build_failure(self):
        self.build_error = subprocess.CalledProcessError(1, "pip wheel")
        delta = self.make_delta({"file.cpp": "new"})
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_build(delta)
        self.cache.commit.assert_called_once()
        self.assertEqual(len(self.commands), 1)
        info = json.loads(self.root.joinpath("build.json").read_text())
        self.assertEqual(info["ccache_before"], "before")
        self.assertEqual(info["ccache_after"], "after")
        self.assertNotIn("wheel_install_seconds", info)

    def test_restores_stale_ninja_timestamps_without_hiding_changed_sources(self):
        build = self.source / "build"
        build.mkdir()
        build.joinpath(".ninja_deps").write_bytes(b"cached dependency records")
        stale = build / "stale.o"
        valid = build / "valid.o"
        for path in (stale, valid):
            path.write_text("compiled")
            os.utime(path, ns=(1_000_000_000, 1_000_007_846))
        self.dependencies = (
            "stale.o: #deps 1, deps mtime 1000000000 (STALE)\n"
            "    ../changed.cpp\n\n"
            "valid.o: #deps 1, deps mtime 1000007846 (VALID)\n"
            "    ../unchanged.cpp\n\n"
            "missing.o: #deps 1, deps mtime 1000000000 (STALE)\n"
            "    ../deleted.cpp\n"
        )

        self.run_build(self.make_delta({"changed.cpp": "new source"}))

        self.assertEqual(stale.stat().st_mtime_ns, 1_000_000_000)
        self.assertEqual(stale.read_text(), "compiled")
        self.assertEqual(valid.stat().st_mtime_ns, 1_000_007_846)
        self.assertFalse(build.joinpath("missing.o").exists())
        self.assertGreater(
            self.source.joinpath("changed.cpp").stat().st_mtime_ns,
            stale.stat().st_mtime_ns,
        )
        info = json.loads(self.root.joinpath("build.json").read_text())
        self.assertEqual(info["restored_build_outputs"], 1)
        self.assertGreaterEqual(info["native_cache_restore_seconds"], 0)

    def test_rejects_ninja_outputs_outside_build_directory(self):
        build = self.source / "build"
        build.mkdir()
        build.joinpath(".ninja_deps").write_bytes(b"cached dependency records")
        self.dependencies = "../version.txt: #deps 0, deps mtime 1 (STALE)\n"
        with self.assertRaisesRegex(ValueError, "Ninja output escapes"):
            self.run_build(self.make_delta())
        self.assertEqual(self.source.joinpath("version.txt").read_text(), "2.17.0a0\n")
        self.assertEqual(self.commands, [])


if __name__ == "__main__":
    run_tests()
