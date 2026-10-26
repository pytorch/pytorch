from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

from tools.ramjet_dependencies import (
    DEPENDENCIES,
    install_dependencies,
    prepare_dependencies,
)

from torch.testing._internal.common_utils import run_tests, TestCase


VISION_COMMIT = "0123456789abcdef" * 2 + "01234567"
TORCH_VERSION = "2.16.0a0+ramjet.123456789abc"


class TestRamjetDependencies(TestCase):
    root: Path

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-dependency-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_dependency_build_preserves_torch_and_cross_compiles_vision(self):
        def run(command, **kwargs):
            self.assertTrue(kwargs["check"])
            self.assertEqual(os.environ["MAX_JOBS"], "16")
            self.assertEqual(os.environ["FORCE_CUDA"], "1")
            self.assertEqual(os.environ["WITH_CUDA"], "1")
            self.assertEqual(os.environ["TORCH_CUDA_ARCH_LIST"], "7.5;8.9")
            self.assertTrue(Path(os.environ["CCACHE_DIR"]).is_dir())
            if "--constraint" in command:
                path = Path(command[command.index("--constraint") + 1])
                self.assertEqual(path.read_text(), f"torch==={TORCH_VERSION}\n")
                self.assertEqual(command[-len(DEPENDENCIES) :], list(DEPENDENCIES))
            elif "wheel" in command:
                self.assertIn("--no-build-isolation", command)
                self.assertIn("--no-deps", command)
                self.assertEqual(
                    command[-1],
                    f"git+https://github.com/pytorch/vision.git@{VISION_COMMIT}",
                )
                directory = Path(command[command.index("--wheel-dir") + 1])
                directory.joinpath("torchvision-test.whl").write_text("wheel")
            elif "install" in command:
                self.assertIn("--no-deps", command)
                self.assertIn("--no-index", command)
                self.assertEqual(Path(command[-1]).read_text(), "wheel")
            else:
                self.assertIn("torchvision.ops.nms", command[-1])

        with (
            mock.patch.dict(os.environ, {"TORCH_CUDA_ARCH_LIST": "7.5;8.9"}),
            mock.patch("tempfile.gettempdir", return_value=str(self.root)),
            mock.patch("importlib.metadata.version", return_value=TORCH_VERSION),
            mock.patch("subprocess.run", side_effect=run) as execute,
        ):
            install_dependencies(VISION_COMMIT)
        self.assertEqual(execute.call_count, 4)

    def test_dependency_install_failure_stops_before_vision_build(self):
        error = subprocess.CalledProcessError(1, ["pip", "install"])
        with (
            mock.patch.dict(os.environ),
            mock.patch("tempfile.gettempdir", return_value=str(self.root)),
            mock.patch("importlib.metadata.version", return_value=TORCH_VERSION),
            mock.patch("subprocess.run", side_effect=error) as execute,
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                install_dependencies(VISION_COMMIT)
        self.assertEqual(execute.call_count, 1)

    def test_dependency_image_uses_frozen_source_and_pinned_vision(self):
        pin = self.root / ".github/ci_commit_pins/vision.txt"
        pin.parent.mkdir(parents=True)
        pin.write_text(VISION_COMMIT + "\n")
        image = mock.Mock()
        snapshots = {"bootstrap": (self.root / "bootstrap", "digest")}

        result = prepare_dependencies(image, self.root, snapshots)

        image.add_local_file.assert_called_once_with(
            self.root / "bootstrap/tools/ramjet_dependencies.py",
            "/opt/ramjet/tools/ramjet_dependencies.py",
            copy=True,
        )
        call = image.add_local_file.return_value.run_function
        call.assert_called_once_with(
            install_dependencies,
            cpu=16,
            memory=65536,
            gpu=None,
            timeout=1800,
            include_source=False,
            kwargs={"vision_commit": VISION_COMMIT},
        )
        self.assertIs(result, call.return_value)

    def test_dependency_image_rejects_mutable_revision(self):
        pin = self.root / ".github/ci_commit_pins/vision.txt"
        pin.parent.mkdir(parents=True)
        pin.write_text("main\n")
        image = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Invalid torchvision commit pin"):
            prepare_dependencies(image, self.root, {})
        image.add_local_file.assert_not_called()


if __name__ == "__main__":
    run_tests()
