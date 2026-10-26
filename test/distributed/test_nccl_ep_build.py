# Owner(s): ["oncall: distributed"]

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from tools import optional_submodules


sys.path.remove(str(REPO_ROOT))


@unittest.skipIf(shutil.which("cmake") is None, "requires CMake")
@unittest.skipIf(shutil.which("git") is None, "requires Git")
@instantiate_parametrized_tests
class TestNcclEpBuild(TestCase):
    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "pytorch"
        self.source.mkdir()
        gitmodules = (REPO_ROOT / ".gitmodules").read_text()
        (self.source / ".gitmodules").write_text(gitmodules)
        for path in re.findall(r"^\s*path\s*=\s*(.+)$", gitmodules, re.MULTILINE):
            if path == "third_party/nccl-extensions":
                continue
            directory = self.source / path
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "CMakeLists.txt").touch()
        asmjit = self.source / "third_party/fbgemm/external/asmjit"
        asmjit.mkdir(parents=True, exist_ok=True)
        (asmjit / "CMakeLists.txt").touch()
        (self.source / "third_party/nccl").mkdir()
        shutil.copytree(
            REPO_ROOT / "third_party/nccl_ep", self.source / "third_party/nccl_ep"
        )
        (self.source / "CMakeLists.txt").write_text(
            """\
cmake_minimum_required(VERSION 3.18)
project(nccl_ep_configuration NONE)
include("${PYTORCH_SOURCE_DIR}/cmake/PreBuildSteps.cmake")
if(USE_NCCL_EP)
  include(ExternalProject)
  set(NCCL_INCLUDE_DIRS "${CMAKE_BINARY_DIR}/system_nccl/include")
  add_custom_target(nccl_external)
  include("${PYTORCH_SOURCE_DIR}/cmake/External/nccl_ep.cmake")
  ExternalProject_Get_Property(nccl_ep_external CMAKE_ARGS)
  get_target_property(deps nccl_ep_external MANUALLY_ADDED_DEPENDENCIES)
  if(NOT deps)
    set(deps "")
  endif()
  file(WRITE "${CMAKE_BINARY_DIR}/ep_config.txt"
    "args=${CMAKE_ARGS}\n"
    "library=${NCCL_EP_LIBRARIES}\n"
    "dependencies=${deps}\n")
endif()
"""
        )

    def _configure(self, *, enabled, system_nccl, env=None):
        result = subprocess.run(
            [
                "cmake",
                "-S",
                str(self.source),
                "-B",
                str(self.root / "build"),
                f"-DPYTORCH_SOURCE_DIR={REPO_ROOT.as_posix()}",
                f"-DUSE_NCCL_EP={'ON' if enabled else 'OFF'}",
                f"-DUSE_SYSTEM_NCCL={'ON' if system_nccl else 'OFF'}",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_disabled_without_sources(self):
        self._configure(enabled=False, system_nccl=True)
        self.assertFalse((self.source / "third_party/nccl-extensions").exists())
        self.assertFalse((self.root / "build/ep_config.txt").exists())

    @parametrize("checkout", ["source_nccl", "system_nccl", "release"])
    def test_checkout_pinned_revision(self, checkout):
        upstream = self.root / "upstream"
        upstream.mkdir()
        subprocess.run(["git", "init", "-q", str(upstream)], check=True)
        (upstream / "nccl_ep").mkdir()
        marker = upstream / "nccl_ep/CMakeLists.txt"
        marker.write_text("# Pinned revision\n")
        subprocess.run(["git", "add", "."], cwd=upstream, check=True)
        commit = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com"]
        commit += ["-c", "commit.gpgsign=false", "commit", "-qm", "Test fixture"]
        subprocess.run(commit, cwd=upstream, check=True)
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
        ).strip()
        (self.source / "third_party/nccl_ep/pin.txt").write_text(revision + "\n")
        marker.write_text("# Newer revision\n")
        subprocess.run(["git", "add", "."], cwd=upstream, check=True)
        subprocess.run(commit, cwd=upstream, check=True)

        gitconfig = self.root / "gitconfig"
        gitconfig.write_text(
            f'[url "{upstream.as_uri()}"]\n'
            "    insteadOf = https://github.com/NVIDIA/nccl-extensions.git\n"
        )
        env = os.environ.copy()
        env.update(
            GIT_CONFIG_GLOBAL=str(gitconfig),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_ALLOW_PROTOCOL="file",
        )
        env.pop("NCCL_INCLUDE_DIR", None)
        if checkout == "release":
            third_party = self.source / "third_party"
            with (
                patch.object(optional_submodules, "third_party_path", third_party),
                patch.dict(os.environ, env),
            ):
                optional_submodules.checkout_nccl_ep()
        else:
            system_nccl = checkout == "system_nccl"
            self._configure(enabled=True, system_nccl=system_nccl, env=env)
            config = dict(
                line.split("=", 1)
                for line in (self.root / "build/ep_config.txt").read_text().splitlines()
            )
            source_dir = (self.source / "third_party/nccl-extensions/nccl_ep").as_posix()
            self.assertIn(f"-DNCCL_EP_SOURCE_DIR={source_dir}", config["args"].split(";"))
            library = "libnccl_ep.so" if system_nccl else "libnccl_ep.a"
            self.assertEqual(Path(config["library"]), self.root / "build/nccl/lib" / library)
            self.assertEqual("nccl_external" in config["dependencies"], not system_nccl)

        checkout_dir = self.source / "third_party/nccl-extensions"
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout_dir, text=True
        ).strip()
        self.assertEqual(actual, revision)
        self.assertEqual(
            (checkout_dir / "nccl_ep/CMakeLists.txt").read_text(), "# Pinned revision\n"
        )


if __name__ == "__main__":
    run_tests()
