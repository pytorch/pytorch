"""Precedence of cmake/EnvVarForwarding.cmake, measured with a real configure.

The rule under test: a forwarded environment variable overrides the cache on
every configure (environment > -D > cache), for prefix names, passthrough names
and aliases alike, with an alias yielding to its canonical name.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "cmake" / "EnvVarForwarding.cmake"

PREFIX_NAME = "USE_ENVFWD_PROBE"
PASSTHROUGH_NAME = "BLAS"
PROBED = [
    PREFIX_NAME,
    PASSTHROUGH_NAME,
    "CUDA_NVCC_EXECUTABLE",
    "CUDA_HOST_COMPILER",
    "CMAKE_CUDA_HOST_COMPILER",
]
# Names the tests set; scrubbed from the ambient environment so a developer's
# shell cannot leak into a configure.
SCRUBBED = PROBED + ["CUDACXX", "CUDAHOSTCXX", "CMAKE_CUDA_COMPILER"]

CMAKELISTS = """
cmake_minimum_required(VERSION 3.27)
project(envfwd NONE)
set(Python_EXECUTABLE "{python}")
include("{module}")
file(WRITE "${{CMAKE_BINARY_DIR}}/resolved.txt" {lines})
"""


@unittest.skipUnless(shutil.which("cmake"), "cmake not found")
class TestEnvVarForwardingPrecedence(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="envfwd-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.src = self.tmp / "src"
        self.build = self.tmp / "build"
        self.src.mkdir()
        lines = " ".join(f'"{n}=${{{n}}}\\n"' for n in PROBED)
        (self.src / "CMakeLists.txt").write_text(
            CMAKELISTS.format(
                python=Path(sys.executable).as_posix(),
                module=MODULE.as_posix(),
                lines=lines,
            )
        )

    def configure(
        self, env: dict[str, str] | None = None, defines: dict[str, str] | None = None
    ) -> dict[str, str]:
        full_env = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
        full_env.update(env or {})
        cmd = ["cmake", "-S", str(self.src), "-B", str(self.build)]
        cmd += [f"-D{k}={v}" for k, v in (defines or {}).items()]
        subprocess.run(cmd, env=full_env, check=True, capture_output=True, text=True)
        resolved = (self.build / "resolved.txt").read_text().splitlines()
        return dict(line.split("=", 1) for line in resolved)

    def forwarded_names(self) -> list[str]:
        lines = (self.build / "env_forwarded.txt").read_text().splitlines()
        return [line.split("=", 1)[0] for line in lines]

    def test_environment_wins_over_flag(self) -> None:
        for name in (PREFIX_NAME, PASSTHROUGH_NAME):
            with self.subTest(name=name):
                got = self.configure(env={name: "env"}, defines={name: "flag"})
                self.assertEqual(got[name], "env")

    def test_environment_reapplies_over_cache(self) -> None:
        for name in (PREFIX_NAME, PASSTHROUGH_NAME):
            with self.subTest(name=name):
                self.configure(env={name: "first"})
                got = self.configure(env={name: "second"})
                self.assertEqual(got[name], "second")

    def test_cache_keeps_value_when_environment_is_silent(self) -> None:
        for name in (PREFIX_NAME, PASSTHROUGH_NAME):
            with self.subTest(name=name):
                self.configure(env={name: "first"})
                got = self.configure()
                self.assertEqual(got[name], "first")

    def test_flag_updates_cache_when_environment_is_silent(self) -> None:
        for name in (PREFIX_NAME, PASSTHROUGH_NAME):
            with self.subTest(name=name):
                self.configure(env={name: "first"})
                got = self.configure(defines={name: "flag"})
                self.assertEqual(got[name], "flag")

    def test_alias_yields_to_canonical_name(self) -> None:
        got = self.configure(env={"CUDACXX": "alias", "CUDA_NVCC_EXECUTABLE": "canon"})
        self.assertEqual(got["CUDA_NVCC_EXECUTABLE"], "canon")

    def test_alias_reapplies_over_cache(self) -> None:
        self.configure(env={"CUDACXX": "first"})
        got = self.configure(env={"CUDACXX": "second"})
        self.assertEqual(got["CUDA_NVCC_EXECUTABLE"], "second")

    def test_first_listed_alias_wins(self) -> None:
        # CUDAHOSTCXX precedes CMAKE_CUDA_HOST_COMPILER in _ENV_ALIASES for
        # CUDA_HOST_COMPILER; the latter is still forwarded under its own name.
        env = {"CUDAHOSTCXX": "one", "CMAKE_CUDA_HOST_COMPILER": "two"}
        got = self.configure(env=env)
        self.assertEqual(got["CUDA_HOST_COMPILER"], "one")
        self.assertEqual(got["CMAKE_CUDA_HOST_COMPILER"], "two")

    def test_forwarded_names_are_recorded(self) -> None:
        self.configure(env={PREFIX_NAME: "x", PASSTHROUGH_NAME: "y", "CUDACXX": "z"})
        names = self.forwarded_names()
        self.assertIn(PREFIX_NAME, names)
        self.assertIn(PASSTHROUGH_NAME, names)
        self.assertIn("CUDA_NVCC_EXECUTABLE", names)


if __name__ == "__main__":
    unittest.main()
