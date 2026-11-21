from __future__ import annotations

import os
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from tools.linter.clang_tidy import generate_build_files


if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


REPO_ROOT = Path(__file__).absolute().parents[2]


def _run_capturing_cmds(func: Callable[[], None]) -> list[list[str]]:
    cmds: list[list[str]] = []
    with (
        mock.patch.object(generate_build_files, "run_cmd", cmds.append),
        mock.patch.dict(os.environ),
    ):
        func()
    return cmds


class TestClangTidyBuildDir(unittest.TestCase):
    def setUp(self) -> None:
        (cmd,) = _run_capturing_cmds(generate_build_files.gen_compile_commands)
        self.build_dir = cmd[cmd.index("-B") + 1]

    def test_isolated_from_the_pip_build_dir(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as f:
            pyproject = tomllib.load(f)
        pip_build_dir = pyproject["tool"]["scikit-build"]["build-dir"]
        self.assertNotEqual(self.build_dir, pip_build_dir)

    def test_generated_aten_headers_land_in_the_same_build_dir(self) -> None:
        cmds = _run_capturing_cmds(generate_build_files.run_autogen)
        torchgen = next(cmd for cmd in cmds if "torchgen.gen" in cmd)
        out_dir = Path(torchgen[torchgen.index("-d") + 1])
        self.assertEqual(out_dir.parts[0], self.build_dir)

    def test_linters_point_at_the_regenerated_build_dir(self) -> None:
        with (REPO_ROOT / ".lintrunner.toml").open("rb") as f:
            config = tomllib.load(f)
        build_dirs = {
            arg.partition("=")[2]
            for linter in config["linter"]
            if linter["code"].startswith("CLANGTIDY")
            for arg in linter["command"]
            if arg.startswith("--build_dir=")
        }
        self.assertEqual(build_dirs, {f"./{self.build_dir}"})


if __name__ == "__main__":
    unittest.main()
