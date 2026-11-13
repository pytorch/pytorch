from __future__ import annotations

import contextlib
import os
import tempfile
import unittest
from unittest import mock

import tools.build_libtorch


# Env vars that influence build_libtorch; run_build clears any not set by a case.
CONTROLLED_ENV_VARS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_GENERATOR",
    "DEBUG",
    "REL_WITH_DEB_INFO",
    "MAX_JOBS",
)


class TestBuildLibtorch(unittest.TestCase):
    """Tests the cmake command lines produced by tools/build_libtorch.py."""

    def run_build(
        self, env: dict[str, str], have_ninja: bool = True
    ) -> tuple[list[str], list[str]]:
        """Runs build_libtorch with mocks; returns (configure_args, build_args)."""

        def which(name: str) -> str | None:
            if name == "cmake":
                return "/usr/bin/cmake"
            if name == "ninja" and have_ninja:
                return "/usr/bin/ninja"
            return None

        with contextlib.ExitStack() as stack:
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(mock.patch.dict(os.environ, env))
            for key in CONTROLLED_ENV_VARS:
                if key not in env:
                    os.environ.pop(key, None)
            stack.enter_context(
                mock.patch.object(tools.build_libtorch.shutil, "which", which)
            )
            check_call = stack.enter_context(
                mock.patch.object(tools.build_libtorch.subprocess, "check_call")
            )
            prev_cwd = os.getcwd()
            os.chdir(tmpdir)
            stack.callback(os.chdir, prev_cwd)
            tools.build_libtorch.build_libtorch(rerun_cmake=False, cmake_only=False)

        (configure_call, build_call) = check_call.mock_calls
        return configure_call.args[0], build_call.args[0]

    def test_build_type(self) -> None:
        cases = [
            ({}, "Release"),
            ({"DEBUG": "1"}, "Debug"),
            ({"REL_WITH_DEB_INFO": "1"}, "RelWithDebInfo"),
            # An explicit CMAKE_BUILD_TYPE wins over the env flags.
            ({"CMAKE_BUILD_TYPE": "MinSizeRel", "DEBUG": "1"}, "MinSizeRel"),
        ]
        for env, want in cases:
            with self.subTest(env=env):
                _, build_args = self.run_build(env)
                self.assertEqual(build_args[build_args.index("--config") + 1], want)

    def test_generator(self) -> None:
        cases = [
            # env, have_ninja, expected -G value (None: no -G passed)
            ({}, True, "Ninja"),
            ({}, False, None),
            ({"CMAKE_GENERATOR": "Unix Makefiles"}, True, "Unix Makefiles"),
        ]
        for env, have_ninja, want in cases:
            with self.subTest(env=env, have_ninja=have_ninja):
                configure_args, _ = self.run_build(env, have_ninja=have_ninja)
                if want is None:
                    self.assertNotIn("-G", configure_args)
                else:
                    generator = configure_args[configure_args.index("-G") + 1]
                    self.assertEqual(generator, want)

    def test_build_jobs(self) -> None:
        cases = [
            # env, have_ninja, expected -j value (None: no -j passed)
            ({"MAX_JOBS": "8"}, True, "8"),
            ({"MAX_JOBS": "7"}, False, "7"),
            ({}, True, None),  # ninja parallelizes by default
            ({}, False, "13"),
            ({"CMAKE_GENERATOR": "Unix Makefiles"}, True, "13"),
        ]
        for env, have_ninja, want in cases:
            with self.subTest(env=env, have_ninja=have_ninja):
                cpu_count = mock.patch.object(
                    tools.build_libtorch.multiprocessing, "cpu_count", return_value=13
                )
                with cpu_count:
                    _, build_args = self.run_build(env, have_ninja=have_ninja)
                if want is None:
                    self.assertNotIn("-j", build_args)
                else:
                    self.assertEqual(build_args[build_args.index("-j") + 1], want)


if __name__ == "__main__":
    unittest.main()
