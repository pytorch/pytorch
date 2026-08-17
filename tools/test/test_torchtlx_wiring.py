#!/usr/bin/env python3
"""Tests for the FBTriton wiring in .ci/pytorch/binary_populate_env.sh.

The claim this fork rests on is that FBTriton is opt-in and the default build
path is unchanged, so that claim is asserted directly: with FBTRITON unset the
emitted PYTORCH_EXTRA_INSTALL_REQUIREMENTS must be byte-identical to what
upstream produces.

    python tools/test/test_torchtlx_wiring.py
    pytest tools/test -o "python_files=test*.py" -k torchtlx

Lives here rather than under tools/torchtlx/ so CI collects it: the lint
workflow runs `pytest tools/test`, and a guard on the default build path only
helps if it runs unattended. That job uses the linter image, so this shells out
to bash and compares strings; it imports neither torch nor triton, and needs no
GPU and no network. Hence plain unittest rather than torch.testing._internal,
matching the rest of tools/test.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".ci/pytorch/binary_populate_env.sh"

# The whole script runs under `set -eux` and needs many CI-only variables, so
# only the Triton requirement block is extracted and evaluated.
BLOCK_START = "/# Set triton version as part of/"
BLOCK_END = "/# Set triton via PYTORCH_EXTRA_INSTALL_REQUIREMENTS for triton xpu/"


def emitted_requirement(build_version: str, fbtriton: bool) -> str:
    """Return the triton requirement binary_populate_env.sh would export."""
    block = subprocess.run(
        ["sed", "-n", f"{BLOCK_START},{BLOCK_END}p", str(SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    script = "\n".join(
        [
            f"export PYTORCH_ROOT={REPO_ROOT}",
            "export PACKAGE_TYPE=manywheel",
            f"export PYTORCH_BUILD_VERSION={build_version}",
            "export PYTORCH_EXTRA_INSTALL_REQUIREMENTS=placeholder",
            "export FBTRITON=1" if fbtriton else "unset FBTRITON",
            block,
            'echo "$PYTORCH_EXTRA_INSTALL_REQUIREMENTS"',
        ]
    )
    res = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if res.returncode != 0:
        raise AssertionError(f"block failed: {res.stderr.strip()[-400:]}")
    return res.stdout.strip().split("|")[-1].strip()


class TestBinaryPopulateEnv(unittest.TestCase):
    ROCM_DEV = "2.15.0.dev20260814+rocm7.0"
    ROCM_REL = "2.15.0+rocm7.0"
    CUDA_DEV = "2.15.0.dev20260814+cu129"
    CUDA_REL = "2.15.0+cu129"

    def _version(self) -> str:
        return (REPO_ROOT / ".ci/docker/triton_version.txt").read_text().strip()

    def _shorthash(self) -> str:
        pin = REPO_ROOT / ".ci/docker/ci_commit_pins/triton.txt"
        return pin.read_text().strip()[:8]

    def test_default_rocm_dev_unchanged(self):
        self.assertEqual(
            emitted_requirement(self.ROCM_DEV, fbtriton=False),
            f"triton-rocm=={self._version()}+git{self._shorthash()}; "
            "platform_system == 'Linux' and python_version < '3.15'",
        )

    def test_default_rocm_release_unchanged(self):
        self.assertEqual(
            emitted_requirement(self.ROCM_REL, fbtriton=False),
            f"triton-rocm~={self._version()}; "
            "platform_system == 'Linux' and python_version < '3.15'",
        )

    def test_default_cuda_dev_unchanged(self):
        self.assertEqual(
            emitted_requirement(self.CUDA_DEV, fbtriton=False),
            f"triton=={self._version()}+git{self._shorthash()}; "
            "platform_system == 'Linux' and python_version < '3.15'",
        )

    def test_default_cuda_release_unchanged(self):
        self.assertEqual(
            emitted_requirement(self.CUDA_REL, fbtriton=False),
            f"triton~={self._version()}; "
            "platform_system == 'Linux' and python_version < '3.15'",
        )

    def test_fbtriton_replaces_package_name(self):
        for build_version in (self.ROCM_REL, self.CUDA_REL):
            with self.subTest(build_version=build_version):
                self.assertEqual(
                    emitted_requirement(build_version, fbtriton=True),
                    f"fbtriton~={self._version()}; "
                    "platform_system == 'Linux' and python_version < '3.15'",
                )

    def test_fbtriton_dev_does_not_request_a_git_local_version(self):
        # FBTriton publishes <ver> and <ver>.devYYYYMMDD, never <ver>+git<sha>,
        # so a shorthash pin would be unsatisfiable. Unlike upstream, whose dev
        # wheels PyTorch builds itself from ci_commit_pins/triton.txt.
        for build_version in (self.ROCM_DEV, self.CUDA_DEV):
            with self.subTest(build_version=build_version):
                got = emitted_requirement(build_version, fbtriton=True)
                self.assertNotIn("+git", got)
                self.assertTrue(got.startswith("fbtriton~="), got)

    def test_no_fbtriton_pin_file(self):
        # It would sit inside .ci/docker/, whose tree hash gates every CI Docker
        # image rebuild, for a value nothing in the image reads.
        self.assertFalse(
            (REPO_ROOT / ".ci/docker/ci_commit_pins/fbtriton.txt").exists()
        )


if __name__ == "__main__":
    if not SCRIPT.exists():
        print(f"cannot find {SCRIPT}", file=sys.stderr)
        raise SystemExit(1)
    unittest.main()
