#!/usr/bin/env python3

import argparse
import os
import subprocess
import re

from datetime import datetime
from distutils.util import strtobool
from pathlib import Path

LEADING_V_PATTERN = re.compile("^v")
TRAILING_RC_PATTERN = re.compile("-rc[0-9]*$")
LEGACY_BASE_VERSION_SUFFIX_PATTERN = re.compile("a0$")

class NoGitTagException(Exception):
    pass

def get_pytorch_root() -> Path:
    return Path(subprocess.check_output(
        ['git', 'rev-parse', '--show-toplevel']
    ).decode('ascii').strip())

def get_tag() -> str:
    root = get_pytorch_root()
    try:
        dirty_tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--exact'],
            cwd=root
        ).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return ""
    # Strip leading v that we typically do when we tag branches
    # ie: v1.7.1 -> 1.7.1
    tag = re.sub(LEADING_V_PATTERN, "", dirty_tag)
    # Strip trailing rc pattern
    # ie: 1.7.1-rc1 -> 1.7.1
    tag = re.sub(TRAILING_RC_PATTERN, "", tag)
    return tag

def get_base_version() -> str:
    root = get_pytorch_root()
    dirty_version = open(root / 'version.txt', 'r').read().strip()
    # Strips trailing a0 from version.txt, not too sure why it's there in the
    # first place
    return re.sub(LEGACY_BASE_VERSION_SUFFIX_PATTERN, "", dirty_version)

class PytorchVersion:
    def __init__(
        self,
        gpu_arch_type: str,
        gpu_arch_version: str,
        no_build_suffix: bool,
    ) -> None:
        self.gpu_arch_type = gpu_arch_type
        self.gpu_arch_version = gpu_arch_version
        self.no_build_suffix = no_build_suffix

    def get_post_build_suffix(self) -> str:
        if self.no_build_suffix:
            return ""
        if self.gpu_arch_type == "cuda":
            return f"+cu{self.gpu_arch_version.replace('.', '')}"
        return f"+{self.gpu_arch_type}{self.gpu_arch_version}"

    def get_release_version(self) -> str:
        if not get_tag():
            raise NoGitTagException(
                "Not on a git tag, are you sure you want a release version?"
            )
        return f"{get_tag()}{self.get_post_build_suffix()}"

    def get_nightly_version(self) -> str:
        date_str = datetime.today().strftime('%Y%m%d')
        build_suffix = self.get_post_build_suffix()
        return f"{get_base_version()}.dev{date_str}{build_suffix}"

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate pytorch version for binary builds"
    )
    parser.add_argument(
        "--no-build-suffix",
        action="store_true",
        help="Whether or not to add a build suffix typically (+cpu)",
        default=strtobool(os.environ.get("NO_BUILD_SUFFIX", "False"))
    )
    parser.add_argument(
        "--gpu-arch-type",
        type=str,
        help="GPU arch you are building for, typically (cpu, cuda, rocm)",
        default=os.environ.get("GPU_ARCH_TYPE", "cpu")
    )
    parser.add_argument(
        "--gpu-arch-version",
        type=str,
        help="GPU arch version, typically (10.2, 4.0), leave blank for CPU",
        default=os.environ.get("GPU_ARCH_VERSION", "")
    )
    args = parser.parse_args()
    version_obj = PytorchVersion(
        args.gpu_arch_type,
        args.gpu_arch_version,
        args.no_build_suffix
    )
    try:
        print(version_obj.get_release_version())
    except NoGitTagException:
        print(version_obj.get_nightly_version())

if __name__ == "__main__":
    main()
