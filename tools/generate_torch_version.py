import argparse
import os
import subprocess
from pathlib import Path
from setuptools import distutils  # type: ignore[import]
from typing import Optional, Union

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

def get_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=get_pytorch_root()
        ).decode('ascii').strip()
    except Exception:
        return 'Unknown'

def get_tag() -> str:
    root = get_pytorch_root()
    # We're on a tag
    am_on_tag = (
        subprocess.run(
            ['git', 'describe', '--tags', '--exact'],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        ).returncode == 0
    )
    tag = ""
    if am_on_tag:
        dirty_tag = subprocess.check_output(
            ['git', 'describe', '--tags', '--exact'],
            cwd=root
        ).decode('ascii').strip()
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

    def get_dev_version(self) -> str:
        return f"{get_base_version()}+git{get_sha()[:7]}"

def get_torch_version(
        gpu_arch_type: str = "",
        gpu_arch_version: str = "",
        no_build_suffix: bool = False,
        is_nightly: bool = False,
        # TODO: Remove these once we can rely fully on gpu_arch_[type,version]
        cuda_version: str = "",
        hip_version: str = "",
) -> str:
    version_obj = PytorchVersion(
        gpu_arch_version=gpu_arch_version or cuda_version or hip_version,
        gpu_arch_type=gpu_arch_type,
        no_build_suffix=no_build_suffix,
    )
    # NOTE: This is a legacy way to grab versions, prefer not using PYTORCH_BUILD_VERSION
    if os.getenv('PYTORCH_BUILD_VERSION'):
         assert os.getenv('PYTORCH_BUILD_NUMBER') is not None
         build_number = int(os.getenv('PYTORCH_BUILD_NUMBER', ""))
         version = os.getenv('PYTORCH_BUILD_VERSION', "")
         if build_number > 1:
             version += '.post' + str(build_number)
    else:
        try:
            version = version_obj.get_release_version()
        except NoGitTagException:
            if is_nightly:
                version = version_obj.get_nightly_version()
            else:
                version = version_obj.get_dev_version()
    return version

def main():
    version_path = get_pytorch_root() / "torch" / "version.py"

    with open(version_path, 'w') as f:
        f.write(f"""__version__ = '{get_torch_version()}'
debug = {bool(args.is_debug)}
cuda = {repr(args.cuda_version or None)}
git_version = '{get_sha()}'
hip = {repr(args.hip_version or None)}
gpu_arch_type = '{args.gpu_arch_type}'
gpu_arch_version = '{args.gpu_arch_version}'""")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate torch/version.py from build and environment metadata.")
    parser.add_argument(
        "--cuda_version",
        type=str,
        default=os.environ.get("CUDA_VERSION", ""),
        help="[DEPRECATED] This option will be removed in the future (post 1.10), prefer --gpu-arch-version with --gpu-arch-type cuda"
    )
    parser.add_argument(
        "--hip_version",
        type=str,
        default=os.environ.get("TORCH_HIP_VERSION", ""),
        help="[DEPRECATED] This option will be removed in the future (post 1.10), prefer --gpu-arch-version with --gpu-arch-type rocm"
    )
    parser.add_argument(
        "--is_debug",
        type=bool,
        default=strtobool(os.environ.get("TORCH_VERSION_DEBUG", "False")),
        help="Whether this build is debug mode or not."
    )
    parser.add_argument(
        "--is_nightly",
        type=bool,
        default=strtobool(os.environ.get("IS_NIGHTLY", "False")),
        help="Whether this build is a nightly or not."
    )
    parser.add_argument(
        "--gpu-arch-type",
        type=str,
        help="GPU arch you are building for, typically (cpu, cuda, rocm)",
        default=os.environ.get("GPU_ARCH_TYPE", "cpu"),
        choices=["cpu", "cuda", "rocm"]
    )
    parser.add_argument(
        "--gpu-arch-version",
        type=str,
        help="GPU arch version, typically (10.2, 4.0), leave blank for CPU",
        default=os.environ.get("GPU_ARCH_VERSION", "")
    )
    parser.add_argument(
        "--no-build-suffix",
        action="store_true",
        help="Whether or not to add a build suffix typically (+cpu)",
        default=strtobool(os.environ.get("NO_BUILD_SUFFIX", "False"))
    )

    args = parser.parse_args()
    main()
