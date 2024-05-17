import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Optional, Union

from setuptools import distutils  # type: ignore[import]


UNKNOWN = "Unknown"
RELEASE_PATTERN = re.compile(r"/v[0-9]+(\.[0-9]+)*(-rc[0-9]+)?/")


def get_sha(pytorch_root: Union[str, Path]) -> str:
    try:
        rev = None
        if os.path.exists(os.path.join(pytorch_root, ".git")):
            rev = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=pytorch_root
            )
        elif os.path.exists(os.path.join(pytorch_root, ".hg")):
            rev = subprocess.check_output(
                ["hg", "identify", "-r", "."], cwd=pytorch_root
            )
        if rev:
            return rev.decode("ascii").strip()
    except Exception:
        pass
    return UNKNOWN


def get_tag(pytorch_root: Union[str, Path]) -> str:
    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--exact"],
            cwd=pytorch_root,
            encoding="ascii",
            capture_output=True,
        ).stdout.strip()
        if RELEASE_PATTERN.match(tag):
            return tag
        else:
            return UNKNOWN
    except Exception:
        return UNKNOWN


def get_torch_version(sha: Optional[str] = None) -> str:
    pytorch_root = Path(__file__).parent.parent
    version = open(pytorch_root / "version.txt").read().strip()

    if os.getenv("PYTORCH_BUILD_VERSION"):
        assert os.getenv("PYTORCH_BUILD_NUMBER") is not None
        build_number = int(os.getenv("PYTORCH_BUILD_NUMBER", ""))
        version = os.getenv("PYTORCH_BUILD_VERSION", "")
        if build_number > 1:
            version += ".post" + str(build_number)
    elif sha != UNKNOWN:
        if sha is None:
            sha = get_sha(pytorch_root)
        version += "+git" + sha[:7]
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate torch/version.py from build and environment metadata."
    )
    parser.add_argument(
        "--is-debug",
        "--is_debug",
        type=distutils.util.strtobool,
        help="Whether this build is debug mode or not.",
    )
    parser.add_argument("--cuda-version", "--cuda_version", type=str)
    parser.add_argument("--hip-version", "--hip_version", type=str)

    args = parser.parse_args()

    assert args.is_debug is not None
    args.cuda_version = None if args.cuda_version == "" else args.cuda_version
    args.hip_version = None if args.hip_version == "" else args.hip_version

    pytorch_root = Path(__file__).parent.parent
    version_path = pytorch_root / "torch" / "version.py"
    # Attempt to get tag first, fall back to sha if a tag was not found
    tagged_version = get_tag(pytorch_root)
    sha = get_sha(pytorch_root)
    if tagged_version == UNKNOWN:
        version = get_torch_version(sha)
    else:
        version = tagged_version

    with open(version_path, "w") as f:
        f.write("from typing import Optional\n\n")
        f.write("__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']\n")
        f.write(f"__version__ = '{version}'\n")
        # NB: This is not 100% accurate, because you could have built the
        # library code with DEBUG, but csrc without DEBUG (in which case
        # this would claim to be a release build when it's not.)
        f.write(f"debug = {repr(bool(args.is_debug))}\n")
        f.write(f"cuda: Optional[str] = {repr(args.cuda_version)}\n")
        f.write(f"git_version = {repr(sha)}\n")
        f.write(f"hip: Optional[str] = {repr(args.hip_version)}\n")
