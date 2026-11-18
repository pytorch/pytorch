"""Which build image a run starts from.

lineage.json in ramjet's directory keeps one lineage per toolchain (CI's image and
Python) and build config: the newest build image and the commit it holds. Each build
starts from the previous build's image, so it only recompiles what changed.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path


TTL = 5 * 24 * 3600  # build images expire this long after they are taken
RENEW_AFTER = 4 * 24 * 3600  # older images are rebuilt so the lineage outlives the TTL
EXPIRES_AFTER = TTL - 3600  # an hour of margin for the sandbox that boots from it

# Read by pip, by CMake's configure step or by the builder's other steps in build.py,
# rather than by ninja
CONFIGURE_FILES = {
    ".gitmodules",
    "pyproject.toml",
    "setup.py",
    "version.txt",
    "requirements-build.txt",
    "scripts/install_triton_wheel.sh",
    ".ci/docker/triton_version.txt",
    ".ci/docker/ci_commit_pins/triton.txt",
    ".ci/pytorch/common_utils.sh",
}
# Read by the build but missing from ninja's list: submodules, each of which git
# lists as one path, and native-AOT stage 2, which runs after ninja
BUILD_DIRS = ("third_party/", "tools/native_aot/", "torch/_native/")
# CMake's globs only find a new source here when it reconfigures
SOURCE_DIRS = ("aten/", "c10/", "caffe2/", "functorch/csrc/", "torch/csrc/")
# The variables PyTorch's build reads that change what it builds; others such as
# CMAKE_PREFIX_PATH or CUDA_PATH hold paths on your machine
CONFIG_PREFIXES = ("BUILD_", "USE_")
CONFIG_VARS = ("CMAKE_BUILD_TYPE", "DEBUG", "REL_WITH_DEB_INFO", "TORCH_CUDA_ARCH_LIST")


@dataclass(frozen=True)
class Build:
    image_id: str
    commit: str
    taken: float  # time.time() when the image was taken


def build_config(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name, value in environ.items()
        if name.startswith(CONFIG_PREFIXES) or name in CONFIG_VARS
    }


def key(toolchain: str, config: dict[str, str]) -> str:
    """One lineage per toolchain and build config. A lineage keeps the tag of CI's
    image it started on, because CI retags its images whenever .ci/docker changes."""
    identity = json.dumps([toolchain, config], sort_keys=True)
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def load(path: Path) -> dict[str, Build]:
    if not path.exists():
        return {}
    saved = json.loads(path.read_text())
    return {name: Build(**build) for name, build in saved.items()}


def save(path: Path, builds: Mapping[str, Build]) -> None:
    scratch = path.with_suffix(".tmp")
    scratch.write_text(json.dumps({name: asdict(b) for name, b in builds.items()}))
    os.replace(scratch, path)


def usable(build: Build | None, now: float) -> Build | None:
    """The build, unless its image is gone or about to expire."""
    return build if build is not None and now - build.taken < EXPIRES_AFTER else None


def due(build: Build, now: float) -> bool:
    """Whether the image is old enough that the next run should rebuild it."""
    return now - build.taken > RENEW_AFTER


def read_by_build(changes: Mapping[str, str], inputs: frozenset[str]) -> list[str]:
    """The changed paths the build reads: a change to one needs the builder, while
    every other change can go straight to a runner."""
    return [
        path
        for path, status in changes.items()
        if path in inputs
        or path in CONFIGURE_FILES
        or path.endswith(("CMakeLists.txt", ".cmake"))
        or path.startswith(BUILD_DIRS)
        or (status == "A" and path.startswith(SOURCE_DIRS))
    ]
