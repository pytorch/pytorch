"""The builder: a 64-core VM sandbox with no GPU. It checks out a commit, builds it
incrementally and saves the whole machine as the next build image."""

import functools
import importlib.util
import re
import shlex
import sys
from pathlib import Path
from types import ModuleType
from typing import NamedTuple, TextIO

import modal
from modal.stream_type import StreamType

from tools.ramjet.common import git, repo_root
from tools.ramjet.lineage import TTL
from tools.ramjet.logger import log, VERBATIM
from tools.ramjet.run import MAX_CORES, MAX_GIB


REGISTRY = "ghcr.io/pytorch/ci-image"
# CI's main image for each CUDA version: Ubuntu jammy, cuDNN 9 and gcc11, which
# .ci/docker/build.sh names py3 for some CUDA versions and py3.11 for others
CI_IMAGE_NAME = re.compile(
    r"(pytorch-linux-jammy-cuda(\d+\.\d+)-cudnn9-py3(?:\.11)?-gcc11)(?![\w.-])"
)
# Blobs arrive with the checkout. A restore can change inode numbers and ctimes,
# and git must not rehash every file for that
CLONE = (
    "git clone -q --filter=blob:none --no-checkout"
    " -c core.checkStat=minimal -c core.trustctime=false"
    " https://github.com/pytorch/pytorch.git /pytorch"
)
# The builder is Modal's largest sandbox. It bills for BUILDER_GIB of memory and
# bursts to MAX_GIB when a build needs it
BUILDER_GIB = 160
BUILDER_TIMEOUT = 4 * 3600  # a cold build takes about an hour; this caps a stuck one
SNAPSHOT_TIMEOUT = 30 * 60  # the first build's snapshot took 35 s
SUBMODULE_JOBS = 16  # parallel clones wait on the network, not the cores
# The compiles every builder in the workspace has done, which ccache looks up after
# the build image's own cache. Its keys hash the compiler, the command line and every
# header, so builds of any config share it. Build images leave Volumes out
CACHE_VOLUME = "ramjet-ccache"
CACHE_MOUNT = "/ccache"
CACHE_DAYS = 14  # each build deletes the entries written longer ago
# The builder's own settings; unlike your build variables, they don't change the result
BUILDER_ENV = {
    "MAX_JOBS": str(MAX_CORES),
    # 16 FlashAttention compiles at up to 5 GiB each fit in half of BUILDER_GIB
    "FLASH_ATTENTION_MAX_JOBS": "16",
    # Jammy's ccache 4.5 knows no other name; later ones also call it REMOTE_STORAGE
    "CCACHE_SECONDARY_STORAGE": f"file:{CACHE_MOUNT}",
    # Not the compiler's mtime: CI rebuilds an image under the same tag, and CUDA's
    # files take the time of each build
    "CCACHE_COMPILERCHECK": "content",
}
# CI's image has Triton only for its own Python, and no stage 2 runtimes: install
# the pinned Triton for the Python being built, and, like .ci/pytorch/build.sh, the
# CUTLASS DSL when stage 2 will run
RUNTIMES = (
    "source .ci/pytorch/common_utils.sh && retry bash scripts/install_triton_wheel.sh"
    ' && if [ "$(python tools/native_aot/build_stage2.py --print-verdict)" = RUN ];'
    " then install_cutlass_dsl; fi"
)


class CIImage(NamedTuple):
    name: str  # without the .ci/docker hash its tags end with
    python: str  # the Python CI installs in it


@functools.cache
def _load(path: str) -> ModuleType:
    """One of the repository's scripts, imported by its path."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, repo_root() / path)
    if spec is None or spec.loader is None:
        log.error("cannot load %s", path)
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ci_python(script: str, name: str) -> str:
    """The Python CI puts in one of its images, read from .ci/docker/build.sh's text."""
    pattern = rf"^\s*{re.escape(name)}\)$(.*?)^\s*;;$"  # the image's case
    case = re.search(pattern, script, re.MULTILINE | re.DOTALL)
    version = case and re.search(r"ANACONDA_PYTHON_VERSION=(\S+)", case[1])
    if not version:
        log.error("no Python for %s in .ci/docker/build.sh", name)
        sys.exit(1)
    return version[1]


def ci_images() -> dict[str, CIImage]:
    """CI's image to build in for each CUDA version with an official arch list,
    oldest version first."""
    text = (repo_root() / ".github/workflows/docker-builds.yml").read_text()
    script = (repo_root() / ".ci/docker/build.sh").read_text()
    archs = _load(".ci/wheel/linux/build_env_setup.py").TORCH_CUDA_ARCH_LIST_TABLE
    names = {cuda: name for name, cuda in CI_IMAGE_NAME.findall(text) if cuda in archs}
    images: dict[str, CIImage] = {}
    for cuda in sorted(names, key=lambda v: [int(part) for part in v.split(".")]):
        name = names[cuda]
        images[cuda] = CIImage(f"{REGISTRY}:{name}", ci_python(script, name))
    return images


def python_versions() -> list[str]:
    """The Python versions PyTorch ships wheels for, free-threaded ones included."""
    matrix = _load(".github/scripts/generate_binary_build_matrix.py")
    return matrix.FULL_PYTHON_VERSIONS


def stable_cuda() -> str:
    """The CUDA version of PyTorch's default wheel."""
    return _load(".github/scripts/generate_binary_build_matrix.py").CUDA_STABLE


def default_config(cuda: str) -> dict[str, str]:
    """The official arch list for this CUDA version."""
    setup = _load(".ci/wheel/linux/build_env_setup.py")
    return {"TORCH_CUDA_ARCH_LIST": setup.torch_cuda_arch_list(cuda, "x86_64")}


def toolchain_id(image: CIImage, python: str) -> str:
    """What sets a lineage's toolchain apart: CI's image, and the Python installed
    in it when that isn't the image's own."""
    return image.name if python == image.python else f"{image.name}+python{python}"


def _with_python(image: modal.Image, python: str) -> modal.Image:
    """The image with another Python, installed the way .ci/docker/ubuntu/Dockerfile
    installs CI's: in a venv that comes first on PATH."""
    return image.dockerfile_commands(
        f"ENV PYTHON_VERSION={python.removesuffix('t')}",
        f"ENV PYTHON_FREETHREADED={'1' if python.endswith('t') else ''}",
        "ENV VENV_PATH=/opt/python-${PYTHON_VERSION}-venv",
        "ENV VIRTUAL_ENV=${VENV_PATH}",
        "ENV PATH=${VENV_PATH}/bin:$PATH",
        "COPY requirements-ci.txt /opt/requirements-ci.txt",
        "COPY common/install_python.sh install_python.sh",
        "RUN bash ./install_python.sh && rm install_python.sh /opt/requirements-ci.txt",
        context_dir=repo_root() / ".ci/docker",
    )


def toolchain(commit: str, image: CIImage, python: str) -> modal.Image:
    """CI's image at this commit, with the Python to build with: its tags are keyed
    by the .ci/docker tree hash."""
    ref = f"{image.name}-{git('rev-parse', f'{commit}:.ci/docker')}"
    built = modal.Image.from_registry(ref)
    if python != image.python:
        built = _with_python(built, python)
    # /opt/cache/bin holds sccache wrappers that point at CI's cache, which has no
    # entries for ramjet's build config; ccache takes their place
    return built.apt_install("ccache").run_commands("rm -rf /opt/cache/bin")


def run(
    builder: modal.Sandbox,
    command: str,
    log_file: TextIO,
    env: dict[str, str] | None = None,
    workdir: str = "/pytorch",
) -> None:
    """One command on the builder. Its output goes unchanged to the build log and to
    your stderr; a failure logs an error and exits."""
    log.info("builder: %s", command)
    shell = ["bash", "-c", f"exec 2>&1; {command}"]  # the log gets stderr too
    process = builder.exec(*shell, workdir=workdir, env=env, bufsize=1)  # by line
    for line in process.stdout:
        log_file.write(line)
        log.info(line.removesuffix("\n"), extra=VERBATIM)
    if process.wait() != 0:
        log.error("builder: %s failed; full log in %s", command, log_file.name)
        sys.exit(1)


def build(
    app: modal.App,
    start: modal.Image,
    bundle: Path,
    config: dict[str, str],
    inputs: Path,
    clone: bool,
) -> str:
    """Build the bundle's commit on a builder booted from `start`, and return the id
    of the image that holds the result. `clone` starts a new lineage: the builder
    clones pytorch/pytorch first."""
    log.info("builder: vm, %d cores, %d-%d GiB", MAX_CORES, BUILDER_GIB, MAX_GIB)
    cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True, version=2)
    builder = modal.Sandbox.create(
        app=app,
        image=start,
        runtime="vm",
        cpu=MAX_CORES,
        memory=(BUILDER_GIB * 1024, MAX_GIB * 1024),
        timeout=BUILDER_TIMEOUT,
        volumes={CACHE_MOUNT: cache},
    )
    log.debug("builder: sandbox %s", builder.object_id)
    env = {**BUILDER_ENV, **config}
    script = bundle.with_name("inputs.py")
    steps = [
        f"git fetch -q {shlex.quote(str(bundle))} refs/worktree/ramjet",
        "git checkout -q --force --detach FETCH_HEAD",
        f"git submodule update -q --init --recursive --depth 1 --jobs {SUBMODULE_JOBS}",
        "pip install -q -r requirements-build.txt",
        "pip install -e . -v --no-build-isolation",
        RUNTIMES,
        "python tools/native_aot/build_stage2.py",
        # Bytecode, so new runners skip compiling it; the few files written for
        # newer Pythons fail, harmlessly
        "python -m compileall -qq -j 0 torch || true",
        f"python {shlex.quote(str(script))} /pytorch {shlex.quote(str(inputs))}",
        # Saves the cache's new entries now rather than at shutdown; failing only
        # loses them
        f"sync {CACHE_MOUNT} || true",
    ]
    try:
        # Delete the cache's old entries alongside the build; nothing reads the output
        old = ["-type", "f", "-mmin", f"+{CACHE_DAYS * 24 * 60}", "-delete"]
        quiet = {"stdout": StreamType.DEVNULL, "stderr": StreamType.DEVNULL}
        builder.exec("find", CACHE_MOUNT, *old, **quiet)
        fs = builder.filesystem
        fs.copy_from_local(bundle, str(bundle))  # same paths as yours
        fs.copy_from_local(Path(__file__).with_name("inputs.py"), str(script))
        with open(bundle.with_name("build.log"), "w") as log_file:
            if clone:
                run(builder, CLONE, log_file, workdir="/")
            for step in steps:
                run(builder, step, log_file, env)
            fs.copy_to_local(str(inputs), inputs)
            # Your files stay out of the image
            run(builder, f"rm -r {shlex.quote(str(bundle.parent))}", log_file)
        log.info("builder: saving the build image")
        return builder.snapshot_filesystem(timeout=SNAPSHOT_TIMEOUT, ttl=TTL).object_id
    finally:
        builder.terminate()
