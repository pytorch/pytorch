#!/usr/bin/env python3
# Much of the logging code here was forked from https://github.com/ezyang/ghstack
# Copyright (c) Edward Z. Yang <ezyang@mit.edu>
r"""Checks out the nightly development version of PyTorch and installs pre-built
binaries into the repo.

You can use this script to check out a new nightly branch with the following::

    $ ./tools/nightly.py checkout -b my-nightly-branch
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

Or if you would like to re-use an existing virtual environment, you can pass in
the prefix argument (--prefix)::

    $ ./tools/nightly.py checkout -b my-nightly-branch -p my-env
    $ source my-env/bin/activate  # or `& .\my-env\Scripts\Activate.ps1` on Windows

To install the nightly binaries built with CUDA, you can pass in the flag --cuda::

    $ ./tools/nightly.py checkout -b my-nightly-branch --cuda
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

You can also use this tool to pull the nightly commits into the current branch as
well. This can be done with::

    $ ./tools/nightly.py pull
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

Pulling will recreate a fresh virtual environment and reinstall the development
dependencies as well as the nightly binaries into the repo directory.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from platform import system as platform_system
from typing import Any, Callable, cast, NamedTuple, TYPE_CHECKING, TypeVar


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator


try:
    from packaging.version import Version
except ImportError:
    Version = None  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).absolute().parent.parent
GITHUB_REMOTE_URL = "https://github.com/pytorch/pytorch.git"
PACKAGES_TO_INSTALL = (
    "torch",
    "numpy",
    "cmake",
    "ninja",
    "packaging",
    "ruff",
    "mypy",
    "pytest",
    "hypothesis",
    "ipython",
    "rich",
    "clang-format",
    "clang-tidy",
    "sphinx",
)
DEFAULT_VENV_DIR = REPO_ROOT / "venv"


LOGGER: logging.Logger | None = None
DATETIME_FORMAT = "%Y-%m-%d_%Hh%Mm%Ss"
SHA1_RE = re.compile(r"(?P<sha1>[0-9a-fA-F]{40})")
USERNAME_PASSWORD_RE = re.compile(r":\/\/(.*?)\@")
LOG_DIRNAME_RE = re.compile(
    r"(?P<datetime>\d{4}-\d\d-\d\d_\d\dh\d\dm\d\ds)_"
    r"(?P<uuid>[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})",
)


PLATFORM = platform_system().replace("Darwin", "macOS")
LINUX = PLATFORM == "Linux"
MACOS = PLATFORM == "macOS"
WINDOWS = PLATFORM == "Windows"
POSIX = LINUX or MACOS


class PipSource(NamedTuple):
    name: str
    index_url: str
    supported_platforms: set[str]
    accelerator: str


PYTORCH_NIGHTLY_PIP_INDEX_URL = "https://download.pytorch.org/whl/nightly"
PIP_SOURCES = {
    "cpu": PipSource(
        name="cpu",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cpu",
        supported_platforms={"Linux", "macOS", "Windows"},
        accelerator="cpu",
    ),
    "cuda-11.8": PipSource(
        name="cuda-11.8",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cu118",
        supported_platforms={"Linux", "Windows"},
        accelerator="cuda",
    ),
    "cuda-12.4": PipSource(
        name="cuda-12.4",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cu124",
        supported_platforms={"Linux", "Windows"},
        accelerator="cuda",
    ),
    "cuda-12.6": PipSource(
        name="cuda-12.6",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/cu126",
        supported_platforms={"Linux", "Windows"},
        accelerator="cuda",
    ),
}


class Formatter(logging.Formatter):
    redactions: dict[str, str]

    def __init__(self, fmt: str | None = None, datefmt: str | None = None) -> None:
        super().__init__(fmt, datefmt)
        self.redactions = {}

    # Remove sensitive information from URLs
    def _filter(self, s: str) -> str:
        s = USERNAME_PASSWORD_RE.sub(r"://<USERNAME>:<PASSWORD>@", s)
        for needle, replace in self.redactions.items():
            s = s.replace(needle, replace)
        return s

    def formatMessage(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO or record.levelno == logging.DEBUG:
            # Log INFO/DEBUG without any adornment
            return record.getMessage()
        else:
            # I'm not sure why, but formatMessage doesn't show up
            # even though it's in the typeshed for Python >3
            return super().formatMessage(record)

    def format(self, record: logging.LogRecord) -> str:
        return self._filter(super().format(record))

    def redact(self, needle: str, replace: str = "<REDACTED>") -> None:
        """Redact specific strings; e.g., authorization tokens.  This won't
        retroactively redact stuff you've already leaked, so make sure
        you redact things as soon as possible.
        """
        # Don't redact empty strings; this will lead to something
        # that looks like s<REDACTED>t<REDACTED>r<REDACTED>...
        if needle == "":
            return
        self.redactions[needle] = replace


@contextlib.contextmanager
def timer(logger: logging.Logger, prefix: str) -> Iterator[None]:
    """Timed context manager"""
    start_time = time.perf_counter()
    yield
    logger.info("%s took %.3f [s]", prefix, time.perf_counter() - start_time)


F = TypeVar("F", bound=Callable[..., Any])


def timed(prefix: str) -> Callable[[F], F]:
    """Decorator for timing functions"""

    def decorator(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = cast(logging.Logger, LOGGER)
            logger.info(prefix)
            with timer(logger, prefix):
                return f(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


class Venv:
    """Virtual environment manager"""

    AGGRESSIVE_UPDATE_PACKAGES = ("pip", "setuptools", "packaging", "wheel")

    def __init__(
        self,
        prefix: Path | str,
        pip_source: PipSource,
        *,
        base_executable: Path | str | None = None,
    ) -> None:
        self.prefix = Path(prefix).absolute()
        self.pip_source = pip_source
        self.base_executable = Path(base_executable or sys.executable).absolute()
        self._executable: Path | None = None
        self._env = {"PIP_EXTRA_INDEX_URL": self.pip_source.index_url}

    def is_venv(self) -> bool:
        """Check if the prefix is a virtual environment."""
        return self.prefix.is_dir() and (self.prefix / "pyvenv.cfg").is_file()

    @property
    def executable(self) -> Path:
        """Get the Python executable for the virtual environment."""
        assert self.is_venv()
        if self._executable is None:
            if WINDOWS:
                executable = self.prefix / "Scripts" / "python.exe"
            else:
                executable = self.prefix / "bin" / "python"
            assert executable.is_file() or executable.is_symlink()
            assert os.access(executable, os.X_OK), f"{executable} is not executable"
            self._executable = executable
        return self._executable

    def site_packages(self, python: Path | str | None = None) -> Path:
        """Get the site-packages directory for the virtual environment."""
        output = self.python(
            "-c",
            "import site; [print(p) for p in site.getsitepackages()]",
            python=python,
            capture_output=True,
        ).stdout
        candidates = list(map(Path, filter(None, map(str.strip, output.splitlines()))))
        candidates = [p for p in candidates if p.is_dir() and p.name == "site-packages"]
        if not candidates:
            raise RuntimeError(
                f"No site-packages directory found for excecutable {python}"
            )
        return candidates[0]

    @property
    def activate_command(self) -> str:
        """Get the command to activate the virtual environment."""
        if WINDOWS:
            # Assume PowerShell
            return f"& {self.prefix / 'Scripts' / 'Activate.ps1'}"
        return f"source {self.prefix}/bin/activate"

    @timed("Creating virtual environment")
    def create(self, *, remove_if_exists: bool = False) -> Path:
        """Create a virtual environment."""
        if self.prefix.exists():
            if remove_if_exists:
                # If the venv directory already exists, remove it first
                if not self.is_venv():
                    raise RuntimeError(
                        f"The path {self.prefix} already exists and is not a virtual environment. "
                        "Please remove it manually or choose a different prefix."
                    )
                if self.prefix in [
                    Path(p).absolute()
                    for p in [
                        sys.prefix,
                        sys.exec_prefix,
                        sys.base_prefix,
                        sys.base_exec_prefix,
                    ]
                ]:
                    raise RuntimeError(
                        f"The path {self.prefix} trying to remove is the same as the interpreter "
                        "to run this script. Please choose a different prefix or deactivate the "
                        "current virtual environment."
                    )
                if self.prefix in [
                    Path(
                        self.base_python(
                            "-c",
                            f"import os, sys; print(os.path.abspath({p}))",
                            capture_output=True,
                        ).stdout.strip()
                    ).absolute()
                    for p in [
                        "sys.prefix",
                        "sys.exec_prefix",
                        "sys.base_prefix",
                        "sys.base_exec_prefix",
                    ]
                ]:
                    raise RuntimeError(
                        f"The Python executable {self.base_executable} trying to remove is the "
                        "same as the interpreter to create the virtual environment. Please choose "
                        "a different prefix or a different Python interpreter."
                    )
                print(f"Removing existing venv: {self.prefix}")
                _remove_existing(self.prefix)

            else:
                raise RuntimeError(f"Path {self.prefix} already exists.")

        print(f"Creating venv (Python {self.base_python_version()}): {self.prefix}")
        self.base_python("-m", "venv", str(self.prefix))
        assert self.is_venv(), "Failed to create virtual environment."
        (self.prefix / ".gitignore").write_text("*\n", encoding="utf-8")
        return self.ensure()

    def ensure(self) -> Path:
        """Ensure the virtual environment exists."""
        if not self.is_venv():
            return self.create(remove_if_exists=True)

        self.pip_install(*self.AGGRESSIVE_UPDATE_PACKAGES, upgrade=True)
        return self.prefix

    def python(
        self,
        *args: str,
        python: Path | str | None = None,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Run a Python command in the virtual environment."""
        if python is None:
            python = self.executable
        cmd = [str(python), *args]
        env = popen_kwargs.pop("env", None) or {}
        return subprocess.run(
            cmd,
            check=True,
            text=True,
            encoding="utf-8",
            env={**self._env, **env},
            **popen_kwargs,
        )

    def base_python(
        self,
        *args: str,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Run a Python command in the base environment."""
        return self.python(*args, python=self.base_executable, **popen_kwargs)

    def python_version(self, *, python: Path | str | None = None) -> str:
        """Get the Python version for the virtual environment."""
        return self.python(
            "-c",
            (
                "import sys; print('{0.major}.{0.minor}.{0.micro}{1}'."
                "format(sys.version_info, getattr(sys, 'abiflags', '')))"
            ),
            python=python,
            capture_output=True,
        ).stdout.strip()

    def base_python_version(self) -> str:
        """Get the Python version for the base environment."""
        return self.python_version(python=self.base_executable)

    def pip(self, *args: str, **popen_kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Run a pip command in the virtual environment."""
        return self.python("-m", "pip", *args, **popen_kwargs)

    @timed("Installing packages")
    def pip_install(
        self,
        *packages: str,
        prerelease: bool = False,
        upgrade: bool = False,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Run a pip install command in the virtual environment."""
        if upgrade:
            args = ["--upgrade", *packages]
            verb = "Upgrading"
        else:
            args = list(packages)
            verb = "Installing"
        if prerelease:
            args = ["--pre", *args]
        print(
            f"{verb} package(s) ({self.pip_source.index_url}): "
            f"{', '.join(map(os.path.basename, packages))}"
        )
        return self.pip("install", *args, **popen_kwargs)

    @timed("Downloading packages")
    def pip_download(
        self,
        *packages: str,
        prerelease: bool = False,
        **popen_kwargs: Any,
    ) -> list[Path]:
        """Download a package in the virtual environment."""
        tmpdir = tempfile.TemporaryDirectory(prefix="pip-download-")
        atexit.register(tmpdir.cleanup)
        tempdir = Path(tmpdir.name).absolute()
        print(
            f"Downloading package(s) ({self.pip_source.index_url}): "
            f"{', '.join(packages)}"
        )
        if prerelease:
            args = ["--pre", *packages]
        else:
            args = list(packages)
        self.pip("download", "--dest", str(tempdir), *args, **popen_kwargs)
        files = list(tempdir.iterdir())
        print(f"Downloaded {len(files)} file(s) to {tempdir}:")
        for file in files:
            print(f"  - {file.name}")
        return files

    def wheel(
        self,
        *args: str,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Run a wheel command in the virtual environment."""
        return self.python("-m", "wheel", *args, **popen_kwargs)

    @timed("Unpacking wheel file")
    def wheel_unpack(
        self,
        wheel: Path | str,
        dest: Path | str,
        **popen_kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Unpack a wheel into a directory."""
        wheel = Path(wheel).absolute()
        dest = Path(dest).absolute()
        assert wheel.is_file() and wheel.suffix.lower() == ".whl"
        return self.wheel("unpack", "--dest", str(dest), str(wheel), **popen_kwargs)

    @contextlib.contextmanager
    def extracted_wheel(self, wheel: Path | str) -> Generator[Path]:
        """Download and extract a wheel into a temporary directory."""
        with tempfile.TemporaryDirectory(prefix="wheel-") as tempdir:
            self.wheel_unpack(wheel, tempdir)
            subdirs = [p for p in Path(tempdir).absolute().iterdir() if p.is_dir()]
            if len(subdirs) != 1:
                raise RuntimeError(
                    f"Expected exactly one directory in {tempdir}, "
                    f"got {[str(d) for d in subdirs]}."
                )
            yield subdirs[0]


def git(*args: str) -> list[str]:
    return ["git", "-C", str(REPO_ROOT), *args]


@functools.lru_cache
def logging_base_dir() -> Path:
    base_dir = REPO_ROOT / "nightly" / "log"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


@functools.lru_cache
def logging_run_dir() -> Path:
    base_dir = logging_base_dir()
    cur_dir = base_dir / f"{datetime.now().strftime(DATETIME_FORMAT)}_{uuid.uuid1()}"
    cur_dir.mkdir(parents=True, exist_ok=True)
    return cur_dir


@functools.lru_cache
def logging_record_argv() -> None:
    s = subprocess.list2cmdline(sys.argv)
    (logging_run_dir() / "argv").write_text(s, encoding="utf-8")


def logging_record_exception(e: BaseException) -> None:
    text = f"{type(e).__name__}: {e}"
    if isinstance(e, subprocess.CalledProcessError):
        text += f"\n\nstdout: {e.stdout}\n\nstderr: {e.stderr}"
    (logging_run_dir() / "exception").write_text(text, encoding="utf-8")


def logging_rotate() -> None:
    log_base = logging_base_dir()
    old_logs = sorted(log_base.iterdir(), reverse=True)
    for stale_log in old_logs[1000:]:
        # Sanity check that it looks like a log
        if LOG_DIRNAME_RE.fullmatch(stale_log.name) is not None:
            shutil.rmtree(stale_log)


@contextlib.contextmanager
def logging_manager(*, debug: bool = False) -> Generator[logging.Logger, None, None]:
    """Setup logging. If a failure starts here we won't
    be able to save the user in a reasonable way.

    Logging structure: there is one logger (the root logger)
    and in processes all events.  There are two handlers:
    stderr (INFO) and file handler (DEBUG).
    """
    formatter = Formatter(fmt="%(levelname)s: %(message)s", datefmt="")
    root_logger = logging.getLogger("pytorch-nightly")
    root_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    log_file = logging_run_dir() / "nightly.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logging_record_argv()

    try:
        logging_rotate()
        print(f"log file: {log_file}")
        yield root_logger
    except Exception as e:
        logging.exception("Fatal exception")
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)
    except BaseException as e:
        # You could logging.debug here to suppress the backtrace
        # entirely, but there is no reason to hide it from technically
        # savvy users.
        logging.info("", exc_info=True)
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)


def check_branch(subcommand: str, branch: str | None) -> str | None:
    """Checks that the branch name can be checked out."""
    if subcommand != "checkout":
        return None
    # first make sure actual branch name was given
    if branch is None:
        return "Branch name to checkout must be supplied with '-b' option"
    # next check that the local repo is clean
    cmd = git("status", "--untracked-files=no", "--porcelain")
    stdout = subprocess.check_output(cmd, text=True, encoding="utf-8")
    if stdout.strip():
        return "Need to have clean working tree to checkout!\n\n" + stdout
    # next check that the branch name doesn't already exist
    cmd = git("show-ref", "--verify", "--quiet", f"refs/heads/{branch}")
    p = subprocess.run(cmd, capture_output=True, check=False)  # type: ignore[assignment]
    if not p.returncode:
        return f"Branch {branch!r} already exists"
    return None


@timed("Installing dependencies")
def install_packages(venv: Venv, packages: Iterable[str]) -> None:
    """Install dependencies to deps environment"""
    # install packages
    packages = list(dict.fromkeys(packages))
    if packages:
        venv.pip_install(*packages)


def _ensure_commit(git_sha1: str) -> None:
    """Make sure that we actually have the commit locally"""
    cmd = git("cat-file", "-e", git_sha1 + r"^{commit}")
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode == 0:
        # we have the commit locally
        return
    # we don't have the commit, must fetch
    cmd = git("fetch", GITHUB_REMOTE_URL, git_sha1)
    subprocess.check_call(cmd)


def _nightly_version(site_dir: Path) -> str:
    # first get the git version from the installed module
    version_file = site_dir / "torch" / "version.py"
    with version_file.open(encoding="utf-8") as f:
        for line in f:
            if not line.startswith("git_version"):
                continue
            git_version = literal_eval(line.partition("=")[2].strip())
            break
        else:
            raise RuntimeError(f"Could not find git_version in {version_file}")

    print(f"Found released git version {git_version}")
    # now cross reference with nightly version
    _ensure_commit(git_version)
    cmd = git("show", "--no-patch", "--format=%s", git_version)
    stdout = subprocess.check_output(cmd, text=True, encoding="utf-8")
    m = SHA1_RE.search(stdout)
    if m is None:
        raise RuntimeError(
            f"Could not find nightly release in git history:\n  {stdout}"
        )
    nightly_version = m.group("sha1")
    print(f"Found nightly release version {nightly_version}")
    # now checkout nightly version
    _ensure_commit(nightly_version)
    return nightly_version


@timed("Checking out nightly PyTorch")
def checkout_nightly_version(branch: str, site_dir: Path) -> None:
    """Get's the nightly version and then checks it out."""
    nightly_version = _nightly_version(site_dir)
    cmd = git("checkout", "-b", branch, nightly_version)
    subprocess.check_call(cmd)


@timed("Pulling nightly PyTorch")
def pull_nightly_version(site_dir: Path) -> None:
    """Fetches the nightly version and then merges it ."""
    nightly_version = _nightly_version(site_dir)
    cmd = git("merge", nightly_version)
    subprocess.check_call(cmd)


def _get_listing_linux(source_dir: Path) -> list[Path]:
    return list(
        itertools.chain(
            source_dir.glob("*.so"),
            (source_dir / "lib").glob("*.so"),
            (source_dir / "lib").glob("*.so.*"),
        )
    )


def _get_listing_macos(source_dir: Path) -> list[Path]:
    # oddly, these are .so files even on Mac
    return list(
        itertools.chain(
            source_dir.glob("*.so"),
            (source_dir / "lib").glob("*.dylib"),
        )
    )


def _get_listing_windows(source_dir: Path) -> list[Path]:
    return list(
        itertools.chain(
            source_dir.glob("*.pyd"),
            (source_dir / "lib").glob("*.lib"),
            (source_dir / "lib").glob(".dll"),
        )
    )


def _glob_pyis(d: Path) -> set[str]:
    return {p.relative_to(d).as_posix() for p in d.rglob("*.pyi")}


def _find_missing_pyi(source_dir: Path, target_dir: Path) -> list[Path]:
    source_pyis = _glob_pyis(source_dir)
    target_pyis = _glob_pyis(target_dir)
    missing_pyis = sorted(source_dir / p for p in (source_pyis - target_pyis))
    return missing_pyis


def _get_listing(source_dir: Path, target_dir: Path) -> list[Path]:
    if LINUX:
        listing = _get_listing_linux(source_dir)
    elif MACOS:
        listing = _get_listing_macos(source_dir)
    elif WINDOWS:
        listing = _get_listing_windows(source_dir)
    else:
        raise RuntimeError(f"Platform {platform_system()!r} not recognized")
    listing.extend(_find_missing_pyi(source_dir, target_dir))
    listing.append(source_dir / "version.py")
    listing.append(source_dir / "testing" / "_internal" / "generated")
    listing.append(source_dir / "bin")
    listing.append(source_dir / "include")
    return listing


def _remove_existing(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _move_single(
    src: Path,
    source_dir: Path,
    target_dir: Path,
    mover: Callable[[Path, Path], None],
    verb: str,
) -> None:
    relpath = src.relative_to(source_dir)
    trg = target_dir / relpath
    _remove_existing(trg)
    # move over new files
    if src.is_dir():
        trg.mkdir(parents=True, exist_ok=True)
        for root, dirs, files in os.walk(src):
            relroot = Path(root).relative_to(src)
            for name in files:
                relname = relroot / name
                s = src / relname
                t = trg / relname
                print(f"{verb} {s} -> {t}")
                mover(s, t)
            for name in dirs:
                (trg / relroot / name).mkdir(parents=True, exist_ok=True)
    else:
        print(f"{verb} {src} -> {trg}")
        mover(src, trg)


def _copy_files(listing: list[Path], source_dir: Path, target_dir: Path) -> None:
    for src in listing:
        _move_single(src, source_dir, target_dir, shutil.copy2, "Copying")


def _link_files(listing: list[Path], source_dir: Path, target_dir: Path) -> None:
    for src in listing:
        _move_single(src, source_dir, target_dir, os.link, "Linking")


@timed("Moving nightly files into repo")
def move_nightly_files(site_dir: Path) -> None:
    """Moves PyTorch files from temporary installed location to repo."""
    # get file listing
    source_dir = site_dir / "torch"
    target_dir = REPO_ROOT / "torch"
    listing = _get_listing(source_dir, target_dir)
    # copy / link files
    if WINDOWS:
        _copy_files(listing, source_dir, target_dir)
    else:
        try:
            _link_files(listing, source_dir, target_dir)
        except Exception:
            _copy_files(listing, source_dir, target_dir)


@timed("Writing pytorch-nightly.pth")
def write_pth(venv: Venv) -> None:
    """Writes Python path file for this dir."""
    (venv.site_packages() / "pytorch-nightly.pth").write_text(
        "# This file was autogenerated by PyTorch's tools/nightly.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{REPO_ROOT}\n",
        encoding="utf-8",
    )


def install(
    *,
    venv: Venv,
    packages: Iterable[str],
    subcommand: str = "checkout",
    branch: str | None = None,
    logger: logging.Logger,
) -> None:
    """Development install of PyTorch"""
    use_existing = subcommand == "checkout"
    if use_existing:
        venv.ensure()
    else:
        venv.create(remove_if_exists=True)

    packages = [p for p in packages if p != "torch"]

    dependencies = venv.pip_download("torch", prerelease=True)
    torch_wheel = [
        dep
        for dep in dependencies
        if dep.name.startswith("torch-") and dep.name.endswith(".whl")
    ]
    if len(torch_wheel) != 1:
        raise RuntimeError(f"Expected exactly one torch wheel, got {torch_wheel}")
    torch_wheel = torch_wheel[0]
    dependencies = [deps for deps in dependencies if deps != torch_wheel]

    install_packages(venv, [*packages, *map(str, dependencies)])

    with venv.extracted_wheel(torch_wheel) as wheel_site_dir:
        if subcommand == "checkout":
            checkout_nightly_version(cast(str, branch), wheel_site_dir)
        elif subcommand == "pull":
            pull_nightly_version(wheel_site_dir)
        else:
            raise ValueError(f"Subcommand {subcommand} must be one of: checkout, pull.")
        move_nightly_files(wheel_site_dir)

    write_pth(venv)
    logger.info(
        "-------\n"
        "PyTorch Development Environment set up!\n"
        "Please activate to enable this environment:\n\n"
        "  $ %s",
        venv.activate_command,
    )


def make_parser() -> argparse.ArgumentParser:
    def find_executable(name: str) -> Path:
        executable = shutil.which(name)
        if executable is None:
            raise argparse.ArgumentTypeError(
                f"Could not find executable {name} in PATH."
            )
        return Path(executable).absolute()

    parser = argparse.ArgumentParser()
    # subcommands
    subcmd = parser.add_subparsers(dest="subcmd", help="subcommand to execute")
    checkout = subcmd.add_parser("checkout", help="checkout a new branch")
    checkout.add_argument(
        "-b",
        "--branch",
        help="Branch name to checkout",
        dest="branch",
        default=None,
        metavar="NAME",
    )
    pull = subcmd.add_parser(
        "pull", help="pulls the nightly commits into the current branch"
    )
    # general arguments
    subparsers = [checkout, pull]
    for subparser in subparsers:
        subparser.add_argument(
            "--python",
            "--base-executable",
            type=find_executable,
            help=(
                "Path to Python interpreter to use for creating the virtual environment. "
                "Defaults to the interpreter running this script."
            ),
            dest="base_executable",
            default=None,
            metavar="PYTHON",
        )
        subparser.add_argument(
            "-p",
            "--prefix",
            type=lambda p: Path(p).absolute(),
            help='Path to virtual environment directory (e.g. "./venv")',
            dest="prefix",
            default=str(DEFAULT_VENV_DIR),
            metavar="PATH",
        )
        subparser.add_argument(
            "-v",
            "--verbose",
            help="Provide debugging info",
            dest="verbose",
            default=False,
            action="store_true",
        )
        subparser.add_argument(
            "--cuda",
            help=(
                "CUDA version to install "
                "(defaults to the latest version available on the platform)"
            ),
            dest="cuda",
            nargs="?",
            default=argparse.SUPPRESS,
            metavar="VERSION",
        )
    return parser


def parse_arguments() -> argparse.Namespace:
    parser = make_parser()
    args = parser.parse_args()
    args.branch = getattr(args, "branch", None)
    return args


def main() -> None:
    """Main entry point"""
    global LOGGER
    args = parse_arguments()
    status = check_branch(args.subcmd, args.branch)
    if status:
        sys.exit(status)

    pip_source = None
    if hasattr(args, "cuda"):
        available_sources = {
            src.name[len("cuda-") :]: src
            for src in PIP_SOURCES.values()
            if src.name.startswith("cuda-") and PLATFORM in src.supported_platforms
        }
        if not available_sources:
            print(f"No CUDA versions available on platform {PLATFORM}.")
            sys.exit(1)
        if args.cuda is not None:
            pip_source = available_sources.get(args.cuda)
            if pip_source is None:
                print(
                    f"CUDA {args.cuda} is not available on platform {PLATFORM}. "
                    f"Available version(s): {', '.join(sorted(available_sources, key=Version))}"
                )
                sys.exit(1)
        else:
            pip_source = available_sources[max(available_sources, key=Version)]
    else:
        pip_source = PIP_SOURCES["cpu"]  # always available

    with logging_manager(debug=args.verbose) as logger:
        LOGGER = logger
        venv = Venv(
            prefix=args.prefix,
            pip_source=pip_source,
            base_executable=args.base_executable,
        )
        install(
            venv=venv,
            packages=PACKAGES_TO_INSTALL,
            subcommand=args.subcmd,
            branch=args.branch,
            logger=logger,
        )


if __name__ == "__main__":
    main()
