#!/usr/bin/env python3
# Much of the logging code here was forked from https://github.com/ezyang/ghstack
# Copyright (c) Edward Z. Yang <ezyang@mit.edu>
r"""Checks out the nightly development version of PyTorch and installs pre-built
binaries into the repo.

You can use this script to check out a new nightly branch with the following::

    $ ./tools/nightly.py checkout -b my-nightly-branch
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

Or if you would like to check out the nightly commit in detached HEAD mode::

    $ ./tools/nightly.py checkout
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

Or if you would like to re-use an existing virtual environment, you can pass in
the prefix argument (--prefix)::

    $ ./tools/nightly.py checkout -b my-nightly-branch -p my-env
    $ source my-env/bin/activate  # or `& .\my-env\Scripts\Activate.ps1` on Windows

To install the nightly binaries built with CUDA, you can pass in the flag --cuda::

    $ ./tools/nightly.py checkout -b my-nightly-branch --cuda
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

To install the nightly binaries built with ROCm, you can pass in the flag --rocm::

    $ ./tools/nightly.py checkout -b my-nightly-branch --rocm
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

You can also use this tool to pull the nightly commits into the current branch as
well. This can be done with::

    $ ./tools/nightly.py pull
    $ source venv/bin/activate  # or `& .\venv\Scripts\Activate.ps1` on Windows

Pulling will recreate a fresh virtual environment and reinstall the development
dependencies as well as the nightly binaries into the repo directory.

To install nightly binaries into your current environment (instead of creating a new venv),
use the --inplace flag::

    $ ./tools/nightly.py checkout --inplace
    $ ./tools/nightly.py pull --inplace

The --inplace flag will place the nightly binaries directly into the source directory
and set up torch to be importable from the current environment.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import functools
import hashlib
import itertools
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
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


@functools.lru_cache
def _find_repo_root() -> Path:
    """Find the root of the git repository where this script is located."""
    script_dir = Path.cwd().absolute()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=script_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).absolute()


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


@functools.lru_cache
def default_venv_dir() -> Path:
    """Get the default virtual environment directory."""
    return _find_repo_root() / "venv"


@functools.lru_cache
def wheel_cache_dir() -> Path:
    """Get the wheel cache directory."""
    if WINDOWS:
        cache_home = Path(
            os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        )
    else:
        cache_home = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    cache_dir = cache_home / "pytorch-nightly" / "wheel-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_wheel_cache_key(wheel_path: Path) -> str:
    """Get a cache key for the wheel file."""
    # Wheel filenames already uniquely identify the content
    # e.g., torch-2.2.0.dev20231201+cpu-cp311-cp311-linux_x86_64.whl
    return wheel_path.stem  # Remove .whl extension


def cleanup_wheel_cache() -> None:
    """Clean up old wheel cache entries, keeping only the 10 most recent."""
    cache_dir = wheel_cache_dir()
    if not cache_dir.exists():
        return

    # Get all cache entries sorted by modification time (newest first)
    cache_entries = []
    for entry in cache_dir.iterdir():
        if entry.is_dir():
            try:
                mtime = entry.stat().st_mtime
                cache_entries.append((mtime, entry))
            except OSError:
                # Entry might have been deleted, skip it
                continue

    cache_entries.sort(reverse=True)  # Sort by mtime, newest first

    # Keep only the 10 most recent entries
    for _, old_entry in cache_entries[10:]:
        try:
            shutil.rmtree(old_entry, ignore_errors=True)
            print(f"Cleaned up old wheel cache entry: {old_entry.name}")
        except OSError:
            # Ignore errors during cleanup
            logging.exception("Couldn't clean old cache entry %s", old_entry.name)  # noqa: LOG015


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
    "rocm-6.4": PipSource(
        name="rocm-6.4",
        index_url=f"{PYTORCH_NIGHTLY_PIP_INDEX_URL}/rocm6.4",
        supported_platforms={"Linux"},
        accelerator="rocm",
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
                f"No site-packages directory found for executable {python}"
            )
        return candidates[0]

    @property
    def activate_script(self) -> Path:
        """Get the activation script for the virtual environment."""
        if WINDOWS:
            # Assume PowerShell
            return self.prefix / "Scripts" / "Activate.ps1"
        # Assume POSIX-compliant shell: Bash, Zsh, etc.
        return self.prefix / "bin" / "activate"

    @property
    def activate_command(self) -> str:
        """Get the command to activate the virtual environment."""
        if WINDOWS:
            # Assume PowerShell
            return f'& "{self.activate_script}"'
        # Assume Bash, Zsh, etc.
        # POSIX standard should use dot `. venv/bin/activate` rather than `source`
        return f"source {shlex.quote(str(self.activate_script))}"

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

        if LINUX:
            activate_script = self.activate_script
            st_mode = activate_script.stat().st_mode
            # The activate script may be read-only and we need to add write permissions
            activate_script.chmod(st_mode | 0o200)
            with activate_script.open(mode="a", encoding="utf-8") as f:
                f.write(
                    "\n"
                    + textwrap.dedent(
                        f"""
                        # Add NVIDIA PyPI packages to LD_LIBRARY_PATH
                        export LD_LIBRARY_PATH="$(
                            {self.executable.name} - <<EOS
                        import glob
                        import itertools
                        import os
                        import site

                        nvidia_libs = [
                            p.rstrip("/")
                            for p in itertools.chain.from_iterable(
                                glob.iglob(f"{{site_dir}}/{{pattern}}/", recursive=True)
                                for site_dir in site.getsitepackages()
                                for pattern in ("nvidia/**/lib", "cu*/**/lib")
                            )
                        ]
                        ld_library_path = os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep)
                        print(os.pathsep.join(dict.fromkeys(nvidia_libs + ld_library_path)))
                        EOS
                        )"
                        """
                    ).strip()
                    + "\n"
                )
            # Change the file mode back
            activate_script.chmod(st_mode)

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
        check = popen_kwargs.pop("check", True)
        return subprocess.run(
            cmd,
            check=check,
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
        no_deps: bool = False,
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
        args = []
        if prerelease:
            args.append("--pre")
        if no_deps:
            args.append("--no-deps")
        args.extend(packages)
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
        """Download and extract a wheel, using disk cache if available."""
        wheel_path = Path(wheel).absolute()
        cache_dir = wheel_cache_dir()
        cache_key = get_wheel_cache_key(wheel_path)
        cached_wheel_dir = cache_dir / cache_key

        # Check if we have a cached extraction
        if cached_wheel_dir.exists():
            subdirs = [p for p in cached_wheel_dir.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                print(f"Using cached wheel extraction: {cached_wheel_dir}")
                yield subdirs[0]
                return
            else:
                # Cache is corrupted, remove it
                shutil.rmtree(cached_wheel_dir, ignore_errors=True)

        # Extract to cache directory
        print(f"Extracting wheel to cache: {cached_wheel_dir}")
        cached_wheel_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.wheel_unpack(wheel_path, cached_wheel_dir)
            subdirs = [p for p in cached_wheel_dir.iterdir() if p.is_dir()]
            if len(subdirs) != 1:
                raise RuntimeError(
                    f"Expected exactly one directory in {cached_wheel_dir}, "
                    f"got {[str(d) for d in subdirs]}."
                )
            # Clean up old cache entries after successful extraction
            cleanup_wheel_cache()
            yield subdirs[0]
        except Exception:
            # Clean up on failure
            shutil.rmtree(cached_wheel_dir, ignore_errors=True)
            raise


def git(*args: str) -> list[str]:
    return ["git", "-C", str(_find_repo_root()), *args]


@functools.lru_cache
def logging_base_dir() -> Path:
    base_dir = _find_repo_root() / "nightly" / "log"
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
        logging.exception("Fatal exception")  # noqa: LOG015
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)
    except BaseException as e:
        # You could logging.debug here to suppress the backtrace
        # entirely, but there is no reason to hide it from technically
        # savvy users.
        logging.info("", exc_info=True)  # noqa: LOG015
        logging_record_exception(e)
        print(f"log file: {log_file}")
        sys.exit(1)


def get_torch_install_info(venv: Venv) -> tuple[str, Path | None]:
    """Get information about the current torch installation using importlib.

    Returns:
        Tuple of (install_type, install_path) where:
        - install_type: 'none', 'regular', 'editable_ours', 'editable_other'
        - install_path: Path to torch installation, or None if not installed
    """
    try:
        # Run this check in the venv's Python environment
        # Change cwd to avoid picking up local torch module
        with tempfile.TemporaryDirectory() as temp_cwd:
            result = venv.python(
                "-c",
                """
import json
from importlib import metadata, util
from pathlib import Path

try:
    spec = util.find_spec("torch")
    if spec is None or spec.origin is None:
        print(json.dumps({"type": "none", "path": None}))
        exit()

    torch_path = Path(spec.origin).parent
    dist = metadata.distribution("torch")

    # Check for PEP-610 direct_url.json for editable installs
    try:
        direct_url_txt = dist.read_text("direct_url.json")
        if direct_url_txt:
            info = json.loads(direct_url_txt)
            if info.get("dir_info", {}).get("editable", False):
                repo_root = Path(str(_find_repo_root())).absolute()
                if torch_path.is_relative_to(repo_root):
                    install_type = "editable_ours"
                else:
                    install_type = "editable_other"
                print(json.dumps({"type": install_type, "path": str(torch_path)}))
                exit()
    except Exception:
        # direct_url.json may not exist or be readable, continue to regular check
        pass

    # If we get here, it's a regular wheel or sdist install
    print(json.dumps({"type": "regular", "path": str(torch_path)}))

except ModuleNotFoundError:
    print(json.dumps({"type": "none", "path": None}))
except Exception as e:
    print(json.dumps({"type": "error", "path": None, "error": str(e)}))
                """,
                capture_output=True,
                check=False,
                cwd=temp_cwd,
            )

        if result.returncode != 0:
            return "none", None

        try:
            info = json.loads(result.stdout.strip())
            install_type = info["type"]
            path_str = info["path"]

            if install_type == "none" or path_str is None:
                return "none", None
            elif install_type == "error":
                return "none", None
            else:
                return install_type, Path(path_str)

        except (json.JSONDecodeError, KeyError):
            return "none", None

    except Exception:
        return "none", None


def check_branch(subcommand: str, branch: str | None) -> str | None:
    """Checks that the branch name can be checked out."""
    if subcommand != "checkout":
        return None
    # next check that the local repo is clean
    cmd = git("status", "--untracked-files=no", "--porcelain")
    stdout = subprocess.check_output(cmd, text=True, encoding="utf-8")
    if stdout.strip():
        return "Need to have clean working tree to checkout!\n\n" + stdout
    # next check that the branch name doesn't already exist (if a branch name is provided)
    if branch is not None:
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
def checkout_nightly_version(branch: str | None, site_dir: Path) -> str:
    """Get's the nightly version and then checks it out.

    Returns the nightly version SHA that was checked out.
    """
    nightly_version = _nightly_version(site_dir)
    if branch is None:
        # Detached mode - explicitly use --detach flag
        cmd = git("checkout", "--detach", nightly_version)
    else:
        # Branch mode
        cmd = git("checkout", "-b", branch, nightly_version)
    subprocess.check_call(cmd)
    return nightly_version


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
                # Writing this much to stdout is expensive
                # print(f"{verb} {s} -> {t}")
                mover(s, t)
            for name in dirs:
                (trg / relroot / name).mkdir(parents=True, exist_ok=True)
    else:
        # print(f"{verb} {src} -> {trg}")
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
    target_dir = _find_repo_root() / "torch"
    listing = _get_listing(source_dir, target_dir)
    # copy / link files
    if WINDOWS:
        _copy_files(listing, source_dir, target_dir)
    else:
        try:
            _link_files(listing, source_dir, target_dir)
        except Exception:
            _copy_files(listing, source_dir, target_dir)


@timed("Installing torch distribution metadata")
def install_torch_metadata(wheel_site_dir: Path, venv: Venv) -> None:
    """Install torch distribution metadata from the wheel."""
    # Find the torch dist-info directory in the extracted wheel
    dist_info_dirs = list(wheel_site_dir.glob("torch-*.dist-info"))
    if len(dist_info_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one torch dist-info directory, got {dist_info_dirs}"
        )

    source_dist_info = dist_info_dirs[0]

    try:
        target_site_packages = venv.site_packages()
    except Exception:
        # Fallback to using current Python's site-packages for inplace mode
        import site

        site_packages_dirs = site.getsitepackages()
        if not site_packages_dirs:
            site_packages_dirs = [site.getusersitepackages()]
        target_site_packages = Path(site_packages_dirs[0])

    target_dist_info = target_site_packages / source_dist_info.name

    # Remove existing dist-info if present
    _remove_existing(target_dist_info)

    # Copy the entire dist-info directory
    shutil.copytree(source_dist_info, target_dist_info)

    # Create direct_url.json to indicate this is an editable install
    direct_url_json = {
        "dir_info": {"editable": True},
        "url": f"file://{_find_repo_root()}",
    }

    direct_url_file = target_dist_info / "direct_url.json"
    direct_url_file.write_text(json.dumps(direct_url_json, indent=2), encoding="utf-8")

    # Update RECORD file to include torch.pth and direct_url.json

    record_file = target_dist_info / "RECORD"
    if record_file.exists():
        # Read existing RECORD entries and filter out torch/ and torchgen/ entries
        existing_records = record_file.read_text(encoding="utf-8").strip().split("\n")
        # Remove entries for torch/ and torchgen/ since we don't install them to site-packages
        filtered_records = [
            record
            for record in existing_records
            if not record.startswith(("torch/", "torchgen/", "functorch/"))
        ]
    else:
        filtered_records = []

    # Add our new files to the RECORD
    torch_pth_path = target_site_packages / "torch.pth"

    # Calculate hash and size for torch.pth
    if torch_pth_path.exists():
        torch_pth_content = torch_pth_path.read_bytes()
        torch_pth_hash = hashlib.sha256(torch_pth_content).hexdigest()
        torch_pth_size = len(torch_pth_content)
        torch_pth_record = f"torch.pth,sha256={torch_pth_hash},{torch_pth_size}"
    else:
        torch_pth_record = "torch.pth,,"

    # Calculate hash and size for direct_url.json
    direct_url_content = direct_url_file.read_bytes()
    direct_url_hash = hashlib.sha256(direct_url_content).hexdigest()
    direct_url_size = len(direct_url_content)
    direct_url_relative = direct_url_file.relative_to(target_site_packages)
    direct_url_record = (
        f"{direct_url_relative},sha256={direct_url_hash},{direct_url_size}"
    )

    # Create new RECORD entries
    new_records = [
        torch_pth_record,
        direct_url_record,
    ]

    # Combine and write updated RECORD
    all_records = filtered_records + new_records
    record_content = "\n".join(all_records) + "\n"
    record_file.write_text(record_content, encoding="utf-8")

    print(f"Installed torch metadata: {target_dist_info}")
    print(f"Created editable install marker: {direct_url_file}")
    print("Updated RECORD file with torch.pth and direct_url.json")


@timed("Writing torch.pth")
def write_pth(venv: Venv) -> None:
    """Writes Python path file for this dir."""
    (venv.site_packages() / "torch.pth").write_text(
        "# This file was autogenerated by PyTorch's tools/nightly.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{_find_repo_root()}\n",
        encoding="utf-8",
    )


def uninstall_torch(venv: Venv, logger: logging.Logger) -> None:
    """Uninstall existing torch installation."""
    logger.info("Uninstalling existing torch installation...")
    try:
        venv.pip("uninstall", "torch", "-y")

    except subprocess.CalledProcessError:
        logger.warning("Failed to uninstall torch", exc_info=True)


@timed("Setting up editable torch install")
def setup_editable_torch(venv: Venv, logger: logging.Logger) -> None:
    """Set up torch to be importable from the repo directory."""
    install_type, install_path = get_torch_install_info(venv)

    if install_type == "editable_ours":
        logger.info(
            "Torch is already set up as editable install pointing to this repository"
        )
        return
    elif install_type == "editable_other":
        logger.info(
            "Found editable torch install pointing to different directory: %s",
            install_path,
        )
        logger.info("Overriding with editable install pointing to this repository")
        uninstall_torch(venv, logger)
    elif install_type == "regular":
        logger.info("Found regular torch install at: %s", install_path)
        logger.info(
            "Uninstalling and replacing with editable install pointing to this repository"
        )
        uninstall_torch(venv, logger)
    else:
        logger.info("No existing torch installation found")

    # Set up .pth file to make our repo's torch importable
    site_packages = venv.site_packages()
    # NOTE: Don't change this to pytorch-nightly, torch is more inline
    # with what pip install -e . would have created
    pth_file = site_packages / "torch.pth"

    # Ensure the site-packages directory exists
    site_packages.mkdir(parents=True, exist_ok=True)

    pth_file.write_text(
        "# This file was autogenerated by PyTorch's tools/nightly.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{_find_repo_root()}\n",
        encoding="utf-8",
    )

    logger.info("Created .pth file at: %s", pth_file)


def install(
    *,
    venv: Venv | None,
    packages: Iterable[str],
    subcommand: str = "checkout",
    branch: str | None = None,
    inplace: bool = False,
    logger: logging.Logger,
) -> None:
    """Development install of PyTorch"""
    if inplace:
        # Inplace mode: use current environment
        if venv is None:
            raise RuntimeError("venv cannot be None in inplace mode")

        logger.info("Using current environment for in-place installation")
        venv.ensure()

        # Set up editable torch install (handles all edge cases)
        setup_editable_torch(venv, logger)
    else:
        # Original venv mode
        if venv is None:
            raise RuntimeError("venv cannot be None in non-inplace mode")

        use_existing = subcommand == "checkout"
        if use_existing:
            venv.ensure()
        else:
            venv.create(remove_if_exists=True)

        packages_list = [p for p in packages if p != "torch"]
        install_packages(venv, packages_list)

    # Common logic for both modes: download wheel and extract binaries
    if venv is None:
        raise RuntimeError("venv cannot be None")

    downloaded_files = venv.pip_download("torch", prerelease=True, no_deps=True)
    torch_wheel = [
        file
        for file in downloaded_files
        if file.name.startswith("torch-") and file.name.endswith(".whl")
    ]
    if len(torch_wheel) != 1:
        raise RuntimeError(f"Expected exactly one torch wheel, got {torch_wheel}")
    torch_wheel = torch_wheel[0]

    nightly_version = None
    with venv.extracted_wheel(torch_wheel) as wheel_site_dir:
        if subcommand == "checkout":
            nightly_version = checkout_nightly_version(branch, wheel_site_dir)
        elif subcommand == "pull":
            pull_nightly_version(wheel_site_dir)
        else:
            raise ValueError(f"Subcommand {subcommand} must be one of: checkout, pull.")
        move_nightly_files(wheel_site_dir)
        # Install torch distribution metadata so importlib can find it
        install_torch_metadata(wheel_site_dir, venv)

    if not inplace:
        write_pth(venv)
        logger.info(
            "-------\n"
            "PyTorch Development Environment set up!\n"
            "Please activate to enable this environment:\n\n"
            "  $ %s",
            venv.activate_command,
        )
    else:
        message = (
            "-------\n"
            "PyTorch nightly binaries installed in-place!\n"
            "The current environment now has nightly PyTorch binaries."
        )

        # Add cherry-pick instructions for checkout --inplace with no -b argument
        if subcommand == "checkout" and branch is None and nightly_version:
            message += (
                "\n\n"
                "To cherry-pick your old commits onto this nightly commit, run:\n\n"
                "  git cherry-pick origin/main..HEAD@{1}"
            )

        logger.info(message)


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
        help="Branch name to checkout (if omitted, checks out in detached HEAD mode)",
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
            default=str(default_venv_dir()),
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
        subparser.add_argument(
            "--rocm",
            help=(
                "ROCm version to install "
                "(defaults to the latest version available on the platform)"
            ),
            dest="rocm",
            nargs="?",
            default=argparse.SUPPRESS,
            metavar="VERSION",
        )
        subparser.add_argument(
            "--inplace",
            help=(
                "Install nightly binaries into the current environment instead of creating a new venv."
            ),
            dest="inplace",
            default=False,
            action="store_true",
        )
    return parser


def parse_arguments() -> argparse.Namespace:
    parser = make_parser()
    args = parser.parse_args()
    args.branch = getattr(args, "branch", None)
    args.inplace = getattr(args, "inplace", False)
    if hasattr(args, "cuda") and hasattr(args, "rocm"):
        parser.error("Cannot specify both CUDA and ROCm versions.")
    return args


def main() -> None:
    """Main entry point"""
    global LOGGER
    args = parse_arguments()
    status = check_branch(args.subcmd, args.branch)
    if status:
        sys.exit(status)

    pip_source = None

    for toolkit in ("CUDA", "ROCm"):
        accel = toolkit.lower()
        if hasattr(args, accel):
            requested = getattr(args, accel)
            available_sources = {
                src.name[len(f"{accel}-") :]: src
                for src in PIP_SOURCES.values()
                if src.name.startswith(f"{accel}-")
                and PLATFORM in src.supported_platforms
            }
            if not available_sources:
                print(f"No {toolkit} versions available on platform {PLATFORM}.")
                sys.exit(1)
            if requested is not None:
                pip_source = available_sources.get(requested)
                if pip_source is None:
                    print(
                        f"{toolkit} {requested} is not available on platform {PLATFORM}. "
                        f"Available version(s): {', '.join(sorted(available_sources, key=Version))}"
                    )
                    sys.exit(1)
            else:
                pip_source = available_sources[max(available_sources, key=Version)]

    if pip_source is None:
        pip_source = PIP_SOURCES["cpu"]  # always available

    with logging_manager(debug=args.verbose) as logger:
        LOGGER = logger

        if args.inplace:
            # For inplace mode, create a temporary Venv object using current Python
            venv = Venv(
                prefix=Path(sys.prefix),  # Use current environment
                pip_source=pip_source,
                base_executable=sys.executable,
            )
        else:
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
            inplace=args.inplace,
            logger=logger,
        )


if __name__ == "__main__":
    main()
