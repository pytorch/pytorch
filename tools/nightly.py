#!/usr/bin/env python3
# Much of the logging code here was forked from https://github.com/ezyang/ghstack
# Copyright (c) Edward Z. Yang <ezyang@mit.edu>
"""Checks out the nightly development version of PyTorch and installs pre-built
binaries into the repo.

You can use this script to check out a new nightly branch with the following::

    $ ./tools/nightly.py checkout -b my-nightly-branch
    $ conda activate pytorch-deps

Or if you would like to re-use an existing conda environment, you can pass in
the regular environment parameters (--name or --prefix)::

    $ ./tools/nightly.py checkout -b my-nightly-branch -n my-env
    $ conda activate my-env

To install the nightly binaries built with CUDA, you can pass in the flag --cuda::

    $ ./tools/nightly.py checkout -b my-nightly-branch --cuda
    $ conda activate pytorch-deps

You can also use this tool to pull the nightly commits into the current branch as
well. This can be done with::

    $ ./tools/nightly.py pull -n my-env
    $ conda activate my-env

Pulling will reinstall the conda dependencies as well as the nightly binaries into
the repo directory.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import glob
import itertools
import json
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
from typing import Any, Callable, cast, Generator, Iterable, Iterator, Sequence, TypeVar


REPO_ROOT = Path(__file__).absolute().parent.parent
GITHUB_REMOTE_URL = "https://github.com/pytorch/pytorch.git"
SPECS_TO_INSTALL = ("pytorch", "mypy", "pytest", "hypothesis", "ipython", "sphinx")
DEFAULT_ENV_NAME = "pytorch-deps"

LOGGER: logging.Logger | None = None
URL_FORMAT = "{base_url}/{platform}/{dist_name}.tar.bz2"
DATETIME_FORMAT = "%Y-%m-%d_%Hh%Mm%Ss"
SHA1_RE = re.compile(r"(?P<sha1>[0-9a-fA-F]{40})")
USERNAME_PASSWORD_RE = re.compile(r":\/\/(.*?)\@")
LOG_DIRNAME_RE = re.compile(
    r"(?P<datetime>\d{4}-\d\d-\d\d_\d\dh\d\dm\d\ds)_"
    r"(?P<uuid>[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})",
)


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
    (logging_run_dir() / "exception").write_text(type(e).__name__, encoding="utf-8")


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
    root_logger = logging.getLogger("conda-pytorch")
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


def check_conda_env_exists(name: str | None = None, prefix: str | None = None) -> bool:
    """Checks that the conda environment exists."""
    if name is not None and prefix is not None:
        raise ValueError("Cannot specify both --name and --prefix")
    if name is None and prefix is None:
        raise ValueError("Must specify either --name or --prefix")

    try:
        cmd = ["conda", "info", "--envs"]
        output = subprocess.check_output(cmd, text=True, encoding="utf-8")
    except subprocess.CalledProcessError:
        logger = cast(logging.Logger, LOGGER)
        logger.warning("Failed to list conda environments", exc_info=True)
        return False

    if name is not None:
        return len(re.findall(rf"^{name}\s+", output, flags=re.MULTILINE)) > 0
    assert prefix is not None
    prefix = Path(prefix).absolute()
    return len(re.findall(rf"\s+{prefix}$", output, flags=re.MULTILINE)) > 0


@contextlib.contextmanager
def timer(logger: logging.Logger, prefix: str) -> Iterator[None]:
    """Timed context manager"""
    start_time = time.perf_counter()
    yield
    logger.info("%s took %.3f [s]", prefix, time.perf_counter() - start_time)


F = TypeVar("F", bound=Callable[..., Any])


def timed(prefix: str) -> Callable[[F], F]:
    """Decorator for timing functions"""

    def dec(f: F) -> F:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = cast(logging.Logger, LOGGER)
            logger.info(prefix)
            with timer(logger, prefix):
                return f(*args, **kwargs)

        return cast(F, wrapper)

    return dec


def _make_channel_args(
    channels: Iterable[str] = ("pytorch-nightly",),
    override_channels: bool = False,
) -> list[str]:
    args = []
    for channel in channels:
        args.extend(["--channel", channel])
    if override_channels:
        args.append("--override-channels")
    return args


@timed("Solving conda environment")
def conda_solve(
    specs: Iterable[str],
    *,
    name: str | None = None,
    prefix: str | None = None,
    channels: Iterable[str] = ("pytorch-nightly",),
    override_channels: bool = False,
) -> tuple[list[str], str, str, bool, list[str]]:
    """Performs the conda solve and splits the deps from the package."""
    # compute what environment to use
    if prefix is not None:
        existing_env = True
        env_opts = ["--prefix", prefix]
    elif name is not None:
        existing_env = True
        env_opts = ["--name", name]
    else:
        # create new environment
        existing_env = False
        env_opts = ["--name", DEFAULT_ENV_NAME]
    # run solve
    if existing_env:
        cmd = [
            "conda",
            "install",
            "--yes",
            "--dry-run",
            "--json",
            *env_opts,
        ]
    else:
        cmd = [
            "conda",
            "create",
            "--yes",
            "--dry-run",
            "--json",
            "--name",
            "__pytorch__",
        ]
    channel_args = _make_channel_args(
        channels=channels,
        override_channels=override_channels,
    )
    cmd.extend(channel_args)
    cmd.extend(specs)
    stdout = subprocess.check_output(cmd, text=True, encoding="utf-8")
    # parse solution
    solve = json.loads(stdout)
    link = solve["actions"]["LINK"]
    deps = []
    pytorch, platform = "", ""
    for pkg in link:
        url = URL_FORMAT.format(**pkg)
        if pkg["name"] == "pytorch":
            pytorch = url
            platform = pkg["platform"]
        else:
            deps.append(url)
    assert pytorch, "PyTorch package not found in solve"
    assert platform, "Platform not found in solve"
    return deps, pytorch, platform, existing_env, env_opts


@timed("Installing dependencies")
def deps_install(deps: list[str], existing_env: bool, env_opts: list[str]) -> None:
    """Install dependencies to deps environment"""
    if not existing_env:
        # first remove previous pytorch-deps env
        if check_conda_env_exists(name=DEFAULT_ENV_NAME):
            cmd = ["conda", "env", "remove", "--yes", *env_opts]
            subprocess.check_output(cmd)
    # install new deps
    install_command = "install" if existing_env else "create"
    cmd = ["conda", install_command, "--yes", "--no-deps", *env_opts, *deps]
    subprocess.check_call(cmd)


@timed("Installing pytorch nightly binaries")
def pytorch_install(url: str) -> tempfile.TemporaryDirectory[str]:
    """Install pytorch into a temporary directory"""
    pytorch_dir = tempfile.TemporaryDirectory(prefix="conda-pytorch-")
    cmd = ["conda", "create", "--yes", "--no-deps", f"--prefix={pytorch_dir.name}", url]
    subprocess.check_call(cmd)
    return pytorch_dir


def _site_packages(dirname: str, platform: str) -> Path:
    if platform.startswith("win"):
        template = os.path.join(dirname, "Lib", "site-packages")
    else:
        template = os.path.join(dirname, "lib", "python*.*", "site-packages")
    return Path(next(glob.iglob(template))).absolute()


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


def _get_listing_osx(source_dir: Path) -> list[Path]:
    # oddly, these are .so files even on Mac
    return list(
        itertools.chain(
            source_dir.glob("*.so"),
            (source_dir / "lib").glob("*.dylib"),
        )
    )


def _get_listing_win(source_dir: Path) -> list[Path]:
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


def _get_listing(source_dir: Path, target_dir: Path, platform: str) -> list[Path]:
    if platform.startswith("linux"):
        listing = _get_listing_linux(source_dir)
    elif platform.startswith("osx"):
        listing = _get_listing_osx(source_dir)
    elif platform.startswith("win"):
        listing = _get_listing_win(source_dir)
    else:
        raise RuntimeError(f"Platform {platform!r} not recognized")
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
def move_nightly_files(site_dir: Path, platform: str) -> None:
    """Moves PyTorch files from temporary installed location to repo."""
    # get file listing
    source_dir = site_dir / "torch"
    target_dir = REPO_ROOT / "torch"
    listing = _get_listing(source_dir, target_dir, platform)
    # copy / link files
    if platform.startswith("win"):
        _copy_files(listing, source_dir, target_dir)
    else:
        try:
            _link_files(listing, source_dir, target_dir)
        except Exception:
            _copy_files(listing, source_dir, target_dir)


def _available_envs() -> dict[str, str]:
    cmd = ["conda", "env", "list"]
    stdout = subprocess.check_output(cmd, text=True, encoding="utf-8")
    envs = {}
    for line in map(str.strip, stdout.splitlines()):
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            # unnamed env
            continue
        envs[parts[0]] = parts[-1]
    return envs


@timed("Writing pytorch-nightly.pth")
def write_pth(env_opts: list[str], platform: str) -> None:
    """Writes Python path file for this dir."""
    env_type, env_dir = env_opts
    if env_type == "--name":
        # have to find directory
        envs = _available_envs()
        env_dir = envs[env_dir]
    site_dir = _site_packages(env_dir, platform)
    (site_dir / "pytorch-nightly.pth").write_text(
        "# This file was autogenerated by PyTorch's tools/nightly.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{REPO_ROOT}\n",
        encoding="utf-8",
    )


def install(
    specs: Iterable[str],
    *,
    logger: logging.Logger,
    subcommand: str = "checkout",
    branch: str | None = None,
    name: str | None = None,
    prefix: str | None = None,
    channels: Iterable[str] = ("pytorch-nightly",),
    override_channels: bool = False,
) -> None:
    """Development install of PyTorch"""
    specs = list(specs)
    deps, pytorch, platform, existing_env, env_opts = conda_solve(
        specs=specs,
        name=name,
        prefix=prefix,
        channels=channels,
        override_channels=override_channels,
    )
    if deps:
        deps_install(deps, existing_env, env_opts)

    with pytorch_install(pytorch) as pytorch_dir:
        site_dir = _site_packages(pytorch_dir, platform)
        if subcommand == "checkout":
            checkout_nightly_version(cast(str, branch), site_dir)
        elif subcommand == "pull":
            pull_nightly_version(site_dir)
        else:
            raise ValueError(f"Subcommand {subcommand} must be one of: checkout, pull.")
        move_nightly_files(site_dir, platform)

    write_pth(env_opts, platform)
    logger.info(
        "-------\nPyTorch Development Environment set up!\nPlease activate to "
        "enable this environment:\n  $ conda activate %s",
        env_opts[1],
    )


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # subcommands
    subcmd = p.add_subparsers(dest="subcmd", help="subcommand to execute")
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
            "-n",
            "--name",
            help="Name of environment",
            dest="name",
            default=None,
            metavar="ENVIRONMENT",
        )
        subparser.add_argument(
            "-p",
            "--prefix",
            help="Full path to environment location (i.e. prefix)",
            dest="prefix",
            default=None,
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
            "--override-channels",
            help="Do not search default or .condarc channels.",
            dest="override_channels",
            default=False,
            action="store_true",
        )
        subparser.add_argument(
            "-c",
            "--channel",
            help=(
                "Additional channel to search for packages. "
                "'pytorch-nightly' will always be prepended to this list."
            ),
            dest="channels",
            action="append",
            metavar="CHANNEL",
        )
        if platform_system() in {"Linux", "Windows"}:
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
    return p


def main(args: Sequence[str] | None = None) -> None:
    """Main entry point"""
    global LOGGER
    p = make_parser()
    ns = p.parse_args(args)
    ns.branch = getattr(ns, "branch", None)
    status = check_branch(ns.subcmd, ns.branch)
    if status:
        sys.exit(status)
    specs = list(SPECS_TO_INSTALL)
    channels = ["pytorch-nightly"]
    if hasattr(ns, "cuda"):
        if ns.cuda is not None:
            specs.append(f"pytorch-cuda={ns.cuda}")
        else:
            specs.append("pytorch-cuda")
        specs.append("pytorch-mutex=*=*cuda*")
        channels.append("nvidia")
    else:
        specs.append("pytorch-mutex=*=*cpu*")
    if ns.channels:
        channels.extend(ns.channels)
    with logging_manager(debug=ns.verbose) as logger:
        LOGGER = logger
        install(
            specs=specs,
            subcommand=ns.subcmd,
            branch=ns.branch,
            name=ns.name,
            prefix=ns.prefix,
            logger=logger,
            channels=channels,
            override_channels=ns.override_channels,
        )


if __name__ == "__main__":
    main()
