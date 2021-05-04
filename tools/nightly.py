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

You can also use this tool to pull the nightly commits into the current branch as
well. This can be done with

    $ ./tools/nightly.py pull -n my-env
    $ conda activate my-env

Pulling will reinstalle the conda dependencies as well as the nightly binaries into
the repo directory.
"""
import os
import re
import sys
import json
import glob
import time
import uuid
import shutil
import logging
import datetime
import tempfile
import functools
import contextlib
import subprocess
from ast import literal_eval
from argparse import ArgumentParser
from typing import Dict, Optional, Iterator


LOGGER = None
URL_FORMAT = "{base_url}/{platform}/{dist_name}.tar.bz2"
DATETIME_FORMAT = "%Y-%m-%d_%Hh%Mm%Ss"
SHA1_RE = re.compile("([0-9a-fA-F]{40})")
USERNAME_PASSWORD_RE = re.compile(r":\/\/(.*?)\@")
LOG_DIRNAME_RE = re.compile(
    r"(\d{4}-\d\d-\d\d_\d\dh\d\dm\d\ds)_" r"[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}"
)
SPECS_TO_INSTALL = ("pytorch", "mypy", "pytest", "hypothesis", "ipython", "sphinx")


class Formatter(logging.Formatter):
    redactions: Dict[str, str]

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
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


@functools.lru_cache()
def logging_base_dir() -> str:
    meta_dir = os.getcwd()
    base_dir = os.path.join(meta_dir, "nightly", "log")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


@functools.lru_cache()
def logging_run_dir() -> str:
    cur_dir = os.path.join(
        logging_base_dir(),
        "{}_{}".format(datetime.datetime.now().strftime(DATETIME_FORMAT), uuid.uuid1()),
    )
    os.makedirs(cur_dir, exist_ok=True)
    return cur_dir


@functools.lru_cache()
def logging_record_argv() -> None:
    s = subprocess.list2cmdline(sys.argv)
    with open(os.path.join(logging_run_dir(), "argv"), "w") as f:
        f.write(s)


def logging_record_exception(e: BaseException) -> None:
    with open(os.path.join(logging_run_dir(), "exception"), "w") as f:
        f.write(type(e).__name__)


def logging_rotate() -> None:
    log_base = logging_base_dir()
    old_logs = os.listdir(log_base)
    old_logs.sort(reverse=True)
    for stale_log in old_logs[1000:]:
        # Sanity check that it looks like a log
        if LOG_DIRNAME_RE.fullmatch(stale_log) is not None:
            shutil.rmtree(os.path.join(log_base, stale_log))


@contextlib.contextmanager
def logging_manager(*, debug: bool = False) -> Iterator[None]:
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

    log_file = os.path.join(logging_run_dir(), "nightly.log")

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


def check_in_repo():
    """Ensures that we are in the PyTorch repo."""
    if not os.path.isfile("setup.py"):
        return "Not in root-level PyTorch repo, no setup.py found"
    with open("setup.py") as f:
        s = f.read()
    if "PyTorch" not in s:
        return "Not in PyTorch repo, 'PyTorch' not found in setup.py"


def check_branch(subcommand, branch):
    """Checks that the branch name can be checked out."""
    if subcommand != "checkout":
        return
    # first make sure actual branch name was given
    if branch is None:
        return "Branch name to checkout must be supplied with '-b' option"
    # next check that the local repo is clean
    cmd = ["git", "status", "--untracked-files=no", "--porcelain"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, universal_newlines=True)
    if p.stdout.strip():
        return "Need to have clean working tree to checkout!\n\n" + p.stdout
    # next check that the branch name doesn't already exist
    cmd = ["git", "show-ref", "--verify", "--quiet", "refs/heads/" + branch]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if not p.returncode:
        return f"Branch {branch!r} already exists"


@contextlib.contextmanager
def timer(logger, prefix):
    """Timed context manager"""
    start_time = time.time()
    yield
    logger.info(f"{prefix} took {time.time() - start_time:.3f} [s]")


def timed(prefix):
    """Decorator for timing functions"""

    def dec(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            global LOGGER
            LOGGER.info(prefix)
            with timer(LOGGER, prefix):
                return f(*args, **kwargs)

        return wrapper

    return dec


def _make_channel_args(channels=("pytorch-nightly",), override_channels=False):
    args = []
    for channel in channels:
        args.append("--channel")
        args.append(channel)
    if override_channels:
        args.append("--override-channels")
    return args


@timed("Solving conda environment")
def conda_solve(
    name=None, prefix=None, channels=("pytorch-nightly",), override_channels=False
):
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
        env_opts = ["--name", "pytorch-deps"]
    # run solve
    if existing_env:
        cmd = [
            "conda",
            "install",
            "--yes",
            "--dry-run",
            "--json",
        ]
        cmd.extend(env_opts)
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
        channels=channels, override_channels=override_channels
    )
    cmd.extend(channel_args)
    cmd.extend(SPECS_TO_INSTALL)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    # parse solution
    solve = json.loads(p.stdout)
    link = solve["actions"]["LINK"]
    deps = []
    for pkg in link:
        url = URL_FORMAT.format(**pkg)
        if pkg["name"] == "pytorch":
            pytorch = url
            platform = pkg["platform"]
        else:
            deps.append(url)
    return deps, pytorch, platform, existing_env, env_opts


@timed("Installing dependencies")
def deps_install(deps, existing_env, env_opts):
    """Install dependencies to deps environment"""
    if not existing_env:
        # first remove previous pytorch-deps env
        cmd = ["conda", "env", "remove", "--yes"] + env_opts
        p = subprocess.run(cmd, check=True)
    # install new deps
    inst_opt = "install" if existing_env else "create"
    cmd = ["conda", inst_opt, "--yes", "--no-deps"] + env_opts + deps
    p = subprocess.run(cmd, check=True)


@timed("Installing pytorch nightly binaries")
def pytorch_install(url):
    """"Install pytorch into a temporary directory"""
    pytdir = tempfile.TemporaryDirectory()
    cmd = ["conda", "create", "--yes", "--no-deps", "--prefix", pytdir.name, url]
    p = subprocess.run(cmd, check=True)
    return pytdir


def _site_packages(dirname, platform):
    if platform.startswith("win"):
        template = os.path.join(dirname, "Lib", "site-packages")
    else:
        template = os.path.join(dirname, "lib", "python*.*", "site-packages")
    spdir = glob.glob(template)[0]
    return spdir


def _ensure_commit(git_sha1):
    """Make sure that we actually have the commit locally"""
    cmd = ["git", "cat-file", "-e", git_sha1 + "^{commit}"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode == 0:
        # we have the commit locally
        return
    # we don't have the commit, must fetch
    cmd = ["git", "fetch", "https://github.com/pytorch/pytorch.git", git_sha1]
    p = subprocess.run(cmd, check=True)


def _nightly_version(spdir):
    # first get the git version from the installed module
    version_fname = os.path.join(spdir, "torch", "version.py")
    with open(version_fname) as f:
        lines = f.read().splitlines()
    for line in lines:
        if not line.startswith("git_version"):
            continue
        git_version = literal_eval(line.partition("=")[2].strip())
        break
    else:
        raise RuntimeError(f"Could not find git_version in {version_fname}")
    print(f"Found released git version {git_version}")
    # now cross reference with nightly version
    _ensure_commit(git_version)
    cmd = ["git", "show", "--no-patch", "--format=%s", git_version]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, universal_newlines=True)
    m = SHA1_RE.search(p.stdout)
    if m is None:
        raise RuntimeError(
            f"Could not find nightly release in git history:\n  {p.stdout}"
        )
    nightly_version = m.group(1)
    print(f"Found nightly release version {nightly_version}")
    # now checkout nightly version
    _ensure_commit(nightly_version)
    return nightly_version


@timed("Checking out nightly PyTorch")
def checkout_nightly_version(branch, spdir):
    """Get's the nightly version and then checks it out."""
    nightly_version = _nightly_version(spdir)
    cmd = ["git", "checkout", "-b", branch, nightly_version]
    p = subprocess.run(cmd, check=True)


@timed("Pulling nightly PyTorch")
def pull_nightly_version(spdir):
    """Fetches the nightly version and then merges it ."""
    nightly_version = _nightly_version(spdir)
    cmd = ["git", "merge", nightly_version]
    p = subprocess.run(cmd, check=True)


def _get_listing_linux(source_dir):
    listing = glob.glob(os.path.join(source_dir, "*.so"))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.so")))
    return listing


def _get_listing_osx(source_dir):
    # oddly, these are .so files even on Mac
    listing = glob.glob(os.path.join(source_dir, "*.so"))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.dylib")))
    return listing


def _get_listing_win(source_dir):
    listing = glob.glob(os.path.join(source_dir, "*.pyd"))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.lib")))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.dll")))
    return listing


def _glob_pyis(d):
    search = os.path.join(d, "**", "*.pyi")
    pyis = {os.path.relpath(p, d) for p in glob.iglob(search)}
    return pyis


def _find_missing_pyi(source_dir, target_dir):
    source_pyis = _glob_pyis(source_dir)
    target_pyis = _glob_pyis(target_dir)
    missing_pyis = [os.path.join(source_dir, p) for p in (source_pyis - target_pyis)]
    missing_pyis.sort()
    return missing_pyis


def _get_listing(source_dir, target_dir, platform):
    if platform.startswith("linux"):
        listing = _get_listing_linux(source_dir)
    elif platform.startswith("osx"):
        listing = _get_listing_osx(source_dir)
    elif platform.startswith("win"):
        listing = _get_listing_win(source_dir)
    else:
        raise RuntimeError(f"Platform {platform!r} not recognized")
    listing.extend(_find_missing_pyi(source_dir, target_dir))
    listing.append(os.path.join(source_dir, "version.py"))
    listing.append(os.path.join(source_dir, "testing", "_internal", "generated"))
    listing.append(os.path.join(source_dir, "bin"))
    listing.append(os.path.join(source_dir, "include"))
    return listing


def _remove_existing(trg, is_dir):
    if os.path.exists(trg):
        if is_dir:
            shutil.rmtree(trg)
        else:
            os.remove(trg)


def _move_single(src, source_dir, target_dir, mover, verb):
    is_dir = os.path.isdir(src)
    relpath = os.path.relpath(src, source_dir)
    trg = os.path.join(target_dir, relpath)
    _remove_existing(trg, is_dir)
    # move over new files
    if is_dir:
        os.makedirs(trg, exist_ok=True)
        for root, dirs, files in os.walk(src):
            relroot = os.path.relpath(root, src)
            for name in files:
                relname = os.path.join(relroot, name)
                s = os.path.join(src, relname)
                t = os.path.join(trg, relname)
                print(f"{verb} {s} -> {t}")
                mover(s, t)
            for name in dirs:
                relname = os.path.join(relroot, name)
                os.makedirs(os.path.join(trg, relname), exist_ok=True)
    else:
        print(f"{verb} {src} -> {trg}")
        mover(src, trg)


def _copy_files(listing, source_dir, target_dir):
    for src in listing:
        _move_single(src, source_dir, target_dir, shutil.copy2, "Copying")


def _link_files(listing, source_dir, target_dir):
    for src in listing:
        _move_single(src, source_dir, target_dir, os.link, "Linking")


@timed("Moving nightly files into repo")
def move_nightly_files(spdir, platform):
    """Moves PyTorch files from temporary installed location to repo."""
    # get file listing
    source_dir = os.path.join(spdir, "torch")
    target_dir = os.path.abspath("torch")
    listing = _get_listing(source_dir, target_dir, platform)
    # copy / link files
    if platform.startswith("win"):
        _copy_files(listing, source_dir, target_dir)
    else:
        try:
            _link_files(listing, source_dir, target_dir)
        except Exception:
            _copy_files(listing, source_dir, target_dir)


def _available_envs():
    cmd = ["conda", "env", "list"]
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    lines = p.stdout.splitlines()
    envs = {}
    for line in map(str.strip, lines):
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            # unnamed env
            continue
        envs[parts[0]] = parts[-1]
    return envs


@timed("Writing pytorch-nightly.pth")
def write_pth(env_opts, platform):
    """Writes Python path file for this dir."""
    env_type, env_dir = env_opts
    if env_type == "--name":
        # have to find directory
        envs = _available_envs()
        env_dir = envs[env_dir]
    spdir = _site_packages(env_dir, platform)
    pth = os.path.join(spdir, "pytorch-nightly.pth")
    s = (
        "# This file was autogenerated by PyTorch's tools/nightly.py\n"
        "# Please delete this file if you no longer need the following development\n"
        "# version of PyTorch to be importable\n"
        f"{os.getcwd()}\n"
    )
    with open(pth, "w") as f:
        f.write(s)


def install(
    subcommand="checkout",
    branch=None,
    name=None,
    prefix=None,
    channels=("pytorch-nightly",),
    override_channels=False,
    logger=None,
):
    """Development install of PyTorch"""
    global LOGGER
    logger = logger or LOGGER
    deps, pytorch, platform, existing_env, env_opts = conda_solve(
        name=name, prefix=prefix, channels=channels, override_channels=override_channels
    )
    if deps:
        deps_install(deps, existing_env, env_opts)
    pytdir = pytorch_install(pytorch)
    spdir = _site_packages(pytdir.name, platform)
    if subcommand == "checkout":
        checkout_nightly_version(branch, spdir)
    elif subcommand == "pull":
        pull_nightly_version(spdir)
    else:
        raise ValueError(f"Subcommand {subcommand} must be one of: checkout, pull.")
    move_nightly_files(spdir, platform)
    write_pth(env_opts, platform)
    pytdir.cleanup()
    logger.info(
        "-------\nPyTorch Development Environment set up!\nPlease activate to "
        f"enable this environment:\n  $ conda activate {env_opts[1]}"
    )


def make_parser():
    p = ArgumentParser("nightly")
    # subcommands
    subcmd = p.add_subparsers(dest="subcmd", help="subcommand to execute")
    co = subcmd.add_parser("checkout", help="checkout a new branch")
    co.add_argument(
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
    subps = [co, pull]
    for subp in subps:
        subp.add_argument(
            "-n",
            "--name",
            help="Name of environment",
            dest="name",
            default=None,
            metavar="ENVIRONMENT",
        )
        subp.add_argument(
            "-p",
            "--prefix",
            help="Full path to environment location (i.e. prefix)",
            dest="prefix",
            default=None,
            metavar="PATH",
        )
        subp.add_argument(
            "-v",
            "--verbose",
            help="Provide debugging info",
            dest="verbose",
            default=False,
            action="store_true",
        )
        subp.add_argument(
            "--override-channels",
            help="Do not search default or .condarc channels.",
            dest="override_channels",
            default=False,
            action="store_true",
        )
        subp.add_argument(
            "-c",
            "--channel",
            help="Additional channel to search for packages. 'pytorch-nightly' will always be prepended to this list.",
            dest="channels",
            action="append",
            metavar="CHANNEL",
        )
    return p


def main(args=None):
    """Main entry point"""
    global LOGGER
    p = make_parser()
    ns = p.parse_args(args)
    ns.branch = getattr(ns, "branch", None)
    status = check_in_repo()
    status = status or check_branch(ns.subcmd, ns.branch)
    if status:
        sys.exit(status)
    channels = ["pytorch-nightly"]
    if ns.channels:
        channels.extend(ns.channels)
    with logging_manager(debug=ns.verbose) as logger:
        LOGGER = logger
        install(
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
