#!/usr/bin/env python3
"""Checks out the nightly development version of PyTorch and installs pre-built
binaries into the repo.
"""
import os
import re
import sys
import json
import glob
import time
import shutil
import logging
import tempfile
import functools
import contextlib
import subprocess
from ast import literal_eval
from argparse import ArgumentParser


LOGGER = logging.getLogger("conda-pytorch")
URL_FORMAT = "{base_url}/{platform}/{dist_name}.tar.bz2"
SHA1_RE = re.compile("([0-9a-fA-F]{40})")
SPECS_TO_INSTALL = ("pytorch", "mypy", "pytest", "ipython", "sphinx")


def init_logging(level=logging.INFO):
    """Start up the logger"""
    logging.basicConfig(level=level)


def check_branch(branch):
    """Checks that the branch name can be checked out."""
    # first make sure actual branch name was given
    if branch is None:
        return "Branch name to checkout must be supplied with '-b' option"
    # next check that the local repo is clean
    cmd = ["git", "status", "--untracked-files=no", "--porcelain"]
    p = subprocess.run(cmd, capture_output=True, check=True, text=True)
    if p.stdout.strip():
        return "Need to have clean working tree to checkout!\n\n" + p.stdout
    # next check that the branch name doesn't already exist
    cmd = ["git", "show-ref", "--verify", "--quiet", "refs/heads/" + branch]
    p = subprocess.run(cmd, capture_output=True, check=False)
    if not p.returncode:
        return f"Branch {branch!r} already exists"


@contextlib.contextmanager
def timer(logger, prefix):
    """Timed conetxt manager"""
    start_time = time.time()
    yield
    logger.info(f"{prefix} took {time.time() - start_time:.3f} [s]")


def timed(prefix):
    """Decorator for timing functions"""
    def dec(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            LOGGER.info(prefix)
            with timer(LOGGER, prefix):
                return f(*args, **kwargs)
        return wrapper
    return dec


@timed("Solving conda environment")
def conda_solve(name=None, prefix=None):
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
        cmd = ["conda", "install", "--yes", "--dry-run", "--json",
               "-c", "pytorch-nightly", "-c", "conda-forge"]
        cmd.extend(env_opts)
    else:
        cmd = ["conda", "create", "--yes", "--dry-run", "--json",
               "--name", "__pytorch__", "-c", "pytorch-nightly", "-c", "conda-forge"]
    cmd.extend(SPECS_TO_INSTALL)
    p = subprocess.run(cmd, capture_output=True, check=True)
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


def _site_packages(pytdir, platform):
    if platform.startswith("win"):
        os.path.join(pytdir.name, "Lib", "site-packages")
    else:
        template = os.path.join(pytdir.name, "lib", "python*.*", "site-packages")
        spdir = glob.glob(template)[0]
    return spdir


def _ensure_commit(git_sha1):
    """Make sure that we actually have the commit locally"""
    cmd = ["git", "cat-file", "-e", git_sha1 + "^{commit}"]
    p = subprocess.run(cmd, capture_output=True, check=False)
    if p.returncode == 0:
        # we have the commit locally
        return
    # we don't have the commit, must fetch
    cmd = ["git", "fetch", "https://github.com/pytorch/pytorch.git", git_sha1]
    p = subprocess.run(cmd, check=True)


@timed("Checking out nightly PyTorch")
def checkout_nightly_version(branch, spdir):
    """Get's the nightly version and then checks it out."""
    # first get the git version from the installed module
    version_fname = os.path.join(spdir, "torch", "version.py")
    with open(version_fname) as f:
        lines = f.read().splitlines()
    for line in lines:
        if not line.startswith('git_version'):
            continue
        git_version = literal_eval(line.partition("=")[2].strip())
        break
    else:
        raise RuntimeError(f"Could not find git_version in {version_fname}")
    print(f"Found released git version {git_version}")
    # now cross refernce with nightly version
    _ensure_commit(git_version)
    cmd = ["git", "show", "--no-patch", "--format=%s", git_version]
    p = subprocess.run(cmd, capture_output=True, check=True, text=True)
    m = SHA1_RE.search(p.stdout)
    if m is None:
        raise RuntimeError(f"Could not find nightly release in git history:\n  {p.stdout}")
    nightly_version = m.group(1)
    print(f"Found nightly release version {nightly_version}")
    # now checkout nightly version
    _ensure_commit(nightly_version)
    cmd = ["git", "checkout", "-b", branch, nightly_version]
    p = subprocess.run(cmd, check=True)


def _get_listing_linux(source_dir):
    listing = glob.glob(os.path.join(source_dir, "*.so"))
    listing.append(os.path.join(source_dir, "version.py"))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.so")))
    listing.append(os.path.join(source_dir, "bin"))
    return listing


def _get_listing_osx(source_dir):
    # oddly, these are .so files even on Mac
    listing = glob.glob(os.path.join(source_dir, "*.so"))
    listing.append(os.path.join(source_dir, "version.py"))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.dylib")))
    listing.append(os.path.join(source_dir, "bin"))
    return listing


def _get_listing_win(source_dir):
    listing = glob.glob(os.path.join(source_dir, "*.pyd"))
    listing.append(os.path.join(source_dir, "version.py"))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.lib")))
    listing.extend(glob.glob(os.path.join(source_dir, "lib", "*.dll")))
    listing.append(os.path.join(source_dir, "bin"))
    return listing


def _get_listing(source_dir, platform):
    if platform.startswith("linux"):
        listing = _get_listing_linux(source_dir)
    elif platform.startswith("osx"):
        listing = _get_listing_osx(source_dir)
    elif platform.startswith("win"):
        listing = _get_listing_win(source_dir)
    else:
        raise RuntimeError(f"Platform {platform!r} not recognized")
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
            relroot = root[len(src):]
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
    listing = _get_listing(source_dir, platform)
    target_dir = os.path.abspath("torch")
    # copy / link files
    if platform.startswith("win"):
        _copy_files(listing, source_dir, target_dir)
    else:
        _link_files(listing, source_dir, target_dir)


def install(branch=None, name=None, prefix=None):
    """Development install of PyTorch"""
    deps, pytorch, platform, existing_env, env_opts = conda_solve(name=name, prefix=prefix)
    if deps:
        deps_install(deps, existing_env, env_opts)
    pytdir = pytorch_install(pytorch)
    spdir = _site_packages(pytdir, platform)
    checkout_nightly_version(branch, spdir)
    move_nightly_files(spdir, platform)
    pytdir.cleanup()
    print("-------\nPyTorch Development Environment set up!\nPlease activate to "
          f"enable this environment:\n  $ conda activate {env_opts[1]}")


def make_parser():
    p = ArgumentParser("nightly-checkout")
    p.add_argument("-b", "--branch", help="Branch name to checkout", dest="branch", default=None,
                   metavar="NAME")
    p.add_argument("-n", "--name", help="Name of environment", dest="name", default=None,
                   metavar="ENVIRONMENT")
    p.add_argument("-p", "--prefix", help="Full path to environment location (i.e. prefix)",
                   dest="prefix", default=None, metavar="PATH")
    return p


def main(args=None):
    """Main entry point"""
    p = make_parser()
    ns = p.parse_args(args)
    init_logging()
    status = check_branch(ns.branch)
    if status:
        sys.exit(status)
    install(branch=ns.branch, name=ns.name, prefix=ns.prefix)


if __name__ == "__main__":
    main()
