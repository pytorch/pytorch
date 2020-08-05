#!/usr/bin/env python3
"""Checks out the nightly development version of PyTorch and installs pre-built
binaries into the repo.
"""
import os
import re
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


LOGGER = logging.getLogger("conda-pytorch")
URL_FORMAT = "{base_url}/{platform}/{dist_name}.tar.bz2"
SHA1_RE = re.compile("([0-9a-fA-F]{40})")


def init_logging(level=logging.INFO):
    """Start up the logger"""
    logging.basicConfig(level=level)


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


@timed("Solving conda envrionment")
def conda_solve():
    """Performs the conda solve and splits the deps from the package."""
    cmd = ["conda", "create", "--yes", "--dry-run", "--json",
           "--name", "__pytorch__", "-c", "pytorch-nightly", "pytorch"]
    p = subprocess.run(cmd, capture_output=True, check=True)
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
    return deps, pytorch, platform


@timed("Installing dependencies")
def deps_install(deps):
    """Install dependencies to deps environment"""
    # first remove previous env
    cmd = ["conda", "env", "remove", "--yes", "--name", "pytorch-deps"]
    p = subprocess.run(cmd, check=True)
    # install new deps
    cmd = ["conda", "create", "--yes", "--no-deps", "--name", "pytorch-deps"] + deps
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
def checkout_nightly_version(spdir):
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
    cmd = ["git", "checkout", nightly_version]
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

def _move_single(src, target_dir, mover, verb):
    is_dir = os.path.isdir(src)
    base = os.path.basename(src)
    trg = os.path.join(target_dir, base)
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


def _copy_files(listing, target_dir):
    for src in listing:
        _move_single(src, target_dir, shutil.copy2, "Copying")


def _link_files(listing, target_dir):
    for src in listing:
        _move_single(src, target_dir, os.link, "Linking")


@timed("Moving nightly files into repo")
def move_nightly_files(spdir, platform):
    """Moves PyTorch files from temporary installed location to repo."""
    # get file listing
    source_dir = os.path.join(spdir, "torch")
    listing = _get_listing(source_dir, platform)
    target_dir = os.path.abspath("torch")
    # copy / link files
    if platform.startswith("win"):
        _copy_files(listing, target_dir)
    else:
        _link_files(listing, target_dir)


def install():
    """Development install of PyTorch"""
    deps, pytorch, platform = conda_solve()
    deps_install(deps)
    pytdir = pytorch_install(pytorch)
    spdir = _site_packages(pytdir, platform)
    checkout_nightly_version(spdir)
    move_nightly_files(spdir, platform)
    pytdir.cleanup()
    print("-------\nPyTorch Development Environment set up!\nPlease activate to "
          "enable this environment:\n  $ conda activate pytorch-deps")


def main(args=None):
    """Main entry point"""
    init_logging()
    install()


if __name__ == "__main__":
    main()
