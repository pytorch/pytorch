from __future__ import annotations

import os.path
import re
import shutil
import tempfile
import zipfile
from glob import iglob

from .._bdist_wheel import bdist_wheel
from ..wheelfile import WheelFile
from . import WheelError

try:
    from setuptools import Distribution
except ImportError:
    from distutils.dist import Distribution

egg_info_re = re.compile(
    r"""
    (?P<name>.+?)-(?P<ver>.+?)
    (-(?P<pyver>py\d\.\d+)
     (-(?P<arch>.+?))?
    )?.egg$""",
    re.VERBOSE,
)


class _bdist_wheel_tag(bdist_wheel):
    # allow the client to override the default generated wheel tag
    # The default bdist_wheel implementation uses python and abi tags
    # of the running python process. This is not suitable for
    # generating/repackaging prebuild binaries.

    full_tag_supplied = False
    full_tag = None  # None or a (pytag, soabitag, plattag) triple

    def get_tag(self):
        if self.full_tag_supplied and self.full_tag is not None:
            return self.full_tag
        else:
            return bdist_wheel.get_tag(self)


def egg2wheel(egg_path: str, dest_dir: str) -> None:
    filename = os.path.basename(egg_path)
    match = egg_info_re.match(filename)
    if not match:
        raise WheelError(f"Invalid egg file name: {filename}")

    egg_info = match.groupdict()
    dir = tempfile.mkdtemp(suffix="_e2w")
    if os.path.isfile(egg_path):
        # assume we have a bdist_egg otherwise
        with zipfile.ZipFile(egg_path) as egg:
            egg.extractall(dir)
    else:
        # support buildout-style installed eggs directories
        for pth in os.listdir(egg_path):
            src = os.path.join(egg_path, pth)
            if os.path.isfile(src):
                shutil.copy2(src, dir)
            else:
                shutil.copytree(src, os.path.join(dir, pth))

    pyver = egg_info["pyver"]
    if pyver:
        pyver = egg_info["pyver"] = pyver.replace(".", "")

    arch = (egg_info["arch"] or "any").replace(".", "_").replace("-", "_")

    # assume all binary eggs are for CPython
    abi = "cp" + pyver[2:] if arch != "any" else "none"

    root_is_purelib = egg_info["arch"] is None
    if root_is_purelib:
        bw = bdist_wheel(Distribution())
    else:
        bw = _bdist_wheel_tag(Distribution())

    bw.root_is_pure = root_is_purelib
    bw.python_tag = pyver
    bw.plat_name_supplied = True
    bw.plat_name = egg_info["arch"] or "any"
    if not root_is_purelib:
        bw.full_tag_supplied = True
        bw.full_tag = (pyver, abi, arch)

    dist_info_dir = os.path.join(dir, "{name}-{ver}.dist-info".format(**egg_info))
    bw.egg2dist(os.path.join(dir, "EGG-INFO"), dist_info_dir)
    bw.write_wheelfile(dist_info_dir, generator="egg2wheel")
    wheel_name = "{name}-{ver}-{pyver}-{}-{}.whl".format(abi, arch, **egg_info)
    with WheelFile(os.path.join(dest_dir, wheel_name), "w") as wf:
        wf.write_files(dir)

    shutil.rmtree(dir)


def parse_wininst_info(wininfo_name: str, egginfo_name: str | None):
    """Extract metadata from filenames.

    Extracts the 4 metadataitems needed (name, version, pyversion, arch) from
    the installer filename and the name of the egg-info directory embedded in
    the zipfile (if any).

    The egginfo filename has the format::

        name-ver(-pyver)(-arch).egg-info

    The installer filename has the format::

        name-ver.arch(-pyver).exe

    Some things to note:

    1. The installer filename is not definitive. An installer can be renamed
       and work perfectly well as an installer. So more reliable data should
       be used whenever possible.
    2. The egg-info data should be preferred for the name and version, because
       these come straight from the distutils metadata, and are mandatory.
    3. The pyver from the egg-info data should be ignored, as it is
       constructed from the version of Python used to build the installer,
       which is irrelevant - the installer filename is correct here (even to
       the point that when it's not there, any version is implied).
    4. The architecture must be taken from the installer filename, as it is
       not included in the egg-info data.
    5. Architecture-neutral installers still have an architecture because the
       installer format itself (being executable) is architecture-specific. We
       should therefore ignore the architecture if the content is pure-python.
    """

    egginfo = None
    if egginfo_name:
        egginfo = egg_info_re.search(egginfo_name)
        if not egginfo:
            raise ValueError(f"Egg info filename {egginfo_name} is not valid")

    # Parse the wininst filename
    # 1. Distribution name (up to the first '-')
    w_name, sep, rest = wininfo_name.partition("-")
    if not sep:
        raise ValueError(f"Installer filename {wininfo_name} is not valid")

    # Strip '.exe'
    rest = rest[:-4]
    # 2. Python version (from the last '-', must start with 'py')
    rest2, sep, w_pyver = rest.rpartition("-")
    if sep and w_pyver.startswith("py"):
        rest = rest2
        w_pyver = w_pyver.replace(".", "")
    else:
        # Not version specific - use py2.py3. While it is possible that
        # pure-Python code is not compatible with both Python 2 and 3, there
        # is no way of knowing from the wininst format, so we assume the best
        # here (the user can always manually rename the wheel to be more
        # restrictive if needed).
        w_pyver = "py2.py3"
    # 3. Version and architecture
    w_ver, sep, w_arch = rest.rpartition(".")
    if not sep:
        raise ValueError(f"Installer filename {wininfo_name} is not valid")

    if egginfo:
        w_name = egginfo.group("name")
        w_ver = egginfo.group("ver")

    return {"name": w_name, "ver": w_ver, "arch": w_arch, "pyver": w_pyver}


def wininst2wheel(path: str, dest_dir: str) -> None:
    with zipfile.ZipFile(path) as bdw:
        # Search for egg-info in the archive
        egginfo_name = None
        for filename in bdw.namelist():
            if ".egg-info" in filename:
                egginfo_name = filename
                break

        info = parse_wininst_info(os.path.basename(path), egginfo_name)

        root_is_purelib = True
        for zipinfo in bdw.infolist():
            if zipinfo.filename.startswith("PLATLIB"):
                root_is_purelib = False
                break
        if root_is_purelib:
            paths = {"purelib": ""}
        else:
            paths = {"platlib": ""}

        dist_info = "{name}-{ver}".format(**info)
        datadir = f"{dist_info}.data/"

        # rewrite paths to trick ZipFile into extracting an egg
        # XXX grab wininst .ini - between .exe, padding, and first zip file.
        members: list[str] = []
        egginfo_name = ""
        for zipinfo in bdw.infolist():
            key, basename = zipinfo.filename.split("/", 1)
            key = key.lower()
            basepath = paths.get(key, None)
            if basepath is None:
                basepath = datadir + key.lower() + "/"
            oldname = zipinfo.filename
            newname = basepath + basename
            zipinfo.filename = newname
            del bdw.NameToInfo[oldname]
            bdw.NameToInfo[newname] = zipinfo
            # Collect member names, but omit '' (from an entry like "PLATLIB/"
            if newname:
                members.append(newname)
            # Remember egg-info name for the egg2dist call below
            if not egginfo_name:
                if newname.endswith(".egg-info"):
                    egginfo_name = newname
                elif ".egg-info/" in newname:
                    egginfo_name, sep, _ = newname.rpartition("/")
        dir = tempfile.mkdtemp(suffix="_b2w")
        bdw.extractall(dir, members)

    # egg2wheel
    abi = "none"
    pyver = info["pyver"]
    arch = (info["arch"] or "any").replace(".", "_").replace("-", "_")
    # Wininst installers always have arch even if they are not
    # architecture-specific (because the format itself is).
    # So, assume the content is architecture-neutral if root is purelib.
    if root_is_purelib:
        arch = "any"
    # If the installer is architecture-specific, it's almost certainly also
    # CPython-specific.
    if arch != "any":
        pyver = pyver.replace("py", "cp")
    wheel_name = "-".join((dist_info, pyver, abi, arch))
    if root_is_purelib:
        bw = bdist_wheel(Distribution())
    else:
        bw = _bdist_wheel_tag(Distribution())

    bw.root_is_pure = root_is_purelib
    bw.python_tag = pyver
    bw.plat_name_supplied = True
    bw.plat_name = info["arch"] or "any"

    if not root_is_purelib:
        bw.full_tag_supplied = True
        bw.full_tag = (pyver, abi, arch)

    dist_info_dir = os.path.join(dir, f"{dist_info}.dist-info")
    bw.egg2dist(os.path.join(dir, egginfo_name), dist_info_dir)
    bw.write_wheelfile(dist_info_dir, generator="wininst2wheel")

    wheel_path = os.path.join(dest_dir, wheel_name)
    with WheelFile(wheel_path, "w") as wf:
        wf.write_files(dir)

    shutil.rmtree(dir)


def convert(files: list[str], dest_dir: str, verbose: bool) -> None:
    for pat in files:
        for installer in iglob(pat):
            if os.path.splitext(installer)[1] == ".egg":
                conv = egg2wheel
            else:
                conv = wininst2wheel

            if verbose:
                print(f"{installer}... ", flush=True)

            conv(installer, dest_dir)
            if verbose:
                print("OK")
