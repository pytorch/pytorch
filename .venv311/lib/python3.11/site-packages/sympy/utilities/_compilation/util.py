from collections import namedtuple
from hashlib import sha256
import os
import shutil
import sys
import fnmatch

from sympy.testing.pytest import XFAIL


def may_xfail(func):
    if sys.platform.lower() == 'darwin' or os.name == 'nt':
        # sympy.utilities._compilation needs more testing on Windows and macOS
        # once those two platforms are reliably supported this xfail decorator
        # may be removed.
        return XFAIL(func)
    else:
        return func


class CompilerNotFoundError(FileNotFoundError):
    pass


class CompileError (Exception):
    """Failure to compile one or more C/C++ source files."""


def get_abspath(path, cwd='.'):
    """ Returns the absolute path.

    Parameters
    ==========

    path : str
        (relative) path.
    cwd : str
        Path to root of relative path.
    """
    if os.path.isabs(path):
        return path
    else:
        if not os.path.isabs(cwd):
            cwd = os.path.abspath(cwd)
        return os.path.abspath(
            os.path.join(cwd, path)
        )


def make_dirs(path):
    """ Create directories (equivalent of ``mkdir -p``). """
    if path[-1] == '/':
        parent = os.path.dirname(path[:-1])
    else:
        parent = os.path.dirname(path)

    if len(parent) > 0:
        if not os.path.exists(parent):
            make_dirs(parent)

    if not os.path.exists(path):
        os.mkdir(path, 0o777)
    else:
        assert os.path.isdir(path)

def missing_or_other_newer(path, other_path, cwd=None):
    """
    Investigate if path is non-existent or older than provided reference
    path.

    Parameters
    ==========
    path: string
        path to path which might be missing or too old
    other_path: string
        reference path
    cwd: string
        working directory (root of relative paths)

    Returns
    =======
    True if path is older or missing.
    """
    cwd = cwd or '.'
    path = get_abspath(path, cwd=cwd)
    other_path = get_abspath(other_path, cwd=cwd)
    if not os.path.exists(path):
        return True
    if os.path.getmtime(other_path) - 1e-6 >= os.path.getmtime(path):
        # 1e-6 is needed because http://stackoverflow.com/questions/17086426/
        return True
    return False

def copy(src, dst, only_update=False, copystat=True, cwd=None,
         dest_is_dir=False, create_dest_dirs=False):
    """ Variation of ``shutil.copy`` with extra options.

    Parameters
    ==========

    src : str
        Path to source file.
    dst : str
        Path to destination.
    only_update : bool
        Only copy if source is newer than destination
        (returns None if it was newer), default: ``False``.
    copystat : bool
        See ``shutil.copystat``. default: ``True``.
    cwd : str
        Path to working directory (root of relative paths).
    dest_is_dir : bool
        Ensures that dst is treated as a directory. default: ``False``
    create_dest_dirs : bool
        Creates directories if needed.

    Returns
    =======

    Path to the copied file.

    """
    if cwd:  # Handle working directory
        if not os.path.isabs(src):
            src = os.path.join(cwd, src)
        if not os.path.isabs(dst):
            dst = os.path.join(cwd, dst)

    if not os.path.exists(src):  # Make sure source file exists
        raise FileNotFoundError("Source: `{}` does not exist".format(src))

    # We accept both (re)naming destination file _or_
    # passing a (possible non-existent) destination directory
    if dest_is_dir:
        if not dst[-1] == '/':
            dst = dst+'/'
    else:
        if os.path.exists(dst) and os.path.isdir(dst):
            dest_is_dir = True

    if dest_is_dir:
        dest_dir = dst
        dest_fname = os.path.basename(src)
        dst = os.path.join(dest_dir, dest_fname)
    else:
        dest_dir = os.path.dirname(dst)

    if not os.path.exists(dest_dir):
        if create_dest_dirs:
            make_dirs(dest_dir)
        else:
            raise FileNotFoundError("You must create directory first.")

    if only_update:
        if not missing_or_other_newer(dst, src):
            return

    if os.path.islink(dst):
        dst = os.path.abspath(os.path.realpath(dst), cwd=cwd)

    shutil.copy(src, dst)
    if copystat:
        shutil.copystat(src, dst)

    return dst

Glob = namedtuple('Glob', 'pathname')
ArbitraryDepthGlob = namedtuple('ArbitraryDepthGlob', 'filename')

def glob_at_depth(filename_glob, cwd=None):
    if cwd is not None:
        cwd = '.'
    globbed = []
    for root, dirs, filenames in os.walk(cwd):
        for fn in filenames:
            # This is not tested:
            if fnmatch.fnmatch(fn, filename_glob):
                globbed.append(os.path.join(root, fn))
    return globbed

def sha256_of_file(path, nblocks=128):
    """ Computes the SHA256 hash of a file.

    Parameters
    ==========

    path : string
        Path to file to compute hash of.
    nblocks : int
        Number of blocks to read per iteration.

    Returns
    =======

    hashlib sha256 hash object. Use ``.digest()`` or ``.hexdigest()``
    on returned object to get binary or hex encoded string.
    """
    sh = sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(nblocks*sh.block_size), b''):
            sh.update(chunk)
    return sh


def sha256_of_string(string):
    """ Computes the SHA256 hash of a string. """
    sh = sha256()
    sh.update(string)
    return sh


def pyx_is_cplus(path):
    """
    Inspect a Cython source file (.pyx) and look for comment line like:

    # distutils: language = c++

    Returns True if such a file is present in the file, else False.
    """
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') and '=' in line:
                splitted = line.split('=')
                if len(splitted) != 2:
                    continue
                lhs, rhs = splitted
                if lhs.strip().split()[-1].lower() == 'language' and \
                       rhs.strip().split()[0].lower() == 'c++':
                            return True
    return False

def import_module_from_file(filename, only_if_newer_than=None):
    """ Imports Python extension (from shared object file)

    Provide a list of paths in `only_if_newer_than` to check
    timestamps of dependencies. import_ raises an ImportError
    if any is newer.

    Word of warning: The OS may cache shared objects which makes
    reimporting same path of an shared object file very problematic.

    It will not detect the new time stamp, nor new checksum, but will
    instead silently use old module. Use unique names for this reason.

    Parameters
    ==========

    filename : str
        Path to shared object.
    only_if_newer_than : iterable of strings
        Paths to dependencies of the shared object.

    Raises
    ======

    ``ImportError`` if any of the files specified in ``only_if_newer_than`` are newer
    than the file given by filename.
    """
    path, name = os.path.split(filename)
    name, ext = os.path.splitext(name)
    name = name.split('.')[0]
    if sys.version_info[0] == 2:
        from imp import find_module, load_module
        fobj, filename, data = find_module(name, [path])
        if only_if_newer_than:
            for dep in only_if_newer_than:
                if os.path.getmtime(filename) < os.path.getmtime(dep):
                    raise ImportError("{} is newer than {}".format(dep, filename))
        mod = load_module(name, fobj, filename, data)
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, filename)
        if spec is None:
            raise ImportError("Failed to import: '%s'" % filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


def find_binary_of_command(candidates):
    """ Finds binary first matching name among candidates.

    Calls ``which`` from shutils for provided candidates and returns
    first hit.

    Parameters
    ==========

    candidates : iterable of str
        Names of candidate commands

    Raises
    ======

    CompilerNotFoundError if no candidates match.
    """
    from shutil import which
    for c in candidates:
        binary_path = which(c)
        if c and binary_path:
            return c, binary_path

    raise CompilerNotFoundError('No binary located for candidates: {}'.format(candidates))


def unique_list(l):
    """ Uniquify a list (skip duplicate items). """
    result = []
    for x in l:
        if x not in result:
            result.append(x)
    return result
