"""Locations where we look for configs, install stuff, etc"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

from __future__ import absolute_import

import os
import os.path
import platform
import site
import sys
import sysconfig
from distutils import sysconfig as distutils_sysconfig
from distutils.command.install import SCHEME_KEYS  # type: ignore
from distutils.command.install import install as distutils_install_command

from pip._internal.models.scheme import Scheme
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.typing import MYPY_CHECK_RUNNING, cast
from pip._internal.utils.virtualenv import running_under_virtualenv

if MYPY_CHECK_RUNNING:
    from typing import Dict, List, Optional, Union

    from distutils.cmd import Command as DistutilsCommand


# Application Directories
USER_CACHE_DIR = appdirs.user_cache_dir("pip")


def get_major_minor_version():
    # type: () -> str
    """
    Return the major-minor version of the current Python as a string, e.g.
    "3.7" or "3.10".
    """
    return '{}.{}'.format(*sys.version_info)


def get_src_prefix():
    # type: () -> str
    if running_under_virtualenv():
        src_prefix = os.path.join(sys.prefix, 'src')
    else:
        # FIXME: keep src in cwd for now (it is not a temporary folder)
        try:
            src_prefix = os.path.join(os.getcwd(), 'src')
        except OSError:
            # In case the current working directory has been renamed or deleted
            sys.exit(
                "The folder you are executing pip from can no longer be found."
            )

    # under macOS + virtualenv sys.prefix is not properly resolved
    # it is something like /path/to/python/bin/..
    return os.path.abspath(src_prefix)


# FIXME doesn't account for venv linked to global site-packages

site_packages = sysconfig.get_path("purelib")  # type: Optional[str]

# This is because of a bug in PyPy's sysconfig module, see
# https://bitbucket.org/pypy/pypy/issues/2506/sysconfig-returns-incorrect-paths
# for more information.
if platform.python_implementation().lower() == "pypy":
    site_packages = distutils_sysconfig.get_python_lib()
try:
    # Use getusersitepackages if this is present, as it ensures that the
    # value is initialised properly.
    user_site = site.getusersitepackages()
except AttributeError:
    user_site = site.USER_SITE

if WINDOWS:
    bin_py = os.path.join(sys.prefix, 'Scripts')
    bin_user = os.path.join(user_site, 'Scripts')
    # buildout uses 'bin' on Windows too?
    if not os.path.exists(bin_py):
        bin_py = os.path.join(sys.prefix, 'bin')
        bin_user = os.path.join(user_site, 'bin')
else:
    bin_py = os.path.join(sys.prefix, 'bin')
    bin_user = os.path.join(user_site, 'bin')

    # Forcing to use /usr/local/bin for standard macOS framework installs
    # Also log to ~/Library/Logs/ for use with the Console.app log viewer
    if sys.platform[:6] == 'darwin' and sys.prefix[:16] == '/System/Library/':
        bin_py = '/usr/local/bin'


def distutils_scheme(
    dist_name, user=False, home=None, root=None, isolated=False, prefix=None
):
    # type:(str, bool, str, str, bool, str) -> Dict[str, str]
    """
    Return a distutils install scheme
    """
    from distutils.dist import Distribution

    dist_args = {'name': dist_name}  # type: Dict[str, Union[str, List[str]]]
    if isolated:
        dist_args["script_args"] = ["--no-user-cfg"]

    d = Distribution(dist_args)
    d.parse_config_files()
    obj = None  # type: Optional[DistutilsCommand]
    obj = d.get_command_obj('install', create=True)
    assert obj is not None
    i = cast(distutils_install_command, obj)
    # NOTE: setting user or home has the side-effect of creating the home dir
    # or user base for installations during finalize_options()
    # ideally, we'd prefer a scheme class that has no side-effects.
    assert not (user and prefix), "user={} prefix={}".format(user, prefix)
    assert not (home and prefix), "home={} prefix={}".format(home, prefix)
    i.user = user or i.user
    if user or home:
        i.prefix = ""
    i.prefix = prefix or i.prefix
    i.home = home or i.home
    i.root = root or i.root
    i.finalize_options()

    scheme = {}
    for key in SCHEME_KEYS:
        scheme[key] = getattr(i, 'install_' + key)

    # install_lib specified in setup.cfg should install *everything*
    # into there (i.e. it takes precedence over both purelib and
    # platlib).  Note, i.install_lib is *always* set after
    # finalize_options(); we only want to override here if the user
    # has explicitly requested it hence going back to the config
    if 'install_lib' in d.get_option_dict('install'):
        scheme.update(dict(purelib=i.install_lib, platlib=i.install_lib))

    if running_under_virtualenv():
        scheme['headers'] = os.path.join(
            i.prefix,
            'include',
            'site',
            'python{}'.format(get_major_minor_version()),
            dist_name,
        )

        if root is not None:
            path_no_drive = os.path.splitdrive(
                os.path.abspath(scheme["headers"]))[1]
            scheme["headers"] = os.path.join(
                root,
                path_no_drive[1:],
            )

    return scheme


def get_scheme(
    dist_name,  # type: str
    user=False,  # type: bool
    home=None,  # type: Optional[str]
    root=None,  # type: Optional[str]
    isolated=False,  # type: bool
    prefix=None,  # type: Optional[str]
):
    # type: (...) -> Scheme
    """
    Get the "scheme" corresponding to the input parameters. The distutils
    documentation provides the context for the available schemes:
    https://docs.python.org/3/install/index.html#alternate-installation

    :param dist_name: the name of the package to retrieve the scheme for, used
        in the headers scheme path
    :param user: indicates to use the "user" scheme
    :param home: indicates to use the "home" scheme and provides the base
        directory for the same
    :param root: root under which other directories are re-based
    :param isolated: equivalent to --no-user-cfg, i.e. do not consider
        ~/.pydistutils.cfg (posix) or ~/pydistutils.cfg (non-posix) for
        scheme paths
    :param prefix: indicates to use the "prefix" scheme and provides the
        base directory for the same
    """
    scheme = distutils_scheme(
        dist_name, user, home, root, isolated, prefix
    )
    return Scheme(
        platlib=scheme["platlib"],
        purelib=scheme["purelib"],
        headers=scheme["headers"],
        scripts=scheme["scripts"],
        data=scheme["data"],
    )
