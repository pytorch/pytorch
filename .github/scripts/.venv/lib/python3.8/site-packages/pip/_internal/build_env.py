"""Build Environment used for isolation during sdist building
"""

import logging
import os
import sys
import textwrap
from collections import OrderedDict
from distutils.sysconfig import get_python_lib
from sysconfig import get_paths

from pip._vendor.pkg_resources import Requirement, VersionConflict, WorkingSet

from pip import __file__ as pip_location
from pip._internal.cli.spinners import open_spinner
from pip._internal.utils.subprocess import call_subprocess
from pip._internal.utils.temp_dir import TempDirectory, tempdir_kinds
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

if MYPY_CHECK_RUNNING:
    from types import TracebackType
    from typing import Tuple, Set, Iterable, Optional, List, Type
    from pip._internal.index.package_finder import PackageFinder

logger = logging.getLogger(__name__)


class _Prefix:

    def __init__(self, path):
        # type: (str) -> None
        self.path = path
        self.setup = False
        self.bin_dir = get_paths(
            'nt' if os.name == 'nt' else 'posix_prefix',
            vars={'base': path, 'platbase': path}
        )['scripts']
        # Note: prefer distutils' sysconfig to get the
        # library paths so PyPy is correctly supported.
        purelib = get_python_lib(plat_specific=False, prefix=path)
        platlib = get_python_lib(plat_specific=True, prefix=path)
        if purelib == platlib:
            self.lib_dirs = [purelib]
        else:
            self.lib_dirs = [purelib, platlib]


class BuildEnvironment(object):
    """Creates and manages an isolated environment to install build deps
    """

    def __init__(self):
        # type: () -> None
        temp_dir = TempDirectory(
            kind=tempdir_kinds.BUILD_ENV, globally_managed=True
        )

        self._prefixes = OrderedDict((
            (name, _Prefix(os.path.join(temp_dir.path, name)))
            for name in ('normal', 'overlay')
        ))

        self._bin_dirs = []  # type: List[str]
        self._lib_dirs = []  # type: List[str]
        for prefix in reversed(list(self._prefixes.values())):
            self._bin_dirs.append(prefix.bin_dir)
            self._lib_dirs.extend(prefix.lib_dirs)

        # Customize site to:
        # - ensure .pth files are honored
        # - prevent access to system site packages
        system_sites = {
            os.path.normcase(site) for site in (
                get_python_lib(plat_specific=False),
                get_python_lib(plat_specific=True),
            )
        }
        self._site_dir = os.path.join(temp_dir.path, 'site')
        if not os.path.exists(self._site_dir):
            os.mkdir(self._site_dir)
        with open(os.path.join(self._site_dir, 'sitecustomize.py'), 'w') as fp:
            fp.write(textwrap.dedent(
                '''
                import os, site, sys

                # First, drop system-sites related paths.
                original_sys_path = sys.path[:]
                known_paths = set()
                for path in {system_sites!r}:
                    site.addsitedir(path, known_paths=known_paths)
                system_paths = set(
                    os.path.normcase(path)
                    for path in sys.path[len(original_sys_path):]
                )
                original_sys_path = [
                    path for path in original_sys_path
                    if os.path.normcase(path) not in system_paths
                ]
                sys.path = original_sys_path

                # Second, add lib directories.
                # ensuring .pth file are processed.
                for path in {lib_dirs!r}:
                    assert not path in sys.path
                    site.addsitedir(path)
                '''
            ).format(system_sites=system_sites, lib_dirs=self._lib_dirs))

    def __enter__(self):
        # type: () -> None
        self._save_env = {
            name: os.environ.get(name, None)
            for name in ('PATH', 'PYTHONNOUSERSITE', 'PYTHONPATH')
        }

        path = self._bin_dirs[:]
        old_path = self._save_env['PATH']
        if old_path:
            path.extend(old_path.split(os.pathsep))

        pythonpath = [self._site_dir]

        os.environ.update({
            'PATH': os.pathsep.join(path),
            'PYTHONNOUSERSITE': '1',
            'PYTHONPATH': os.pathsep.join(pythonpath),
        })

    def __exit__(
        self,
        exc_type,  # type: Optional[Type[BaseException]]
        exc_val,  # type: Optional[BaseException]
        exc_tb  # type: Optional[TracebackType]
    ):
        # type: (...) -> None
        for varname, old_value in self._save_env.items():
            if old_value is None:
                os.environ.pop(varname, None)
            else:
                os.environ[varname] = old_value

    def check_requirements(self, reqs):
        # type: (Iterable[str]) -> Tuple[Set[Tuple[str, str]], Set[str]]
        """Return 2 sets:
            - conflicting requirements: set of (installed, wanted) reqs tuples
            - missing requirements: set of reqs
        """
        missing = set()
        conflicting = set()
        if reqs:
            ws = WorkingSet(self._lib_dirs)
            for req in reqs:
                try:
                    if ws.find(Requirement.parse(req)) is None:
                        missing.add(req)
                except VersionConflict as e:
                    conflicting.add((str(e.args[0].as_requirement()),
                                     str(e.args[1])))
        return conflicting, missing

    def install_requirements(
        self,
        finder,  # type: PackageFinder
        requirements,  # type: Iterable[str]
        prefix_as_string,  # type: str
        message  # type: str
    ):
        # type: (...) -> None
        prefix = self._prefixes[prefix_as_string]
        assert not prefix.setup
        prefix.setup = True
        if not requirements:
            return
        args = [
            sys.executable, os.path.dirname(pip_location), 'install',
            '--ignore-installed', '--no-user', '--prefix', prefix.path,
            '--no-warn-script-location',
        ]  # type: List[str]
        if logger.getEffectiveLevel() <= logging.DEBUG:
            args.append('-v')
        for format_control in ('no_binary', 'only_binary'):
            formats = getattr(finder.format_control, format_control)
            args.extend(('--' + format_control.replace('_', '-'),
                         ','.join(sorted(formats or {':none:'}))))

        index_urls = finder.index_urls
        if index_urls:
            args.extend(['-i', index_urls[0]])
            for extra_index in index_urls[1:]:
                args.extend(['--extra-index-url', extra_index])
        else:
            args.append('--no-index')
        for link in finder.find_links:
            args.extend(['--find-links', link])

        for host in finder.trusted_hosts:
            args.extend(['--trusted-host', host])
        if finder.allow_all_prereleases:
            args.append('--pre')
        if finder.prefer_binary:
            args.append('--prefer-binary')
        args.append('--')
        args.extend(requirements)
        with open_spinner(message) as spinner:
            call_subprocess(args, spinner=spinner)


class NoOpBuildEnvironment(BuildEnvironment):
    """A no-op drop-in replacement for BuildEnvironment
    """

    def __init__(self):
        # type: () -> None
        pass

    def __enter__(self):
        # type: () -> None
        pass

    def __exit__(
        self,
        exc_type,  # type: Optional[Type[BaseException]]
        exc_val,  # type: Optional[BaseException]
        exc_tb  # type: Optional[TracebackType]
    ):
        # type: (...) -> None
        pass

    def cleanup(self):
        # type: () -> None
        pass

    def install_requirements(
        self,
        finder,  # type: PackageFinder
        requirements,  # type: Iterable[str]
        prefix_as_string,  # type: str
        message  # type: str
    ):
        # type: (...) -> None
        raise NotImplementedError()
