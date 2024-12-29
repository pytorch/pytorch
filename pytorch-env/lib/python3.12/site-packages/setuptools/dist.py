from __future__ import annotations

import io
import itertools
import numbers
import os
import re
import sys
from glob import iglob
from pathlib import Path
from typing import TYPE_CHECKING, MutableMapping

import distutils.cmd
import distutils.command
import distutils.core
import distutils.dist
import distutils.log
from distutils.debug import DEBUG
from distutils.errors import DistutilsOptionError, DistutilsSetupError
from distutils.fancy_getopt import translate_longopt
from distutils.util import strtobool

from more_itertools import partition, unique_everseen
from ordered_set import OrderedSet
from packaging.markers import InvalidMarker, Marker
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import Version

from . import _entry_points
from . import _reqs
from . import command as _  # noqa  -- imported for side-effects
from ._importlib import metadata
from .config import setupcfg, pyprojecttoml
from .discovery import ConfigDiscovery
from .monkey import get_unpatched
from .warnings import InformationOnly, SetuptoolsDeprecationWarning

__all__ = ['Distribution']

sequence = tuple, list


def check_importable(dist, attr, value):
    try:
        ep = metadata.EntryPoint(value=value, name=None, group=None)
        assert not ep.extras
    except (TypeError, ValueError, AttributeError, AssertionError) as e:
        raise DistutilsSetupError(
            "%r must be importable 'module:attrs' string (got %r)" % (attr, value)
        ) from e


def assert_string_list(dist, attr, value):
    """Verify that value is a string list"""
    try:
        # verify that value is a list or tuple to exclude unordered
        # or single-use iterables
        assert isinstance(value, (list, tuple))
        # verify that elements of value are strings
        assert ''.join(value) != value
    except (TypeError, ValueError, AttributeError, AssertionError) as e:
        raise DistutilsSetupError(
            "%r must be a list of strings (got %r)" % (attr, value)
        ) from e


def check_nsp(dist, attr, value):
    """Verify that namespace packages are valid"""
    ns_packages = value
    assert_string_list(dist, attr, ns_packages)
    for nsp in ns_packages:
        if not dist.has_contents_for(nsp):
            raise DistutilsSetupError(
                "Distribution contains no modules or packages for "
                + "namespace package %r" % nsp
            )
        parent, sep, child = nsp.rpartition('.')
        if parent and parent not in ns_packages:
            distutils.log.warn(
                "WARNING: %r is declared as a package namespace, but %r"
                " is not: please correct this in setup.py",
                nsp,
                parent,
            )
        SetuptoolsDeprecationWarning.emit(
            "The namespace_packages parameter is deprecated.",
            "Please replace its usage with implicit namespaces (PEP 420).",
            see_docs="references/keywords.html#keyword-namespace-packages",
            # TODO: define due_date, it may break old packages that are no longer
            # maintained (e.g. sphinxcontrib extensions) when installed from source.
            # Warning officially introduced in May 2022, however the deprecation
            # was mentioned much earlier in the docs (May 2020, see #2149).
        )


def check_extras(dist, attr, value):
    """Verify that extras_require mapping is valid"""
    try:
        list(itertools.starmap(_check_extra, value.items()))
    except (TypeError, ValueError, AttributeError) as e:
        raise DistutilsSetupError(
            "'extras_require' must be a dictionary whose values are "
            "strings or lists of strings containing valid project/version "
            "requirement specifiers."
        ) from e


def _check_extra(extra, reqs):
    name, sep, marker = extra.partition(':')
    try:
        _check_marker(marker)
    except InvalidMarker:
        msg = f"Invalid environment marker: {marker} ({extra!r})"
        raise DistutilsSetupError(msg) from None
    list(_reqs.parse(reqs))


def _check_marker(marker):
    if not marker:
        return
    m = Marker(marker)
    m.evaluate()


def assert_bool(dist, attr, value):
    """Verify that value is True, False, 0, or 1"""
    if bool(value) != value:
        tmpl = "{attr!r} must be a boolean value (got {value!r})"
        raise DistutilsSetupError(tmpl.format(attr=attr, value=value))


def invalid_unless_false(dist, attr, value):
    if not value:
        DistDeprecationWarning.emit(f"{attr} is ignored.")
        # TODO: should there be a `due_date` here?
        return
    raise DistutilsSetupError(f"{attr} is invalid.")


def check_requirements(dist, attr, value):
    """Verify that install_requires is a valid requirements list"""
    try:
        list(_reqs.parse(value))
        if isinstance(value, (dict, set)):
            raise TypeError("Unordered types are not allowed")
    except (TypeError, ValueError) as error:
        tmpl = (
            "{attr!r} must be a string or list of strings "
            "containing valid project/version requirement specifiers; {error}"
        )
        raise DistutilsSetupError(tmpl.format(attr=attr, error=error)) from error


def check_specifier(dist, attr, value):
    """Verify that value is a valid version specifier"""
    try:
        SpecifierSet(value)
    except (InvalidSpecifier, AttributeError) as error:
        tmpl = "{attr!r} must be a string containing valid version specifiers; {error}"
        raise DistutilsSetupError(tmpl.format(attr=attr, error=error)) from error


def check_entry_points(dist, attr, value):
    """Verify that entry_points map is parseable"""
    try:
        _entry_points.load(value)
    except Exception as e:
        raise DistutilsSetupError(e) from e


def check_package_data(dist, attr, value):
    """Verify that value is a dictionary of package names to glob lists"""
    if not isinstance(value, dict):
        raise DistutilsSetupError(
            "{!r} must be a dictionary mapping package names to lists of "
            "string wildcard patterns".format(attr)
        )
    for k, v in value.items():
        if not isinstance(k, str):
            raise DistutilsSetupError(
                "keys of {!r} dict must be strings (got {!r})".format(attr, k)
            )
        assert_string_list(dist, 'values of {!r} dict'.format(attr), v)


def check_packages(dist, attr, value):
    for pkgname in value:
        if not re.match(r'\w+(\.\w+)*', pkgname):
            distutils.log.warn(
                "WARNING: %r not a valid package name; please use only "
                ".-separated package names in setup.py",
                pkgname,
            )


if TYPE_CHECKING:
    # Work around a mypy issue where type[T] can't be used as a base: https://github.com/python/mypy/issues/10962
    _Distribution = distutils.core.Distribution
else:
    _Distribution = get_unpatched(distutils.core.Distribution)


class Distribution(_Distribution):
    """Distribution with support for tests and package data

    This is an enhanced version of 'distutils.dist.Distribution' that
    effectively adds the following new optional keyword arguments to 'setup()':

     'install_requires' -- a string or sequence of strings specifying project
        versions that the distribution requires when installed, in the format
        used by 'pkg_resources.require()'.  They will be installed
        automatically when the package is installed.  If you wish to use
        packages that are not available in PyPI, or want to give your users an
        alternate download location, you can add a 'find_links' option to the
        '[easy_install]' section of your project's 'setup.cfg' file, and then
        setuptools will scan the listed web pages for links that satisfy the
        requirements.

     'extras_require' -- a dictionary mapping names of optional "extras" to the
        additional requirement(s) that using those extras incurs. For example,
        this::

            extras_require = dict(reST = ["docutils>=0.3", "reSTedit"])

        indicates that the distribution can optionally provide an extra
        capability called "reST", but it can only be used if docutils and
        reSTedit are installed.  If the user installs your package using
        EasyInstall and requests one of your extras, the corresponding
        additional requirements will be installed if needed.

     'package_data' -- a dictionary mapping package names to lists of filenames
        or globs to use to find data files contained in the named packages.
        If the dictionary has filenames or globs listed under '""' (the empty
        string), those names will be searched for in every package, in addition
        to any names for the specific package.  Data files found using these
        names/globs will be installed along with the package, in the same
        location as the package.  Note that globs are allowed to reference
        the contents of non-package subdirectories, as long as you use '/' as
        a path separator.  (Globs are automatically converted to
        platform-specific paths at runtime.)

    In addition to these new keywords, this class also has several new methods
    for manipulating the distribution's contents.  For example, the 'include()'
    and 'exclude()' methods can be thought of as in-place add and subtract
    commands that add or remove packages, modules, extensions, and so on from
    the distribution.
    """

    _DISTUTILS_UNSUPPORTED_METADATA = {
        'long_description_content_type': lambda: None,
        'project_urls': dict,
        'provides_extras': OrderedSet,
        'license_file': lambda: None,
        'license_files': lambda: None,
        'install_requires': list,
        'extras_require': dict,
    }

    # Used by build_py, editable_wheel and install_lib commands for legacy namespaces
    namespace_packages: list[str]  #: :meta private: DEPRECATED

    def __init__(self, attrs: MutableMapping | None = None) -> None:
        have_package_data = hasattr(self, "package_data")
        if not have_package_data:
            self.package_data: dict[str, list[str]] = {}
        attrs = attrs or {}
        self.dist_files: list[tuple[str, str, str]] = []
        # Filter-out setuptools' specific options.
        self.src_root = attrs.pop("src_root", None)
        self.dependency_links = attrs.pop('dependency_links', [])
        self.setup_requires = attrs.pop('setup_requires', [])
        for ep in metadata.entry_points(group='distutils.setup_keywords'):
            vars(self).setdefault(ep.name, None)

        metadata_only = set(self._DISTUTILS_UNSUPPORTED_METADATA)
        metadata_only -= {"install_requires", "extras_require"}
        dist_attrs = {k: v for k, v in attrs.items() if k not in metadata_only}
        _Distribution.__init__(self, dist_attrs)

        # Private API (setuptools-use only, not restricted to Distribution)
        # Stores files that are referenced by the configuration and need to be in the
        # sdist (e.g. `version = file: VERSION.txt`)
        self._referenced_files: set[str] = set()

        self.set_defaults = ConfigDiscovery(self)

        self._set_metadata_defaults(attrs)

        self.metadata.version = self._normalize_version(self.metadata.version)
        self._finalize_requires()

    def _validate_metadata(self):
        required = {"name"}
        provided = {
            key
            for key in vars(self.metadata)
            if getattr(self.metadata, key, None) is not None
        }
        missing = required - provided

        if missing:
            msg = f"Required package metadata is missing: {missing}"
            raise DistutilsSetupError(msg)

    def _set_metadata_defaults(self, attrs):
        """
        Fill-in missing metadata fields not supported by distutils.
        Some fields may have been set by other tools (e.g. pbr).
        Those fields (vars(self.metadata)) take precedence to
        supplied attrs.
        """
        for option, default in self._DISTUTILS_UNSUPPORTED_METADATA.items():
            vars(self.metadata).setdefault(option, attrs.get(option, default()))

    @staticmethod
    def _normalize_version(version):
        from . import sic

        if isinstance(version, numbers.Number):
            # Some people apparently take "version number" too literally :)
            version = str(version)
        elif isinstance(version, sic) or version is None:
            return version

        normalized = str(Version(version))
        if version != normalized:
            InformationOnly.emit(f"Normalizing '{version}' to '{normalized}'")
            return normalized
        return version

    def _finalize_requires(self):
        """
        Set `metadata.python_requires` and fix environment markers
        in `install_requires` and `extras_require`.
        """
        if getattr(self, 'python_requires', None):
            self.metadata.python_requires = self.python_requires

        self._normalize_requires()
        self.metadata.install_requires = self.install_requires
        self.metadata.extras_require = self.extras_require

        if self.extras_require:
            for extra in self.extras_require.keys():
                # Setuptools allows a weird "<name>:<env markers> syntax for extras
                extra = extra.split(':')[0]
                if extra:
                    self.metadata.provides_extras.add(extra)

    def _normalize_requires(self):
        """Make sure requirement-related attributes exist and are normalized"""
        install_requires = getattr(self, "install_requires", None) or []
        extras_require = getattr(self, "extras_require", None) or {}
        self.install_requires = list(map(str, _reqs.parse(install_requires)))
        self.extras_require = {
            k: list(map(str, _reqs.parse(v or []))) for k, v in extras_require.items()
        }

    def _finalize_license_files(self) -> None:
        """Compute names of all license files which should be included."""
        license_files: list[str] | None = self.metadata.license_files
        patterns: list[str] = license_files if license_files else []

        license_file: str | None = self.metadata.license_file
        if license_file and license_file not in patterns:
            patterns.append(license_file)

        if license_files is None and license_file is None:
            # Default patterns match the ones wheel uses
            # See https://wheel.readthedocs.io/en/stable/user_guide.html
            # -> 'Including license files in the generated wheel file'
            patterns = ['LICEN[CS]E*', 'COPYING*', 'NOTICE*', 'AUTHORS*']

        self.metadata.license_files = list(
            unique_everseen(self._expand_patterns(patterns))
        )

    @staticmethod
    def _expand_patterns(patterns):
        """
        >>> list(Distribution._expand_patterns(['LICENSE']))
        ['LICENSE']
        >>> list(Distribution._expand_patterns(['pyproject.toml', 'LIC*']))
        ['pyproject.toml', 'LICENSE']
        """
        return (
            path
            for pattern in patterns
            for path in sorted(iglob(pattern))
            if not path.endswith('~') and os.path.isfile(path)
        )

    # FIXME: 'Distribution._parse_config_files' is too complex (14)
    def _parse_config_files(self, filenames=None):  # noqa: C901
        """
        Adapted from distutils.dist.Distribution.parse_config_files,
        this method provides the same functionality in subtly-improved
        ways.
        """
        from configparser import ConfigParser

        # Ignore install directory options if we have a venv
        ignore_options = (
            []
            if sys.prefix == sys.base_prefix
            else [
                'install-base',
                'install-platbase',
                'install-lib',
                'install-platlib',
                'install-purelib',
                'install-headers',
                'install-scripts',
                'install-data',
                'prefix',
                'exec-prefix',
                'home',
                'user',
                'root',
            ]
        )

        ignore_options = frozenset(ignore_options)

        if filenames is None:
            filenames = self.find_config_files()

        if DEBUG:
            self.announce("Distribution.parse_config_files():")

        parser = ConfigParser()
        parser.optionxform = str
        for filename in filenames:
            with open(filename, encoding='utf-8') as reader:
                if DEBUG:
                    self.announce("  reading {filename}".format(**locals()))
                parser.read_file(reader)
            for section in parser.sections():
                options = parser.options(section)
                opt_dict = self.get_option_dict(section)

                for opt in options:
                    if opt == '__name__' or opt in ignore_options:
                        continue

                    val = parser.get(section, opt)
                    opt = self.warn_dash_deprecation(opt, section)
                    opt = self.make_option_lowercase(opt, section)
                    opt_dict[opt] = (filename, val)

            # Make the ConfigParser forget everything (so we retain
            # the original filenames that options come from)
            parser.__init__()

        if 'global' not in self.command_options:
            return

        # If there was a "global" section in the config file, use it
        # to set Distribution options.

        for opt, (src, val) in self.command_options['global'].items():
            alias = self.negative_opt.get(opt)
            if alias:
                val = not strtobool(val)
            elif opt in ('verbose', 'dry_run'):  # ugh!
                val = strtobool(val)

            try:
                setattr(self, alias or opt, val)
            except ValueError as e:
                raise DistutilsOptionError(e) from e

    def warn_dash_deprecation(self, opt, section):
        if section in (
            'options.extras_require',
            'options.data_files',
        ):
            return opt

        underscore_opt = opt.replace('-', '_')
        commands = list(
            itertools.chain(
                distutils.command.__all__,
                self._setuptools_commands(),
            )
        )
        if (
            not section.startswith('options')
            and section != 'metadata'
            and section not in commands
        ):
            return underscore_opt

        if '-' in opt:
            SetuptoolsDeprecationWarning.emit(
                "Invalid dash-separated options",
                f"""
                Usage of dash-separated {opt!r} will not be supported in future
                versions. Please use the underscore name {underscore_opt!r} instead.
                """,
                see_docs="userguide/declarative_config.html",
                due_date=(2024, 9, 26),
                # Warning initially introduced in 3 Mar 2021
            )
        return underscore_opt

    def _setuptools_commands(self):
        try:
            entry_points = metadata.distribution('setuptools').entry_points
            return {ep.name for ep in entry_points}  # Avoid newer API for compatibility
        except metadata.PackageNotFoundError:
            # during bootstrapping, distribution doesn't exist
            return []

    def make_option_lowercase(self, opt, section):
        if section != 'metadata' or opt.islower():
            return opt

        lowercase_opt = opt.lower()
        SetuptoolsDeprecationWarning.emit(
            "Invalid uppercase configuration",
            f"""
            Usage of uppercase key {opt!r} in {section!r} will not be supported in
            future versions. Please use lowercase {lowercase_opt!r} instead.
            """,
            see_docs="userguide/declarative_config.html",
            due_date=(2024, 9, 26),
            # Warning initially introduced in 6 Mar 2021
        )
        return lowercase_opt

    # FIXME: 'Distribution._set_command_options' is too complex (14)
    def _set_command_options(self, command_obj, option_dict=None):  # noqa: C901
        """
        Set the options for 'command_obj' from 'option_dict'.  Basically
        this means copying elements of a dictionary ('option_dict') to
        attributes of an instance ('command').

        'command_obj' must be a Command instance.  If 'option_dict' is not
        supplied, uses the standard option dictionary for this command
        (from 'self.command_options').

        (Adopted from distutils.dist.Distribution._set_command_options)
        """
        command_name = command_obj.get_command_name()
        if option_dict is None:
            option_dict = self.get_option_dict(command_name)

        if DEBUG:
            self.announce("  setting options for '%s' command:" % command_name)
        for option, (source, value) in option_dict.items():
            if DEBUG:
                self.announce("    %s = %s (from %s)" % (option, value, source))
            try:
                bool_opts = [translate_longopt(o) for o in command_obj.boolean_options]
            except AttributeError:
                bool_opts = []
            try:
                neg_opt = command_obj.negative_opt
            except AttributeError:
                neg_opt = {}

            try:
                is_string = isinstance(value, str)
                if option in neg_opt and is_string:
                    setattr(command_obj, neg_opt[option], not strtobool(value))
                elif option in bool_opts and is_string:
                    setattr(command_obj, option, strtobool(value))
                elif hasattr(command_obj, option):
                    setattr(command_obj, option, value)
                else:
                    raise DistutilsOptionError(
                        "error in %s: command '%s' has no such option '%s'"
                        % (source, command_name, option)
                    )
            except ValueError as e:
                raise DistutilsOptionError(e) from e

    def _get_project_config_files(self, filenames):
        """Add default file and split between INI and TOML"""
        tomlfiles = []
        standard_project_metadata = Path(self.src_root or os.curdir, "pyproject.toml")
        if filenames is not None:
            parts = partition(lambda f: Path(f).suffix == ".toml", filenames)
            filenames = list(parts[0])  # 1st element => predicate is False
            tomlfiles = list(parts[1])  # 2nd element => predicate is True
        elif standard_project_metadata.exists():
            tomlfiles = [standard_project_metadata]
        return filenames, tomlfiles

    def parse_config_files(self, filenames=None, ignore_option_errors=False):
        """Parses configuration files from various levels
        and loads configuration.
        """
        inifiles, tomlfiles = self._get_project_config_files(filenames)

        self._parse_config_files(filenames=inifiles)

        setupcfg.parse_configuration(
            self, self.command_options, ignore_option_errors=ignore_option_errors
        )
        for filename in tomlfiles:
            pyprojecttoml.apply_configuration(self, filename, ignore_option_errors)

        self._finalize_requires()
        self._finalize_license_files()

    def fetch_build_eggs(self, requires):
        """Resolve pre-setup requirements"""
        from .installer import _fetch_build_eggs

        return _fetch_build_eggs(self, requires)

    def finalize_options(self):
        """
        Allow plugins to apply arbitrary operations to the
        distribution. Each hook may optionally define a 'order'
        to influence the order of execution. Smaller numbers
        go first and the default is 0.
        """
        group = 'setuptools.finalize_distribution_options'

        def by_order(hook):
            return getattr(hook, 'order', 0)

        defined = metadata.entry_points(group=group)
        filtered = itertools.filterfalse(self._removed, defined)
        loaded = map(lambda e: e.load(), filtered)
        for ep in sorted(loaded, key=by_order):
            ep(self)

    @staticmethod
    def _removed(ep):
        """
        When removing an entry point, if metadata is loaded
        from an older version of Setuptools, that removed
        entry point will attempt to be loaded and will fail.
        See #2765 for more details.
        """
        removed = {
            # removed 2021-09-05
            '2to3_doctests',
        }
        return ep.name in removed

    def _finalize_setup_keywords(self):
        for ep in metadata.entry_points(group='distutils.setup_keywords'):
            value = getattr(self, ep.name, None)
            if value is not None:
                ep.load()(self, ep.name, value)

    def get_egg_cache_dir(self):
        from . import windows_support

        egg_cache_dir = os.path.join(os.curdir, '.eggs')
        if not os.path.exists(egg_cache_dir):
            os.mkdir(egg_cache_dir)
            windows_support.hide_file(egg_cache_dir)
            readme_txt_filename = os.path.join(egg_cache_dir, 'README.txt')
            with open(readme_txt_filename, 'w', encoding="utf-8") as f:
                f.write(
                    'This directory contains eggs that were downloaded '
                    'by setuptools to build, test, and run plug-ins.\n\n'
                )
                f.write(
                    'This directory caches those eggs to prevent '
                    'repeated downloads.\n\n'
                )
                f.write('However, it is safe to delete this directory.\n\n')

        return egg_cache_dir

    def fetch_build_egg(self, req):
        """Fetch an egg needed for building"""
        from .installer import fetch_build_egg

        return fetch_build_egg(self, req)

    def get_command_class(self, command):
        """Pluggable version of get_command_class()"""
        if command in self.cmdclass:
            return self.cmdclass[command]

        # Special case bdist_wheel so it's never loaded from "wheel"
        if command == 'bdist_wheel':
            from .command.bdist_wheel import bdist_wheel

            return bdist_wheel

        eps = metadata.entry_points(group='distutils.commands', name=command)
        for ep in eps:
            self.cmdclass[command] = cmdclass = ep.load()
            return cmdclass
        else:
            return _Distribution.get_command_class(self, command)

    def print_commands(self):
        for ep in metadata.entry_points(group='distutils.commands'):
            if ep.name not in self.cmdclass:
                cmdclass = ep.load()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.print_commands(self)

    def get_command_list(self):
        for ep in metadata.entry_points(group='distutils.commands'):
            if ep.name not in self.cmdclass:
                cmdclass = ep.load()
                self.cmdclass[ep.name] = cmdclass
        return _Distribution.get_command_list(self)

    def include(self, **attrs):
        """Add items to distribution that are named in keyword arguments

        For example, 'dist.include(py_modules=["x"])' would add 'x' to
        the distribution's 'py_modules' attribute, if it was not already
        there.

        Currently, this method only supports inclusion for attributes that are
        lists or tuples.  If you need to add support for adding to other
        attributes in this or a subclass, you can add an '_include_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'include()'.  So, 'dist.include(foo={"bar":"baz"})'
        will try to call 'dist._include_foo({"bar":"baz"})', which can then
        handle whatever special inclusion logic is needed.
        """
        for k, v in attrs.items():
            include = getattr(self, '_include_' + k, None)
            if include:
                include(v)
            else:
                self._include_misc(k, v)

    def exclude_package(self, package):
        """Remove packages, modules, and extensions in named package"""

        pfx = package + '.'
        if self.packages:
            self.packages = [
                p for p in self.packages if p != package and not p.startswith(pfx)
            ]

        if self.py_modules:
            self.py_modules = [
                p for p in self.py_modules if p != package and not p.startswith(pfx)
            ]

        if self.ext_modules:
            self.ext_modules = [
                p
                for p in self.ext_modules
                if p.name != package and not p.name.startswith(pfx)
            ]

    def has_contents_for(self, package):
        """Return true if 'exclude_package(package)' would do something"""

        pfx = package + '.'

        for p in self.iter_distribution_names():
            if p == package or p.startswith(pfx):
                return True

        return False

    def _exclude_misc(self, name, value):
        """Handle 'exclude()' for list/tuple attrs without a special handler"""
        if not isinstance(value, sequence):
            raise DistutilsSetupError(
                "%s: setting must be a list or tuple (%r)" % (name, value)
            )
        try:
            old = getattr(self, name)
        except AttributeError as e:
            raise DistutilsSetupError("%s: No such distribution setting" % name) from e
        if old is not None and not isinstance(old, sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        elif old:
            setattr(self, name, [item for item in old if item not in value])

    def _include_misc(self, name, value):
        """Handle 'include()' for list/tuple attrs without a special handler"""

        if not isinstance(value, sequence):
            raise DistutilsSetupError("%s: setting must be a list (%r)" % (name, value))
        try:
            old = getattr(self, name)
        except AttributeError as e:
            raise DistutilsSetupError("%s: No such distribution setting" % name) from e
        if old is None:
            setattr(self, name, value)
        elif not isinstance(old, sequence):
            raise DistutilsSetupError(
                name + ": this setting cannot be changed via include/exclude"
            )
        else:
            new = [item for item in value if item not in old]
            setattr(self, name, old + new)

    def exclude(self, **attrs):
        """Remove items from distribution that are named in keyword arguments

        For example, 'dist.exclude(py_modules=["x"])' would remove 'x' from
        the distribution's 'py_modules' attribute.  Excluding packages uses
        the 'exclude_package()' method, so all of the package's contained
        packages, modules, and extensions are also excluded.

        Currently, this method only supports exclusion from attributes that are
        lists or tuples.  If you need to add support for excluding from other
        attributes in this or a subclass, you can add an '_exclude_X' method,
        where 'X' is the name of the attribute.  The method will be called with
        the value passed to 'exclude()'.  So, 'dist.exclude(foo={"bar":"baz"})'
        will try to call 'dist._exclude_foo({"bar":"baz"})', which can then
        handle whatever special exclusion logic is needed.
        """
        for k, v in attrs.items():
            exclude = getattr(self, '_exclude_' + k, None)
            if exclude:
                exclude(v)
            else:
                self._exclude_misc(k, v)

    def _exclude_packages(self, packages):
        if not isinstance(packages, sequence):
            raise DistutilsSetupError(
                "packages: setting must be a list or tuple (%r)" % (packages,)
            )
        list(map(self.exclude_package, packages))

    def _parse_command_opts(self, parser, args):
        # Remove --with-X/--without-X options when processing command args
        self.global_options = self.__class__.global_options
        self.negative_opt = self.__class__.negative_opt

        # First, expand any aliases
        command = args[0]
        aliases = self.get_option_dict('aliases')
        while command in aliases:
            src, alias = aliases[command]
            del aliases[command]  # ensure each alias can expand only once!
            import shlex

            args[:1] = shlex.split(alias, True)
            command = args[0]

        nargs = _Distribution._parse_command_opts(self, parser, args)

        # Handle commands that want to consume all remaining arguments
        cmd_class = self.get_command_class(command)
        if getattr(cmd_class, 'command_consumes_arguments', None):
            self.get_option_dict(command)['args'] = ("command line", nargs)
            if nargs is not None:
                return []

        return nargs

    def get_cmdline_options(self):
        """Return a '{cmd: {opt:val}}' map of all command-line options

        Option names are all long, but do not include the leading '--', and
        contain dashes rather than underscores.  If the option doesn't take
        an argument (e.g. '--quiet'), the 'val' is 'None'.

        Note that options provided by config files are intentionally excluded.
        """

        d = {}

        for cmd, opts in self.command_options.items():
            for opt, (src, val) in opts.items():
                if src != "command line":
                    continue

                opt = opt.replace('_', '-')

                if val == 0:
                    cmdobj = self.get_command_obj(cmd)
                    neg_opt = self.negative_opt.copy()
                    neg_opt.update(getattr(cmdobj, 'negative_opt', {}))
                    for neg, pos in neg_opt.items():
                        if pos == opt:
                            opt = neg
                            val = None
                            break
                    else:
                        raise AssertionError("Shouldn't be able to get here")

                elif val == 1:
                    val = None

                d.setdefault(cmd, {})[opt] = val

        return d

    def iter_distribution_names(self):
        """Yield all packages, modules, and extension names in distribution"""

        yield from self.packages or ()

        yield from self.py_modules or ()

        for ext in self.ext_modules or ():
            if isinstance(ext, tuple):
                name, buildinfo = ext
            else:
                name = ext.name
            if name.endswith('module'):
                name = name[:-6]
            yield name

    def handle_display_options(self, option_order):
        """If there were any non-global "display-only" options
        (--help-commands or the metadata display options) on the command
        line, display the requested info and return true; else return
        false.
        """
        import sys

        if self.help_commands:
            return _Distribution.handle_display_options(self, option_order)

        # Stdout may be StringIO (e.g. in tests)
        if not isinstance(sys.stdout, io.TextIOWrapper):
            return _Distribution.handle_display_options(self, option_order)

        # Don't wrap stdout if utf-8 is already the encoding. Provides
        #  workaround for #334.
        if sys.stdout.encoding.lower() in ('utf-8', 'utf8'):
            return _Distribution.handle_display_options(self, option_order)

        # Print metadata in UTF-8 no matter the platform
        encoding = sys.stdout.encoding
        sys.stdout.reconfigure(encoding='utf-8')
        try:
            return _Distribution.handle_display_options(self, option_order)
        finally:
            sys.stdout.reconfigure(encoding=encoding)

    def run_command(self, command):
        self.set_defaults()
        # Postpone defaults until all explicit configuration is considered
        # (setup() args, config files, command line and plugins)

        super().run_command(command)


class DistDeprecationWarning(SetuptoolsDeprecationWarning):
    """Class for warning about deprecations in dist in
    setuptools. Not ignored by default, unlike DeprecationWarning."""
