# mypy: allow-untyped-defs
"""Command line options, ini-file and conftest.py processing."""

from __future__ import annotations

import argparse
import collections.abc
import copy
import dataclasses
import enum
from functools import lru_cache
import glob
import importlib.metadata
import inspect
import os
import pathlib
import re
import shlex
import sys
from textwrap import dedent
import types
from types import FunctionType
from typing import Any
from typing import Callable
from typing import cast
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import TextIO
from typing import Type
from typing import TYPE_CHECKING
import warnings

import pluggy
from pluggy import HookimplMarker
from pluggy import HookimplOpts
from pluggy import HookspecMarker
from pluggy import HookspecOpts
from pluggy import PluginManager

from .compat import PathAwareHookProxy
from .exceptions import PrintHelp as PrintHelp
from .exceptions import UsageError as UsageError
from .findpaths import determine_setup
from _pytest import __version__
import _pytest._code
from _pytest._code import ExceptionInfo
from _pytest._code import filter_traceback
from _pytest._code.code import TracebackStyle
from _pytest._io import TerminalWriter
from _pytest.config.argparsing import Argument
from _pytest.config.argparsing import Parser
import _pytest.deprecated
import _pytest.hookspec
from _pytest.outcomes import fail
from _pytest.outcomes import Skipped
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportMode
from _pytest.pathlib import resolve_package_path
from _pytest.pathlib import safe_exists
from _pytest.stash import Stash
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import warn_explicit_for


if TYPE_CHECKING:
    from _pytest.cacheprovider import Cache
    from _pytest.terminal import TerminalReporter


_PluggyPlugin = object
"""A type to represent plugin objects.

Plugins can be any namespace, so we can't narrow it down much, but we use an
alias to make the intent clear.

Ideally this type would be provided by pluggy itself.
"""


hookimpl = HookimplMarker("pytest")
hookspec = HookspecMarker("pytest")


@final
class ExitCode(enum.IntEnum):
    """Encodes the valid exit codes by pytest.

    Currently users and plugins may supply other exit codes as well.

    .. versionadded:: 5.0
    """

    #: Tests passed.
    OK = 0
    #: Tests failed.
    TESTS_FAILED = 1
    #: pytest was interrupted.
    INTERRUPTED = 2
    #: An internal error got in the way.
    INTERNAL_ERROR = 3
    #: pytest was misused.
    USAGE_ERROR = 4
    #: pytest couldn't find tests.
    NO_TESTS_COLLECTED = 5


class ConftestImportFailure(Exception):
    def __init__(
        self,
        path: pathlib.Path,
        *,
        cause: Exception,
    ) -> None:
        self.path = path
        self.cause = cause

    def __str__(self) -> str:
        return f"{type(self.cause).__name__}: {self.cause} (from {self.path})"


def filter_traceback_for_conftest_import_failure(
    entry: _pytest._code.TracebackEntry,
) -> bool:
    """Filter tracebacks entries which point to pytest internals or importlib.

    Make a special case for importlib because we use it to import test modules and conftest files
    in _pytest.pathlib.import_path.
    """
    return filter_traceback(entry) and "importlib" not in str(entry.path).split(os.sep)


def main(
    args: list[str] | os.PathLike[str] | None = None,
    plugins: Sequence[str | _PluggyPlugin] | None = None,
) -> int | ExitCode:
    """Perform an in-process test run.

    :param args:
        List of command line arguments. If `None` or not given, defaults to reading
        arguments directly from the process command line (:data:`sys.argv`).
    :param plugins: List of plugin objects to be auto-registered during initialization.

    :returns: An exit code.
    """
    old_pytest_version = os.environ.get("PYTEST_VERSION")
    try:
        os.environ["PYTEST_VERSION"] = __version__
        try:
            config = _prepareconfig(args, plugins)
        except ConftestImportFailure as e:
            exc_info = ExceptionInfo.from_exception(e.cause)
            tw = TerminalWriter(sys.stderr)
            tw.line(f"ImportError while loading conftest '{e.path}'.", red=True)
            exc_info.traceback = exc_info.traceback.filter(
                filter_traceback_for_conftest_import_failure
            )
            exc_repr = (
                exc_info.getrepr(style="short", chain=False)
                if exc_info.traceback
                else exc_info.exconly()
            )
            formatted_tb = str(exc_repr)
            for line in formatted_tb.splitlines():
                tw.line(line.rstrip(), red=True)
            return ExitCode.USAGE_ERROR
        else:
            try:
                ret: ExitCode | int = config.hook.pytest_cmdline_main(config=config)
                try:
                    return ExitCode(ret)
                except ValueError:
                    return ret
            finally:
                config._ensure_unconfigure()
    except UsageError as e:
        tw = TerminalWriter(sys.stderr)
        for msg in e.args:
            tw.line(f"ERROR: {msg}\n", red=True)
        return ExitCode.USAGE_ERROR
    finally:
        if old_pytest_version is None:
            os.environ.pop("PYTEST_VERSION", None)
        else:
            os.environ["PYTEST_VERSION"] = old_pytest_version


def console_main() -> int:
    """The CLI entry point of pytest.

    This function is not meant for programmable use; use `main()` instead.
    """
    # https://docs.python.org/3/library/signal.html#note-on-sigpipe
    try:
        code = main()
        sys.stdout.flush()
        return code
    except BrokenPipeError:
        # Python flushes standard streams on exit; redirect remaining output
        # to devnull to avoid another BrokenPipeError at shutdown
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 1  # Python exits with error code 1 on EPIPE


class cmdline:  # compatibility namespace
    main = staticmethod(main)


def filename_arg(path: str, optname: str) -> str:
    """Argparse type validator for filename arguments.

    :path: Path of filename.
    :optname: Name of the option.
    """
    if os.path.isdir(path):
        raise UsageError(f"{optname} must be a filename, given: {path}")
    return path


def directory_arg(path: str, optname: str) -> str:
    """Argparse type validator for directory arguments.

    :path: Path of directory.
    :optname: Name of the option.
    """
    if not os.path.isdir(path):
        raise UsageError(f"{optname} must be a directory, given: {path}")
    return path


# Plugins that cannot be disabled via "-p no:X" currently.
essential_plugins = (
    "mark",
    "main",
    "runner",
    "fixtures",
    "helpconfig",  # Provides -p.
)

default_plugins = (
    *essential_plugins,
    "python",
    "terminal",
    "debugging",
    "unittest",
    "capture",
    "skipping",
    "legacypath",
    "tmpdir",
    "monkeypatch",
    "recwarn",
    "pastebin",
    "assertion",
    "junitxml",
    "doctest",
    "cacheprovider",
    "freeze_support",
    "setuponly",
    "setupplan",
    "stepwise",
    "warnings",
    "logging",
    "reports",
    "python_path",
    "unraisableexception",
    "threadexception",
    "faulthandler",
)

builtin_plugins = set(default_plugins)
builtin_plugins.add("pytester")
builtin_plugins.add("pytester_assertions")


def get_config(
    args: list[str] | None = None,
    plugins: Sequence[str | _PluggyPlugin] | None = None,
) -> Config:
    # subsequent calls to main will create a fresh instance
    pluginmanager = PytestPluginManager()
    config = Config(
        pluginmanager,
        invocation_params=Config.InvocationParams(
            args=args or (),
            plugins=plugins,
            dir=pathlib.Path.cwd(),
        ),
    )

    if args is not None:
        # Handle any "-p no:plugin" args.
        pluginmanager.consider_preparse(args, exclude_only=True)

    for spec in default_plugins:
        pluginmanager.import_plugin(spec)

    return config


def get_plugin_manager() -> PytestPluginManager:
    """Obtain a new instance of the
    :py:class:`pytest.PytestPluginManager`, with default plugins
    already loaded.

    This function can be used by integration with other tools, like hooking
    into pytest to run tests into an IDE.
    """
    return get_config().pluginmanager


def _prepareconfig(
    args: list[str] | os.PathLike[str] | None = None,
    plugins: Sequence[str | _PluggyPlugin] | None = None,
) -> Config:
    if args is None:
        args = sys.argv[1:]
    elif isinstance(args, os.PathLike):
        args = [os.fspath(args)]
    elif not isinstance(args, list):
        msg = (  # type:ignore[unreachable]
            "`args` parameter expected to be a list of strings, got: {!r} (type: {})"
        )
        raise TypeError(msg.format(args, type(args)))

    config = get_config(args, plugins)
    pluginmanager = config.pluginmanager
    try:
        if plugins:
            for plugin in plugins:
                if isinstance(plugin, str):
                    pluginmanager.consider_pluginarg(plugin)
                else:
                    pluginmanager.register(plugin)
        config = pluginmanager.hook.pytest_cmdline_parse(
            pluginmanager=pluginmanager, args=args
        )
        return config
    except BaseException:
        config._ensure_unconfigure()
        raise


def _get_directory(path: pathlib.Path) -> pathlib.Path:
    """Get the directory of a path - itself if already a directory."""
    if path.is_file():
        return path.parent
    else:
        return path


def _get_legacy_hook_marks(
    method: Any,
    hook_type: str,
    opt_names: tuple[str, ...],
) -> dict[str, bool]:
    if TYPE_CHECKING:
        # abuse typeguard from importlib to avoid massive method type union that's lacking an alias
        assert inspect.isroutine(method)
    known_marks: set[str] = {m.name for m in getattr(method, "pytestmark", [])}
    must_warn: list[str] = []
    opts: dict[str, bool] = {}
    for opt_name in opt_names:
        opt_attr = getattr(method, opt_name, AttributeError)
        if opt_attr is not AttributeError:
            must_warn.append(f"{opt_name}={opt_attr}")
            opts[opt_name] = True
        elif opt_name in known_marks:
            must_warn.append(f"{opt_name}=True")
            opts[opt_name] = True
        else:
            opts[opt_name] = False
    if must_warn:
        hook_opts = ", ".join(must_warn)
        message = _pytest.deprecated.HOOK_LEGACY_MARKING.format(
            type=hook_type,
            fullname=method.__qualname__,
            hook_opts=hook_opts,
        )
        warn_explicit_for(cast(FunctionType, method), message)
    return opts


@final
class PytestPluginManager(PluginManager):
    """A :py:class:`pluggy.PluginManager <pluggy.PluginManager>` with
    additional pytest-specific functionality:

    * Loading plugins from the command line, ``PYTEST_PLUGINS`` env variable and
      ``pytest_plugins`` global variables found in plugins being loaded.
    * ``conftest.py`` loading during start-up.
    """

    def __init__(self) -> None:
        import _pytest.assertion

        super().__init__("pytest")

        # -- State related to local conftest plugins.
        # All loaded conftest modules.
        self._conftest_plugins: set[types.ModuleType] = set()
        # All conftest modules applicable for a directory.
        # This includes the directory's own conftest modules as well
        # as those of its parent directories.
        self._dirpath2confmods: dict[pathlib.Path, list[types.ModuleType]] = {}
        # Cutoff directory above which conftests are no longer discovered.
        self._confcutdir: pathlib.Path | None = None
        # If set, conftest loading is skipped.
        self._noconftest = False

        # _getconftestmodules()'s call to _get_directory() causes a stat
        # storm when it's called potentially thousands of times in a test
        # session (#9478), often with the same path, so cache it.
        self._get_directory = lru_cache(256)(_get_directory)

        # plugins that were explicitly skipped with pytest.skip
        # list of (module name, skip reason)
        # previously we would issue a warning when a plugin was skipped, but
        # since we refactored warnings as first citizens of Config, they are
        # just stored here to be used later.
        self.skipped_plugins: list[tuple[str, str]] = []

        self.add_hookspecs(_pytest.hookspec)
        self.register(self)
        if os.environ.get("PYTEST_DEBUG"):
            err: IO[str] = sys.stderr
            encoding: str = getattr(err, "encoding", "utf8")
            try:
                err = open(
                    os.dup(err.fileno()),
                    mode=err.mode,
                    buffering=1,
                    encoding=encoding,
                )
            except Exception:
                pass
            self.trace.root.setwriter(err.write)
            self.enable_tracing()

        # Config._consider_importhook will set a real object if required.
        self.rewrite_hook = _pytest.assertion.DummyRewriteHook()
        # Used to know when we are importing conftests after the pytest_configure stage.
        self._configured = False

    def parse_hookimpl_opts(
        self, plugin: _PluggyPlugin, name: str
    ) -> HookimplOpts | None:
        """:meta private:"""
        # pytest hooks are always prefixed with "pytest_",
        # so we avoid accessing possibly non-readable attributes
        # (see issue #1073).
        if not name.startswith("pytest_"):
            return None
        # Ignore names which cannot be hooks.
        if name == "pytest_plugins":
            return None

        opts = super().parse_hookimpl_opts(plugin, name)
        if opts is not None:
            return opts

        method = getattr(plugin, name)
        # Consider only actual functions for hooks (#3775).
        if not inspect.isroutine(method):
            return None
        # Collect unmarked hooks as long as they have the `pytest_' prefix.
        return _get_legacy_hook_marks(  # type: ignore[return-value]
            method, "impl", ("tryfirst", "trylast", "optionalhook", "hookwrapper")
        )

    def parse_hookspec_opts(self, module_or_class, name: str) -> HookspecOpts | None:
        """:meta private:"""
        opts = super().parse_hookspec_opts(module_or_class, name)
        if opts is None:
            method = getattr(module_or_class, name)
            if name.startswith("pytest_"):
                opts = _get_legacy_hook_marks(  # type: ignore[assignment]
                    method,
                    "spec",
                    ("firstresult", "historic"),
                )
        return opts

    def register(self, plugin: _PluggyPlugin, name: str | None = None) -> str | None:
        if name in _pytest.deprecated.DEPRECATED_EXTERNAL_PLUGINS:
            warnings.warn(
                PytestConfigWarning(
                    "{} plugin has been merged into the core, "
                    "please remove it from your requirements.".format(
                        name.replace("_", "-")
                    )
                )
            )
            return None
        plugin_name = super().register(plugin, name)
        if plugin_name is not None:
            self.hook.pytest_plugin_registered.call_historic(
                kwargs=dict(
                    plugin=plugin,
                    plugin_name=plugin_name,
                    manager=self,
                )
            )

            if isinstance(plugin, types.ModuleType):
                self.consider_module(plugin)
        return plugin_name

    def getplugin(self, name: str):
        # Support deprecated naming because plugins (xdist e.g.) use it.
        plugin: _PluggyPlugin | None = self.get_plugin(name)
        return plugin

    def hasplugin(self, name: str) -> bool:
        """Return whether a plugin with the given name is registered."""
        return bool(self.get_plugin(name))

    def pytest_configure(self, config: Config) -> None:
        """:meta private:"""
        # XXX now that the pluginmanager exposes hookimpl(tryfirst...)
        # we should remove tryfirst/trylast as markers.
        config.addinivalue_line(
            "markers",
            "tryfirst: mark a hook implementation function such that the "
            "plugin machinery will try to call it first/as early as possible. "
            "DEPRECATED, use @pytest.hookimpl(tryfirst=True) instead.",
        )
        config.addinivalue_line(
            "markers",
            "trylast: mark a hook implementation function such that the "
            "plugin machinery will try to call it last/as late as possible. "
            "DEPRECATED, use @pytest.hookimpl(trylast=True) instead.",
        )
        self._configured = True

    #
    # Internal API for local conftest plugin handling.
    #
    def _set_initial_conftests(
        self,
        args: Sequence[str | pathlib.Path],
        pyargs: bool,
        noconftest: bool,
        rootpath: pathlib.Path,
        confcutdir: pathlib.Path | None,
        invocation_dir: pathlib.Path,
        importmode: ImportMode | str,
        *,
        consider_namespace_packages: bool,
    ) -> None:
        """Load initial conftest files given a preparsed "namespace".

        As conftest files may add their own command line options which have
        arguments ('--my-opt somepath') we might get some false positives.
        All builtin and 3rd party plugins will have been loaded, however, so
        common options will not confuse our logic here.
        """
        self._confcutdir = (
            absolutepath(invocation_dir / confcutdir) if confcutdir else None
        )
        self._noconftest = noconftest
        self._using_pyargs = pyargs
        foundanchor = False
        for initial_path in args:
            path = str(initial_path)
            # remove node-id syntax
            i = path.find("::")
            if i != -1:
                path = path[:i]
            anchor = absolutepath(invocation_dir / path)

            # Ensure we do not break if what appears to be an anchor
            # is in fact a very long option (#10169, #11394).
            if safe_exists(anchor):
                self._try_load_conftest(
                    anchor,
                    importmode,
                    rootpath,
                    consider_namespace_packages=consider_namespace_packages,
                )
                foundanchor = True
        if not foundanchor:
            self._try_load_conftest(
                invocation_dir,
                importmode,
                rootpath,
                consider_namespace_packages=consider_namespace_packages,
            )

    def _is_in_confcutdir(self, path: pathlib.Path) -> bool:
        """Whether to consider the given path to load conftests from."""
        if self._confcutdir is None:
            return True
        # The semantics here are literally:
        #   Do not load a conftest if it is found upwards from confcut dir.
        # But this is *not* the same as:
        #   Load only conftests from confcutdir or below.
        # At first glance they might seem the same thing, however we do support use cases where
        # we want to load conftests that are not found in confcutdir or below, but are found
        # in completely different directory hierarchies like packages installed
        # in out-of-source trees.
        # (see #9767 for a regression where the logic was inverted).
        return path not in self._confcutdir.parents

    def _try_load_conftest(
        self,
        anchor: pathlib.Path,
        importmode: str | ImportMode,
        rootpath: pathlib.Path,
        *,
        consider_namespace_packages: bool,
    ) -> None:
        self._loadconftestmodules(
            anchor,
            importmode,
            rootpath,
            consider_namespace_packages=consider_namespace_packages,
        )
        # let's also consider test* subdirs
        if anchor.is_dir():
            for x in anchor.glob("test*"):
                if x.is_dir():
                    self._loadconftestmodules(
                        x,
                        importmode,
                        rootpath,
                        consider_namespace_packages=consider_namespace_packages,
                    )

    def _loadconftestmodules(
        self,
        path: pathlib.Path,
        importmode: str | ImportMode,
        rootpath: pathlib.Path,
        *,
        consider_namespace_packages: bool,
    ) -> None:
        if self._noconftest:
            return

        directory = self._get_directory(path)

        # Optimization: avoid repeated searches in the same directory.
        # Assumes always called with same importmode and rootpath.
        if directory in self._dirpath2confmods:
            return

        clist = []
        for parent in reversed((directory, *directory.parents)):
            if self._is_in_confcutdir(parent):
                conftestpath = parent / "conftest.py"
                if conftestpath.is_file():
                    mod = self._importconftest(
                        conftestpath,
                        importmode,
                        rootpath,
                        consider_namespace_packages=consider_namespace_packages,
                    )
                    clist.append(mod)
        self._dirpath2confmods[directory] = clist

    def _getconftestmodules(self, path: pathlib.Path) -> Sequence[types.ModuleType]:
        directory = self._get_directory(path)
        return self._dirpath2confmods.get(directory, ())

    def _rget_with_confmod(
        self,
        name: str,
        path: pathlib.Path,
    ) -> tuple[types.ModuleType, Any]:
        modules = self._getconftestmodules(path)
        for mod in reversed(modules):
            try:
                return mod, getattr(mod, name)
            except AttributeError:
                continue
        raise KeyError(name)

    def _importconftest(
        self,
        conftestpath: pathlib.Path,
        importmode: str | ImportMode,
        rootpath: pathlib.Path,
        *,
        consider_namespace_packages: bool,
    ) -> types.ModuleType:
        conftestpath_plugin_name = str(conftestpath)
        existing = self.get_plugin(conftestpath_plugin_name)
        if existing is not None:
            return cast(types.ModuleType, existing)

        # conftest.py files there are not in a Python package all have module
        # name "conftest", and thus conflict with each other. Clear the existing
        # before loading the new one, otherwise the existing one will be
        # returned from the module cache.
        pkgpath = resolve_package_path(conftestpath)
        if pkgpath is None:
            try:
                del sys.modules[conftestpath.stem]
            except KeyError:
                pass

        try:
            mod = import_path(
                conftestpath,
                mode=importmode,
                root=rootpath,
                consider_namespace_packages=consider_namespace_packages,
            )
        except Exception as e:
            assert e.__traceback__ is not None
            raise ConftestImportFailure(conftestpath, cause=e) from e

        self._check_non_top_pytest_plugins(mod, conftestpath)

        self._conftest_plugins.add(mod)
        dirpath = conftestpath.parent
        if dirpath in self._dirpath2confmods:
            for path, mods in self._dirpath2confmods.items():
                if dirpath in path.parents or path == dirpath:
                    if mod in mods:
                        raise AssertionError(
                            f"While trying to load conftest path {conftestpath!s}, "
                            f"found that the module {mod} is already loaded with path {mod.__file__}. "
                            "This is not supposed to happen. Please report this issue to pytest."
                        )
                    mods.append(mod)
        self.trace(f"loading conftestmodule {mod!r}")
        self.consider_conftest(mod, registration_name=conftestpath_plugin_name)
        return mod

    def _check_non_top_pytest_plugins(
        self,
        mod: types.ModuleType,
        conftestpath: pathlib.Path,
    ) -> None:
        if (
            hasattr(mod, "pytest_plugins")
            and self._configured
            and not self._using_pyargs
        ):
            msg = (
                "Defining 'pytest_plugins' in a non-top-level conftest is no longer supported:\n"
                "It affects the entire test suite instead of just below the conftest as expected.\n"
                "  {}\n"
                "Please move it to a top level conftest file at the rootdir:\n"
                "  {}\n"
                "For more information, visit:\n"
                "  https://docs.pytest.org/en/stable/deprecations.html#pytest-plugins-in-non-top-level-conftest-files"
            )
            fail(msg.format(conftestpath, self._confcutdir), pytrace=False)

    #
    # API for bootstrapping plugin loading
    #
    #

    def consider_preparse(
        self, args: Sequence[str], *, exclude_only: bool = False
    ) -> None:
        """:meta private:"""
        i = 0
        n = len(args)
        while i < n:
            opt = args[i]
            i += 1
            if isinstance(opt, str):
                if opt == "-p":
                    try:
                        parg = args[i]
                    except IndexError:
                        return
                    i += 1
                elif opt.startswith("-p"):
                    parg = opt[2:]
                else:
                    continue
                parg = parg.strip()
                if exclude_only and not parg.startswith("no:"):
                    continue
                self.consider_pluginarg(parg)

    def consider_pluginarg(self, arg: str) -> None:
        """:meta private:"""
        if arg.startswith("no:"):
            name = arg[3:]
            if name in essential_plugins:
                raise UsageError(f"plugin {name} cannot be disabled")

            # PR #4304: remove stepwise if cacheprovider is blocked.
            if name == "cacheprovider":
                self.set_blocked("stepwise")
                self.set_blocked("pytest_stepwise")

            self.set_blocked(name)
            if not name.startswith("pytest_"):
                self.set_blocked("pytest_" + name)
        else:
            name = arg
            # Unblock the plugin.
            self.unblock(name)
            if not name.startswith("pytest_"):
                self.unblock("pytest_" + name)
            self.import_plugin(arg, consider_entry_points=True)

    def consider_conftest(
        self, conftestmodule: types.ModuleType, registration_name: str
    ) -> None:
        """:meta private:"""
        self.register(conftestmodule, name=registration_name)

    def consider_env(self) -> None:
        """:meta private:"""
        self._import_plugin_specs(os.environ.get("PYTEST_PLUGINS"))

    def consider_module(self, mod: types.ModuleType) -> None:
        """:meta private:"""
        self._import_plugin_specs(getattr(mod, "pytest_plugins", []))

    def _import_plugin_specs(
        self, spec: None | types.ModuleType | str | Sequence[str]
    ) -> None:
        plugins = _get_plugin_specs_as_list(spec)
        for import_spec in plugins:
            self.import_plugin(import_spec)

    def import_plugin(self, modname: str, consider_entry_points: bool = False) -> None:
        """Import a plugin with ``modname``.

        If ``consider_entry_points`` is True, entry point names are also
        considered to find a plugin.
        """
        # Most often modname refers to builtin modules, e.g. "pytester",
        # "terminal" or "capture".  Those plugins are registered under their
        # basename for historic purposes but must be imported with the
        # _pytest prefix.
        assert isinstance(
            modname, str
        ), f"module name as text required, got {modname!r}"
        if self.is_blocked(modname) or self.get_plugin(modname) is not None:
            return

        importspec = "_pytest." + modname if modname in builtin_plugins else modname
        self.rewrite_hook.mark_rewrite(importspec)

        if consider_entry_points:
            loaded = self.load_setuptools_entrypoints("pytest11", name=modname)
            if loaded:
                return

        try:
            __import__(importspec)
        except ImportError as e:
            raise ImportError(
                f'Error importing plugin "{modname}": {e.args[0]}'
            ).with_traceback(e.__traceback__) from e

        except Skipped as e:
            self.skipped_plugins.append((modname, e.msg or ""))
        else:
            mod = sys.modules[importspec]
            self.register(mod, modname)


def _get_plugin_specs_as_list(
    specs: None | types.ModuleType | str | Sequence[str],
) -> list[str]:
    """Parse a plugins specification into a list of plugin names."""
    # None means empty.
    if specs is None:
        return []
    # Workaround for #3899 - a submodule which happens to be called "pytest_plugins".
    if isinstance(specs, types.ModuleType):
        return []
    # Comma-separated list.
    if isinstance(specs, str):
        return specs.split(",") if specs else []
    # Direct specification.
    if isinstance(specs, collections.abc.Sequence):
        return list(specs)
    raise UsageError(
        f"Plugins may be specified as a sequence or a ','-separated string of plugin names. Got: {specs!r}"
    )


class Notset:
    def __repr__(self):
        return "<NOTSET>"


notset = Notset()


def _iter_rewritable_modules(package_files: Iterable[str]) -> Iterator[str]:
    """Given an iterable of file names in a source distribution, return the "names" that should
    be marked for assertion rewrite.

    For example the package "pytest_mock/__init__.py" should be added as "pytest_mock" in
    the assertion rewrite mechanism.

    This function has to deal with dist-info based distributions and egg based distributions
    (which are still very much in use for "editable" installs).

    Here are the file names as seen in a dist-info based distribution:

        pytest_mock/__init__.py
        pytest_mock/_version.py
        pytest_mock/plugin.py
        pytest_mock.egg-info/PKG-INFO

    Here are the file names as seen in an egg based distribution:

        src/pytest_mock/__init__.py
        src/pytest_mock/_version.py
        src/pytest_mock/plugin.py
        src/pytest_mock.egg-info/PKG-INFO
        LICENSE
        setup.py

    We have to take in account those two distribution flavors in order to determine which
    names should be considered for assertion rewriting.

    More information:
        https://github.com/pytest-dev/pytest-mock/issues/167
    """
    package_files = list(package_files)
    seen_some = False
    for fn in package_files:
        is_simple_module = "/" not in fn and fn.endswith(".py")
        is_package = fn.count("/") == 1 and fn.endswith("__init__.py")
        if is_simple_module:
            module_name, _ = os.path.splitext(fn)
            # we ignore "setup.py" at the root of the distribution
            # as well as editable installation finder modules made by setuptools
            if module_name != "setup" and not module_name.startswith("__editable__"):
                seen_some = True
                yield module_name
        elif is_package:
            package_name = os.path.dirname(fn)
            seen_some = True
            yield package_name

    if not seen_some:
        # At this point we did not find any packages or modules suitable for assertion
        # rewriting, so we try again by stripping the first path component (to account for
        # "src" based source trees for example).
        # This approach lets us have the common case continue to be fast, as egg-distributions
        # are rarer.
        new_package_files = []
        for fn in package_files:
            parts = fn.split("/")
            new_fn = "/".join(parts[1:])
            if new_fn:
                new_package_files.append(new_fn)
        if new_package_files:
            yield from _iter_rewritable_modules(new_package_files)


@final
class Config:
    """Access to configuration values, pluginmanager and plugin hooks.

    :param PytestPluginManager pluginmanager:
        A pytest PluginManager.

    :param InvocationParams invocation_params:
        Object containing parameters regarding the :func:`pytest.main`
        invocation.
    """

    @final
    @dataclasses.dataclass(frozen=True)
    class InvocationParams:
        """Holds parameters passed during :func:`pytest.main`.

        The object attributes are read-only.

        .. versionadded:: 5.1

        .. note::

            Note that the environment variable ``PYTEST_ADDOPTS`` and the ``addopts``
            ini option are handled by pytest, not being included in the ``args`` attribute.

            Plugins accessing ``InvocationParams`` must be aware of that.
        """

        args: tuple[str, ...]
        """The command-line arguments as passed to :func:`pytest.main`."""
        plugins: Sequence[str | _PluggyPlugin] | None
        """Extra plugins, might be `None`."""
        dir: pathlib.Path
        """The directory from which :func:`pytest.main` was invoked. :type: pathlib.Path"""

        def __init__(
            self,
            *,
            args: Iterable[str],
            plugins: Sequence[str | _PluggyPlugin] | None,
            dir: pathlib.Path,
        ) -> None:
            object.__setattr__(self, "args", tuple(args))
            object.__setattr__(self, "plugins", plugins)
            object.__setattr__(self, "dir", dir)

    class ArgsSource(enum.Enum):
        """Indicates the source of the test arguments.

        .. versionadded:: 7.2
        """

        #: Command line arguments.
        ARGS = enum.auto()
        #: Invocation directory.
        INVOCATION_DIR = enum.auto()
        INCOVATION_DIR = INVOCATION_DIR  # backwards compatibility alias
        #: 'testpaths' configuration value.
        TESTPATHS = enum.auto()

    # Set by cacheprovider plugin.
    cache: Cache

    def __init__(
        self,
        pluginmanager: PytestPluginManager,
        *,
        invocation_params: InvocationParams | None = None,
    ) -> None:
        from .argparsing import FILE_OR_DIR
        from .argparsing import Parser

        if invocation_params is None:
            invocation_params = self.InvocationParams(
                args=(), plugins=None, dir=pathlib.Path.cwd()
            )

        self.option = argparse.Namespace()
        """Access to command line option as attributes.

        :type: argparse.Namespace
        """

        self.invocation_params = invocation_params
        """The parameters with which pytest was invoked.

        :type: InvocationParams
        """

        _a = FILE_OR_DIR
        self._parser = Parser(
            usage=f"%(prog)s [options] [{_a}] [{_a}] [...]",
            processopt=self._processopt,
            _ispytest=True,
        )
        self.pluginmanager = pluginmanager
        """The plugin manager handles plugin registration and hook invocation.

        :type: PytestPluginManager
        """

        self.stash = Stash()
        """A place where plugins can store information on the config for their
        own use.

        :type: Stash
        """
        # Deprecated alias. Was never public. Can be removed in a few releases.
        self._store = self.stash

        self.trace = self.pluginmanager.trace.root.get("config")
        self.hook: pluggy.HookRelay = PathAwareHookProxy(self.pluginmanager.hook)  # type: ignore[assignment]
        self._inicache: dict[str, Any] = {}
        self._override_ini: Sequence[str] = ()
        self._opt2dest: dict[str, str] = {}
        self._cleanup: list[Callable[[], None]] = []
        self.pluginmanager.register(self, "pytestconfig")
        self._configured = False
        self.hook.pytest_addoption.call_historic(
            kwargs=dict(parser=self._parser, pluginmanager=self.pluginmanager)
        )
        self.args_source = Config.ArgsSource.ARGS
        self.args: list[str] = []

    @property
    def rootpath(self) -> pathlib.Path:
        """The path to the :ref:`rootdir <rootdir>`.

        :type: pathlib.Path

        .. versionadded:: 6.1
        """
        return self._rootpath

    @property
    def inipath(self) -> pathlib.Path | None:
        """The path to the :ref:`configfile <configfiles>`.

        .. versionadded:: 6.1
        """
        return self._inipath

    def add_cleanup(self, func: Callable[[], None]) -> None:
        """Add a function to be called when the config object gets out of
        use (usually coinciding with pytest_unconfigure)."""
        self._cleanup.append(func)

    def _do_configure(self) -> None:
        assert not self._configured
        self._configured = True
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            self.hook.pytest_configure.call_historic(kwargs=dict(config=self))

    def _ensure_unconfigure(self) -> None:
        if self._configured:
            self._configured = False
            self.hook.pytest_unconfigure(config=self)
            self.hook.pytest_configure._call_history = []
        while self._cleanup:
            fin = self._cleanup.pop()
            fin()

    def get_terminal_writer(self) -> TerminalWriter:
        terminalreporter: TerminalReporter | None = self.pluginmanager.get_plugin(
            "terminalreporter"
        )
        assert terminalreporter is not None
        return terminalreporter._tw

    def pytest_cmdline_parse(
        self, pluginmanager: PytestPluginManager, args: list[str]
    ) -> Config:
        try:
            self.parse(args)
        except UsageError:
            # Handle --version and --help here in a minimal fashion.
            # This gets done via helpconfig normally, but its
            # pytest_cmdline_main is not called in case of errors.
            if getattr(self.option, "version", False) or "--version" in args:
                from _pytest.helpconfig import showversion

                showversion(self)
            elif (
                getattr(self.option, "help", False) or "--help" in args or "-h" in args
            ):
                self._parser._getparser().print_help()
                sys.stdout.write(
                    "\nNOTE: displaying only minimal help due to UsageError.\n\n"
                )

            raise

        return self

    def notify_exception(
        self,
        excinfo: ExceptionInfo[BaseException],
        option: argparse.Namespace | None = None,
    ) -> None:
        if option and getattr(option, "fulltrace", False):
            style: TracebackStyle = "long"
        else:
            style = "native"
        excrepr = excinfo.getrepr(
            funcargs=True, showlocals=getattr(option, "showlocals", False), style=style
        )
        res = self.hook.pytest_internalerror(excrepr=excrepr, excinfo=excinfo)
        if not any(res):
            for line in str(excrepr).split("\n"):
                sys.stderr.write(f"INTERNALERROR> {line}\n")
                sys.stderr.flush()

    def cwd_relative_nodeid(self, nodeid: str) -> str:
        # nodeid's are relative to the rootpath, compute relative to cwd.
        if self.invocation_params.dir != self.rootpath:
            base_path_part, *nodeid_part = nodeid.split("::")
            # Only process path part
            fullpath = self.rootpath / base_path_part
            relative_path = bestrelpath(self.invocation_params.dir, fullpath)

            nodeid = "::".join([relative_path, *nodeid_part])
        return nodeid

    @classmethod
    def fromdictargs(cls, option_dict, args) -> Config:
        """Constructor usable for subprocesses."""
        config = get_config(args)
        config.option.__dict__.update(option_dict)
        config.parse(args, addopts=False)
        for x in config.option.plugins:
            config.pluginmanager.consider_pluginarg(x)
        return config

    def _processopt(self, opt: Argument) -> None:
        for name in opt._short_opts + opt._long_opts:
            self._opt2dest[name] = opt.dest

        if hasattr(opt, "default"):
            if not hasattr(self.option, opt.dest):
                setattr(self.option, opt.dest, opt.default)

    @hookimpl(trylast=True)
    def pytest_load_initial_conftests(self, early_config: Config) -> None:
        # We haven't fully parsed the command line arguments yet, so
        # early_config.args it not set yet. But we need it for
        # discovering the initial conftests. So "pre-run" the logic here.
        # It will be done for real in `parse()`.
        args, args_source = early_config._decide_args(
            args=early_config.known_args_namespace.file_or_dir,
            pyargs=early_config.known_args_namespace.pyargs,
            testpaths=early_config.getini("testpaths"),
            invocation_dir=early_config.invocation_params.dir,
            rootpath=early_config.rootpath,
            warn=False,
        )
        self.pluginmanager._set_initial_conftests(
            args=args,
            pyargs=early_config.known_args_namespace.pyargs,
            noconftest=early_config.known_args_namespace.noconftest,
            rootpath=early_config.rootpath,
            confcutdir=early_config.known_args_namespace.confcutdir,
            invocation_dir=early_config.invocation_params.dir,
            importmode=early_config.known_args_namespace.importmode,
            consider_namespace_packages=early_config.getini(
                "consider_namespace_packages"
            ),
        )

    def _initini(self, args: Sequence[str]) -> None:
        ns, unknown_args = self._parser.parse_known_and_unknown_args(
            args, namespace=copy.copy(self.option)
        )
        rootpath, inipath, inicfg = determine_setup(
            inifile=ns.inifilename,
            args=ns.file_or_dir + unknown_args,
            rootdir_cmd_arg=ns.rootdir or None,
            invocation_dir=self.invocation_params.dir,
        )
        self._rootpath = rootpath
        self._inipath = inipath
        self.inicfg = inicfg
        self._parser.extra_info["rootdir"] = str(self.rootpath)
        self._parser.extra_info["inifile"] = str(self.inipath)
        self._parser.addini("addopts", "Extra command line options", "args")
        self._parser.addini("minversion", "Minimally required pytest version")
        self._parser.addini(
            "required_plugins",
            "Plugins that must be present for pytest to run",
            type="args",
            default=[],
        )
        self._override_ini = ns.override_ini or ()

    def _consider_importhook(self, args: Sequence[str]) -> None:
        """Install the PEP 302 import hook if using assertion rewriting.

        Needs to parse the --assert=<mode> option from the commandline
        and find all the installed plugins to mark them for rewriting
        by the importhook.
        """
        ns, unknown_args = self._parser.parse_known_and_unknown_args(args)
        mode = getattr(ns, "assertmode", "plain")
        if mode == "rewrite":
            import _pytest.assertion

            try:
                hook = _pytest.assertion.install_importhook(self)
            except SystemError:
                mode = "plain"
            else:
                self._mark_plugins_for_rewrite(hook)
        self._warn_about_missing_assertion(mode)

    def _mark_plugins_for_rewrite(self, hook) -> None:
        """Given an importhook, mark for rewrite any top-level
        modules or packages in the distribution package for
        all pytest plugins."""
        self.pluginmanager.rewrite_hook = hook

        if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
            # We don't autoload from distribution package entry points,
            # no need to continue.
            return

        package_files = (
            str(file)
            for dist in importlib.metadata.distributions()
            if any(ep.group == "pytest11" for ep in dist.entry_points)
            for file in dist.files or []
        )

        for name in _iter_rewritable_modules(package_files):
            hook.mark_rewrite(name)

    def _validate_args(self, args: list[str], via: str) -> list[str]:
        """Validate known args."""
        self._parser._config_source_hint = via  # type: ignore
        try:
            self._parser.parse_known_and_unknown_args(
                args, namespace=copy.copy(self.option)
            )
        finally:
            del self._parser._config_source_hint  # type: ignore

        return args

    def _decide_args(
        self,
        *,
        args: list[str],
        pyargs: bool,
        testpaths: list[str],
        invocation_dir: pathlib.Path,
        rootpath: pathlib.Path,
        warn: bool,
    ) -> tuple[list[str], ArgsSource]:
        """Decide the args (initial paths/nodeids) to use given the relevant inputs.

        :param warn: Whether can issue warnings.

        :returns: The args and the args source. Guaranteed to be non-empty.
        """
        if args:
            source = Config.ArgsSource.ARGS
            result = args
        else:
            if invocation_dir == rootpath:
                source = Config.ArgsSource.TESTPATHS
                if pyargs:
                    result = testpaths
                else:
                    result = []
                    for path in testpaths:
                        result.extend(sorted(glob.iglob(path, recursive=True)))
                    if testpaths and not result:
                        if warn:
                            warning_text = (
                                "No files were found in testpaths; "
                                "consider removing or adjusting your testpaths configuration. "
                                "Searching recursively from the current directory instead."
                            )
                            self.issue_config_time_warning(
                                PytestConfigWarning(warning_text), stacklevel=3
                            )
            else:
                result = []
            if not result:
                source = Config.ArgsSource.INVOCATION_DIR
                result = [str(invocation_dir)]
        return result, source

    def _preparse(self, args: list[str], addopts: bool = True) -> None:
        if addopts:
            env_addopts = os.environ.get("PYTEST_ADDOPTS", "")
            if len(env_addopts):
                args[:] = (
                    self._validate_args(shlex.split(env_addopts), "via PYTEST_ADDOPTS")
                    + args
                )
        self._initini(args)
        if addopts:
            args[:] = (
                self._validate_args(self.getini("addopts"), "via addopts config") + args
            )

        self.known_args_namespace = self._parser.parse_known_args(
            args, namespace=copy.copy(self.option)
        )
        self._checkversion()
        self._consider_importhook(args)
        self.pluginmanager.consider_preparse(args, exclude_only=False)
        if not os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
            # Don't autoload from distribution package entry point. Only
            # explicitly specified plugins are going to be loaded.
            self.pluginmanager.load_setuptools_entrypoints("pytest11")
        self.pluginmanager.consider_env()

        self.known_args_namespace = self._parser.parse_known_args(
            args, namespace=copy.copy(self.known_args_namespace)
        )

        self._validate_plugins()
        self._warn_about_skipped_plugins()

        if self.known_args_namespace.confcutdir is None:
            if self.inipath is not None:
                confcutdir = str(self.inipath.parent)
            else:
                confcutdir = str(self.rootpath)
            self.known_args_namespace.confcutdir = confcutdir
        try:
            self.hook.pytest_load_initial_conftests(
                early_config=self, args=args, parser=self._parser
            )
        except ConftestImportFailure as e:
            if self.known_args_namespace.help or self.known_args_namespace.version:
                # we don't want to prevent --help/--version to work
                # so just let is pass and print a warning at the end
                self.issue_config_time_warning(
                    PytestConfigWarning(f"could not load initial conftests: {e.path}"),
                    stacklevel=2,
                )
            else:
                raise

    @hookimpl(wrapper=True)
    def pytest_collection(self) -> Generator[None, object, object]:
        # Validate invalid ini keys after collection is done so we take in account
        # options added by late-loading conftest files.
        try:
            return (yield)
        finally:
            self._validate_config_options()

    def _checkversion(self) -> None:
        import pytest

        minver = self.inicfg.get("minversion", None)
        if minver:
            # Imported lazily to improve start-up time.
            from packaging.version import Version

            if not isinstance(minver, str):
                raise pytest.UsageError(
                    f"{self.inipath}: 'minversion' must be a single value"
                )

            if Version(minver) > Version(pytest.__version__):
                raise pytest.UsageError(
                    f"{self.inipath}: 'minversion' requires pytest-{minver}, actual pytest-{pytest.__version__}'"
                )

    def _validate_config_options(self) -> None:
        for key in sorted(self._get_unknown_ini_keys()):
            self._warn_or_fail_if_strict(f"Unknown config option: {key}\n")

    def _validate_plugins(self) -> None:
        required_plugins = sorted(self.getini("required_plugins"))
        if not required_plugins:
            return

        # Imported lazily to improve start-up time.
        from packaging.requirements import InvalidRequirement
        from packaging.requirements import Requirement
        from packaging.version import Version

        plugin_info = self.pluginmanager.list_plugin_distinfo()
        plugin_dist_info = {dist.project_name: dist.version for _, dist in plugin_info}

        missing_plugins = []
        for required_plugin in required_plugins:
            try:
                req = Requirement(required_plugin)
            except InvalidRequirement:
                missing_plugins.append(required_plugin)
                continue

            if req.name not in plugin_dist_info:
                missing_plugins.append(required_plugin)
            elif not req.specifier.contains(
                Version(plugin_dist_info[req.name]), prereleases=True
            ):
                missing_plugins.append(required_plugin)

        if missing_plugins:
            raise UsageError(
                "Missing required plugins: {}".format(", ".join(missing_plugins)),
            )

    def _warn_or_fail_if_strict(self, message: str) -> None:
        if self.known_args_namespace.strict_config:
            raise UsageError(message)

        self.issue_config_time_warning(PytestConfigWarning(message), stacklevel=3)

    def _get_unknown_ini_keys(self) -> list[str]:
        parser_inicfg = self._parser._inidict
        return [name for name in self.inicfg if name not in parser_inicfg]

    def parse(self, args: list[str], addopts: bool = True) -> None:
        # Parse given cmdline arguments into this config object.
        assert (
            self.args == []
        ), "can only parse cmdline args at most once per Config object"
        self.hook.pytest_addhooks.call_historic(
            kwargs=dict(pluginmanager=self.pluginmanager)
        )
        self._preparse(args, addopts=addopts)
        self._parser.after_preparse = True  # type: ignore
        try:
            args = self._parser.parse_setoption(
                args, self.option, namespace=self.option
            )
            self.args, self.args_source = self._decide_args(
                args=args,
                pyargs=self.known_args_namespace.pyargs,
                testpaths=self.getini("testpaths"),
                invocation_dir=self.invocation_params.dir,
                rootpath=self.rootpath,
                warn=True,
            )
        except PrintHelp:
            pass

    def issue_config_time_warning(self, warning: Warning, stacklevel: int) -> None:
        """Issue and handle a warning during the "configure" stage.

        During ``pytest_configure`` we can't capture warnings using the ``catch_warnings_for_item``
        function because it is not possible to have hook wrappers around ``pytest_configure``.

        This function is mainly intended for plugins that need to issue warnings during
        ``pytest_configure`` (or similar stages).

        :param warning: The warning instance.
        :param stacklevel: stacklevel forwarded to warnings.warn.
        """
        if self.pluginmanager.is_blocked("warnings"):
            return

        cmdline_filters = self.known_args_namespace.pythonwarnings or []
        config_filters = self.getini("filterwarnings")

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always", type(warning))
            apply_warning_filters(config_filters, cmdline_filters)
            warnings.warn(warning, stacklevel=stacklevel)

        if records:
            frame = sys._getframe(stacklevel - 1)
            location = frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name
            self.hook.pytest_warning_recorded.call_historic(
                kwargs=dict(
                    warning_message=records[0],
                    when="config",
                    nodeid="",
                    location=location,
                )
            )

    def addinivalue_line(self, name: str, line: str) -> None:
        """Add a line to an ini-file option. The option must have been
        declared but might not yet be set in which case the line becomes
        the first line in its value."""
        x = self.getini(name)
        assert isinstance(x, list)
        x.append(line)  # modifies the cached list inline

    def getini(self, name: str):
        """Return configuration value from an :ref:`ini file <configfiles>`.

        If a configuration value is not defined in an
        :ref:`ini file <configfiles>`, then the ``default`` value provided while
        registering the configuration through
        :func:`parser.addini <pytest.Parser.addini>` will be returned.
        Please note that you can even provide ``None`` as a valid
        default value.

        If ``default`` is not provided while registering using
        :func:`parser.addini <pytest.Parser.addini>`, then a default value
        based on the ``type`` parameter passed to
        :func:`parser.addini <pytest.Parser.addini>` will be returned.
        The default values based on ``type`` are:
        ``paths``, ``pathlist``, ``args`` and ``linelist`` : empty list ``[]``
        ``bool`` : ``False``
        ``string`` : empty string ``""``

        If neither the ``default`` nor the ``type`` parameter is passed
        while registering the configuration through
        :func:`parser.addini <pytest.Parser.addini>`, then the configuration
        is treated as a string and a default empty string '' is returned.

        If the specified name hasn't been registered through a prior
        :func:`parser.addini <pytest.Parser.addini>` call (usually from a
        plugin), a ValueError is raised.
        """
        try:
            return self._inicache[name]
        except KeyError:
            self._inicache[name] = val = self._getini(name)
            return val

    # Meant for easy monkeypatching by legacypath plugin.
    # Can be inlined back (with no cover removed) once legacypath is gone.
    def _getini_unknown_type(self, name: str, type: str, value: str | list[str]):
        msg = f"unknown configuration type: {type}"
        raise ValueError(msg, value)  # pragma: no cover

    def _getini(self, name: str):
        try:
            description, type, default = self._parser._inidict[name]
        except KeyError as e:
            raise ValueError(f"unknown configuration value: {name!r}") from e
        override_value = self._get_override_ini_value(name)
        if override_value is None:
            try:
                value = self.inicfg[name]
            except KeyError:
                return default
        else:
            value = override_value
        # Coerce the values based on types.
        #
        # Note: some coercions are only required if we are reading from .ini files, because
        # the file format doesn't contain type information, but when reading from toml we will
        # get either str or list of str values (see _parse_ini_config_from_pyproject_toml).
        # For example:
        #
        #   ini:
        #     a_line_list = "tests acceptance"
        #   in this case, we need to split the string to obtain a list of strings.
        #
        #   toml:
        #     a_line_list = ["tests", "acceptance"]
        #   in this case, we already have a list ready to use.
        #
        if type == "paths":
            dp = (
                self.inipath.parent
                if self.inipath is not None
                else self.invocation_params.dir
            )
            input_values = shlex.split(value) if isinstance(value, str) else value
            return [dp / x for x in input_values]
        elif type == "args":
            return shlex.split(value) if isinstance(value, str) else value
        elif type == "linelist":
            if isinstance(value, str):
                return [t for t in map(lambda x: x.strip(), value.split("\n")) if t]
            else:
                return value
        elif type == "bool":
            return _strtobool(str(value).strip())
        elif type == "string":
            return value
        elif type is None:
            return value
        else:
            return self._getini_unknown_type(name, type, value)

    def _getconftest_pathlist(
        self, name: str, path: pathlib.Path
    ) -> list[pathlib.Path] | None:
        try:
            mod, relroots = self.pluginmanager._rget_with_confmod(name, path)
        except KeyError:
            return None
        assert mod.__file__ is not None
        modpath = pathlib.Path(mod.__file__).parent
        values: list[pathlib.Path] = []
        for relroot in relroots:
            if isinstance(relroot, os.PathLike):
                relroot = pathlib.Path(relroot)
            else:
                relroot = relroot.replace("/", os.sep)
                relroot = absolutepath(modpath / relroot)
            values.append(relroot)
        return values

    def _get_override_ini_value(self, name: str) -> str | None:
        value = None
        # override_ini is a list of "ini=value" options.
        # Always use the last item if multiple values are set for same ini-name,
        # e.g. -o foo=bar1 -o foo=bar2 will set foo to bar2.
        for ini_config in self._override_ini:
            try:
                key, user_ini_value = ini_config.split("=", 1)
            except ValueError as e:
                raise UsageError(
                    f"-o/--override-ini expects option=value style (got: {ini_config!r})."
                ) from e
            else:
                if key == name:
                    value = user_ini_value
        return value

    def getoption(self, name: str, default=notset, skip: bool = False):
        """Return command line option value.

        :param name: Name of the option.  You may also specify
            the literal ``--OPT`` option instead of the "dest" option name.
        :param default: Default value if no option of that name exists.
        :param skip: If True, raise pytest.skip if option does not exists
            or has a None value.
        """
        name = self._opt2dest.get(name, name)
        try:
            val = getattr(self.option, name)
            if val is None and skip:
                raise AttributeError(name)
            return val
        except AttributeError as e:
            if default is not notset:
                return default
            if skip:
                import pytest

                pytest.skip(f"no {name!r} option found")
            raise ValueError(f"no option named {name!r}") from e

    def getvalue(self, name: str, path=None):
        """Deprecated, use getoption() instead."""
        return self.getoption(name)

    def getvalueorskip(self, name: str, path=None):
        """Deprecated, use getoption(skip=True) instead."""
        return self.getoption(name, skip=True)

    #: Verbosity type for failed assertions (see :confval:`verbosity_assertions`).
    VERBOSITY_ASSERTIONS: Final = "assertions"
    #: Verbosity type for test case execution (see :confval:`verbosity_test_cases`).
    VERBOSITY_TEST_CASES: Final = "test_cases"
    _VERBOSITY_INI_DEFAULT: Final = "auto"

    def get_verbosity(self, verbosity_type: str | None = None) -> int:
        r"""Retrieve the verbosity level for a fine-grained verbosity type.

        :param verbosity_type: Verbosity type to get level for. If a level is
            configured for the given type, that value will be returned. If the
            given type is not a known verbosity type, the global verbosity
            level will be returned. If the given type is None (default), the
            global verbosity level will be returned.

        To configure a level for a fine-grained verbosity type, the
        configuration file should have a setting for the configuration name
        and a numeric value for the verbosity level. A special value of "auto"
        can be used to explicitly use the global verbosity level.

        Example:

        .. code-block:: ini

            # content of pytest.ini
            [pytest]
            verbosity_assertions = 2

        .. code-block:: console

            pytest -v

        .. code-block:: python

            print(config.get_verbosity())  # 1
            print(config.get_verbosity(Config.VERBOSITY_ASSERTIONS))  # 2
        """
        global_level = self.getoption("verbose", default=0)
        assert isinstance(global_level, int)
        if verbosity_type is None:
            return global_level

        ini_name = Config._verbosity_ini_name(verbosity_type)
        if ini_name not in self._parser._inidict:
            return global_level

        level = self.getini(ini_name)
        if level == Config._VERBOSITY_INI_DEFAULT:
            return global_level

        return int(level)

    @staticmethod
    def _verbosity_ini_name(verbosity_type: str) -> str:
        return f"verbosity_{verbosity_type}"

    @staticmethod
    def _add_verbosity_ini(parser: Parser, verbosity_type: str, help: str) -> None:
        """Add a output verbosity configuration option for the given output type.

        :param parser: Parser for command line arguments and ini-file values.
        :param verbosity_type: Fine-grained verbosity category.
        :param help: Description of the output this type controls.

        The value should be retrieved via a call to
        :py:func:`config.get_verbosity(type) <pytest.Config.get_verbosity>`.
        """
        parser.addini(
            Config._verbosity_ini_name(verbosity_type),
            help=help,
            type="string",
            default=Config._VERBOSITY_INI_DEFAULT,
        )

    def _warn_about_missing_assertion(self, mode: str) -> None:
        if not _assertion_supported():
            if mode == "plain":
                warning_text = (
                    "ASSERTIONS ARE NOT EXECUTED"
                    " and FAILING TESTS WILL PASS.  Are you"
                    " using python -O?"
                )
            else:
                warning_text = (
                    "assertions not in test modules or"
                    " plugins will be ignored"
                    " because assert statements are not executed "
                    "by the underlying Python interpreter "
                    "(are you using python -O?)\n"
                )
            self.issue_config_time_warning(
                PytestConfigWarning(warning_text),
                stacklevel=3,
            )

    def _warn_about_skipped_plugins(self) -> None:
        for module_name, msg in self.pluginmanager.skipped_plugins:
            self.issue_config_time_warning(
                PytestConfigWarning(f"skipped plugin {module_name!r}: {msg}"),
                stacklevel=2,
            )


def _assertion_supported() -> bool:
    try:
        assert False
    except AssertionError:
        return True
    else:
        return False  # type: ignore[unreachable]


def create_terminal_writer(
    config: Config, file: TextIO | None = None
) -> TerminalWriter:
    """Create a TerminalWriter instance configured according to the options
    in the config object.

    Every code which requires a TerminalWriter object and has access to a
    config object should use this function.
    """
    tw = TerminalWriter(file=file)

    if config.option.color == "yes":
        tw.hasmarkup = True
    elif config.option.color == "no":
        tw.hasmarkup = False

    if config.option.code_highlight == "yes":
        tw.code_highlight = True
    elif config.option.code_highlight == "no":
        tw.code_highlight = False

    return tw


def _strtobool(val: str) -> bool:
    """Convert a string representation of truth to True or False.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    .. note:: Copied from distutils.util.
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


@lru_cache(maxsize=50)
def parse_warning_filter(
    arg: str, *, escape: bool
) -> tuple[warnings._ActionKind, str, type[Warning], str, int]:
    """Parse a warnings filter string.

    This is copied from warnings._setoption with the following changes:

    * Does not apply the filter.
    * Escaping is optional.
    * Raises UsageError so we get nice error messages on failure.
    """
    __tracebackhide__ = True
    error_template = dedent(
        f"""\
        while parsing the following warning configuration:

          {arg}

        This error occurred:

        {{error}}
        """
    )

    parts = arg.split(":")
    if len(parts) > 5:
        doc_url = (
            "https://docs.python.org/3/library/warnings.html#describing-warning-filters"
        )
        error = dedent(
            f"""\
            Too many fields ({len(parts)}), expected at most 5 separated by colons:

              action:message:category:module:line

            For more information please consult: {doc_url}
            """
        )
        raise UsageError(error_template.format(error=error))

    while len(parts) < 5:
        parts.append("")
    action_, message, category_, module, lineno_ = (s.strip() for s in parts)
    try:
        action: warnings._ActionKind = warnings._getaction(action_)  # type: ignore[attr-defined]
    except warnings._OptionError as e:
        raise UsageError(error_template.format(error=str(e))) from None
    try:
        category: type[Warning] = _resolve_warning_category(category_)
    except Exception:
        exc_info = ExceptionInfo.from_current()
        exception_text = exc_info.getrepr(style="native")
        raise UsageError(error_template.format(error=exception_text)) from None
    if message and escape:
        message = re.escape(message)
    if module and escape:
        module = re.escape(module) + r"\Z"
    if lineno_:
        try:
            lineno = int(lineno_)
            if lineno < 0:
                raise ValueError("number is negative")
        except ValueError as e:
            raise UsageError(
                error_template.format(error=f"invalid lineno {lineno_!r}: {e}")
            ) from None
    else:
        lineno = 0
    return action, message, category, module, lineno


def _resolve_warning_category(category: str) -> type[Warning]:
    """
    Copied from warnings._getcategory, but changed so it lets exceptions (specially ImportErrors)
    propagate so we can get access to their tracebacks (#9218).
    """
    __tracebackhide__ = True
    if not category:
        return Warning

    if "." not in category:
        import builtins as m

        klass = category
    else:
        module, _, klass = category.rpartition(".")
        m = __import__(module, None, None, [klass])
    cat = getattr(m, klass)
    if not issubclass(cat, Warning):
        raise UsageError(f"{cat} is not a Warning subclass")
    return cast(Type[Warning], cat)


def apply_warning_filters(
    config_filters: Iterable[str], cmdline_filters: Iterable[str]
) -> None:
    """Applies pytest-configured filters to the warnings module"""
    # Filters should have this precedence: cmdline options, config.
    # Filters should be applied in the inverse order of precedence.
    for arg in config_filters:
        warnings.filterwarnings(*parse_warning_filter(arg, escape=False))

    for arg in cmdline_filters:
        warnings.filterwarnings(*parse_warning_filter(arg, escape=True))
