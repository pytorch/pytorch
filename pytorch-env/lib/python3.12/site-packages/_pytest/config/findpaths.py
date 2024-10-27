from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Iterable
from typing import Sequence

import iniconfig

from .exceptions import UsageError
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.pathlib import safe_exists


def _parse_ini_config(path: Path) -> iniconfig.IniConfig:
    """Parse the given generic '.ini' file using legacy IniConfig parser, returning
    the parsed object.

    Raise UsageError if the file cannot be parsed.
    """
    try:
        return iniconfig.IniConfig(str(path))
    except iniconfig.ParseError as exc:
        raise UsageError(str(exc)) from exc


def load_config_dict_from_file(
    filepath: Path,
) -> dict[str, str | list[str]] | None:
    """Load pytest configuration from the given file path, if supported.

    Return None if the file does not contain valid pytest configuration.
    """
    # Configuration from ini files are obtained from the [pytest] section, if present.
    if filepath.suffix == ".ini":
        iniconfig = _parse_ini_config(filepath)

        if "pytest" in iniconfig:
            return dict(iniconfig["pytest"].items())
        else:
            # "pytest.ini" files are always the source of configuration, even if empty.
            if filepath.name == "pytest.ini":
                return {}

    # '.cfg' files are considered if they contain a "[tool:pytest]" section.
    elif filepath.suffix == ".cfg":
        iniconfig = _parse_ini_config(filepath)

        if "tool:pytest" in iniconfig.sections:
            return dict(iniconfig["tool:pytest"].items())
        elif "pytest" in iniconfig.sections:
            # If a setup.cfg contains a "[pytest]" section, we raise a failure to indicate users that
            # plain "[pytest]" sections in setup.cfg files is no longer supported (#3086).
            fail(CFG_PYTEST_SECTION.format(filename="setup.cfg"), pytrace=False)

    # '.toml' files are considered if they contain a [tool.pytest.ini_options] table.
    elif filepath.suffix == ".toml":
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib

        toml_text = filepath.read_text(encoding="utf-8")
        try:
            config = tomllib.loads(toml_text)
        except tomllib.TOMLDecodeError as exc:
            raise UsageError(f"{filepath}: {exc}") from exc

        result = config.get("tool", {}).get("pytest", {}).get("ini_options", None)
        if result is not None:
            # TOML supports richer data types than ini files (strings, arrays, floats, ints, etc),
            # however we need to convert all scalar values to str for compatibility with the rest
            # of the configuration system, which expects strings only.
            def make_scalar(v: object) -> str | list[str]:
                return v if isinstance(v, list) else str(v)

            return {k: make_scalar(v) for k, v in result.items()}

    return None


def locate_config(
    invocation_dir: Path,
    args: Iterable[Path],
) -> tuple[Path | None, Path | None, dict[str, str | list[str]]]:
    """Search in the list of arguments for a valid ini-file for pytest,
    and return a tuple of (rootdir, inifile, cfg-dict)."""
    config_names = [
        "pytest.ini",
        ".pytest.ini",
        "pyproject.toml",
        "tox.ini",
        "setup.cfg",
    ]
    args = [x for x in args if not str(x).startswith("-")]
    if not args:
        args = [invocation_dir]
    found_pyproject_toml: Path | None = None
    for arg in args:
        argpath = absolutepath(arg)
        for base in (argpath, *argpath.parents):
            for config_name in config_names:
                p = base / config_name
                if p.is_file():
                    if p.name == "pyproject.toml" and found_pyproject_toml is None:
                        found_pyproject_toml = p
                    ini_config = load_config_dict_from_file(p)
                    if ini_config is not None:
                        return base, p, ini_config
    if found_pyproject_toml is not None:
        return found_pyproject_toml.parent, found_pyproject_toml, {}
    return None, None, {}


def get_common_ancestor(
    invocation_dir: Path,
    paths: Iterable[Path],
) -> Path:
    common_ancestor: Path | None = None
    for path in paths:
        if not path.exists():
            continue
        if common_ancestor is None:
            common_ancestor = path
        else:
            if common_ancestor in path.parents or path == common_ancestor:
                continue
            elif path in common_ancestor.parents:
                common_ancestor = path
            else:
                shared = commonpath(path, common_ancestor)
                if shared is not None:
                    common_ancestor = shared
    if common_ancestor is None:
        common_ancestor = invocation_dir
    elif common_ancestor.is_file():
        common_ancestor = common_ancestor.parent
    return common_ancestor


def get_dirs_from_args(args: Iterable[str]) -> list[Path]:
    def is_option(x: str) -> bool:
        return x.startswith("-")

    def get_file_part_from_node_id(x: str) -> str:
        return x.split("::")[0]

    def get_dir_from_path(path: Path) -> Path:
        if path.is_dir():
            return path
        return path.parent

    # These look like paths but may not exist
    possible_paths = (
        absolutepath(get_file_part_from_node_id(arg))
        for arg in args
        if not is_option(arg)
    )

    return [get_dir_from_path(path) for path in possible_paths if safe_exists(path)]


CFG_PYTEST_SECTION = "[pytest] section in {filename} files is no longer supported, change to [tool:pytest] instead."


def determine_setup(
    *,
    inifile: str | None,
    args: Sequence[str],
    rootdir_cmd_arg: str | None,
    invocation_dir: Path,
) -> tuple[Path, Path | None, dict[str, str | list[str]]]:
    """Determine the rootdir, inifile and ini configuration values from the
    command line arguments.

    :param inifile:
        The `--inifile` command line argument, if given.
    :param args:
        The free command line arguments.
    :param rootdir_cmd_arg:
        The `--rootdir` command line argument, if given.
    :param invocation_dir:
        The working directory when pytest was invoked.
    """
    rootdir = None
    dirs = get_dirs_from_args(args)
    if inifile:
        inipath_ = absolutepath(inifile)
        inipath: Path | None = inipath_
        inicfg = load_config_dict_from_file(inipath_) or {}
        if rootdir_cmd_arg is None:
            rootdir = inipath_.parent
    else:
        ancestor = get_common_ancestor(invocation_dir, dirs)
        rootdir, inipath, inicfg = locate_config(invocation_dir, [ancestor])
        if rootdir is None and rootdir_cmd_arg is None:
            for possible_rootdir in (ancestor, *ancestor.parents):
                if (possible_rootdir / "setup.py").is_file():
                    rootdir = possible_rootdir
                    break
            else:
                if dirs != [ancestor]:
                    rootdir, inipath, inicfg = locate_config(invocation_dir, dirs)
                if rootdir is None:
                    rootdir = get_common_ancestor(
                        invocation_dir, [invocation_dir, ancestor]
                    )
                    if is_fs_root(rootdir):
                        rootdir = ancestor
    if rootdir_cmd_arg:
        rootdir = absolutepath(os.path.expandvars(rootdir_cmd_arg))
        if not rootdir.is_dir():
            raise UsageError(
                f"Directory '{rootdir}' not found. Check your '--rootdir' option."
            )
    assert rootdir is not None
    return rootdir, inipath, inicfg or {}


def is_fs_root(p: Path) -> bool:
    r"""
    Return True if the given path is pointing to the root of the
    file system ("/" on Unix and "C:\\" on Windows for example).
    """
    return os.path.splitdrive(str(p))[1] == os.sep
