# mypy: allow-untyped-defs
from __future__ import annotations

import argparse
from gettext import gettext
import os
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import final
from typing import List
from typing import Literal
from typing import Mapping
from typing import NoReturn
from typing import Sequence

import _pytest._io
from _pytest.config.exceptions import UsageError
from _pytest.deprecated import check_ispytest


FILE_OR_DIR = "file_or_dir"


class NotSet:
    def __repr__(self) -> str:
        return "<notset>"


NOT_SET = NotSet()


@final
class Parser:
    """Parser for command line arguments and ini-file values.

    :ivar extra_info: Dict of generic param -> value to display in case
        there's an error processing the command line arguments.
    """

    prog: str | None = None

    def __init__(
        self,
        usage: str | None = None,
        processopt: Callable[[Argument], None] | None = None,
        *,
        _ispytest: bool = False,
    ) -> None:
        check_ispytest(_ispytest)
        self._anonymous = OptionGroup("Custom options", parser=self, _ispytest=True)
        self._groups: list[OptionGroup] = []
        self._processopt = processopt
        self._usage = usage
        self._inidict: dict[str, tuple[str, str | None, Any]] = {}
        self._ininames: list[str] = []
        self.extra_info: dict[str, Any] = {}

    def processoption(self, option: Argument) -> None:
        if self._processopt:
            if option.dest:
                self._processopt(option)

    def getgroup(
        self, name: str, description: str = "", after: str | None = None
    ) -> OptionGroup:
        """Get (or create) a named option Group.

        :param name: Name of the option group.
        :param description: Long description for --help output.
        :param after: Name of another group, used for ordering --help output.
        :returns: The option group.

        The returned group object has an ``addoption`` method with the same
        signature as :func:`parser.addoption <pytest.Parser.addoption>` but
        will be shown in the respective group in the output of
        ``pytest --help``.
        """
        for group in self._groups:
            if group.name == name:
                return group
        group = OptionGroup(name, description, parser=self, _ispytest=True)
        i = 0
        for i, grp in enumerate(self._groups):
            if grp.name == after:
                break
        self._groups.insert(i + 1, group)
        return group

    def addoption(self, *opts: str, **attrs: Any) -> None:
        """Register a command line option.

        :param opts:
            Option names, can be short or long options.
        :param attrs:
            Same attributes as the argparse library's :meth:`add_argument()
            <argparse.ArgumentParser.add_argument>` function accepts.

        After command line parsing, options are available on the pytest config
        object via ``config.option.NAME`` where ``NAME`` is usually set
        by passing a ``dest`` attribute, for example
        ``addoption("--long", dest="NAME", ...)``.
        """
        self._anonymous.addoption(*opts, **attrs)

    def parse(
        self,
        args: Sequence[str | os.PathLike[str]],
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        from _pytest._argcomplete import try_argcomplete

        self.optparser = self._getparser()
        try_argcomplete(self.optparser)
        strargs = [os.fspath(x) for x in args]
        return self.optparser.parse_args(strargs, namespace=namespace)

    def _getparser(self) -> MyOptionParser:
        from _pytest._argcomplete import filescompleter

        optparser = MyOptionParser(self, self.extra_info, prog=self.prog)
        groups = [*self._groups, self._anonymous]
        for group in groups:
            if group.options:
                desc = group.description or group.name
                arggroup = optparser.add_argument_group(desc)
                for option in group.options:
                    n = option.names()
                    a = option.attrs()
                    arggroup.add_argument(*n, **a)
        file_or_dir_arg = optparser.add_argument(FILE_OR_DIR, nargs="*")
        # bash like autocompletion for dirs (appending '/')
        # Type ignored because typeshed doesn't know about argcomplete.
        file_or_dir_arg.completer = filescompleter  # type: ignore
        return optparser

    def parse_setoption(
        self,
        args: Sequence[str | os.PathLike[str]],
        option: argparse.Namespace,
        namespace: argparse.Namespace | None = None,
    ) -> list[str]:
        parsedoption = self.parse(args, namespace=namespace)
        for name, value in parsedoption.__dict__.items():
            setattr(option, name, value)
        return cast(List[str], getattr(parsedoption, FILE_OR_DIR))

    def parse_known_args(
        self,
        args: Sequence[str | os.PathLike[str]],
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Parse the known arguments at this point.

        :returns: An argparse namespace object.
        """
        return self.parse_known_and_unknown_args(args, namespace=namespace)[0]

    def parse_known_and_unknown_args(
        self,
        args: Sequence[str | os.PathLike[str]],
        namespace: argparse.Namespace | None = None,
    ) -> tuple[argparse.Namespace, list[str]]:
        """Parse the known arguments at this point, and also return the
        remaining unknown arguments.

        :returns:
            A tuple containing an argparse namespace object for the known
            arguments, and a list of the unknown arguments.
        """
        optparser = self._getparser()
        strargs = [os.fspath(x) for x in args]
        return optparser.parse_known_args(strargs, namespace=namespace)

    def addini(
        self,
        name: str,
        help: str,
        type: Literal["string", "paths", "pathlist", "args", "linelist", "bool"]
        | None = None,
        default: Any = NOT_SET,
    ) -> None:
        """Register an ini-file option.

        :param name:
            Name of the ini-variable.
        :param type:
            Type of the variable. Can be:

                * ``string``: a string
                * ``bool``: a boolean
                * ``args``: a list of strings, separated as in a shell
                * ``linelist``: a list of strings, separated by line breaks
                * ``paths``: a list of :class:`pathlib.Path`, separated as in a shell
                * ``pathlist``: a list of ``py.path``, separated as in a shell

            For ``paths`` and ``pathlist`` types, they are considered relative to the ini-file.
            In case the execution is happening without an ini-file defined,
            they will be considered relative to the current working directory (for example with ``--override-ini``).

            .. versionadded:: 7.0
                The ``paths`` variable type.

            .. versionadded:: 8.1
                Use the current working directory to resolve ``paths`` and ``pathlist`` in the absence of an ini-file.

            Defaults to ``string`` if ``None`` or not passed.
        :param default:
            Default value if no ini-file option exists but is queried.

        The value of ini-variables can be retrieved via a call to
        :py:func:`config.getini(name) <pytest.Config.getini>`.
        """
        assert type in (None, "string", "paths", "pathlist", "args", "linelist", "bool")
        if default is NOT_SET:
            default = get_ini_default_for_type(type)

        self._inidict[name] = (help, type, default)
        self._ininames.append(name)


def get_ini_default_for_type(
    type: Literal["string", "paths", "pathlist", "args", "linelist", "bool"] | None,
) -> Any:
    """
    Used by addini to get the default value for a given ini-option type, when
    default is not supplied.
    """
    if type is None:
        return ""
    elif type in ("paths", "pathlist", "args", "linelist"):
        return []
    elif type == "bool":
        return False
    else:
        return ""


class ArgumentError(Exception):
    """Raised if an Argument instance is created with invalid or
    inconsistent arguments."""

    def __init__(self, msg: str, option: Argument | str) -> None:
        self.msg = msg
        self.option_id = str(option)

    def __str__(self) -> str:
        if self.option_id:
            return f"option {self.option_id}: {self.msg}"
        else:
            return self.msg


class Argument:
    """Class that mimics the necessary behaviour of optparse.Option.

    It's currently a least effort implementation and ignoring choices
    and integer prefixes.

    https://docs.python.org/3/library/optparse.html#optparse-standard-option-types
    """

    def __init__(self, *names: str, **attrs: Any) -> None:
        """Store params in private vars for use in add_argument."""
        self._attrs = attrs
        self._short_opts: list[str] = []
        self._long_opts: list[str] = []
        try:
            self.type = attrs["type"]
        except KeyError:
            pass
        try:
            # Attribute existence is tested in Config._processopt.
            self.default = attrs["default"]
        except KeyError:
            pass
        self._set_opt_strings(names)
        dest: str | None = attrs.get("dest")
        if dest:
            self.dest = dest
        elif self._long_opts:
            self.dest = self._long_opts[0][2:].replace("-", "_")
        else:
            try:
                self.dest = self._short_opts[0][1:]
            except IndexError as e:
                self.dest = "???"  # Needed for the error repr.
                raise ArgumentError("need a long or short option", self) from e

    def names(self) -> list[str]:
        return self._short_opts + self._long_opts

    def attrs(self) -> Mapping[str, Any]:
        # Update any attributes set by processopt.
        attrs = "default dest help".split()
        attrs.append(self.dest)
        for attr in attrs:
            try:
                self._attrs[attr] = getattr(self, attr)
            except AttributeError:
                pass
        return self._attrs

    def _set_opt_strings(self, opts: Sequence[str]) -> None:
        """Directly from optparse.

        Might not be necessary as this is passed to argparse later on.
        """
        for opt in opts:
            if len(opt) < 2:
                raise ArgumentError(
                    f"invalid option string {opt!r}: "
                    "must be at least two characters long",
                    self,
                )
            elif len(opt) == 2:
                if not (opt[0] == "-" and opt[1] != "-"):
                    raise ArgumentError(
                        f"invalid short option string {opt!r}: "
                        "must be of the form -x, (x any non-dash char)",
                        self,
                    )
                self._short_opts.append(opt)
            else:
                if not (opt[0:2] == "--" and opt[2] != "-"):
                    raise ArgumentError(
                        f"invalid long option string {opt!r}: "
                        "must start with --, followed by non-dash",
                        self,
                    )
                self._long_opts.append(opt)

    def __repr__(self) -> str:
        args: list[str] = []
        if self._short_opts:
            args += ["_short_opts: " + repr(self._short_opts)]
        if self._long_opts:
            args += ["_long_opts: " + repr(self._long_opts)]
        args += ["dest: " + repr(self.dest)]
        if hasattr(self, "type"):
            args += ["type: " + repr(self.type)]
        if hasattr(self, "default"):
            args += ["default: " + repr(self.default)]
        return "Argument({})".format(", ".join(args))


class OptionGroup:
    """A group of options shown in its own section."""

    def __init__(
        self,
        name: str,
        description: str = "",
        parser: Parser | None = None,
        *,
        _ispytest: bool = False,
    ) -> None:
        check_ispytest(_ispytest)
        self.name = name
        self.description = description
        self.options: list[Argument] = []
        self.parser = parser

    def addoption(self, *opts: str, **attrs: Any) -> None:
        """Add an option to this group.

        If a shortened version of a long option is specified, it will
        be suppressed in the help. ``addoption('--twowords', '--two-words')``
        results in help showing ``--two-words`` only, but ``--twowords`` gets
        accepted **and** the automatic destination is in ``args.twowords``.

        :param opts:
            Option names, can be short or long options.
        :param attrs:
            Same attributes as the argparse library's :meth:`add_argument()
            <argparse.ArgumentParser.add_argument>` function accepts.
        """
        conflict = set(opts).intersection(
            name for opt in self.options for name in opt.names()
        )
        if conflict:
            raise ValueError(f"option names {conflict} already added")
        option = Argument(*opts, **attrs)
        self._addoption_instance(option, shortupper=False)

    def _addoption(self, *opts: str, **attrs: Any) -> None:
        option = Argument(*opts, **attrs)
        self._addoption_instance(option, shortupper=True)

    def _addoption_instance(self, option: Argument, shortupper: bool = False) -> None:
        if not shortupper:
            for opt in option._short_opts:
                if opt[0] == "-" and opt[1].islower():
                    raise ValueError("lowercase shortoptions reserved")
        if self.parser:
            self.parser.processoption(option)
        self.options.append(option)


class MyOptionParser(argparse.ArgumentParser):
    def __init__(
        self,
        parser: Parser,
        extra_info: dict[str, Any] | None = None,
        prog: str | None = None,
    ) -> None:
        self._parser = parser
        super().__init__(
            prog=prog,
            usage=parser._usage,
            add_help=False,
            formatter_class=DropShorterLongHelpFormatter,
            allow_abbrev=False,
            fromfile_prefix_chars="@",
        )
        # extra_info is a dict of (param -> value) to display if there's
        # an usage error to provide more contextual information to the user.
        self.extra_info = extra_info if extra_info else {}

    def error(self, message: str) -> NoReturn:
        """Transform argparse error message into UsageError."""
        msg = f"{self.prog}: error: {message}"

        if hasattr(self._parser, "_config_source_hint"):
            msg = f"{msg} ({self._parser._config_source_hint})"

        raise UsageError(self.format_usage() + msg)

    # Type ignored because typeshed has a very complex type in the superclass.
    def parse_args(  # type: ignore
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Allow splitting of positional arguments."""
        parsed, unrecognized = self.parse_known_args(args, namespace)
        if unrecognized:
            for arg in unrecognized:
                if arg and arg[0] == "-":
                    lines = [
                        "unrecognized arguments: {}".format(" ".join(unrecognized))
                    ]
                    for k, v in sorted(self.extra_info.items()):
                        lines.append(f"  {k}: {v}")
                    self.error("\n".join(lines))
            getattr(parsed, FILE_OR_DIR).extend(unrecognized)
        return parsed

    if sys.version_info < (3, 9):  # pragma: no cover
        # Backport of https://github.com/python/cpython/pull/14316 so we can
        # disable long --argument abbreviations without breaking short flags.
        def _parse_optional(
            self, arg_string: str
        ) -> tuple[argparse.Action | None, str, str | None] | None:
            if not arg_string:
                return None
            if arg_string[0] not in self.prefix_chars:
                return None
            if arg_string in self._option_string_actions:
                action = self._option_string_actions[arg_string]
                return action, arg_string, None
            if len(arg_string) == 1:
                return None
            if "=" in arg_string:
                option_string, explicit_arg = arg_string.split("=", 1)
                if option_string in self._option_string_actions:
                    action = self._option_string_actions[option_string]
                    return action, option_string, explicit_arg
            if self.allow_abbrev or not arg_string.startswith("--"):
                option_tuples = self._get_option_tuples(arg_string)
                if len(option_tuples) > 1:
                    msg = gettext(
                        "ambiguous option: %(option)s could match %(matches)s"
                    )
                    options = ", ".join(option for _, option, _ in option_tuples)
                    self.error(msg % {"option": arg_string, "matches": options})
                elif len(option_tuples) == 1:
                    (option_tuple,) = option_tuples
                    return option_tuple
            if self._negative_number_matcher.match(arg_string):
                if not self._has_negative_number_optionals:
                    return None
            if " " in arg_string:
                return None
            return None, arg_string, None


class DropShorterLongHelpFormatter(argparse.HelpFormatter):
    """Shorten help for long options that differ only in extra hyphens.

    - Collapse **long** options that are the same except for extra hyphens.
    - Shortcut if there are only two options and one of them is a short one.
    - Cache result on the action object as this is called at least 2 times.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Use more accurate terminal width.
        if "width" not in kwargs:
            kwargs["width"] = _pytest._io.get_terminal_width()
        super().__init__(*args, **kwargs)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        orgstr = super()._format_action_invocation(action)
        if orgstr and orgstr[0] != "-":  # only optional arguments
            return orgstr
        res: str | None = getattr(action, "_formatted_action_invocation", None)
        if res:
            return res
        options = orgstr.split(", ")
        if len(options) == 2 and (len(options[0]) == 2 or len(options[1]) == 2):
            # a shortcut for '-h, --help' or '--abc', '-a'
            action._formatted_action_invocation = orgstr  # type: ignore
            return orgstr
        return_list = []
        short_long: dict[str, str] = {}
        for option in options:
            if len(option) == 2 or option[2] == " ":
                continue
            if not option.startswith("--"):
                raise ArgumentError(
                    f'long optional argument without "--": [{option}]', option
                )
            xxoption = option[2:]
            shortened = xxoption.replace("-", "")
            if shortened not in short_long or len(short_long[shortened]) < len(
                xxoption
            ):
                short_long[shortened] = xxoption
        # now short_long has been filled out to the longest with dashes
        # **and** we keep the right option ordering from add_argument
        for option in options:
            if len(option) == 2 or option[2] == " ":
                return_list.append(option)
            if option[2:] == short_long.get(option.replace("-", "")):
                return_list.append(option.replace(" ", "=", 1))
        formatted_action_invocation = ", ".join(return_list)
        action._formatted_action_invocation = formatted_action_invocation  # type: ignore
        return formatted_action_invocation

    def _split_lines(self, text, width):
        """Wrap lines after splitting on original newlines.

        This allows to have explicit line breaks in the help text.
        """
        import textwrap

        lines = []
        for line in text.splitlines():
            lines.extend(textwrap.wrap(line.strip(), width))
        return lines
