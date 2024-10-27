# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
.. _hypothesis-cli:

----------------
hypothesis[cli]
----------------

::

    $ hypothesis --help
    Usage: hypothesis [OPTIONS] COMMAND [ARGS]...

    Options:
      --version   Show the version and exit.
      -h, --help  Show this message and exit.

    Commands:
      codemod  `hypothesis codemod` refactors deprecated or inefficient code.
      fuzz     [hypofuzz] runs tests with an adaptive coverage-guided fuzzer.
      write    `hypothesis write` writes property-based tests for you!

This module requires the :pypi:`click` package, and provides Hypothesis' command-line
interface, for e.g. :doc:`'ghostwriting' tests <ghostwriter>` via the terminal.
It's also where `HypoFuzz <https://hypofuzz.com/>`__ adds the :command:`hypothesis fuzz`
command (`learn more about that here <https://hypofuzz.com/docs/quickstart.html>`__).
"""

import builtins
import importlib
import inspect
import sys
import types
from difflib import get_close_matches
from functools import partial
from multiprocessing import Pool
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

MESSAGE = """
The Hypothesis command-line interface requires the `{}` package,
which you do not have installed.  Run:

    python -m pip install --upgrade 'hypothesis[cli]'

and try again.
"""

try:
    import click
except ImportError:

    def main():
        """If `click` is not installed, tell the user to install it then exit."""
        sys.stderr.write(MESSAGE.format("click"))
        sys.exit(1)

else:
    # Ensure that Python scripts in the current working directory are importable,
    # on the principle that Ghostwriter should 'just work' for novice users.  Note
    # that we append rather than prepend to the module search path, so this will
    # never shadow the stdlib or installed packages.
    sys.path.append(".")

    @click.group(context_settings={"help_option_names": ("-h", "--help")})
    @click.version_option()
    def main():
        pass

    def obj_name(s: str) -> object:
        """This "type" imports whatever object is named by a dotted string."""
        s = s.strip()
        if "/" in s or "\\" in s:
            raise click.UsageError(
                "Remember that the ghostwriter should be passed the name of a module, not a path."
            ) from None
        try:
            return importlib.import_module(s)
        except ImportError:
            pass
        classname = None
        if "." not in s:
            modulename, module, funcname = "builtins", builtins, s
        else:
            modulename, funcname = s.rsplit(".", 1)
            try:
                module = importlib.import_module(modulename)
            except ImportError as err:
                try:
                    modulename, classname = modulename.rsplit(".", 1)
                    module = importlib.import_module(modulename)
                except (ImportError, ValueError):
                    if s.endswith(".py"):
                        raise click.UsageError(
                            "Remember that the ghostwriter should be passed the name of a module, not a file."
                        ) from None
                    raise click.UsageError(
                        f"Failed to import the {modulename} module for introspection.  "
                        "Check spelling and your Python import path, or use the Python API?"
                    ) from err

        def describe_close_matches(
            module_or_class: types.ModuleType, objname: str
        ) -> str:
            public_names = [
                name for name in vars(module_or_class) if not name.startswith("_")
            ]
            matches = get_close_matches(objname, public_names)
            if matches:
                return f"  Closest matches: {matches!r}"
            else:
                return ""

        if classname is None:
            try:
                return getattr(module, funcname)
            except AttributeError as err:
                if funcname == "py":
                    # Likely attempted to pass a local file (Eg., "myscript.py") instead of a module name
                    raise click.UsageError(
                        "Remember that the ghostwriter should be passed the name of a module, not a file."
                        f"\n\tTry: hypothesis write {s[:-3]}"
                    ) from None
                raise click.UsageError(
                    f"Found the {modulename!r} module, but it doesn't have a "
                    f"{funcname!r} attribute."
                    + describe_close_matches(module, funcname)
                ) from err
        else:
            try:
                func_class = getattr(module, classname)
            except AttributeError as err:
                raise click.UsageError(
                    f"Found the {modulename!r} module, but it doesn't have a "
                    f"{classname!r} class." + describe_close_matches(module, classname)
                ) from err
            try:
                return getattr(func_class, funcname)
            except AttributeError as err:
                if inspect.isclass(func_class):
                    func_class_is = "class"
                else:
                    func_class_is = "attribute"
                raise click.UsageError(
                    f"Found the {modulename!r} module and {classname!r} {func_class_is}, "
                    f"but it doesn't have a {funcname!r} attribute."
                    + describe_close_matches(func_class, funcname)
                ) from err

    def _refactor(func, fname):
        try:
            oldcode = Path(fname).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as err:
            # Permissions or encoding issue, or file deleted, etc.
            return f"skipping {fname!r} due to {err}"

        if "hypothesis" not in oldcode:
            return  # This is a fast way to avoid running slow no-op codemods

        try:
            newcode = func(oldcode)
        except Exception as err:
            from libcst import ParserSyntaxError

            if isinstance(err, ParserSyntaxError):
                from hypothesis.extra._patching import indent

                msg = indent(str(err).replace("\n\n", "\n"), "    ").strip()
                return f"skipping {fname!r} due to {msg}"
            raise

        if newcode != oldcode:
            Path(fname).write_text(newcode, encoding="utf-8")

    @main.command()  # type: ignore  # Click adds the .command attribute
    @click.argument("path", type=str, required=True, nargs=-1)
    def codemod(path):
        """`hypothesis codemod` refactors deprecated or inefficient code.

        It adapts `python -m libcst.tool`, removing many features and config options
        which are rarely relevant for this purpose.  If you need more control, we
        encourage you to use the libcst CLI directly; if not this one is easier.

        PATH is the file(s) or directories of files to format in place, or
        "-" to read from stdin and write to stdout.
        """
        try:
            from libcst.codemod import gather_files

            from hypothesis.extra import codemods
        except ImportError:
            sys.stderr.write(
                "You are missing required dependencies for this option.  Run:\n\n"
                "    python -m pip install --upgrade hypothesis[codemods]\n\n"
                "and try again."
            )
            sys.exit(1)

        # Special case for stdin/stdout usage
        if "-" in path:
            if len(path) > 1:
                raise Exception(
                    "Cannot specify multiple paths when reading from stdin!"
                )
            print("Codemodding from stdin", file=sys.stderr)
            print(codemods.refactor(sys.stdin.read()))
            return 0

        # Find all the files to refactor, and then codemod them
        files = gather_files(path)
        errors = set()
        if len(files) <= 1:
            errors.add(_refactor(codemods.refactor, *files))
        else:
            with Pool() as pool:
                for msg in pool.imap_unordered(
                    partial(_refactor, codemods.refactor), files
                ):
                    errors.add(msg)
        errors.discard(None)
        for msg in errors:
            print(msg, file=sys.stderr)
        return 1 if errors else 0

    @main.command()  # type: ignore  # Click adds the .command attribute
    @click.argument("func", type=obj_name, required=True, nargs=-1)
    @click.option(
        "--roundtrip",
        "writer",
        flag_value="roundtrip",
        help="start by testing write/read or encode/decode!",
    )
    @click.option(
        "--equivalent",
        "writer",
        flag_value="equivalent",
        help="very useful when optimising or refactoring code",
    )
    @click.option(
        "--errors-equivalent",
        "writer",
        flag_value="errors-equivalent",
        help="--equivalent, but also allows consistent errors",
    )
    @click.option(
        "--idempotent",
        "writer",
        flag_value="idempotent",
        help="check that f(x) == f(f(x))",
    )
    @click.option(
        "--binary-op",
        "writer",
        flag_value="binary_operation",
        help="associativity, commutativity, identity element",
    )
    # Note: we deliberately omit a --ufunc flag, because the magic()
    # detection of ufuncs is both precise and complete.
    @click.option(
        "--style",
        type=click.Choice(["pytest", "unittest"]),
        default="pytest" if pytest else "unittest",
        help="pytest-style function, or unittest-style method?",
    )
    @click.option(
        "-e",
        "--except",
        "except_",
        type=obj_name,
        multiple=True,
        help="dotted name of exception(s) to ignore",
    )
    @click.option(
        "--annotate/--no-annotate",
        default=None,
        help="force ghostwritten tests to be type-annotated (or not).  "
        "By default, match the code to test.",
    )
    def write(func, writer, except_, style, annotate):  # \b disables autowrap
        """`hypothesis write` writes property-based tests for you!

        Type annotations are helpful but not required for our advanced introspection
        and templating logic.  Try running the examples below to see how it works:

        \b
            hypothesis write gzip
            hypothesis write numpy.matmul
            hypothesis write pandas.from_dummies
            hypothesis write re.compile --except re.error
            hypothesis write --equivalent ast.literal_eval eval
            hypothesis write --roundtrip json.dumps json.loads
            hypothesis write --style=unittest --idempotent sorted
            hypothesis write --binary-op operator.add
        """
        # NOTE: if you want to call this function from Python, look instead at the
        # ``hypothesis.extra.ghostwriter`` module.  Click-decorated functions have
        # a different calling convention, and raise SystemExit instead of returning.
        kwargs = {"except_": except_ or (), "style": style, "annotate": annotate}
        if writer is None:
            writer = "magic"
        elif writer == "idempotent" and len(func) > 1:
            raise click.UsageError("Test functions for idempotence one at a time.")
        elif writer == "roundtrip" and len(func) == 1:
            writer = "idempotent"
        elif "equivalent" in writer and len(func) == 1:
            writer = "fuzz"
        if writer == "errors-equivalent":
            writer = "equivalent"
            kwargs["allow_same_errors"] = True

        try:
            from hypothesis.extra import ghostwriter
        except ImportError:
            sys.stderr.write(MESSAGE.format("black"))
            sys.exit(1)

        code = getattr(ghostwriter, writer)(*func, **kwargs)
        try:
            from rich.console import Console
            from rich.syntax import Syntax

            from hypothesis.utils.terminal import guess_background_color
        except ImportError:
            print(code)
        else:
            try:
                theme = "default" if guess_background_color() == "light" else "monokai"
                code = Syntax(code, "python", background_color="default", theme=theme)
                Console().print(code, soft_wrap=True)
            except Exception:
                print("# Error while syntax-highlighting code", file=sys.stderr)
                print(code)
