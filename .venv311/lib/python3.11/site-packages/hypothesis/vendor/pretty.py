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
Python advanced pretty printer.  This pretty printer is intended to
replace the old `pprint` python module which does not allow developers
to provide their own pretty print callbacks.
This module is based on ruby's `prettyprint.rb` library by `Tanaka Akira`.
Example Usage
-------------
To get a string of the output use `pretty`::
    from pretty import pretty
    string = pretty(complex_object)
Extending
---------
The pretty library allows developers to add pretty printing rules for their
own objects.  This process is straightforward.  All you have to do is to
add a `_repr_pretty_` method to your object and call the methods on the
pretty printer passed::
    class MyObject(object):
        def _repr_pretty_(self, p, cycle):
            ...
Here is an example implementation of a `_repr_pretty_` method for a list
subclass::
    class MyList(list):
        def _repr_pretty_(self, p, cycle):
            if cycle:
                p.text('MyList(...)')
            else:
                with p.group(8, 'MyList([', '])'):
                    for idx, item in enumerate(self):
                        if idx:
                            p.text(',')
                            p.breakable()
                        p.pretty(item)
The `cycle` parameter is `True` if pretty detected a cycle.  You *have* to
react to that or the result is an infinite loop.  `p.text()` just adds
non breaking text to the output, `p.breakable()` either adds a whitespace
or breaks here.  If you pass it an argument it's used instead of the
default space.  `p.pretty` prettyprints another object using the pretty print
method.
The first parameter to the `group` function specifies the extra indentation
of the next line.  In this example the next item will either be on the same
line (if the items are short enough) or aligned with the right edge of the
opening bracket of `MyList`.
If you just want to indent something you can use the group function
without open / close parameters.  You can also use this code::
    with p.indent(2):
        ...
Inheritance diagram:
.. inheritance-diagram:: IPython.lib.pretty
   :parts: 3
:copyright: 2007 by Armin Ronacher.
            Portions (c) 2009 by Robert Kern.
:license: BSD License.
"""

import ast
import datetime
import re
import struct
import sys
import types
import warnings
from collections import Counter, OrderedDict, defaultdict, deque
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager, suppress
from enum import Enum, Flag
from functools import partial
from io import StringIO, TextIOBase
from math import copysign, isnan
from typing import TYPE_CHECKING, Any, Optional, TypeAlias, TypeVar

if TYPE_CHECKING:
    from hypothesis.control import BuildContext

T = TypeVar("T")
PrettyPrintFunction: TypeAlias = Callable[[Any, "RepresentationPrinter", bool], None]

__all__ = [
    "IDKey",
    "RepresentationPrinter",
    "pretty",
]


def _safe_getattr(obj: object, attr: str, default: Any | None = None) -> Any:
    """Safe version of getattr.

    Same as getattr, but will return ``default`` on any Exception,
    rather than raising.

    """
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def pretty(obj: object, *, cycle: bool = False) -> str:
    """Pretty print the object's representation."""
    printer = RepresentationPrinter()
    printer.pretty(obj, cycle=cycle)
    return printer.getvalue()


class IDKey:
    def __init__(self, value: object):
        self.value = value

    def __hash__(self) -> int:
        return hash((type(self), id(self.value)))

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, type(self)) and id(self.value) == id(__o.value)


class RepresentationPrinter:
    """Special pretty printer that has a `pretty` method that calls the pretty
    printer for a python object.

    This class stores processing data on `self` so you must *never* use
    this class in a threaded environment.  Always lock it or
    reinstantiate it.

    """

    def __init__(
        self,
        output: TextIOBase | None = None,
        *,
        context: Optional["BuildContext"] = None,
    ) -> None:
        """Optionally pass the output stream and the current build context.

        We use the context to represent objects constructed by strategies by showing
        *how* they were constructed, and add annotations showing which parts of the
        minimal failing example can vary without changing the test result.
        """
        self.broken: bool = False
        self.output: TextIOBase = StringIO() if output is None else output
        self.max_width: int = 79
        self.max_seq_length: int = 1000
        self.output_width: int = 0
        self.buffer_width: int = 0
        self.buffer: deque[Breakable | Text] = deque()

        root_group = Group(0)
        self.group_stack = [root_group]
        self.group_queue = GroupQueue(root_group)
        self.indentation: int = 0

        self.stack: list[int] = []
        self.singleton_pprinters: dict[int, PrettyPrintFunction] = {}
        self.type_pprinters: dict[type, PrettyPrintFunction] = {}
        self.deferred_pprinters: dict[tuple[str, str], PrettyPrintFunction] = {}
        # If IPython has been imported, load up their pretty-printer registry
        if "IPython.lib.pretty" in sys.modules:
            ipp = sys.modules["IPython.lib.pretty"]
            self.singleton_pprinters.update(ipp._singleton_pprinters)
            self.type_pprinters.update(ipp._type_pprinters)
            self.deferred_pprinters.update(ipp._deferred_type_pprinters)
        # If there's overlap between our pprinters and IPython's, we'll use ours.
        self.singleton_pprinters.update(_singleton_pprinters)
        self.type_pprinters.update(_type_pprinters)
        self.deferred_pprinters.update(_deferred_type_pprinters)

        # for which-parts-matter, we track a mapping from the (start_idx, end_idx)
        # of slices into the minimal failing example; this is per-interesting_origin
        # but we report each separately so that's someone else's problem here.
        # Invocations of self.repr_call() can report the slice for each argument,
        # which will then be used to look up the relevant comment if any.
        self.known_object_printers: dict[IDKey, list[PrettyPrintFunction]]
        self.slice_comments: dict[tuple[int, int], str]
        if context is None:
            self.known_object_printers = defaultdict(list)
            self.slice_comments = {}
        else:
            self.known_object_printers = context.known_object_printers
            self.slice_comments = context.data.slice_comments
        assert all(isinstance(k, IDKey) for k in self.known_object_printers)

    def pretty(self, obj: object, *, cycle: bool = False) -> None:
        """Pretty print the given object."""
        obj_id = id(obj)
        cycle = cycle or obj_id in self.stack
        self.stack.append(obj_id)
        try:
            with self.group():
                obj_class = _safe_getattr(obj, "__class__", None) or type(obj)
                # First try to find registered singleton printers for the type.
                try:
                    printer = self.singleton_pprinters[obj_id]
                except (TypeError, KeyError):
                    pass
                else:
                    return printer(obj, self, cycle)

                # Look for the _repr_pretty_ method which allows users
                # to define custom pretty printing.
                # Some objects automatically create any requested
                # attribute. Try to ignore most of them by checking for
                # callability.
                pretty_method = _safe_getattr(obj, "_repr_pretty_", None)
                if callable(pretty_method):
                    return pretty_method(self, cycle)

                # Next walk the mro and check for either:
                #   1) a registered printer
                #   2) a _repr_pretty_ method
                for cls in obj_class.__mro__:
                    if cls in self.type_pprinters:
                        # printer registered in self.type_pprinters
                        return self.type_pprinters[cls](obj, self, cycle)
                    else:
                        # Check if the given class is specified in the deferred type
                        # registry; move it to the regular type registry if so.
                        key = (
                            _safe_getattr(cls, "__module__", None),
                            _safe_getattr(cls, "__name__", None),
                        )
                        if key in self.deferred_pprinters:
                            # Move the printer over to the regular registry.
                            printer = self.deferred_pprinters.pop(key)
                            self.type_pprinters[cls] = printer
                            return printer(obj, self, cycle)
                        else:
                            if hasattr(cls, "__attrs_attrs__"):  # pragma: no cover
                                return pprint_fields(
                                    obj,
                                    self,
                                    cycle,
                                    [at.name for at in cls.__attrs_attrs__ if at.init],
                                )
                            if hasattr(cls, "__dataclass_fields__"):
                                return pprint_fields(
                                    obj,
                                    self,
                                    cycle,
                                    [
                                        k
                                        for k, v in cls.__dataclass_fields__.items()
                                        if v.init
                                    ],
                                )
                # Now check for object-specific printers which show how this
                # object was constructed (a Hypothesis special feature).
                printers = self.known_object_printers[IDKey(obj)]
                if len(printers) == 1:
                    return printers[0](obj, self, cycle)
                elif printers:
                    # We've ended up with multiple registered functions for the same
                    # object, which must have been returned from multiple calls due to
                    # e.g. memoization.  If they all return the same string, we'll use
                    # the first; otherwise we'll pretend that *none* were registered.
                    #
                    # It's annoying, but still seems to be the best option for which-
                    # parts-matter too, as unreportable results aren't very useful.
                    strs = set()
                    for f in printers:
                        p = RepresentationPrinter()
                        f(obj, p, cycle)
                        strs.add(p.getvalue())
                    if len(strs) == 1:
                        return printers[0](obj, self, cycle)

                # A user-provided repr. Find newlines and replace them with p.break_()
                return _repr_pprint(obj, self, cycle)
        finally:
            self.stack.pop()

    def _break_outer_groups(self) -> None:
        while self.max_width < self.output_width + self.buffer_width:
            group = self.group_queue.deq()
            if not group:
                return
            while group.breakables:
                x = self.buffer.popleft()
                self.output_width = x.output(self.output, self.output_width)
                self.buffer_width -= x.width
            while self.buffer and isinstance(self.buffer[0], Text):
                x = self.buffer.popleft()
                self.output_width = x.output(self.output, self.output_width)
                self.buffer_width -= x.width

    def text(self, obj: str) -> None:
        """Add literal text to the output."""
        width = len(obj)
        if self.buffer:
            text = self.buffer[-1]
            if not isinstance(text, Text):
                text = Text()
                self.buffer.append(text)
            text.add(obj, width)
            self.buffer_width += width
            self._break_outer_groups()
        else:
            self.output.write(obj)
            self.output_width += width

    def breakable(self, sep: str = " ") -> None:
        """Add a breakable separator to the output.

        This does not mean that it will automatically break here.  If no
        breaking on this position takes place the `sep` is inserted
        which default to one space.

        """
        width = len(sep)
        group = self.group_stack[-1]
        if group.want_break:
            self.flush()
            self.output.write("\n" + " " * self.indentation)
            self.output_width = self.indentation
            self.buffer_width = 0
        else:
            self.buffer.append(Breakable(sep, width, self))
            self.buffer_width += width
            self._break_outer_groups()

    def break_(self) -> None:
        """Explicitly insert a newline into the output, maintaining correct
        indentation."""
        self.flush()
        self.output.write("\n" + " " * self.indentation)
        self.output_width = self.indentation
        self.buffer_width = 0

    @contextmanager
    def indent(self, indent: int) -> Generator[None, None, None]:
        """`with`-statement support for indenting/dedenting."""
        self.indentation += indent
        try:
            yield
        finally:
            self.indentation -= indent

    @contextmanager
    def group(
        self, indent: int = 0, open: str = "", close: str = ""
    ) -> Generator[None, None, None]:
        """Context manager for an indented group.

            with p.group(1, '{', '}'):

        The first parameter specifies the indentation for the next line
        (usually the width of the opening text), the second and third the
        opening and closing delimiters.
        """
        self.begin_group(indent=indent, open=open)
        try:
            yield
        finally:
            self.end_group(dedent=indent, close=close)

    def begin_group(self, indent: int = 0, open: str = "") -> None:
        """Use the `with group(...) context manager instead.

        The begin_group() and end_group() methods are for IPython compatibility only;
        see https://github.com/HypothesisWorks/hypothesis/issues/3721 for details.
        """
        if open:
            self.text(open)
        group = Group(self.group_stack[-1].depth + 1)
        self.group_stack.append(group)
        self.group_queue.enq(group)
        self.indentation += indent

    def end_group(self, dedent: int = 0, close: str = "") -> None:
        """See begin_group()."""
        self.indentation -= dedent
        group = self.group_stack.pop()
        if not group.breakables:
            self.group_queue.remove(group)
        if close:
            self.text(close)

    def _enumerate(self, seq: Iterable[T]) -> Generator[tuple[int, T], None, None]:
        """Like enumerate, but with an upper limit on the number of items."""
        for idx, x in enumerate(seq):
            if self.max_seq_length and idx >= self.max_seq_length:
                self.text(",")
                self.breakable()
                self.text("...")
                return
            yield idx, x

    def flush(self) -> None:
        """Flush data that is left in the buffer."""
        for data in self.buffer:
            self.output_width += data.output(self.output, self.output_width)
        self.buffer.clear()
        self.buffer_width = 0

    def getvalue(self) -> str:
        assert isinstance(self.output, StringIO)
        self.flush()
        return self.output.getvalue()

    def maybe_repr_known_object_as_call(
        self,
        obj: object,
        cycle: bool,
        name: str,
        args: Sequence[object],
        kwargs: dict[str, object],
    ) -> None:
        # pprint this object as a call, _unless_ the call would be invalid syntax
        # and the repr would be valid and there are not comments on arguments.
        if cycle:
            return self.text("<...>")
        # Since we don't yet track comments for sub-argument parts, we omit the
        # "if no comments" condition here for now.  Add it when we revive
        # https://github.com/HypothesisWorks/hypothesis/pull/3624/
        with suppress(Exception):
            # Check whether the repr is valid syntax:
            ast.parse(repr(obj))
            # Given that the repr is valid syntax, check the call:
            p = RepresentationPrinter()
            p.stack = self.stack.copy()
            p.known_object_printers = self.known_object_printers
            p.repr_call(name, args, kwargs)
            # If the call is not valid syntax, use the repr
            try:
                ast.parse(p.getvalue())
            except Exception:
                return _repr_pprint(obj, self, cycle)
        return self.repr_call(name, args, kwargs)

    def repr_call(
        self,
        func_name: str,
        args: Sequence[object],
        kwargs: dict[str, object],
        *,
        force_split: bool | None = None,
        arg_slices: dict[str, tuple[int, int]] | None = None,
        leading_comment: str | None = None,
        avoid_realization: bool = False,
    ) -> None:
        """Helper function to represent a function call.

        - func_name, args, and kwargs should all be pretty obvious.
        - If split_lines, we'll force one-argument-per-line; otherwise we'll place
          calls that fit on a single line (and split otherwise).
        - arg_slices is a mapping from pos-idx or keyword to (start_idx, end_idx)
          of the Conjecture buffer, by which we can look up comments to add.
        """
        assert isinstance(func_name, str)
        if func_name.startswith(("lambda:", "lambda ")):
            func_name = f"({func_name})"
        self.text(func_name)
        all_args = [(None, v) for v in args] + list(kwargs.items())
        # int indicates the position of a positional argument, rather than a keyword
        # argument. Currently no callers use this; see #3624.
        comments: dict[int | str, object] = {
            k: self.slice_comments[v]
            for k, v in (arg_slices or {}).items()
            if v in self.slice_comments
        }

        if leading_comment or any(k in comments for k, _ in all_args):
            # We have to split one arg per line in order to leave comments on them.
            force_split = True
        if force_split is None:
            # We're OK with printing this call on a single line, but will it fit?
            # If not, we'd rather fall back to one-argument-per-line instead.
            p = RepresentationPrinter()
            p.stack = self.stack.copy()
            p.known_object_printers = self.known_object_printers
            p.repr_call("_" * self.output_width, args, kwargs, force_split=False)
            s = p.getvalue()
            force_split = "\n" in s

        with self.group(indent=4, open="(", close=""):
            for i, (k, v) in enumerate(all_args):
                if force_split:
                    if i == 0 and leading_comment:
                        self.break_()
                        self.text(leading_comment)
                    self.break_()
                else:
                    assert leading_comment is None  # only passed by top-level report
                    self.breakable(" " if i else "")
                if k:
                    self.text(f"{k}=")
                if avoid_realization:
                    self.text("<symbolic>")
                else:
                    self.pretty(v)
                if force_split or i + 1 < len(all_args):
                    self.text(",")
                comment = None
                if k is not None:
                    comment = comments.get(i) or comments.get(k)
                if comment:
                    self.text(f"  # {comment}")
        if all_args and force_split:
            self.break_()
        self.text(")")  # after dedent


class Printable:
    def output(self, stream: TextIOBase, output_width: int) -> int:  # pragma: no cover
        raise NotImplementedError


class Text(Printable):
    def __init__(self) -> None:
        self.objs: list[str] = []
        self.width: int = 0

    def output(self, stream: TextIOBase, output_width: int) -> int:
        for obj in self.objs:
            stream.write(obj)
        return output_width + self.width

    def add(self, obj: str, width: int) -> None:
        self.objs.append(obj)
        self.width += width


class Breakable(Printable):
    def __init__(self, seq: str, width: int, pretty: RepresentationPrinter) -> None:
        self.obj = seq
        self.width = width
        self.pretty = pretty
        self.indentation = pretty.indentation
        self.group = pretty.group_stack[-1]
        self.group.breakables.append(self)

    def output(self, stream: TextIOBase, output_width: int) -> int:
        self.group.breakables.popleft()
        if self.group.want_break:
            stream.write("\n" + " " * self.indentation)
            return self.indentation
        if not self.group.breakables:
            self.pretty.group_queue.remove(self.group)
        stream.write(self.obj)
        return output_width + self.width


class Group(Printable):
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.breakables: deque[Breakable] = deque()
        self.want_break: bool = False


class GroupQueue:
    def __init__(self, *groups: Group) -> None:
        self.queue: list[list[Group]] = []
        for group in groups:
            self.enq(group)

    def enq(self, group: Group) -> None:
        depth = group.depth
        while depth > len(self.queue) - 1:
            self.queue.append([])
        self.queue[depth].append(group)

    def deq(self) -> Group | None:
        for stack in self.queue:
            for idx, group in enumerate(reversed(stack)):
                if group.breakables:
                    del stack[idx]
                    group.want_break = True
                    return group
            for group in stack:
                group.want_break = True
            del stack[:]
        return None

    def remove(self, group: Group) -> None:
        try:
            self.queue[group.depth].remove(group)
        except ValueError:
            pass


def _seq_pprinter_factory(start: str, end: str, basetype: type) -> PrettyPrintFunction:
    """Factory that returns a pprint function useful for sequences.

    Used by the default pprint for tuples, dicts, and lists.
    """

    def inner(
        obj: tuple[object] | list[object], p: RepresentationPrinter, cycle: bool
    ) -> None:
        typ = type(obj)
        if (
            basetype is not None
            and typ is not basetype
            and typ.__repr__ != basetype.__repr__  # type: ignore[comparison-overlap]
        ):
            # If the subclass provides its own repr, use it instead.
            return p.text(typ.__repr__(obj))

        if cycle:
            return p.text(start + "..." + end)
        step = len(start)
        with p.group(step, start, end):
            for idx, x in p._enumerate(obj):
                if idx:
                    p.text(",")
                    p.breakable()
                p.pretty(x)
            if len(obj) == 1 and type(obj) is tuple:
                # Special case for 1-item tuples.
                p.text(",")

    return inner


def get_class_name(cls: type[object]) -> str:
    class_name = _safe_getattr(cls, "__qualname__", cls.__name__)
    assert isinstance(class_name, str)
    return class_name


def _set_pprinter_factory(
    start: str, end: str, basetype: type[object]
) -> PrettyPrintFunction:
    """Factory that returns a pprint function useful for sets and
    frozensets."""

    def inner(
        obj: set[Any] | frozenset[Any],
        p: RepresentationPrinter,
        cycle: bool,
    ) -> None:
        typ = type(obj)
        if (
            basetype is not None
            and typ is not basetype
            and typ.__repr__ != basetype.__repr__
        ):
            # If the subclass provides its own repr, use it instead.
            return p.text(typ.__repr__(obj))

        if cycle:
            return p.text(start + "..." + end)
        if not obj:
            # Special case.
            p.text(get_class_name(basetype) + "()")
        else:
            step = len(start)
            with p.group(step, start, end):
                # Like dictionary keys, try to sort the items if there aren't too many
                items: Iterable[object] = obj
                if not (p.max_seq_length and len(obj) >= p.max_seq_length):
                    try:
                        items = sorted(obj)
                    except Exception:
                        # Sometimes the items don't sort.
                        pass
                for idx, x in p._enumerate(items):
                    if idx:
                        p.text(",")
                        p.breakable()
                    p.pretty(x)

    return inner


def _dict_pprinter_factory(
    start: str, end: str, basetype: type[object] | None = None
) -> PrettyPrintFunction:
    """Factory that returns a pprint function used by the default pprint of
    dicts and dict proxies."""

    def inner(obj: dict[object, object], p: RepresentationPrinter, cycle: bool) -> None:
        typ = type(obj)
        if (
            basetype is not None
            and typ is not basetype
            and typ.__repr__ != basetype.__repr__
        ):
            # If the subclass provides its own repr, use it instead.
            return p.text(typ.__repr__(obj))

        if cycle:
            return p.text("{...}")
        with (
            p.group(1, start, end),
            # If the dict contains both "" and b"" (empty string and empty bytes), we
            # ignore the BytesWarning raised by `python -bb` mode.  We can't use
            # `.items()` because it might be a non-`dict` type of mapping.
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", BytesWarning)
            for idx, key in p._enumerate(obj):
                if idx:
                    p.text(",")
                    p.breakable()
                p.pretty(key)
                p.text(": ")
                p.pretty(obj[key])

    inner.__name__ = f"_dict_pprinter_factory({start!r}, {end!r}, {basetype!r})"
    return inner


def _super_pprint(obj: Any, p: RepresentationPrinter, cycle: bool) -> None:
    """The pprint for the super type."""
    with p.group(8, "<super: ", ">"):
        p.pretty(obj.__thisclass__)
        p.text(",")
        p.breakable()
        p.pretty(obj.__self__)


def _re_pattern_pprint(obj: re.Pattern, p: RepresentationPrinter, cycle: bool) -> None:
    """The pprint function for regular expression patterns."""
    p.text("re.compile(")
    pattern = repr(obj.pattern)
    if pattern[:1] in "uU":  # pragma: no cover
        pattern = pattern[1:]
        prefix = "ur"
    else:
        prefix = "r"
    pattern = prefix + pattern.replace("\\\\", "\\")
    p.text(pattern)
    if obj.flags:
        p.text(",")
        p.breakable()
        done_one = False
        for flag in (
            "TEMPLATE",
            "IGNORECASE",
            "LOCALE",
            "MULTILINE",
            "DOTALL",
            "UNICODE",
            "VERBOSE",
            "DEBUG",
        ):
            if obj.flags & getattr(re, flag, 0):
                if done_one:
                    p.text("|")
                p.text("re." + flag)
                done_one = True
    p.text(")")


def _type_pprint(obj: type[object], p: RepresentationPrinter, cycle: bool) -> None:
    """The pprint for classes and types."""
    # Heap allocated types might not have the module attribute,
    # and others may set it to None.

    # Checks for a __repr__ override in the metaclass
    # != rather than is not because pypy compatibility
    if type(obj).__repr__ != type.__repr__:  # type: ignore[comparison-overlap]
        _repr_pprint(obj, p, cycle)
        return

    mod = _safe_getattr(obj, "__module__", None)
    try:
        name = obj.__qualname__
    except Exception:  # pragma: no cover
        name = obj.__name__
        if not isinstance(name, str):
            name = "<unknown type>"

    if mod in (None, "__builtin__", "builtins", "exceptions"):
        p.text(name)
    else:
        p.text(mod + "." + name)


def _repr_pprint(obj: object, p: RepresentationPrinter, cycle: bool) -> None:
    """A pprint that just redirects to the normal repr function."""
    # Find newlines and replace them with p.break_()
    output = repr(obj)
    for idx, output_line in enumerate(output.splitlines()):
        if idx:
            p.break_()
        p.text(output_line)


def pprint_fields(
    obj: object, p: RepresentationPrinter, cycle: bool, fields: Iterable[str]
) -> None:
    name = get_class_name(obj.__class__)
    if cycle:
        return p.text(f"{name}(...)")
    with p.group(1, name + "(", ")"):
        for idx, field in enumerate(fields):
            if idx:
                p.text(",")
                p.breakable()
            p.text(field)
            p.text("=")
            p.pretty(getattr(obj, field))


def _function_pprint(
    obj: types.FunctionType | types.BuiltinFunctionType | types.MethodType,
    p: RepresentationPrinter,
    cycle: bool,
) -> None:
    """Base pprint for all functions and builtin functions."""
    from hypothesis.internal.reflection import get_pretty_function_description

    p.text(get_pretty_function_description(obj))


def _exception_pprint(
    obj: BaseException, p: RepresentationPrinter, cycle: bool
) -> None:
    """Base pprint for all exceptions."""
    name = getattr(obj.__class__, "__qualname__", obj.__class__.__name__)
    if obj.__class__.__module__ not in ("exceptions", "builtins"):
        name = f"{obj.__class__.__module__}.{name}"
    step = len(name) + 1
    with p.group(step, name + "(", ")"):
        for idx, arg in enumerate(getattr(obj, "args", ())):
            if idx:
                p.text(",")
                p.breakable()
            p.pretty(arg)


def _repr_integer(obj: int, p: RepresentationPrinter, cycle: bool) -> None:
    if abs(obj) < 1_000_000_000:
        p.text(repr(obj))
    elif abs(obj) < 10**640:
        # add underscores for integers over ten decimal digits
        p.text(f"{obj:#_d}")
    else:
        # for very very large integers, use hex because power-of-two bases are cheaper
        # https://docs.python.org/3/library/stdtypes.html#integer-string-conversion-length-limitation
        p.text(f"{obj:#_x}")


def _repr_float_counting_nans(
    obj: float, p: RepresentationPrinter, cycle: bool
) -> None:
    if isnan(obj):
        if struct.pack("!d", abs(obj)) != struct.pack("!d", float("nan")):
            show = hex(*struct.unpack("Q", struct.pack("d", obj)))
            return p.text(f"struct.unpack('d', struct.pack('Q', {show}))[0]")
        elif copysign(1.0, obj) == -1.0:
            return p.text("-nan")
    p.text(repr(obj))


#: printers for builtin types
_type_pprinters: dict[type, PrettyPrintFunction] = {
    int: _repr_integer,
    float: _repr_float_counting_nans,
    str: _repr_pprint,
    tuple: _seq_pprinter_factory("(", ")", tuple),
    list: _seq_pprinter_factory("[", "]", list),
    dict: _dict_pprinter_factory("{", "}", dict),
    set: _set_pprinter_factory("{", "}", set),
    frozenset: _set_pprinter_factory("frozenset({", "})", frozenset),
    super: _super_pprint,
    re.Pattern: _re_pattern_pprint,
    type: _type_pprint,
    types.FunctionType: _function_pprint,
    types.BuiltinFunctionType: _function_pprint,
    types.MethodType: _function_pprint,
    datetime.datetime: _repr_pprint,
    datetime.timedelta: _repr_pprint,
    BaseException: _exception_pprint,
    slice: _repr_pprint,
    range: _repr_pprint,
    bytes: _repr_pprint,
}

#: printers for types specified by name
_deferred_type_pprinters: dict[tuple[str, str], PrettyPrintFunction] = {}


def for_type_by_name(
    type_module: str, type_name: str, func: PrettyPrintFunction
) -> PrettyPrintFunction | None:
    """Add a pretty printer for a type specified by the module and name of a
    type rather than the type object itself."""
    key = (type_module, type_name)
    oldfunc = _deferred_type_pprinters.get(key)
    _deferred_type_pprinters[key] = func
    return oldfunc


#: printers for the default singletons
_singleton_pprinters: dict[int, PrettyPrintFunction] = dict.fromkeys(
    map(id, [None, True, False, Ellipsis, NotImplemented]), _repr_pprint
)


def _defaultdict_pprint(
    obj: defaultdict[object, object], p: RepresentationPrinter, cycle: bool
) -> None:
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + "(", ")"):
        if cycle:
            p.text("...")
        else:
            p.pretty(obj.default_factory)
            p.text(",")
            p.breakable()
            p.pretty(dict(obj))


def _ordereddict_pprint(
    obj: OrderedDict[object, object], p: RepresentationPrinter, cycle: bool
) -> None:
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + "(", ")"):
        if cycle:
            p.text("...")
        elif obj:
            p.pretty(list(obj.items()))


def _deque_pprint(obj: deque[object], p: RepresentationPrinter, cycle: bool) -> None:
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + "(", ")"):
        if cycle:
            p.text("...")
        else:
            p.pretty(list(obj))


def _counter_pprint(
    obj: Counter[object], p: RepresentationPrinter, cycle: bool
) -> None:
    name = obj.__class__.__name__
    with p.group(len(name) + 1, name + "(", ")"):
        if cycle:
            p.text("...")
        elif obj:
            p.pretty(dict(obj))


def _repr_dataframe(
    obj: object, p: RepresentationPrinter, cycle: bool
) -> None:  # pragma: no cover
    with p.indent(4):
        p.break_()
        _repr_pprint(obj, p, cycle)
    p.break_()


def _repr_enum(obj: Enum, p: RepresentationPrinter, cycle: bool) -> None:
    tname = get_class_name(type(obj))
    if isinstance(obj, Flag):
        p.text(
            " | ".join(f"{tname}.{x.name}" for x in type(obj) if x & obj == x)
            or f"{tname}({obj.value!r})"  # if no matching members
        )
    else:
        p.text(f"{tname}.{obj.name}")


class _ReprDots:
    def __repr__(self) -> str:
        return "..."


def _repr_partial(obj: partial[Any], p: RepresentationPrinter, cycle: bool) -> None:
    args, kw = obj.args, obj.keywords
    if cycle:
        args, kw = (_ReprDots(),), {}
    p.repr_call(pretty(type(obj)), (obj.func, *args), kw)


for_type_by_name("collections", "defaultdict", _defaultdict_pprint)
for_type_by_name("collections", "OrderedDict", _ordereddict_pprint)
for_type_by_name("ordereddict", "OrderedDict", _ordereddict_pprint)
for_type_by_name("collections", "deque", _deque_pprint)
for_type_by_name("collections", "Counter", _counter_pprint)
for_type_by_name("pandas.core.frame", "DataFrame", _repr_dataframe)
for_type_by_name("enum", "Enum", _repr_enum)
for_type_by_name("functools", "partial", _repr_partial)
