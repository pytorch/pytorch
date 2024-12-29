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
Write patches which add @example() decorators for discovered test cases.

Requires `hypothesis[codemods,ghostwriter]` installed, i.e. black and libcst.

This module is used by Hypothesis' builtin pytest plugin for failing examples
discovered during testing, and by HypoFuzz for _covering_ examples discovered
during fuzzing.
"""

import ast
import difflib
import hashlib
import inspect
import re
import sys
import types
from ast import literal_eval
from contextlib import suppress
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import libcst as cst
from libcst import matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.metadata import ExpressionContext, ExpressionContextProvider

from hypothesis.configuration import storage_directory
from hypothesis.version import __version__

try:
    import black
except ImportError:
    black = None  # type: ignore

HEADER = f"""\
From HEAD Mon Sep 17 00:00:00 2001
From: Hypothesis {__version__} <no-reply@hypothesis.works>
Date: {{when:%a, %d %b %Y %H:%M:%S}}
Subject: [PATCH] {{msg}}

---
"""
FAIL_MSG = "discovered failure"
_space_only_re = re.compile("^ +$", re.MULTILINE)
_leading_space_re = re.compile("(^[ ]*)(?:[^ \n])", re.MULTILINE)


def dedent(text):
    # Simplified textwrap.dedent, for valid Python source code only
    text = _space_only_re.sub("", text)
    prefix = min(_leading_space_re.findall(text), key=len)
    return re.sub(r"(?m)^" + prefix, "", text), prefix


def indent(text: str, prefix: str) -> str:
    return "".join(prefix + line for line in text.splitlines(keepends=True))


class AddExamplesCodemod(VisitorBasedCodemodCommand):
    DESCRIPTION = "Add explicit examples to failing tests."

    def __init__(self, context, fn_examples, strip_via=(), dec="example", width=88):
        """Add @example() decorator(s) for failing test(s).

        `code` is the source code of the module where the test functions are defined.
        `fn_examples` is a dict of function name to list-of-failing-examples.
        """
        assert fn_examples, "This codemod does nothing without fn_examples."
        super().__init__(context)

        self.decorator_func = cst.parse_expression(dec)
        self.line_length = width
        value_in_strip_via = m.MatchIfTrue(lambda x: literal_eval(x.value) in strip_via)
        self.strip_matching = m.Call(
            m.Attribute(m.Call(), m.Name("via")),
            [m.Arg(m.SimpleString() & value_in_strip_via)],
        )

        # Codemod the failing examples to Call nodes usable as decorators
        self.fn_examples = {
            k: tuple(d for x in nodes if (d := self.__call_node_to_example_dec(*x)))
            for k, nodes in fn_examples.items()
        }

    def __call_node_to_example_dec(self, node, via):
        # If we have black installed, remove trailing comma, _unless_ there's a comment
        node = node.with_changes(
            func=self.decorator_func,
            args=(
                [
                    a.with_changes(
                        comma=(
                            a.comma
                            if m.findall(a.comma, m.Comment())
                            else cst.MaybeSentinel.DEFAULT
                        )
                    )
                    for a in node.args
                ]
                if black
                else node.args
            ),
        )
        via = cst.Call(
            func=cst.Attribute(node, cst.Name("via")),
            args=[cst.Arg(cst.SimpleString(repr(via)))],
        )
        if black:  # pragma: no branch
            try:
                pretty = black.format_str(
                    cst.Module([]).code_for_node(via),
                    mode=black.FileMode(line_length=self.line_length),
                )
            except (ImportError, AttributeError):  # pragma: no cover
                return None  # See https://github.com/psf/black/pull/4224
            via = cst.parse_expression(pretty.strip())
        return cst.Decorator(via)

    def leave_FunctionDef(self, _, updated_node):
        return updated_node.with_changes(
            # TODO: improve logic for where in the list to insert this decorator
            decorators=tuple(
                d
                for d in updated_node.decorators
                # `findall()` to see through the identity function workaround on py38
                if not m.findall(d, self.strip_matching)
            )
            + self.fn_examples.get(updated_node.name.value, ())
        )


def get_patch_for(func, failing_examples, *, strip_via=()):
    # Skip this if we're unable to find the location or source of this function.
    try:
        module = sys.modules[func.__module__]
        fname = Path(module.__file__).relative_to(Path.cwd())
        before = inspect.getsource(func)
    except Exception:
        return None

    modules_in_test_scope = sorted(
        (
            (k, v)
            for (k, v) in module.__dict__.items()
            if isinstance(v, types.ModuleType)
        ),
        key=lambda kv: len(kv[1].__name__),
    )

    # The printed examples might include object reprs which are invalid syntax,
    # so we parse here and skip over those.  If _none_ are valid, there's no patch.
    call_nodes = []
    for ex, via in set(failing_examples):
        with suppress(Exception):
            node = cst.parse_module(ex)
            the_call = node.body[0].body[0].value
            assert isinstance(the_call, cst.Call), the_call
            # Check for st.data(), which doesn't support explicit examples
            data = m.Arg(m.Call(m.Name("data"), args=[m.Arg(m.Ellipsis())]))
            if m.matches(the_call, m.Call(args=[m.ZeroOrMore(), data, m.ZeroOrMore()])):
                return None

            # Many reprs use the unqualified name of the type, e.g. np.array()
            # -> array([...]), so here we find undefined names and look them up
            # on each module which was in the test's global scope.
            names = {}
            for anode in ast.walk(ast.parse(ex, "eval")):
                if (
                    isinstance(anode, ast.Name)
                    and isinstance(anode.ctx, ast.Load)
                    and anode.id not in names
                    and anode.id not in module.__dict__
                ):
                    for k, v in modules_in_test_scope:
                        if anode.id in v.__dict__:
                            names[anode.id] = cst.parse_expression(f"{k}.{anode.id}")
                            break

            # LibCST doesn't track Load()/Store() state of names by default, so we have
            # to do a bit of a dance here, *and* explicitly handle keyword arguments
            # which are treated as Load() context - but even if that's fixed later
            # we'll still want to support older versions.
            with suppress(Exception):
                wrapper = cst.metadata.MetadataWrapper(node)
                kwarg_names = {
                    a.keyword for a in m.findall(wrapper, m.Arg(keyword=m.Name()))
                }
                node = m.replace(
                    wrapper,
                    m.Name(value=m.MatchIfTrue(names.__contains__))
                    & m.MatchMetadata(ExpressionContextProvider, ExpressionContext.LOAD)
                    & m.MatchIfTrue(lambda n, k=kwarg_names: n not in k),
                    replacement=lambda node, _, ns=names: ns[node.value],
                )
            node = node.body[0].body[0].value
            assert isinstance(node, cst.Call), node
            call_nodes.append((node, via))

    if not call_nodes:
        return None

    if (
        module.__dict__.get("hypothesis") is sys.modules["hypothesis"]
        and "given" not in module.__dict__  # more reliably present than `example`
    ):
        decorator_func = "hypothesis.example"
    else:
        decorator_func = "example"

    # Do the codemod and return a triple containing location and replacement info.
    dedented, prefix = dedent(before)
    try:
        node = cst.parse_module(dedented)
    except Exception:  # pragma: no cover
        # inspect.getsource() sometimes returns a decorator alone, which is invalid
        return None
    after = AddExamplesCodemod(
        CodemodContext(),
        fn_examples={func.__name__: call_nodes},
        strip_via=strip_via,
        dec=decorator_func,
        width=88 - len(prefix),  # to match Black's default formatting
    ).transform_module(node)
    return (str(fname), before, indent(after.code, prefix=prefix))


def make_patch(triples, *, msg="Hypothesis: add explicit examples", when=None):
    """Create a patch for (fname, before, after) triples."""
    assert triples, "attempted to create empty patch"
    when = when or datetime.now(tz=timezone.utc)

    by_fname = {}
    for fname, before, after in triples:
        by_fname.setdefault(Path(fname), []).append((before, after))

    diffs = [HEADER.format(msg=msg, when=when)]
    for fname, changes in sorted(by_fname.items()):
        source_before = source_after = fname.read_text(encoding="utf-8")
        for before, after in changes:
            source_after = source_after.replace(before.rstrip(), after.rstrip(), 1)
        ud = difflib.unified_diff(
            source_before.splitlines(keepends=True),
            source_after.splitlines(keepends=True),
            fromfile=f"./{fname}",  # git strips the first part of the path by default
            tofile=f"./{fname}",
        )
        diffs.append("".join(ud))
    return "".join(diffs)


def save_patch(patch: str, *, slug: str = "") -> Path:  # pragma: no cover
    assert re.fullmatch(r"|[a-z]+-", slug), f"malformed {slug=}"
    now = date.today().isoformat()
    cleaned = re.sub(r"^Date: .+?$", "", patch, count=1, flags=re.MULTILINE)
    hash8 = hashlib.sha1(cleaned.encode()).hexdigest()[:8]
    fname = Path(storage_directory("patches", f"{now}--{slug}{hash8}.patch"))
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(patch, encoding="utf-8")
    return fname.relative_to(Path.cwd())


def gc_patches(slug: str = "") -> None:  # pragma: no cover
    cutoff = date.today() - timedelta(days=7)
    for fname in Path(storage_directory("patches")).glob(
        f"????-??-??--{slug}????????.patch"
    ):
        if date.fromisoformat(fname.stem.split("--")[0]) < cutoff:
            fname.unlink()
