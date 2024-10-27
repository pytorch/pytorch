import ast
import os
import re
import string
import sys
import traceback
import unittest
import difflib
from typing import Any, Callable, Dict, List, Match, Tuple, Optional

# NB: We do not internally use this property for anything, but it
# is preserved for BC reasons
ACCEPT = os.getenv('EXPECTTEST_ACCEPT')

LINENO_AT_START = sys.version_info >= (3, 8)


def nth_line(src: str, lineno: int) -> int:
    """
    Compute the starting index of the n-th line (where n is 1-indexed)

    >>> nth_line("aaa\\nbb\\nc", 2)
    4
    """
    assert lineno >= 1
    pos = 0
    for _ in range(lineno - 1):
        pos = src.find("\n", pos) + 1
    return pos


def nth_eol(src: str, lineno: int) -> int:
    """
    Compute the ending index of the n-th line (before the newline,
    where n is 1-indexed)

    >>> nth_eol("aaa\\nbb\\nc", 2)
    6
    """
    assert lineno >= 1
    pos = -1
    for _ in range(lineno):
        pos = src.find("\n", pos + 1)
        if pos == -1:
            return len(src)
    return pos


def normalize_nl(t: str) -> str:
    return t.replace("\r\n", "\n").replace("\r", "\n")


def escape_trailing_quote(s: str, quote: str) -> str:
    if s and s[-1] == quote:
        return s[:-1] + "\\" + quote
    else:
        return s


class EditHistory:
    state: Dict[str, List[Tuple[int, int]]]
    seen: Dict[str, Dict[int, str]]

    def __init__(self) -> None:
        self.state = {}
        self.seen = {}

    def reload_file(self, fn: str) -> None:
        """
        The idea is that if you reload a file, the line numbers
        from traceback are now up to date, but we do NOT want to clear
        out seen list as it will tell us if we are expecting the
        same line multiple times.  Instead, we need to adjust
        the seen list for the new world order.
        """
        new_seen = {}
        for seen_loc, seen_str in self.seen.get(fn, {}).items():
            new_seen[self.adjust_lineno(fn, seen_loc)] = seen_str

        self.seen[fn] = new_seen
        self.state.pop(fn, None)

    def adjust_lineno(self, fn: str, lineno: int) -> int:
        if fn not in self.state:
            return lineno
        for edit_loc, edit_diff in self.state[fn]:
            if lineno > edit_loc:
                lineno += edit_diff
        return lineno

    def seen_file(self, fn: str) -> bool:
        return fn in self.state

    def seen_edit(self, fn: str, lineno: int) -> Optional[str]:
        return self.seen.get(fn, {}).get(lineno, None)

    def record_edit(self, fn: str, lineno: int, delta: int, expect: str) -> None:
        self.state.setdefault(fn, []).append((lineno, delta))
        self.seen.setdefault(fn, {})[lineno] = expect


EDIT_HISTORY = EditHistory()


def ok_for_raw_triple_quoted_string(s: str, quote: str) -> bool:
    """
    Is this string representable inside a raw triple-quoted string?
    Due to the fact that backslashes are always treated literally,
    some strings are not representable.

    >>> ok_for_raw_triple_quoted_string("blah", quote="'")
    True
    >>> ok_for_raw_triple_quoted_string("'", quote="'")
    False
    >>> ok_for_raw_triple_quoted_string("a ''' b", quote="'")
    False
    """
    return quote * 3 not in s and (not s or s[-1] not in [quote, "\\"])


RE_EXPECT = re.compile(
    (r"(?P<raw>r?)" r"(?P<quote>'''|" r'""")' r"(?P<body>.*?)" r"(?P=quote)"), re.DOTALL
)


def replace_string_literal(
    src: str, start_lineno: int, end_lineno: int, new_string: str
) -> Tuple[str, int]:
    r"""
    Replace a triple quoted string literal with new contents.
    Only handles printable ASCII correctly at the moment.  This
    will preserve the quote style of the original string, and
    makes a best effort to preserve raw-ness (unless it is impossible
    to do so.)

    Returns a tuple of the replaced string, as well as a delta of
    number of lines added/removed.

    >>> replace_string_literal("'''arf'''", 1, 1, "barf")
    ("'''barf'''", 0)
    >>> r = replace_string_literal("  moo = '''arf'''", 1, 1, "'a'\n\\b\n")
    >>> print(r[0])
      moo = '''\
    'a'
    \\b
    '''
    >>> r[1]
    3
    >>> replace_string_literal("  moo = '''\\\narf'''", 1, 2, "'a'\n\\b\n")[1]
    2
    >>> print(replace_string_literal("    f('''\"\"\"''')", 1, 1, "a ''' b")[0])
        f('''a \'\'\' b''')
    """
    # Haven't implemented correct escaping for non-printable characters
    assert all(c in string.printable for c in new_string), repr(new_string)

    new_string = normalize_nl(new_string)

    delta = [new_string.count("\n")]
    if delta[0] > 0:
        delta[0] += 1  # handle the extra \\\n

    assert start_lineno <= end_lineno
    start = nth_line(src, start_lineno)
    end = nth_eol(src, end_lineno)
    assert start <= end

    def replace(m: Match[str]) -> str:
        s = new_string
        raw = m.group("raw") == "r"
        if not raw or not ok_for_raw_triple_quoted_string(s, quote=m.group("quote")[0]):
            raw = False
            s = s.replace("\\", "\\\\")
            if m.group("quote") == "'''":
                s = escape_trailing_quote(s, "'").replace("'''", r"\'\'\'")
            else:
                s = escape_trailing_quote(s, '"').replace('"""', r"\"\"\"")

        new_body = "\\\n" + s if "\n" in s and not raw else s
        delta[0] -= m.group("body").count("\n")
        return "".join(
            [
                "r" if raw else "",
                m.group("quote"),
                new_body,
                m.group("quote"),
            ]
        )

    return (
        src[:start] + RE_EXPECT.sub(replace, src[start:end], count=1) + src[end:],
        delta[0],
    )


def replace_many(rep: Dict[str, str], text: str) -> str:
    rep = {re.escape(k): v for k, v in rep.items()}
    pattern = re.compile("|".join(rep.keys()))
    return pattern.sub(lambda m: rep[re.escape(m.group(0))], text)


def assert_eq(expect: str, actual: str, *, msg: str) -> None:
    # TODO: improve this
    if actual != expect:
        diff = "".join(
            difflib.unified_diff(
                expect.splitlines(True),
                actual.splitlines(True),
                fromfile="expect.txt",
                tofile="actual.txt",
            )
        )
        raise AssertionError(
            f"Mismatch between actual and expect strings:\n\n{diff}\n\n{msg}"
        )


def assert_expected_inline(
    actual: str,
    expect: str,
    skip: int = 0,
    *,
    expect_filters: Optional[Dict[str, str]] = None,
    assert_eq: Any = assert_eq,
    debug_id: str = "",
) -> None:
    """
    Assert that actual is equal to expect.  The expect argument
    MUST be a string literal (triple-quoted strings OK), and will
    get updated directly in source when you run the test suite
    with EXPECTTEST_ACCEPT=1.

    If you want to write a helper function that makes use of
    assertExpectedInline (e.g., expect is not a string literal),
    set the skip argument to how many function calls we should
    skip to find the string literal to update.
    """
    if expect_filters is not None:
        actual = replace_many(expect_filters, actual)

    # NB: Intentionally do not use ACCEPT global variable;
    # reaccessing environment here allows for modification
    # of os.environ to be picked up
    if os.getenv("EXPECTTEST_ACCEPT"):
        if actual != expect:
            # current frame and parent frame, plus any requested skip
            tb = traceback.extract_stack(limit=2 + skip)
            fn, lineno, _, _ = tb[0]
            debug_suffix = "" if not debug_id else f" for {debug_id}"
            if (prev_actual := EDIT_HISTORY.seen_edit(fn, lineno)) is not None:
                assert_eq(
                    actual,
                    prev_actual,
                    msg=f"Uh oh, accepting different values{debug_suffix} at {fn}:{lineno}.  Are you running a parametrized test?  If so, you need a separate assertExpectedInline invocation per distinct output.",
                )
                print(
                    "Skipping already accepted output{} at {}:{}".format(
                        debug_suffix, fn, lineno
                    )
                )
                return
            print("Accepting new output{} at {}:{}".format(debug_suffix, fn, lineno))
            with open(fn, "r+") as f:
                old = f.read()
                old_ast = ast.parse(old)

                # NB: it's only the traceback line numbers that are wrong;
                # we reread the file every time we write to it, so AST's
                # line numbers are correct
                lineno = EDIT_HISTORY.adjust_lineno(fn, lineno)

                # Conservative assumption to start
                start_lineno = lineno
                end_lineno = lineno
                # Try to give a more accurate bounds based on AST
                # NB: this walk is in no specified order (in practice it's
                # breadth first)
                for n in ast.walk(old_ast):
                    if isinstance(n, ast.Expr):
                        if hasattr(n, "end_lineno"):
                            assert LINENO_AT_START
                            if n.lineno == start_lineno:
                                end_lineno = n.end_lineno  # type: ignore[attr-defined]
                        else:
                            if n.lineno == end_lineno:
                                start_lineno = n.lineno

                new, delta = replace_string_literal(
                    old, start_lineno, end_lineno, actual
                )

                assert old != new, (
                    f"Failed to substitute string at {fn}:{lineno}; did you use triple quotes?  "
                    "If this is unexpected, please file a bug report at "
                    "https://github.com/ezyang/expecttest/issues/new "
                    f"with the contents of the source file near {fn}:{lineno}"
                )

                # Only write the backup file the first time we hit the
                # file
                if not EDIT_HISTORY.seen_file(fn):
                    with open(fn + ".bak", "w") as f_bak:
                        f_bak.write(old)
                f.seek(0)
                f.truncate(0)

                f.write(new)

            EDIT_HISTORY.record_edit(fn, lineno, delta, actual)
    else:
        help_text = (
            "To accept the new output, re-run test with "
            "envvar EXPECTTEST_ACCEPT=1 (we recommend "
            "staging/committing your changes before doing this)"
        )
        assert_eq(expect, actual, msg=help_text)


class TestCase(unittest.TestCase):
    longMessage = True
    _expect_filters: Dict[str, str]

    def substituteExpected(self, pattern: str, replacement: str) -> None:
        if not hasattr(self, "_expect_filters"):
            self._expect_filters = {}

            def expect_filters_cleanup() -> None:
                del self._expect_filters

            self.addCleanup(expect_filters_cleanup)
        if pattern in self._expect_filters:
            raise RuntimeError(
                "Cannot remap {} to {} (existing mapping is {})".format(
                    pattern, replacement, self._expect_filters[pattern]
                )
            )
        self._expect_filters[pattern] = replacement

    def assertExpectedInline(self, actual: str, expect: str, skip: int = 0) -> None:
        """
        Assert that actual is equal to expect.  The expect argument
        MUST be a string literal (triple-quoted strings OK), and will
        get updated directly in source when you run the test suite
        with EXPECTTEST_ACCEPT=1.

        If you want to write a helper function that makes use of
        assertExpectedInline (e.g., expect is not a string literal),
        set the skip argument to how many function calls we should
        skip to find the string literal to update.
        """
        assert_expected_inline(
            actual,
            expect,
            skip=skip + 1,
            expect_filters=getattr(self, "_expect_filters", None),
            debug_id=self.id(),
            assert_eq=self.assertMultiLineEqualMaybeCppStack,
        )

    def assertExpectedRaisesInline(
        self,
        exc_type: Any,
        callable: Callable[..., Any],
        expect: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Like assertExpectedInline, but tests the str() representation of
        the raised exception from callable.  The raised exeption must
        be exc_type.
        """
        try:
            callable(*args, **kwargs)
        except exc_type as e:
            self.assertExpectedInline(str(e), expect, skip=1)
            return
        # Don't put this in the try block; the AssertionError will catch it
        self.fail(msg="Did not raise when expected to")

    def assertMultiLineEqualMaybeCppStack(
        self, expect: str, actual: str, *args: Any, **kwargs: Any
    ) -> None:
        cpp_stack_header = "\nException raised from"
        if cpp_stack_header in actual:
            actual = actual.rsplit(cpp_stack_header, maxsplit=1)[0]
        if hasattr(self, "assertMultiLineEqual"):
            self.assertMultiLineEqual(expect, actual, *args, **kwargs)
        else:
            self.assertEqual(expect, actual, *args, **kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
