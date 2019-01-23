import re
import unittest
import traceback
import os
import string


ACCEPT = os.getenv('EXPECTTEST_ACCEPT')


def nth_line(src, lineno):
    """
    Compute the starting index of the n-th line (where n is 1-indexed)

    >>> nth_line("aaa\\nbb\\nc", 2)
    4
    """
    assert lineno >= 1
    pos = 0
    for _ in range(lineno - 1):
        pos = src.find('\n', pos) + 1
    return pos


def nth_eol(src, lineno):
    """
    Compute the ending index of the n-th line (before the newline,
    where n is 1-indexed)

    >>> nth_eol("aaa\\nbb\\nc", 2)
    6
    """
    assert lineno >= 1
    pos = -1
    for _ in range(lineno):
        pos = src.find('\n', pos + 1)
        if pos == -1:
            return len(src)
    return pos


def normalize_nl(t):
    return t.replace('\r\n', '\n').replace('\r', '\n')


def escape_trailing_quote(s, quote):
    if s and s[-1] == quote:
        return s[:-1] + '\\' + quote
    else:
        return s


class EditHistory(object):
    def __init__(self):
        self.state = {}

    def adjust_lineno(self, fn, lineno):
        if fn not in self.state:
            return lineno
        for edit_loc, edit_diff in self.state[fn]:
            if lineno > edit_loc:
                lineno += edit_diff
        return lineno

    def seen_file(self, fn):
        return fn in self.state

    def record_edit(self, fn, lineno, delta):
        self.state.setdefault(fn, []).append((lineno, delta))


EDIT_HISTORY = EditHistory()


def ok_for_raw_triple_quoted_string(s, quote):
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
    return quote * 3 not in s and (not s or s[-1] not in [quote, '\\'])


# This operates on the REVERSED string (that's why suffix is first)
RE_EXPECT = re.compile(r"^(?P<suffix>[^\n]*?)"
                       r"(?P<quote>'''|" r'""")'
                       r"(?P<body>.*?)"
                       r"(?P=quote)"
                       r"(?P<raw>r?)", re.DOTALL)


def replace_string_literal(src, lineno, new_string):
    r"""
    Replace a triple quoted string literal with new contents.
    Only handles printable ASCII correctly at the moment.  This
    will preserve the quote style of the original string, and
    makes a best effort to preserve raw-ness (unless it is impossible
    to do so.)

    Returns a tuple of the replaced string, as well as a delta of
    number of lines added/removed.

    >>> replace_string_literal("'''arf'''", 1, "barf")
    ("'''barf'''", 0)
    >>> r = replace_string_literal("  moo = '''arf'''", 1, "'a'\n\\b\n")
    >>> print(r[0])
      moo = '''\
    'a'
    \\b
    '''
    >>> r[1]
    3
    >>> replace_string_literal("  moo = '''\\\narf'''", 2, "'a'\n\\b\n")[1]
    2
    >>> print(replace_string_literal("    f('''\"\"\"''')", 1, "a ''' b")[0])
        f('''a \'\'\' b''')
    """
    # Haven't implemented correct escaping for non-printable characters
    assert all(c in string.printable for c in new_string)
    i = nth_eol(src, lineno)
    new_string = normalize_nl(new_string)

    delta = [new_string.count("\n")]
    if delta[0] > 0:
        delta[0] += 1  # handle the extra \\\n

    def replace(m):
        s = new_string
        raw = m.group('raw') == 'r'
        if not raw or not ok_for_raw_triple_quoted_string(s, quote=m.group('quote')[0]):
            raw = False
            s = s.replace('\\', '\\\\')
            if m.group('quote') == "'''":
                s = escape_trailing_quote(s, "'").replace("'''", r"\'\'\'")
            else:
                s = escape_trailing_quote(s, '"').replace('"""', r'\"\"\"')

        new_body = "\\\n" + s if "\n" in s and not raw else s
        delta[0] -= m.group('body').count("\n")

        return ''.join([m.group('suffix'),
                        m.group('quote'),
                        new_body[::-1],
                        m.group('quote'),
                        'r' if raw else '',
                        ])

    # Having to do this in reverse is very irritating, but it's the
    # only way to make the non-greedy matches work correctly.
    return (RE_EXPECT.sub(replace, src[:i][::-1], count=1)[::-1] + src[i:], delta[0])


class TestCase(unittest.TestCase):
    longMessage = True

    def assertExpectedInline(self, actual, expect, skip=0):
        if ACCEPT:
            if actual != expect:
                # current frame and parent frame, plus any requested skip
                tb = traceback.extract_stack(limit=2 + skip)
                fn, lineno, _, _ = tb[0]
                print("Accepting new output for {} at {}:{}".format(self.id(), fn, lineno))
                with open(fn, 'r+') as f:
                    old = f.read()

                    # compute the change in lineno
                    lineno = EDIT_HISTORY.adjust_lineno(fn, lineno)
                    new, delta = replace_string_literal(old, lineno, actual)

                    assert old != new, "Failed to substitute string at {}:{}".format(fn, lineno)

                    # Only write the backup file the first time we hit the
                    # file
                    if not EDIT_HISTORY.seen_file(fn):
                        with open(fn + ".bak", 'w') as f_bak:
                            f_bak.write(old)
                    f.seek(0)
                    f.truncate(0)

                    f.write(new)

                EDIT_HISTORY.record_edit(fn, lineno, delta)
        else:
            help_text = ("To accept the new output, re-run test with "
                         "envvar EXPECTTEST_ACCEPT=1 (we recommend "
                         "staging/committing your changes before doing this)")
            if hasattr(self, "assertMultiLineEqual"):
                self.assertMultiLineEqual(expect, actual, msg=help_text)
            else:
                self.assertEqual(expect, actual, msg=help_text)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
