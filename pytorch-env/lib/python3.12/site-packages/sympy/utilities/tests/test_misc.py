from textwrap import dedent
import sys
from subprocess import Popen, PIPE
import os

from sympy.core.singleton import S
from sympy.testing.pytest import (raises, warns_deprecated_sympy,
                                  skip_under_pyodide)
from sympy.utilities.misc import (translate, replace, ordinal, rawlines,
                                  strlines, as_int, find_executable)
from sympy.external import import_module

pyodide_js = import_module('pyodide_js')


def test_translate():
    abc = 'abc'
    assert translate(abc, None, 'a') == 'bc'
    assert translate(abc, None, '') == 'abc'
    assert translate(abc, {'a': 'x'}, 'c') == 'xb'
    assert translate(abc, {'a': 'bc'}, 'c') == 'bcb'
    assert translate(abc, {'ab': 'x'}, 'c') == 'x'
    assert translate(abc, {'ab': ''}, 'c') == ''
    assert translate(abc, {'bc': 'x'}, 'c') == 'ab'
    assert translate(abc, {'abc': 'x', 'a': 'y'}) == 'x'
    u = chr(4096)
    assert translate(abc, 'a', 'x', u) == 'xbc'
    assert (u in translate(abc, 'a', u, u)) is True


def test_replace():
    assert replace('abc', ('a', 'b')) == 'bbc'
    assert replace('abc', {'a': 'Aa'}) == 'Aabc'
    assert replace('abc', ('a', 'b'), ('c', 'C')) == 'bbC'


def test_ordinal():
    assert ordinal(-1) == '-1st'
    assert ordinal(0) == '0th'
    assert ordinal(1) == '1st'
    assert ordinal(2) == '2nd'
    assert ordinal(3) == '3rd'
    assert all(ordinal(i).endswith('th') for i in range(4, 21))
    assert ordinal(100) == '100th'
    assert ordinal(101) == '101st'
    assert ordinal(102) == '102nd'
    assert ordinal(103) == '103rd'
    assert ordinal(104) == '104th'
    assert ordinal(200) == '200th'
    assert all(ordinal(i) == str(i) + 'th' for i in range(-220, -203))


def test_rawlines():
    assert rawlines('a a\na') == "dedent('''\\\n    a a\n    a''')"
    assert rawlines('a a') == "'a a'"
    assert rawlines(strlines('\\le"ft')) == (
        '(\n'
        "    '(\\n'\n"
        '    \'r\\\'\\\\le"ft\\\'\\n\'\n'
        "    ')'\n"
        ')')


def test_strlines():
    q = 'this quote (") is in the middle'
    # the following assert rhs was prepared with
    # print(rawlines(strlines(q, 10)))
    assert strlines(q, 10) == dedent('''\
        (
        'this quo'
        'te (") i'
        's in the'
        ' middle'
        )''')
    assert q == (
        'this quo'
        'te (") i'
        's in the'
        ' middle'
        )
    q = "this quote (') is in the middle"
    assert strlines(q, 20) == dedent('''\
        (
        "this quote (') is "
        "in the middle"
        )''')
    assert strlines('\\left') == (
        '(\n'
        "r'\\left'\n"
        ')')
    assert strlines('\\left', short=True) == r"r'\left'"
    assert strlines('\\le"ft') == (
        '(\n'
        'r\'\\le"ft\'\n'
        ')')
    q = 'this\nother line'
    assert strlines(q) == rawlines(q)


def test_translate_args():
    try:
        translate(None, None, None, 'not_none')
    except ValueError:
        pass # Exception raised successfully
    else:
        assert False

    assert translate('s', None, None, None) == 's'

    try:
        translate('s', 'a', 'bc')
    except ValueError:
        pass # Exception raised successfully
    else:
        assert False


@skip_under_pyodide("Cannot create subprocess under pyodide.")
def test_debug_output():
    env = os.environ.copy()
    env['SYMPY_DEBUG'] = 'True'
    cmd = 'from sympy import *; x = Symbol("x"); print(integrate((1-cos(x))/x, x))'
    cmdline = [sys.executable, '-c', cmd]
    proc = Popen(cmdline, env=env, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    out = out.decode('ascii') # utf-8?
    err = err.decode('ascii')
    expected = 'substituted: -x*(1 - cos(x)), u: 1/x, u_var: _u'
    assert expected in err, err


def test_as_int():
    raises(ValueError, lambda : as_int(True))
    raises(ValueError, lambda : as_int(1.1))
    raises(ValueError, lambda : as_int([]))
    raises(ValueError, lambda : as_int(S.NaN))
    raises(ValueError, lambda : as_int(S.Infinity))
    raises(ValueError, lambda : as_int(S.NegativeInfinity))
    raises(ValueError, lambda : as_int(S.ComplexInfinity))
    # for the following, limited precision makes int(arg) == arg
    # but the int value is not necessarily what a user might have
    # expected; Q.prime is more nuanced in its response for
    # expressions which might be complex representations of an
    # integer. This is not -- by design -- as_ints role.
    raises(ValueError, lambda : as_int(1e23))
    raises(ValueError, lambda : as_int(S('1.'+'0'*20+'1')))
    assert as_int(True, strict=False) == 1

def test_deprecated_find_executable():
    with warns_deprecated_sympy():
        find_executable('python')
