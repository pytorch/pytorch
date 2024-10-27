"""Tests of tools for setting up interactive IPython sessions. """

from sympy.interactive.session import (init_ipython_session,
    enable_automatic_symbols, enable_automatic_int_sympification)

from sympy.core import Symbol, Rational, Integer
from sympy.external import import_module
from sympy.testing.pytest import raises

# TODO: The code below could be made more granular with something like:
#
# @requires('IPython', version=">=0.11")
# def test_automatic_symbols(ipython):

ipython = import_module("IPython", min_module_version="0.11")

if not ipython:
    #bin/test will not execute any tests now
    disabled = True

# WARNING: These tests will modify the existing IPython environment. IPython
# uses a single instance for its interpreter, so there is no way to isolate
# the test from another IPython session. It also means that if this test is
# run twice in the same Python session it will fail. This isn't usually a
# problem because the test suite is run in a subprocess by default, but if the
# tests are run with subprocess=False it can pollute the current IPython
# session. See the discussion in issue #15149.

def test_automatic_symbols():
    # NOTE: Because of the way the hook works, you have to use run_cell(code,
    # True).  This means that the code must have no Out, or it will be printed
    # during the tests.
    app = init_ipython_session()
    app.run_cell("from sympy import *")

    enable_automatic_symbols(app)

    symbol = "verylongsymbolname"
    assert symbol not in app.user_ns
    app.run_cell("a = %s" % symbol, True)
    assert symbol not in app.user_ns
    app.run_cell("a = type(%s)" % symbol, True)
    assert app.user_ns['a'] == Symbol
    app.run_cell("%s = Symbol('%s')" % (symbol, symbol), True)
    assert symbol in app.user_ns

    # Check that built-in names aren't overridden
    app.run_cell("a = all == __builtin__.all", True)
    assert "all" not in app.user_ns
    assert app.user_ns['a'] is True

    # Check that SymPy names aren't overridden
    app.run_cell("import sympy")
    app.run_cell("a = factorial == sympy.factorial", True)
    assert app.user_ns['a'] is True


def test_int_to_Integer():
    # XXX: Warning, don't test with == here.  0.5 == Rational(1, 2) is True!
    app = init_ipython_session()
    app.run_cell("from sympy import Integer")
    app.run_cell("a = 1")
    assert isinstance(app.user_ns['a'], int)

    enable_automatic_int_sympification(app)
    app.run_cell("a = 1/2")
    assert isinstance(app.user_ns['a'], Rational)
    app.run_cell("a = 1")
    assert isinstance(app.user_ns['a'], Integer)
    app.run_cell("a = int(1)")
    assert isinstance(app.user_ns['a'], int)
    app.run_cell("a = (1/\n2)")
    assert app.user_ns['a'] == Rational(1, 2)
    # TODO: How can we test that the output of a SyntaxError is the original
    # input, not the transformed input?


def test_ipythonprinting():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import Symbol")

    # Printing without printing extension
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] == "pi"
        assert app.user_ns['a2']['text/plain'] == "pi**2"
    else:
        assert app.user_ns['a'][0]['text/plain'] == "pi"
        assert app.user_ns['a2'][0]['text/plain'] == "pi**2"

    # Load printing extension
    app.run_cell("from sympy import init_printing")
    app.run_cell("init_printing()")
    # Printing with printing extension
    app.run_cell("a = format(Symbol('pi'))")
    app.run_cell("a2 = format(Symbol('pi')**2)")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] in ('\N{GREEK SMALL LETTER PI}', 'pi')
        assert app.user_ns['a2']['text/plain'] in (' 2\n\N{GREEK SMALL LETTER PI} ', '  2\npi ')
    else:
        assert app.user_ns['a'][0]['text/plain'] in ('\N{GREEK SMALL LETTER PI}', 'pi')
        assert app.user_ns['a2'][0]['text/plain'] in (' 2\n\N{GREEK SMALL LETTER PI} ', '  2\npi ')


def test_print_builtin_option():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import Symbol")
    app.run_cell("from sympy import init_printing")

    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        text = app.user_ns['a'][0]['text/plain']
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    # XXX: How can we make this ignore the terminal width? This test fails if
    # the terminal is too narrow.
    assert text in ("{pi: 3.14, n_i: 3}",
                    '{n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3, \N{GREEK SMALL LETTER PI}: 3.14}',
                    "{n_i: 3, pi: 3.14}",
                    '{\N{GREEK SMALL LETTER PI}: 3.14, n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3}')

    # If we enable the default printing, then the dictionary's should render
    # as a LaTeX version of the whole dict: ${\pi: 3.14, n_i: 3}$
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("init_printing(use_latex=True)")
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        latex = app.user_ns['a']['text/latex']
    else:
        text = app.user_ns['a'][0]['text/plain']
        latex = app.user_ns['a'][0]['text/latex']
    assert text in ("{pi: 3.14, n_i: 3}",
                    '{n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3, \N{GREEK SMALL LETTER PI}: 3.14}',
                    "{n_i: 3, pi: 3.14}",
                    '{\N{GREEK SMALL LETTER PI}: 3.14, n\N{LATIN SUBSCRIPT SMALL LETTER I}: 3}')
    assert latex == r'$\displaystyle \left\{ n_{i} : 3, \  \pi : 3.14\right\}$'

    # Objects with an _latex overload should also be handled by our tuple
    # printer.
    app.run_cell("""\
    class WithOverload:
        def _latex(self, printer):
            return r"\\LaTeX"
    """)
    app.run_cell("a = format((WithOverload(),))")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        latex = app.user_ns['a']['text/latex']
    else:
        latex = app.user_ns['a'][0]['text/latex']
    assert latex == r'$\displaystyle \left( \LaTeX,\right)$'

    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("init_printing(use_latex=True, print_builtin=False)")
    app.run_cell("a = format({Symbol('pi'): 3.14, Symbol('n_i'): 3})")
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        text = app.user_ns['a']['text/plain']
        raises(KeyError, lambda: app.user_ns['a']['text/latex'])
    else:
        text = app.user_ns['a'][0]['text/plain']
        raises(KeyError, lambda: app.user_ns['a'][0]['text/latex'])
    # Note : In Python 3 we have one text type: str which holds Unicode data
    # and two byte types bytes and bytearray.
    # Python 3.3.3 + IPython 0.13.2 gives: '{n_i: 3, pi: 3.14}'
    # Python 3.3.3 + IPython 1.1.0 gives: '{n_i: 3, pi: 3.14}'
    assert text in ("{pi: 3.14, n_i: 3}", "{n_i: 3, pi: 3.14}")


def test_builtin_containers():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("from sympy import init_printing, Matrix")
    app.run_cell('init_printing(use_latex=True, use_unicode=False)')

    # Make sure containers that shouldn't pretty print don't.
    app.run_cell('a = format((True, False))')
    app.run_cell('import sys')
    app.run_cell('b = format(sys.flags)')
    app.run_cell('c = format((Matrix([1, 2]),))')
    # Deal with API change starting at IPython 1.0
    if int(ipython.__version__.split(".")[0]) < 1:
        assert app.user_ns['a']['text/plain'] ==  '(True, False)'
        assert 'text/latex' not in app.user_ns['a']
        assert app.user_ns['b']['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b']
        assert app.user_ns['c']['text/plain'] == \
"""\
 [1]  \n\
([ ],)
 [2]  \
"""
        assert app.user_ns['c']['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'
    else:
        assert app.user_ns['a'][0]['text/plain'] ==  '(True, False)'
        assert 'text/latex' not in app.user_ns['a'][0]
        assert app.user_ns['b'][0]['text/plain'][:10] == 'sys.flags('
        assert 'text/latex' not in app.user_ns['b'][0]
        assert app.user_ns['c'][0]['text/plain'] == \
"""\
 [1]  \n\
([ ],)
 [2]  \
"""
        assert app.user_ns['c'][0]['text/latex'] == '$\\displaystyle \\left( \\left[\\begin{matrix}1\\\\2\\end{matrix}\\right],\\right)$'

def test_matplotlib_bad_latex():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("import IPython")
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("from sympy import init_printing, Matrix")
    app.run_cell("init_printing(use_latex='matplotlib')")

    # The png formatter is not enabled by default in this context
    app.run_cell("inst.display_formatter.formatters['image/png'].enabled = True")

    # Make sure no warnings are raised by IPython
    app.run_cell("import warnings")
    # IPython.core.formatters.FormatterWarning was introduced in IPython 2.0
    if int(ipython.__version__.split(".")[0]) < 2:
        app.run_cell("warnings.simplefilter('error')")
    else:
        app.run_cell("warnings.simplefilter('error', IPython.core.formatters.FormatterWarning)")

    # This should not raise an exception
    app.run_cell("a = format(Matrix([1, 2, 3]))")

    # issue 9799
    app.run_cell("from sympy import Piecewise, Symbol, Eq")
    app.run_cell("x = Symbol('x'); pw = format(Piecewise((1, Eq(x, 0)), (0, True)))")


def test_override_repr_latex():
    # Initialize and setup IPython session
    app = init_ipython_session()
    app.run_cell("import IPython")
    app.run_cell("ip = get_ipython()")
    app.run_cell("inst = ip.instance()")
    app.run_cell("format = inst.display_formatter.format")
    app.run_cell("inst.display_formatter.formatters['text/latex'].enabled = True")
    app.run_cell("from sympy import init_printing")
    app.run_cell("from sympy import Symbol")
    app.run_cell("init_printing(use_latex=True)")
    app.run_cell("""\
    class SymbolWithOverload(Symbol):
        def _repr_latex_(self):
            return r"Hello " + super()._repr_latex_() + " world"
    """)
    app.run_cell("a = format(SymbolWithOverload('s'))")

    if int(ipython.__version__.split(".")[0]) < 1:
        latex = app.user_ns['a']['text/latex']
    else:
        latex = app.user_ns['a'][0]['text/latex']
    assert latex == r'Hello $\displaystyle s$ world'
