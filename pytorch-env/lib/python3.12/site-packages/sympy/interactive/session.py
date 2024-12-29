"""Tools for setting up interactive sessions. """

from sympy.external.gmpy import GROUND_TYPES
from sympy.external.importtools import version_tuple

from sympy.interactive.printing import init_printing

from sympy.utilities.misc import ARCH

preexec_source = """\
from sympy import *
x, y, z, t = symbols('x y z t')
k, m, n = symbols('k m n', integer=True)
f, g, h = symbols('f g h', cls=Function)
init_printing()
"""

verbose_message = """\
These commands were executed:
%(source)s
Documentation can be found at https://docs.sympy.org/%(version)s
"""

no_ipython = """\
Could not locate IPython. Having IPython installed is greatly recommended.
See http://ipython.scipy.org for more details. If you use Debian/Ubuntu,
just install the 'ipython' package and start isympy again.
"""


def _make_message(ipython=True, quiet=False, source=None):
    """Create a banner for an interactive session. """
    from sympy import __version__ as sympy_version
    from sympy import SYMPY_DEBUG

    import sys
    import os

    if quiet:
        return ""

    python_version = "%d.%d.%d" % sys.version_info[:3]

    if ipython:
        shell_name = "IPython"
    else:
        shell_name = "Python"

    info = ['ground types: %s' % GROUND_TYPES]

    cache = os.getenv('SYMPY_USE_CACHE')

    if cache is not None and cache.lower() == 'no':
        info.append('cache: off')

    if SYMPY_DEBUG:
        info.append('debugging: on')

    args = shell_name, sympy_version, python_version, ARCH, ', '.join(info)
    message = "%s console for SymPy %s (Python %s-%s) (%s)\n" % args

    if source is None:
        source = preexec_source

    _source = ""

    for line in source.split('\n')[:-1]:
        if not line:
            _source += '\n'
        else:
            _source += '>>> ' + line + '\n'

    doc_version = sympy_version
    if 'dev' in doc_version:
        doc_version = "dev"
    else:
        doc_version = "%s/" % doc_version

    message += '\n' + verbose_message % {'source': _source,
                                         'version': doc_version}

    return message


def int_to_Integer(s):
    """
    Wrap integer literals with Integer.

    This is based on the decistmt example from
    https://docs.python.org/3/library/tokenize.html.

    Only integer literals are converted.  Float literals are left alone.

    Examples
    ========

    >>> from sympy import Integer # noqa: F401
    >>> from sympy.interactive.session import int_to_Integer
    >>> s = '1.2 + 1/2 - 0x12 + a1'
    >>> int_to_Integer(s)
    '1.2 +Integer (1 )/Integer (2 )-Integer (0x12 )+a1 '
    >>> s = 'print (1/2)'
    >>> int_to_Integer(s)
    'print (Integer (1 )/Integer (2 ))'
    >>> exec(s)
    0.5
    >>> exec(int_to_Integer(s))
    1/2
    """
    from tokenize import generate_tokens, untokenize, NUMBER, NAME, OP
    from io import StringIO

    def _is_int(num):
        """
        Returns true if string value num (with token NUMBER) represents an integer.
        """
        # XXX: Is there something in the standard library that will do this?
        if '.' in num or 'j' in num.lower() or 'e' in num.lower():
            return False
        return True

    result = []
    g = generate_tokens(StringIO(s).readline)  # tokenize the string
    for toknum, tokval, _, _, _ in g:
        if toknum == NUMBER and _is_int(tokval):  # replace NUMBER tokens
            result.extend([
                (NAME, 'Integer'),
                (OP, '('),
                (NUMBER, tokval),
                (OP, ')')
            ])
        else:
            result.append((toknum, tokval))
    return untokenize(result)


def enable_automatic_int_sympification(shell):
    """
    Allow IPython to automatically convert integer literals to Integer.
    """
    import ast
    old_run_cell = shell.run_cell

    def my_run_cell(cell, *args, **kwargs):
        try:
            # Check the cell for syntax errors.  This way, the syntax error
            # will show the original input, not the transformed input.  The
            # downside here is that IPython magic like %timeit will not work
            # with transformed input (but on the other hand, IPython magic
            # that doesn't expect transformed input will continue to work).
            ast.parse(cell)
        except SyntaxError:
            pass
        else:
            cell = int_to_Integer(cell)
        return old_run_cell(cell, *args, **kwargs)

    shell.run_cell = my_run_cell


def enable_automatic_symbols(shell):
    """Allow IPython to automatically create symbols (``isympy -a``). """
    # XXX: This should perhaps use tokenize, like int_to_Integer() above.
    # This would avoid re-executing the code, which can lead to subtle
    # issues.  For example:
    #
    # In [1]: a = 1
    #
    # In [2]: for i in range(10):
    #    ...:     a += 1
    #    ...:
    #
    # In [3]: a
    # Out[3]: 11
    #
    # In [4]: a = 1
    #
    # In [5]: for i in range(10):
    #    ...:     a += 1
    #    ...:     print b
    #    ...:
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    # b
    #
    # In [6]: a
    # Out[6]: 12
    #
    # Note how the for loop is executed again because `b` was not defined, but `a`
    # was already incremented once, so the result is that it is incremented
    # multiple times.

    import re
    re_nameerror = re.compile(
        "name '(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)' is not defined")

    def _handler(self, etype, value, tb, tb_offset=None):
        """Handle :exc:`NameError` exception and allow injection of missing symbols. """
        if etype is NameError and tb.tb_next and not tb.tb_next.tb_next:
            match = re_nameerror.match(str(value))

            if match is not None:
                # XXX: Make sure Symbol is in scope. Otherwise you'll get infinite recursion.
                self.run_cell("%(symbol)s = Symbol('%(symbol)s')" %
                              {'symbol': match.group("symbol")}, store_history=False)

                try:
                    code = self.user_ns['In'][-1]
                except (KeyError, IndexError):
                    pass
                else:
                    self.run_cell(code, store_history=False)
                    return None
                finally:
                    self.run_cell("del %s" % match.group("symbol"),
                                  store_history=False)

        stb = self.InteractiveTB.structured_traceback(
            etype, value, tb, tb_offset=tb_offset)
        self._showtraceback(etype, value, stb)

    shell.set_custom_exc((NameError,), _handler)


def init_ipython_session(shell=None, argv=[], auto_symbols=False, auto_int_to_Integer=False):
    """Construct new IPython session. """
    import IPython

    if version_tuple(IPython.__version__) >= version_tuple('0.11'):
        if not shell:
            # use an app to parse the command line, and init config
            # IPython 1.0 deprecates the frontend module, so we import directly
            # from the terminal module to prevent a deprecation message from being
            # shown.
            if version_tuple(IPython.__version__) >= version_tuple('1.0'):
                from IPython.terminal import ipapp
            else:
                from IPython.frontend.terminal import ipapp
            app = ipapp.TerminalIPythonApp()

            # don't draw IPython banner during initialization:
            app.display_banner = False
            app.initialize(argv)

            shell = app.shell

        if auto_symbols:
            enable_automatic_symbols(shell)
        if auto_int_to_Integer:
            enable_automatic_int_sympification(shell)

        return shell
    else:
        from IPython.Shell import make_IPython
        return make_IPython(argv)


def init_python_session():
    """Construct new Python session. """
    from code import InteractiveConsole

    class SymPyConsole(InteractiveConsole):
        """An interactive console with readline support. """

        def __init__(self):
            ns_locals = {}
            InteractiveConsole.__init__(self, locals=ns_locals)
            try:
                import rlcompleter
                import readline
            except ImportError:
                pass
            else:
                import os
                import atexit

                readline.set_completer(rlcompleter.Completer(ns_locals).complete)
                readline.parse_and_bind('tab: complete')

                if hasattr(readline, 'read_history_file'):
                    history = os.path.expanduser('~/.sympy-history')

                    try:
                        readline.read_history_file(history)
                    except OSError:
                        pass

                    atexit.register(readline.write_history_file, history)

    return SymPyConsole()


def init_session(ipython=None, pretty_print=True, order=None,
                 use_unicode=None, use_latex=None, quiet=False, auto_symbols=False,
                 auto_int_to_Integer=False, str_printer=None, pretty_printer=None,
                 latex_printer=None, argv=[]):
    """
    Initialize an embedded IPython or Python session. The IPython session is
    initiated with the --pylab option, without the numpy imports, so that
    matplotlib plotting can be interactive.

    Parameters
    ==========

    pretty_print: boolean
        If True, use pretty_print to stringify;
        if False, use sstrrepr to stringify.
    order: string or None
        There are a few different settings for this parameter:
        lex (default), which is lexographic order;
        grlex, which is graded lexographic order;
        grevlex, which is reversed graded lexographic order;
        old, which is used for compatibility reasons and for long expressions;
        None, which sets it to lex.
    use_unicode: boolean or None
        If True, use unicode characters;
        if False, do not use unicode characters.
    use_latex: boolean or None
        If True, use latex rendering if IPython GUI's;
        if False, do not use latex rendering.
    quiet: boolean
        If True, init_session will not print messages regarding its status;
        if False, init_session will print messages regarding its status.
    auto_symbols: boolean
        If True, IPython will automatically create symbols for you.
        If False, it will not.
        The default is False.
    auto_int_to_Integer: boolean
        If True, IPython will automatically wrap int literals with Integer, so
        that things like 1/2 give Rational(1, 2).
        If False, it will not.
        The default is False.
    ipython: boolean or None
        If True, printing will initialize for an IPython console;
        if False, printing will initialize for a normal console;
        The default is None, which automatically determines whether we are in
        an ipython instance or not.
    str_printer: function, optional, default=None
        A custom string printer function. This should mimic
        sympy.printing.sstrrepr().
    pretty_printer: function, optional, default=None
        A custom pretty printer. This should mimic sympy.printing.pretty().
    latex_printer: function, optional, default=None
        A custom LaTeX printer. This should mimic sympy.printing.latex()
        This should mimic sympy.printing.latex().
    argv: list of arguments for IPython
        See sympy.bin.isympy for options that can be used to initialize IPython.

    See Also
    ========

    sympy.interactive.printing.init_printing: for examples and the rest of the parameters.


    Examples
    ========

    >>> from sympy import init_session, Symbol, sin, sqrt
    >>> sin(x) #doctest: +SKIP
    NameError: name 'x' is not defined
    >>> init_session() #doctest: +SKIP
    >>> sin(x) #doctest: +SKIP
    sin(x)
    >>> sqrt(5) #doctest: +SKIP
      ___
    \\/ 5
    >>> init_session(pretty_print=False) #doctest: +SKIP
    >>> sqrt(5) #doctest: +SKIP
    sqrt(5)
    >>> y + x + y**2 + x**2 #doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_session(order='grlex') #doctest: +SKIP
    >>> y + x + y**2 + x**2 #doctest: +SKIP
    x**2 + y**2 + x + y
    >>> init_session(order='grevlex') #doctest: +SKIP
    >>> y * x**2 + x * y**2 #doctest: +SKIP
    x**2*y + x*y**2
    >>> init_session(order='old') #doctest: +SKIP
    >>> x**2 + y**2 + x + y #doctest: +SKIP
    x + y + x**2 + y**2
    >>> theta = Symbol('theta') #doctest: +SKIP
    >>> theta #doctest: +SKIP
    theta
    >>> init_session(use_unicode=True) #doctest: +SKIP
    >>> theta # doctest: +SKIP
    \u03b8
    """
    import sys

    in_ipython = False

    if ipython is not False:
        try:
            import IPython
        except ImportError:
            if ipython is True:
                raise RuntimeError("IPython is not available on this system")
            ip = None
        else:
            try:
                from IPython import get_ipython
                ip = get_ipython()
            except ImportError:
                ip = None
        in_ipython = bool(ip)
        if ipython is None:
            ipython = in_ipython

    if ipython is False:
        ip = init_python_session()
        mainloop = ip.interact
    else:
        ip = init_ipython_session(ip, argv=argv, auto_symbols=auto_symbols,
                                  auto_int_to_Integer=auto_int_to_Integer)

        if version_tuple(IPython.__version__) >= version_tuple('0.11'):
            # runsource is gone, use run_cell instead, which doesn't
            # take a symbol arg.  The second arg is `store_history`,
            # and False means don't add the line to IPython's history.
            ip.runsource = lambda src, symbol='exec': ip.run_cell(src, False)

            # Enable interactive plotting using pylab.
            try:
                ip.enable_pylab(import_all=False)
            except Exception:
                # Causes an import error if matplotlib is not installed.
                # Causes other errors (depending on the backend) if there
                # is no display, or if there is some problem in the
                # backend, so we have a bare "except Exception" here
                pass
        if not in_ipython:
            mainloop = ip.mainloop

    if auto_symbols and (not ipython or version_tuple(IPython.__version__) < version_tuple('0.11')):
        raise RuntimeError("automatic construction of symbols is possible only in IPython 0.11 or above")
    if auto_int_to_Integer and (not ipython or version_tuple(IPython.__version__) < version_tuple('0.11')):
        raise RuntimeError("automatic int to Integer transformation is possible only in IPython 0.11 or above")

    _preexec_source = preexec_source

    ip.runsource(_preexec_source, symbol='exec')
    init_printing(pretty_print=pretty_print, order=order,
                  use_unicode=use_unicode, use_latex=use_latex, ip=ip,
                  str_printer=str_printer, pretty_printer=pretty_printer,
                  latex_printer=latex_printer)

    message = _make_message(ipython, quiet, _preexec_source)

    if not in_ipython:
        print(message)
        mainloop()
        sys.exit('Exiting ...')
    else:
        print(message)
        import atexit
        atexit.register(lambda: print("Exiting ...\n"))
