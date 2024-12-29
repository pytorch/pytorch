"""Tools for setting up printing in interactive sessions. """

from sympy.external.importtools import version_tuple
from io import BytesIO

from sympy.printing.latex import latex as default_latex
from sympy.printing.preview import preview
from sympy.utilities.misc import debug
from sympy.printing.defaults import Printable


def _init_python_printing(stringify_func, **settings):
    """Setup printing in Python interactive session. """
    import sys
    import builtins

    def _displayhook(arg):
        """Python's pretty-printer display hook.

           This function was adapted from:

            https://www.python.org/dev/peps/pep-0217/

        """
        if arg is not None:
            builtins._ = None
            print(stringify_func(arg, **settings))
            builtins._ = arg

    sys.displayhook = _displayhook


def _init_ipython_printing(ip, stringify_func, use_latex, euler, forecolor,
                           backcolor, fontsize, latex_mode, print_builtin,
                           latex_printer, scale, **settings):
    """Setup printing in IPython interactive session. """
    try:
        from IPython.lib.latextools import latex_to_png
    except ImportError:
        pass

    # Guess best font color if none was given based on the ip.colors string.
    # From the IPython documentation:
    #   It has four case-insensitive values: 'nocolor', 'neutral', 'linux',
    #   'lightbg'. The default is neutral, which should be legible on either
    #   dark or light terminal backgrounds. linux is optimised for dark
    #   backgrounds and lightbg for light ones.
    if forecolor is None:
        color = ip.colors.lower()
        if color == 'lightbg':
            forecolor = 'Black'
        elif color == 'linux':
            forecolor = 'White'
        else:
            # No idea, go with gray.
            forecolor = 'Gray'
        debug("init_printing: Automatic foreground color:", forecolor)

    if use_latex == "svg":
        extra_preamble = "\n\\special{color %s}" % forecolor
    else:
        extra_preamble = ""

    imagesize = 'tight'
    offset = "0cm,0cm"
    resolution = round(150*scale)
    dvi = r"-T %s -D %d -bg %s -fg %s -O %s" % (
        imagesize, resolution, backcolor, forecolor, offset)
    dvioptions = dvi.split()

    svg_scale = 150/72*scale
    dvioptions_svg = ["--no-fonts", "--scale={}".format(svg_scale)]

    debug("init_printing: DVIOPTIONS:", dvioptions)
    debug("init_printing: DVIOPTIONS_SVG:", dvioptions_svg)

    latex = latex_printer or default_latex

    def _print_plain(arg, p, cycle):
        """caller for pretty, for use in IPython 0.11"""
        if _can_print(arg):
            p.text(stringify_func(arg))
        else:
            p.text(IPython.lib.pretty.pretty(arg))

    def _preview_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='png', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions, fontsize=fontsize)
        except Exception as e:
            # IPython swallows exceptions
            debug("png printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        return exprbuffer.getvalue()

    def _svg_wrapper(o):
        exprbuffer = BytesIO()
        try:
            preview(o, output='svg', viewer='BytesIO', euler=euler,
                    outputbuffer=exprbuffer, extra_preamble=extra_preamble,
                    dvioptions=dvioptions_svg, fontsize=fontsize)
        except Exception as e:
            # IPython swallows exceptions
            debug("svg printing:", "_preview_wrapper exception raised:",
                  repr(e))
            raise
        return exprbuffer.getvalue().decode('utf-8')

    def _matplotlib_wrapper(o):
        # mathtext can't render some LaTeX commands. For example, it can't
        # render any LaTeX environments such as array or matrix. So here we
        # ensure that if mathtext fails to render, we return None.
        try:
            try:
                return latex_to_png(o, color=forecolor, scale=scale)
            except TypeError: #  Old IPython version without color and scale
                return latex_to_png(o)
        except ValueError as e:
            debug('matplotlib exception caught:', repr(e))
            return None


    # Hook methods for builtin SymPy printers
    printing_hooks = ('_latex', '_sympystr', '_pretty', '_sympyrepr')


    def _can_print(o):
        """Return True if type o can be printed with one of the SymPy printers.

        If o is a container type, this is True if and only if every element of
        o can be printed in this way.
        """

        try:
            # If you're adding another type, make sure you add it to printable_types
            # later in this file as well

            builtin_types = (list, tuple, set, frozenset)
            if isinstance(o, builtin_types):
                # If the object is a custom subclass with a custom str or
                # repr, use that instead.
                if (type(o).__str__ not in (i.__str__ for i in builtin_types) or
                    type(o).__repr__ not in (i.__repr__ for i in builtin_types)):
                    return False
                return all(_can_print(i) for i in o)
            elif isinstance(o, dict):
                return all(_can_print(i) and _can_print(o[i]) for i in o)
            elif isinstance(o, bool):
                return False
            elif isinstance(o, Printable):
                # types known to SymPy
                return True
            elif any(hasattr(o, hook) for hook in printing_hooks):
                # types which add support themselves
                return True
            elif isinstance(o, (float, int)) and print_builtin:
                return True
            return False
        except RuntimeError:
            return False
            # This is in case maximum recursion depth is reached.
            # Since RecursionError is for versions of Python 3.5+
            # so this is to guard against RecursionError for older versions.

    def _print_latex_png(o):
        """
        A function that returns a png rendered by an external latex
        distribution, falling back to matplotlib rendering
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _preview_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' Falling back to matplotlib backend')
                if latex_mode != 'inline':
                    s = latex(o, mode='inline', **settings)
                return _matplotlib_wrapper(s)

    def _print_latex_svg(o):
        """
        A function that returns a svg rendered by an external latex
        distribution, no fallback available.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                s = '$\\displaystyle %s$' % s
            try:
                return _svg_wrapper(s)
            except RuntimeError as e:
                debug('preview failed with:', repr(e),
                      ' No fallback available.')

    def _print_latex_matplotlib(o):
        """
        A function that returns a png rendered by mathtext
        """
        if _can_print(o):
            s = latex(o, mode='inline', **settings)
            return _matplotlib_wrapper(s)

    def _print_latex_text(o):
        """
        A function to generate the latex representation of SymPy expressions.
        """
        if _can_print(o):
            s = latex(o, mode=latex_mode, **settings)
            if latex_mode == 'plain':
                return '$\\displaystyle %s$' % s
            return s

    def _result_display(self, arg):
        """IPython's pretty-printer display hook, for use in IPython 0.10

           This function was adapted from:

            ipython/IPython/hooks.py:155

        """
        if self.rc.pprint:
            out = stringify_func(arg)

            if '\n' in out:
                print()

            print(out)
        else:
            print(repr(arg))

    import IPython
    if version_tuple(IPython.__version__) >= version_tuple('0.11'):

        # Printable is our own type, so we handle it with methods instead of
        # the approach required by builtin types. This allows downstream
        # packages to override the methods in their own subclasses of Printable,
        # which avoids the effects of gh-16002.
        printable_types = [float, tuple, list, set, frozenset, dict, int]

        plaintext_formatter = ip.display_formatter.formatters['text/plain']

        # Exception to the rule above: IPython has better dispatching rules
        # for plaintext printing (xref ipython/ipython#8938), and we can't
        # use `_repr_pretty_` without hitting a recursion error in _print_plain.
        for cls in printable_types + [Printable]:
            plaintext_formatter.for_type(cls, _print_plain)

        svg_formatter = ip.display_formatter.formatters['image/svg+xml']
        if use_latex in ('svg', ):
            debug("init_printing: using svg formatter")
            for cls in printable_types:
                svg_formatter.for_type(cls, _print_latex_svg)
            Printable._repr_svg_ = _print_latex_svg
        else:
            debug("init_printing: not using any svg formatter")
            for cls in printable_types:
                # Better way to set this, but currently does not work in IPython
                #png_formatter.for_type(cls, None)
                if cls in svg_formatter.type_printers:
                    svg_formatter.type_printers.pop(cls)
            Printable._repr_svg_ = Printable._repr_disabled

        png_formatter = ip.display_formatter.formatters['image/png']
        if use_latex in (True, 'png'):
            debug("init_printing: using png formatter")
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_png)
            Printable._repr_png_ = _print_latex_png
        elif use_latex == 'matplotlib':
            debug("init_printing: using matplotlib formatter")
            for cls in printable_types:
                png_formatter.for_type(cls, _print_latex_matplotlib)
            Printable._repr_png_ = _print_latex_matplotlib
        else:
            debug("init_printing: not using any png formatter")
            for cls in printable_types:
                # Better way to set this, but currently does not work in IPython
                #png_formatter.for_type(cls, None)
                if cls in png_formatter.type_printers:
                    png_formatter.type_printers.pop(cls)
            Printable._repr_png_ = Printable._repr_disabled

        latex_formatter = ip.display_formatter.formatters['text/latex']
        if use_latex in (True, 'mathjax'):
            debug("init_printing: using mathjax formatter")
            for cls in printable_types:
                latex_formatter.for_type(cls, _print_latex_text)
            Printable._repr_latex_ = _print_latex_text
        else:
            debug("init_printing: not using text/latex formatter")
            for cls in printable_types:
                # Better way to set this, but currently does not work in IPython
                #latex_formatter.for_type(cls, None)
                if cls in latex_formatter.type_printers:
                    latex_formatter.type_printers.pop(cls)
            Printable._repr_latex_ = Printable._repr_disabled

    else:
        ip.set_hook('result_display', _result_display)

def _is_ipython(shell):
    """Is a shell instance an IPython shell?"""
    # shortcut, so we don't import IPython if we don't have to
    from sys import modules
    if 'IPython' not in modules:
        return False
    try:
        from IPython.core.interactiveshell import InteractiveShell
    except ImportError:
        # IPython < 0.11
        try:
            from IPython.iplib import InteractiveShell
        except ImportError:
            # Reaching this points means IPython has changed in a backward-incompatible way
            # that we don't know about. Warn?
            return False
    return isinstance(shell, InteractiveShell)

# Used by the doctester to override the default for no_global
NO_GLOBAL = False

def init_printing(pretty_print=True, order=None, use_unicode=None,
                  use_latex=None, wrap_line=None, num_columns=None,
                  no_global=False, ip=None, euler=False, forecolor=None,
                  backcolor='Transparent', fontsize='10pt',
                  latex_mode='plain', print_builtin=True,
                  str_printer=None, pretty_printer=None,
                  latex_printer=None, scale=1.0, **settings):
    r"""
    Initializes pretty-printer depending on the environment.

    Parameters
    ==========

    pretty_print : bool, default=True
        If ``True``, use :func:`~.pretty_print` to stringify or the provided pretty
        printer; if ``False``, use :func:`~.sstrrepr` to stringify or the provided string
        printer.
    order : string or None, default='lex'
        There are a few different settings for this parameter:
        ``'lex'`` (default), which is lexographic order;
        ``'grlex'``, which is graded lexographic order;
        ``'grevlex'``, which is reversed graded lexographic order;
        ``'old'``, which is used for compatibility reasons and for long expressions;
        ``None``, which sets it to lex.
    use_unicode : bool or None, default=None
        If ``True``, use unicode characters;
        if ``False``, do not use unicode characters;
        if ``None``, make a guess based on the environment.
    use_latex : string, bool, or None, default=None
        If ``True``, use default LaTeX rendering in GUI interfaces (png and
        mathjax);
        if ``False``, do not use LaTeX rendering;
        if ``None``, make a guess based on the environment;
        if ``'png'``, enable LaTeX rendering with an external LaTeX compiler,
        falling back to matplotlib if external compilation fails;
        if ``'matplotlib'``, enable LaTeX rendering with matplotlib;
        if ``'mathjax'``, enable LaTeX text generation, for example MathJax
        rendering in IPython notebook or text rendering in LaTeX documents;
        if ``'svg'``, enable LaTeX rendering with an external latex compiler,
        no fallback
    wrap_line : bool
        If True, lines will wrap at the end; if False, they will not wrap
        but continue as one line. This is only relevant if ``pretty_print`` is
        True.
    num_columns : int or None, default=None
        If ``int``, number of columns before wrapping is set to num_columns; if
        ``None``, number of columns before wrapping is set to terminal width.
        This is only relevant if ``pretty_print`` is ``True``.
    no_global : bool, default=False
        If ``True``, the settings become system wide;
        if ``False``, use just for this console/session.
    ip : An interactive console
        This can either be an instance of IPython,
        or a class that derives from code.InteractiveConsole.
    euler : bool, optional, default=False
        Loads the euler package in the LaTeX preamble for handwritten style
        fonts (https://www.ctan.org/pkg/euler).
    forecolor : string or None, optional, default=None
        DVI setting for foreground color. ``None`` means that either ``'Black'``,
        ``'White'``, or ``'Gray'`` will be selected based on a guess of the IPython
        terminal color setting. See notes.
    backcolor : string, optional, default='Transparent'
        DVI setting for background color. See notes.
    fontsize : string or int, optional, default='10pt'
        A font size to pass to the LaTeX documentclass function in the
        preamble. Note that the options are limited by the documentclass.
        Consider using scale instead.
    latex_mode : string, optional, default='plain'
        The mode used in the LaTeX printer. Can be one of:
        ``{'inline'|'plain'|'equation'|'equation*'}``.
    print_builtin : boolean, optional, default=True
        If ``True`` then floats and integers will be printed. If ``False`` the
        printer will only print SymPy types.
    str_printer : function, optional, default=None
        A custom string printer function. This should mimic
        :func:`~.sstrrepr()`.
    pretty_printer : function, optional, default=None
        A custom pretty printer. This should mimic :func:`~.pretty()`.
    latex_printer : function, optional, default=None
        A custom LaTeX printer. This should mimic :func:`~.latex()`.
    scale : float, optional, default=1.0
        Scale the LaTeX output when using the ``'png'`` or ``'svg'`` backends.
        Useful for high dpi screens.
    settings :
        Any additional settings for the ``latex`` and ``pretty`` commands can
        be used to fine-tune the output.

    Examples
    ========

    >>> from sympy.interactive import init_printing
    >>> from sympy import Symbol, sqrt
    >>> from sympy.abc import x, y
    >>> sqrt(5)
    sqrt(5)
    >>> init_printing(pretty_print=True) # doctest: +SKIP
    >>> sqrt(5) # doctest: +SKIP
      ___
    \/ 5
    >>> theta = Symbol('theta') # doctest: +SKIP
    >>> init_printing(use_unicode=True) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    \u03b8
    >>> init_printing(use_unicode=False) # doctest: +SKIP
    >>> theta # doctest: +SKIP
    theta
    >>> init_printing(order='lex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grlex') # doctest: +SKIP
    >>> str(y + x + y**2 + x**2) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(order='grevlex') # doctest: +SKIP
    >>> str(y * x**2 + x * y**2) # doctest: +SKIP
    x**2*y + x*y**2
    >>> init_printing(order='old') # doctest: +SKIP
    >>> str(x**2 + y**2 + x + y) # doctest: +SKIP
    x**2 + x + y**2 + y
    >>> init_printing(num_columns=10) # doctest: +SKIP
    >>> x**2 + x + y**2 + y # doctest: +SKIP
    x + y +
    x**2 + y**2

    Notes
    =====

    The foreground and background colors can be selected when using ``'png'`` or
    ``'svg'`` LaTeX rendering. Note that before the ``init_printing`` command is
    executed, the LaTeX rendering is handled by the IPython console and not SymPy.

    The colors can be selected among the 68 standard colors known to ``dvips``,
    for a list see [1]_. In addition, the background color can be
    set to  ``'Transparent'`` (which is the default value).

    When using the ``'Auto'`` foreground color, the guess is based on the
    ``colors`` variable in the IPython console, see [2]_. Hence, if
    that variable is set correctly in your IPython console, there is a high
    chance that the output will be readable, although manual settings may be
    needed.


    References
    ==========

    .. [1] https://en.wikibooks.org/wiki/LaTeX/Colors#The_68_standard_colors_known_to_dvips

    .. [2] https://ipython.readthedocs.io/en/stable/config/details.html#terminal-colors

    See Also
    ========

    sympy.printing.latex
    sympy.printing.pretty

    """
    import sys
    from sympy.printing.printer import Printer

    if pretty_print:
        if pretty_printer is not None:
            stringify_func = pretty_printer
        else:
            from sympy.printing import pretty as stringify_func
    else:
        if str_printer is not None:
            stringify_func = str_printer
        else:
            from sympy.printing import sstrrepr as stringify_func

    # Even if ip is not passed, double check that not in IPython shell
    in_ipython = False
    if ip is None:
        try:
            ip = get_ipython()
        except NameError:
            pass
        else:
            in_ipython = (ip is not None)

    if ip and not in_ipython:
        in_ipython = _is_ipython(ip)

    if in_ipython and pretty_print:
        try:
            import IPython
            # IPython 1.0 deprecates the frontend module, so we import directly
            # from the terminal module to prevent a deprecation message from being
            # shown.
            if version_tuple(IPython.__version__) >= version_tuple('1.0'):
                from IPython.terminal.interactiveshell import TerminalInteractiveShell
            else:
                from IPython.frontend.terminal.interactiveshell import TerminalInteractiveShell
            from code import InteractiveConsole
        except ImportError:
            pass
        else:
            # This will be True if we are in the qtconsole or notebook
            if not isinstance(ip, (InteractiveConsole, TerminalInteractiveShell)) \
                    and 'ipython-console' not in ''.join(sys.argv):
                if use_unicode is None:
                    debug("init_printing: Setting use_unicode to True")
                    use_unicode = True
                if use_latex is None:
                    debug("init_printing: Setting use_latex to True")
                    use_latex = True

    if not NO_GLOBAL and not no_global:
        Printer.set_global_settings(order=order, use_unicode=use_unicode,
                                    wrap_line=wrap_line, num_columns=num_columns)
    else:
        _stringify_func = stringify_func

        if pretty_print:
            stringify_func = lambda expr, **settings: \
                             _stringify_func(expr, order=order,
                                             use_unicode=use_unicode,
                                             wrap_line=wrap_line,
                                             num_columns=num_columns,
                                             **settings)
        else:
            stringify_func = \
                lambda expr, **settings: _stringify_func(
                    expr, order=order, **settings)

    if in_ipython:
        mode_in_settings = settings.pop("mode", None)
        if mode_in_settings:
            debug("init_printing: Mode is not able to be set due to internals"
                  "of IPython printing")
        _init_ipython_printing(ip, stringify_func, use_latex, euler,
                               forecolor, backcolor, fontsize, latex_mode,
                               print_builtin, latex_printer, scale,
                               **settings)
    else:
        _init_python_printing(stringify_func, **settings)
