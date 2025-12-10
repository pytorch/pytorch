"""Printing subsystem driver

SymPy's printing system works the following way: Any expression can be
passed to a designated Printer who then is responsible to return an
adequate representation of that expression.

**The basic concept is the following:**

1.  Let the object print itself if it knows how.
2.  Take the best fitting method defined in the printer.
3.  As fall-back use the emptyPrinter method for the printer.

Which Method is Responsible for Printing?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The whole printing process is started by calling ``.doprint(expr)`` on the printer
which you want to use. This method looks for an appropriate method which can
print the given expression in the given style that the printer defines.
While looking for the method, it follows these steps:

1.  **Let the object print itself if it knows how.**

    The printer looks for a specific method in every object. The name of that method
    depends on the specific printer and is defined under ``Printer.printmethod``.
    For example, StrPrinter calls ``_sympystr`` and LatexPrinter calls ``_latex``.
    Look at the documentation of the printer that you want to use.
    The name of the method is specified there.

    This was the original way of doing printing in sympy. Every class had
    its own latex, mathml, str and repr methods, but it turned out that it
    is hard to produce a high quality printer, if all the methods are spread
    out that far. Therefore all printing code was combined into the different
    printers, which works great for built-in SymPy objects, but not that
    good for user defined classes where it is inconvenient to patch the
    printers.

2.  **Take the best fitting method defined in the printer.**

    The printer loops through expr classes (class + its bases), and tries
    to dispatch the work to ``_print_<EXPR_CLASS>``

    e.g., suppose we have the following class hierarchy::

            Basic
            |
            Atom
            |
            Number
            |
        Rational

    then, for ``expr=Rational(...)``, the Printer will try
    to call printer methods in the order as shown in the figure below::

        p._print(expr)
        |
        |-- p._print_Rational(expr)
        |
        |-- p._print_Number(expr)
        |
        |-- p._print_Atom(expr)
        |
        `-- p._print_Basic(expr)

    if ``._print_Rational`` method exists in the printer, then it is called,
    and the result is returned back. Otherwise, the printer tries to call
    ``._print_Number`` and so on.

3.  **As a fall-back use the emptyPrinter method for the printer.**

    As fall-back ``self.emptyPrinter`` will be called with the expression. If
    not defined in the Printer subclass this will be the same as ``str(expr)``.

.. _printer_example:

Example of Custom Printer
^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, we have a printer which prints the derivative of a function
in a shorter form.

.. code-block:: python

    from sympy.core.symbol import Symbol
    from sympy.printing.latex import LatexPrinter, print_latex
    from sympy.core.function import UndefinedFunction, Function


    class MyLatexPrinter(LatexPrinter):
        \"\"\"Print derivative of a function of symbols in a shorter form.
        \"\"\"
        def _print_Derivative(self, expr):
            function, *vars = expr.args
            if not isinstance(type(function), UndefinedFunction) or \\
               not all(isinstance(i, Symbol) for i in vars):
                return super()._print_Derivative(expr)

            # If you want the printer to work correctly for nested
            # expressions then use self._print() instead of str() or latex().
            # See the example of nested modulo below in the custom printing
            # method section.
            return "{}_{{{}}}".format(
                self._print(Symbol(function.func.__name__)),
                            ''.join(self._print(i) for i in vars))


    def print_my_latex(expr):
        \"\"\" Most of the printers define their own wrappers for print().
        These wrappers usually take printer settings. Our printer does not have
        any settings.
        \"\"\"
        print(MyLatexPrinter().doprint(expr))


    y = Symbol("y")
    x = Symbol("x")
    f = Function("f")
    expr = f(x, y).diff(x, y)

    # Print the expression using the normal latex printer and our custom
    # printer.
    print_latex(expr)
    print_my_latex(expr)

The output of the code above is::

    \\frac{\\partial^{2}}{\\partial x\\partial y}  f{\\left(x,y \\right)}
    f_{xy}

.. _printer_method_example:

Example of Custom Printing Method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the example below, the latex printing of the modulo operator is modified.
This is done by overriding the method ``_latex`` of ``Mod``.

>>> from sympy import Symbol, Mod, Integer, print_latex

>>> # Always use printer._print()
>>> class ModOp(Mod):
...     def _latex(self, printer):
...         a, b = [printer._print(i) for i in self.args]
...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

Comparing the output of our custom operator to the builtin one:

>>> x = Symbol('x')
>>> m = Symbol('m')
>>> print_latex(Mod(x, m))
x \\bmod m
>>> print_latex(ModOp(x, m))
\\operatorname{Mod}{\\left(x, m\\right)}

Common mistakes
~~~~~~~~~~~~~~~
It's important to always use ``self._print(obj)`` to print subcomponents of
an expression when customizing a printer. Mistakes include:

1.  Using ``self.doprint(obj)`` instead:

    >>> # This example does not work properly, as only the outermost call may use
    >>> # doprint.
    >>> class ModOpModeWrong(Mod):
    ...     def _latex(self, printer):
    ...         a, b = [printer.doprint(i) for i in self.args]
    ...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

    This fails when the ``mode`` argument is passed to the printer:

    >>> print_latex(ModOp(x, m), mode='inline')  # ok
    $\\operatorname{Mod}{\\left(x, m\\right)}$
    >>> print_latex(ModOpModeWrong(x, m), mode='inline')  # bad
    $\\operatorname{Mod}{\\left($x$, $m$\\right)}$

2.  Using ``str(obj)`` instead:

    >>> class ModOpNestedWrong(Mod):
    ...     def _latex(self, printer):
    ...         a, b = [str(i) for i in self.args]
    ...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

    This fails on nested objects:

    >>> # Nested modulo.
    >>> print_latex(ModOp(ModOp(x, m), Integer(7)))  # ok
    \\operatorname{Mod}{\\left(\\operatorname{Mod}{\\left(x, m\\right)}, 7\\right)}
    >>> print_latex(ModOpNestedWrong(ModOpNestedWrong(x, m), Integer(7)))  # bad
    \\operatorname{Mod}{\\left(ModOpNestedWrong(x, m), 7\\right)}

3.  Using ``LatexPrinter()._print(obj)`` instead.

    >>> from sympy.printing.latex import LatexPrinter
    >>> class ModOpSettingsWrong(Mod):
    ...     def _latex(self, printer):
    ...         a, b = [LatexPrinter()._print(i) for i in self.args]
    ...         return r"\\operatorname{Mod}{\\left(%s, %s\\right)}" % (a, b)

    This causes all the settings to be discarded in the subobjects. As an
    example, the ``full_prec`` setting which shows floats to full precision is
    ignored:

    >>> from sympy import Float
    >>> print_latex(ModOp(Float(1) * x, m), full_prec=True)  # ok
    \\operatorname{Mod}{\\left(1.00000000000000 x, m\\right)}
    >>> print_latex(ModOpSettingsWrong(Float(1) * x, m), full_prec=True)  # bad
    \\operatorname{Mod}{\\left(1.0 x, m\\right)}

"""

from __future__ import annotations
import sys
from typing import Any, Type
import inspect
from contextlib import contextmanager
from functools import cmp_to_key, update_wrapper

from sympy.core.add import Add
from sympy.core.basic import Basic

from sympy.core.function import AppliedUndef, UndefinedFunction, Function



@contextmanager
def printer_context(printer, **kwargs):
    original = printer._context.copy()
    try:
        printer._context.update(kwargs)
        yield
    finally:
        printer._context = original


class Printer:
    """ Generic printer

    Its job is to provide infrastructure for implementing new printers easily.

    If you want to define your custom Printer or your custom printing method
    for your custom class then see the example above: printer_example_ .
    """

    _global_settings: dict[str, Any] = {}

    _default_settings: dict[str, Any] = {}

    # must be initialized to pass tests and cannot be set to '| None' to pass mypy
    printmethod = None  # type: str

    @classmethod
    def _get_initial_settings(cls):
        settings = cls._default_settings.copy()
        for key, val in cls._global_settings.items():
            if key in cls._default_settings:
                settings[key] = val
        return settings

    def __init__(self, settings=None):
        self._str = str

        self._settings = self._get_initial_settings()
        self._context = {}  # mutable during printing

        if settings is not None:
            self._settings.update(settings)

            if len(self._settings) > len(self._default_settings):
                for key in self._settings:
                    if key not in self._default_settings:
                        raise TypeError("Unknown setting '%s'." % key)

        # _print_level is the number of times self._print() was recursively
        # called. See StrPrinter._print_Float() for an example of usage
        self._print_level = 0

    @classmethod
    def set_global_settings(cls, **settings):
        """Set system-wide printing settings. """
        for key, val in settings.items():
            if val is not None:
                cls._global_settings[key] = val

    @property
    def order(self):
        if 'order' in self._settings:
            return self._settings['order']
        else:
            raise AttributeError("No order defined.")

    def doprint(self, expr):
        """Returns printer's representation for expr (as a string)"""
        return self._str(self._print(expr))

    def _print(self, expr, **kwargs) -> str:
        """Internal dispatcher

        Tries the following concepts to print an expression:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
        self._print_level += 1
        try:
            # If the printer defines a name for a printing method
            # (Printer.printmethod) and the object knows for itself how it
            # should be printed, use that method.
            if self.printmethod and hasattr(expr, self.printmethod):
                if not (isinstance(expr, type) and issubclass(expr, Basic)):
                    return getattr(expr, self.printmethod)(self, **kwargs)

            # See if the class of expr is known, or if one of its super
            # classes is known, and use that print function
            # Exception: ignore the subclasses of Undefined, so that, e.g.,
            # Function('gamma') does not get dispatched to _print_gamma
            classes = type(expr).__mro__
            if AppliedUndef in classes:
                classes = classes[classes.index(AppliedUndef):]
            if UndefinedFunction in classes:
                classes = classes[classes.index(UndefinedFunction):]
            # Another exception: if someone subclasses a known function, e.g.,
            # gamma, and changes the name, then ignore _print_gamma
            if Function in classes:
                i = classes.index(Function)
                classes = tuple(c for c in classes[:i] if \
                    c.__name__ == classes[0].__name__ or \
                    c.__name__.endswith("Base")) + classes[i:]
            for cls in classes:
                printmethodname = '_print_' + cls.__name__
                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            # Unknown object, fall back to the emptyPrinter.
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1

    def emptyPrinter(self, expr):
        return str(expr)

    def _as_ordered_terms(self, expr, order=None):
        """A compatibility function for ordering terms in Add. """
        order = order or self.order

        if order == 'old':
            return sorted(Add.make_args(expr), key=cmp_to_key(self._compare_pretty))
        elif order == 'none':
            return list(expr.args)
        else:
            return expr.as_ordered_terms(order=order)

    def _compare_pretty(self, a, b):
        """return -1, 0, 1 if a is canonically less, equal or
        greater than b. This is used when 'order=old' is selected
        for printing. This puts Order last, orders Rationals
        according to value, puts terms in order wrt the power of
        the last power appearing in a term. Ties are broken using
        Basic.compare.
        """
        from sympy.core.numbers import Rational
        from sympy.core.symbol import Wild
        from sympy.series.order import Order
        if isinstance(a, Order) and not isinstance(b, Order):
            return 1
        if not isinstance(a, Order) and isinstance(b, Order):
            return -1

        if isinstance(a, Rational) and isinstance(b, Rational):
            l = a.p * b.q
            r = b.p * a.q
            return (l > r) - (l < r)
        else:
            p1, p2, p3 = Wild("p1"), Wild("p2"), Wild("p3")
            r_a = a.match(p1 * p2**p3)
            if r_a and p3 in r_a:
                a3 = r_a[p3]
                r_b = b.match(p1 * p2**p3)
                if r_b and p3 in r_b:
                    b3 = r_b[p3]
                    c = Basic.compare(a3, b3)
                    if c != 0:
                        return c

        # break ties
        return Basic.compare(a, b)


class _PrintFunction:
    """
    Function wrapper to replace ``**settings`` in the signature with printer defaults
    """
    def __init__(self, f, print_cls: Type[Printer]):
        # find all the non-setting arguments
        params = list(inspect.signature(f).parameters.values())
        assert params.pop(-1).kind == inspect.Parameter.VAR_KEYWORD
        self.__other_params = params

        self.__print_cls = print_cls
        update_wrapper(self, f)

    def __reduce__(self):
        # Since this is used as a decorator, it replaces the original function.
        # The default pickling will try to pickle self.__wrapped__ and fail
        # because the wrapped function can't be retrieved by name.
        return self.__wrapped__.__qualname__

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)

    @property
    def __signature__(self) -> inspect.Signature:
        settings = self.__print_cls._get_initial_settings()
        return inspect.Signature(
            parameters=self.__other_params + [
                inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY, default=v)
                for k, v in settings.items()
            ],
            return_annotation=self.__wrapped__.__annotations__.get('return', inspect.Signature.empty)  # type:ignore
        )


def print_function(print_cls):
    """ A decorator to replace kwargs with the printer settings in __signature__ """
    def decorator(f):
        if sys.version_info < (3, 9):
            # We have to create a subclass so that `help` actually shows the docstring in older Python versions.
            # IPython and Sphinx do not need this, only a raw Python console.
            cls = type(f'{f.__qualname__}_PrintFunction', (_PrintFunction,), {"__doc__": f.__doc__})
        else:
            cls = _PrintFunction
        return cls(f, print_cls)
    return decorator
