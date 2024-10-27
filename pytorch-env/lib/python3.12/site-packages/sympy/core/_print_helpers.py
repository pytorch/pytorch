"""
Base class to provide str and repr hooks that `init_printing` can overwrite.

This is exposed publicly in the `printing.defaults` module,
but cannot be defined there without causing circular imports.
"""

class Printable:
    """
    The default implementation of printing for SymPy classes.

    This implements a hack that allows us to print elements of built-in
    Python containers in a readable way. Natively Python uses ``repr()``
    even if ``str()`` was explicitly requested. Mix in this trait into
    a class to get proper default printing.

    This also adds support for LaTeX printing in jupyter notebooks.
    """

    # Since this class is used as a mixin we set empty slots. That means that
    # instances of any subclasses that use slots will not need to have a
    # __dict__.
    __slots__ = ()

    # Note, we always use the default ordering (lex) in __str__ and __repr__,
    # regardless of the global setting. See issue 5487.
    def __str__(self):
        from sympy.printing.str import sstr
        return sstr(self, order=None)

    __repr__ = __str__

    def _repr_disabled(self):
        """
        No-op repr function used to disable jupyter display hooks.

        When :func:`sympy.init_printing` is used to disable certain display
        formats, this function is copied into the appropriate ``_repr_*_``
        attributes.

        While we could just set the attributes to `None``, doing it this way
        allows derived classes to call `super()`.
        """
        return None

    # We don't implement _repr_png_ here because it would add a large amount of
    # data to any notebook containing SymPy expressions, without adding
    # anything useful to the notebook. It can still enabled manually, e.g.,
    # for the qtconsole, with init_printing().
    _repr_png_ = _repr_disabled

    _repr_svg_ = _repr_disabled

    def _repr_latex_(self):
        """
        IPython/Jupyter LaTeX printing

        To change the behavior of this (e.g., pass in some settings to LaTeX),
        use init_printing(). init_printing() will also enable LaTeX printing
        for built in numeric types like ints and container types that contain
        SymPy objects, like lists and dictionaries of expressions.
        """
        from sympy.printing.latex import latex
        s = latex(self, mode='plain')
        return "$\\displaystyle %s$" % s
