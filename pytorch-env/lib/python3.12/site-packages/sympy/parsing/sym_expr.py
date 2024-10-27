from sympy.printing import pycode, ccode, fcode
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on

lfortran = import_module('lfortran')
cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

if lfortran:
    from sympy.parsing.fortran.fortran_parser import src_to_sympy
if cin:
    from sympy.parsing.c.c_parser import parse_c

@doctest_depends_on(modules=['lfortran', 'clang.cindex'])
class SymPyExpression:  # type: ignore
    """Class to store and handle SymPy expressions

    This class will hold SymPy Expressions and handle the API for the
    conversion to and from different languages.

    It works with the C and the Fortran Parser to generate SymPy expressions
    which are stored here and which can be converted to multiple language's
    source code.

    Notes
    =====

    The module and its API are currently under development and experimental
    and can be changed during development.

    The Fortran parser does not support numeric assignments, so all the
    variables have been Initialized to zero.

    The module also depends on external dependencies:

    - LFortran which is required to use the Fortran parser
    - Clang which is required for the C parser

    Examples
    ========

    Example of parsing C code:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... int a,b;
    ... float c = 2, d =4;
    ... '''
    >>> a = SymPyExpression(src, 'c')
    >>> a.return_expr()
    [Declaration(Variable(a, type=intc)),
    Declaration(Variable(b, type=intc)),
    Declaration(Variable(c, type=float32, value=2.0)),
    Declaration(Variable(d, type=float32, value=4.0))]

    An example of variable definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src2 = '''
    ... integer :: a, b, c, d
    ... real :: p, q, r, s
    ... '''
    >>> p = SymPyExpression()
    >>> p.convert_to_expr(src2, 'f')
    >>> p.convert_to_c()
    ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0']

    An example of Assignment:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src3 = '''
    ... integer :: a, b, c, d, e
    ... d = a + b - c
    ... e = b * d + c * e / a
    ... '''
    >>> p = SymPyExpression(src3, 'f')
    >>> p.convert_to_python()
    ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'e = 0', 'd = a + b - c', 'e = b*d + c*e/a']

    An example of function definition:

    >>> from sympy.parsing.sym_expr import SymPyExpression
    >>> src = '''
    ... integer function f(a,b)
    ... integer, intent(in) :: a, b
    ... integer :: r
    ... end function
    ... '''
    >>> a = SymPyExpression(src, 'f')
    >>> a.convert_to_python()
    ['def f(a, b):\\n   f = 0\\n    r = 0\\n    return f']

    """

    def __init__(self, source_code = None, mode = None):
        """Constructor for SymPyExpression class"""
        super().__init__()
        if not(mode or source_code):
            self._expr = []
        elif mode:
            if source_code:
                if mode.lower() == 'f':
                    if lfortran:
                        self._expr = src_to_sympy(source_code)
                    else:
                        raise ImportError("LFortran is not installed, cannot parse Fortran code")
                elif mode.lower() == 'c':
                    if cin:
                        self._expr = parse_c(source_code)
                    else:
                        raise ImportError("Clang is not installed, cannot parse C code")
                else:
                    raise NotImplementedError(
                        'Parser for specified language is not implemented'
                    )
            else:
                raise ValueError('Source code not present')
        else:
            raise ValueError('Please specify a mode for conversion')

    def convert_to_expr(self, src_code, mode):
        """Converts the given source code to SymPy Expressions

        Attributes
        ==========

        src_code : String
            the source code or filename of the source code that is to be
            converted

        mode: String
            the mode to determine which parser is to be used according to
            the language of the source code
            f or F for Fortran
            c or C for C/C++

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b) result(r)
        ... integer, intent(in) :: a, b
        ... integer :: x
        ... r = a + b -x
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(r, type=integer, value=0)),
        Declaration(Variable(x, type=integer, value=0)),
        Assignment(Variable(r), a + b - x),
        Return(Variable(r))
        ))]




        """
        if mode.lower() == 'f':
            if lfortran:
                self._expr = src_to_sympy(src_code)
            else:
                raise ImportError("LFortran is not installed, cannot parse Fortran code")
        elif mode.lower() == 'c':
            if cin:
                self._expr = parse_c(src_code)
            else:
                raise ImportError("Clang is not installed, cannot parse C code")
        else:
            raise NotImplementedError(
                "Parser for specified language has not been implemented"
            )

    def convert_to_python(self):
        """Returns a list with Python code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_python()
        ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'p = 0.0', 'q = 0.0', 'r = 0.0', 's = 0.0', 'c = a/b', 'd = c/a', 's = p/q', 'r = q/p']

        """
        self._pycode = []
        for iter in self._expr:
            self._pycode.append(pycode(iter))
        return self._pycode

    def convert_to_c(self):
        """Returns a list with the c source code for the SymPy expressions


        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src2, 'f')
        >>> p.convert_to_c()
        ['int a = 0', 'int b = 0', 'int c = 0', 'int d = 0', 'double p = 0.0', 'double q = 0.0', 'double r = 0.0', 'double s = 0.0', 'c = a/b;', 'd = c/a;', 's = p/q;', 'r = q/p;']

        """
        self._ccode = []
        for iter in self._expr:
            self._ccode.append(ccode(iter))
        return self._ccode

    def convert_to_fortran(self):
        """Returns a list with the fortran source code for the SymPy expressions

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src2 = '''
        ... integer :: a, b, c, d
        ... real :: p, q, r, s
        ... c = a/b
        ... d = c/a
        ... s = p/q
        ... r = q/p
        ... '''
        >>> p = SymPyExpression(src2, 'f')
        >>> p.convert_to_fortran()
        ['      integer*4 a', '      integer*4 b', '      integer*4 c', '      integer*4 d', '      real*8 p', '      real*8 q', '      real*8 r', '      real*8 s', '      c = a/b', '      d = c/a', '      s = p/q', '      r = q/p']

        """
        self._fcode = []
        for iter in self._expr:
            self._fcode.append(fcode(iter))
        return self._fcode

    def return_expr(self):
        """Returns the expression list

        Examples
        ========

        >>> from sympy.parsing.sym_expr import SymPyExpression
        >>> src3 = '''
        ... integer function f(a,b)
        ... integer, intent(in) :: a, b
        ... integer :: r
        ... r = a+b
        ... f = r
        ... end function
        ... '''
        >>> p = SymPyExpression()
        >>> p.convert_to_expr(src3, 'f')
        >>> p.return_expr()
        [FunctionDefinition(integer, name=f, parameters=(Variable(a), Variable(b)), body=CodeBlock(
        Declaration(Variable(f, type=integer, value=0)),
        Declaration(Variable(r, type=integer, value=0)),
        Assignment(Variable(f), Variable(r)),
        Return(Variable(f))
        ))]

        """
        return self._expr
