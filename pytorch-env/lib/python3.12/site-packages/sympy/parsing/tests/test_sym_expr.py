from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises
from sympy.external import import_module

lfortran = import_module('lfortran')
cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

if lfortran and cin:
    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Declaration, FloatType)
    from sympy.core import Integer, Float
    from sympy.core.symbol import Symbol

    expr1 = SymPyExpression()
    src = """\
    integer :: a, b, c, d
    real :: p, q, r, s
    """

    def test_c_parse():
        src1 = """\
        int a, b = 4;
        float c, d = 2.4;
        """
        expr1.convert_to_expr(src1, 'c')
        ls = expr1.return_expr()

        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc'))
            )
        )
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(4)
            )
        )
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
            )
        )
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.3999999999999999', precision=53)
            )
        )


    def test_fortran_parse():
        expr = SymPyExpression(src, 'f')
        ls = expr.return_expr()

        assert ls[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[2] == Declaration(
            Variable(
                Symbol('c'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[3] == Declaration(
            Variable(
                Symbol('d'),
                type=IntBaseType(String('integer')),
                value=Integer(0)
            )
        )
        assert ls[4] == Declaration(
            Variable(
                Symbol('p'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        assert ls[5] == Declaration(
            Variable(
                Symbol('q'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        assert ls[6] == Declaration(
            Variable(
                Symbol('r'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )
        assert ls[7] == Declaration(
            Variable(
                Symbol('s'),
                type=FloatBaseType(String('real')),
                value=Float('0.0', precision=53)
            )
        )


    def test_convert_py():
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        exp_py = expr1.convert_to_python()
        assert exp_py == [
            'a = 0',
            'b = 0',
            'c = 0',
            'd = 0',
            'p = 0.0',
            'q = 0.0',
            'r = 0.0',
            's = 0.0',
            'a = b + c',
            's = p*q/r'
        ]


    def test_convert_fort():
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        exp_fort = expr1.convert_to_fortran()
        assert exp_fort == [
            '      integer*4 a',
            '      integer*4 b',
            '      integer*4 c',
            '      integer*4 d',
            '      real*8 p',
            '      real*8 q',
            '      real*8 r',
            '      real*8 s',
            '      a = b + c',
            '      s = p*q/r'
        ]


    def test_convert_c():
        src1 = (
            src +
            """\
            a = b + c
            s = p * q / r
            """
        )
        expr1.convert_to_expr(src1, 'f')
        exp_c = expr1.convert_to_c()
        assert exp_c == [
            'int a = 0',
            'int b = 0',
            'int c = 0',
            'int d = 0',
            'double p = 0.0',
            'double q = 0.0',
            'double r = 0.0',
            'double s = 0.0',
            'a = b + c;',
            's = p*q/r;'
        ]


    def test_exceptions():
        src = 'int a;'
        raises(ValueError, lambda: SymPyExpression(src))
        raises(ValueError, lambda: SymPyExpression(mode = 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(src, mode = 'd'))

elif not lfortran and not cin:
    def test_raise():
        raises(ImportError, lambda: SymPyExpression('int a;', 'c'))
        raises(ImportError, lambda: SymPyExpression('integer :: a', 'f'))
