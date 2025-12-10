from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module

cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

if cin:
    from sympy.codegen.ast import (Variable, String, Return,
        FunctionDefinition, Integer, Float, Declaration, CodeBlock,
        FunctionPrototype, FunctionCall, NoneToken, Assignment, Type,
        IntBaseType, SignedIntType, UnsignedIntType, FloatType,
        AddAugmentedAssignment, SubAugmentedAssignment,
        MulAugmentedAssignment, DivAugmentedAssignment,
        ModAugmentedAssignment, While)
    from sympy.codegen.cnodes import (PreDecrement, PostDecrement,
        PreIncrement, PostIncrement)
    from sympy.core import (Add, Mul, Mod, Pow, Rational,
        StrictLessThan, LessThan, StrictGreaterThan, GreaterThan,
        Equality, Unequality)
    from sympy.logic.boolalg import And, Not, Or
    from sympy.core.symbol import Symbol
    from sympy.logic.boolalg import (false, true)
    import os

    def test_variable():
        c_src1 = (
            'int a;' + '\n' +
            'int b;' + '\n'
        )
        c_src2 = (
            'float a;' + '\n'
            + 'float b;' + '\n'
        )
        c_src3 = (
            'int a;' + '\n' +
            'float b;' + '\n' +
            'int c;'
        )
        c_src4 = (
            'int x = 1, y = 6.78;' + '\n' +
            'float p = 2, q = 9.67;'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()

        assert res1[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc'))
            )
        )

        assert res1[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc'))
            )
        )

        assert res2[0] == Declaration(
            Variable(
                Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
            )
        )
        assert res2[1] == Declaration(
            Variable(
                Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
            )
        )

        assert res3[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc'))
            )
        )

        assert res3[1] == Declaration(
            Variable(
                Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
            )
        )

        assert res3[2] == Declaration(
            Variable(
                Symbol('c'),
                type=IntBaseType(String('intc'))
            )
        )

        assert res4[0] == Declaration(
            Variable(
                Symbol('x'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
            )
        )

        assert res4[1] == Declaration(
            Variable(
                Symbol('y'),
                type=IntBaseType(String('intc')),
                value=Integer(6)
            )
        )

        assert res4[2] == Declaration(
            Variable(
                Symbol('p'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.0', precision=53)
            )
        )

        assert res4[3] == Declaration(
            Variable(
                Symbol('q'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('9.67', precision=53)
            )
        )


    def test_int():
        c_src1 = 'int a = 1;'
        c_src2 = (
            'int a = 1;' + '\n' +
            'int b = 2;' + '\n'
        )
        c_src3 = 'int a = 2.345, b = 5.67;'
        c_src4 = 'int p = 6, q = 23.45;'
        c_src5 = "int x = '0', y = 'a';"
        c_src6 = "int r = true, s = false;"

        # cin.TypeKind.UCHAR
        c_src_type1 = (
            "signed char a = 1, b = 5.1;"
            )

        # cin.TypeKind.SHORT
        c_src_type2 = (
            "short a = 1, b = 5.1;"
            "signed short c = 1, d = 5.1;"
            "short int e = 1, f = 5.1;"
            "signed short int g = 1, h = 5.1;"
            )

        # cin.TypeKind.INT
        c_src_type3 = (
            "signed int a = 1, b = 5.1;"
            "int c = 1, d = 5.1;"
            )

        # cin.TypeKind.LONG
        c_src_type4 = (
            "long a = 1, b = 5.1;"
            "long int c = 1, d = 5.1;"
            )

        # cin.TypeKind.UCHAR
        c_src_type5 = "unsigned char a = 1, b = 5.1;"

        # cin.TypeKind.USHORT
        c_src_type6 = (
            "unsigned short a = 1, b = 5.1;"
            "unsigned short int c = 1, d = 5.1;"
            )

        # cin.TypeKind.UINT
        c_src_type7 = "unsigned int a = 1, b = 5.1;"

        # cin.TypeKind.ULONG
        c_src_type8 = (
            "unsigned long a = 1, b = 5.1;"
            "unsigned long int c = 1, d = 5.1;"
            )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()
        res5 = SymPyExpression(c_src5, 'c').return_expr()
        res6 = SymPyExpression(c_src6, 'c').return_expr()

        res_type1 = SymPyExpression(c_src_type1, 'c').return_expr()
        res_type2 = SymPyExpression(c_src_type2, 'c').return_expr()
        res_type3 = SymPyExpression(c_src_type3, 'c').return_expr()
        res_type4 = SymPyExpression(c_src_type4, 'c').return_expr()
        res_type5 = SymPyExpression(c_src_type5, 'c').return_expr()
        res_type6 = SymPyExpression(c_src_type6, 'c').return_expr()
        res_type7 = SymPyExpression(c_src_type7, 'c').return_expr()
        res_type8 = SymPyExpression(c_src_type8, 'c').return_expr()

        assert res1[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
            )
        )

        assert res2[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
            )
        )

        assert res2[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
            )
        )

        assert res3[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
            )
        )

        assert res3[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(5)
            )
        )

        assert res4[0] == Declaration(
            Variable(
                Symbol('p'),
                type=IntBaseType(String('intc')),
                value=Integer(6)
            )
        )

        assert res4[1] == Declaration(
            Variable(
                Symbol('q'),
                type=IntBaseType(String('intc')),
                value=Integer(23)
            )
        )

        assert res5[0] == Declaration(
            Variable(
                Symbol('x'),
                type=IntBaseType(String('intc')),
                value=Integer(48)
            )
        )

        assert res5[1] == Declaration(
            Variable(
                Symbol('y'),
                type=IntBaseType(String('intc')),
                value=Integer(97)
            )
        )

        assert res6[0] == Declaration(
            Variable(
                Symbol('r'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
            )
        )

        assert res6[1] == Declaration(
            Variable(
                Symbol('s'),
                type=IntBaseType(String('intc')),
                value=Integer(0)
            )
        )

        assert res_type1[0] == Declaration(
            Variable(
                Symbol('a'),
                type=SignedIntType(
                    String('int8'),
                    nbits=Integer(8)
                    ),
                value=Integer(1)
                )
            )

        assert res_type1[1] == Declaration(
            Variable(
                Symbol('b'),
                type=SignedIntType(
                    String('int8'),
                    nbits=Integer(8)
                    ),
                value=Integer(5)
                )
            )

        assert res_type2[0] == Declaration(
            Variable(
                Symbol('a'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(1)
                )
            )

        assert res_type2[1] == Declaration(
            Variable(
                Symbol('b'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(5)
                )
            )

        assert res_type2[2] == Declaration(
            Variable(Symbol('c'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(1)
                )
            )

        assert res_type2[3] == Declaration(
            Variable(
                Symbol('d'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(5)
                )
            )

        assert res_type2[4] == Declaration(
            Variable(
                Symbol('e'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(1)
                )
            )

        assert res_type2[5] == Declaration(
            Variable(
                Symbol('f'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(5)
                )
            )

        assert res_type2[6] == Declaration(
            Variable(
                Symbol('g'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(1)
                )
            )

        assert res_type2[7] == Declaration(
            Variable(
                Symbol('h'),
                type=SignedIntType(
                    String('int16'),
                    nbits=Integer(16)
                    ),
                value=Integer(5)
                )
            )

        assert res_type3[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res_type3[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(5)
                )
            )

        assert res_type3[2] == Declaration(
            Variable(
                Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res_type3[3] == Declaration(
            Variable(
                Symbol('d'),
                type=IntBaseType(String('intc')),
                value=Integer(5)
                )
            )

        assert res_type4[0] == Declaration(
            Variable(
                Symbol('a'),
                type=SignedIntType(
                    String('int64'),
                    nbits=Integer(64)
                    ),
                value=Integer(1)
                )
            )

        assert res_type4[1] == Declaration(
            Variable(
                Symbol('b'),
                type=SignedIntType(
                    String('int64'),
                    nbits=Integer(64)
                    ),
                value=Integer(5)
                )
            )

        assert res_type4[2] == Declaration(
            Variable(
                Symbol('c'),
                type=SignedIntType(
                    String('int64'),
                    nbits=Integer(64)
                    ),
                value=Integer(1)
                )
            )

        assert res_type4[3] == Declaration(
            Variable(
                Symbol('d'),
                type=SignedIntType(
                    String('int64'),
                    nbits=Integer(64)
                    ),
                value=Integer(5)
                )
            )

        assert res_type5[0] == Declaration(
            Variable(
                Symbol('a'),
                type=UnsignedIntType(
                    String('uint8'),
                    nbits=Integer(8)
                    ),
                value=Integer(1)
                )
            )

        assert res_type5[1] == Declaration(
            Variable(
                Symbol('b'),
                type=UnsignedIntType(
                    String('uint8'),
                    nbits=Integer(8)
                    ),
                value=Integer(5)
                )
            )

        assert res_type6[0] == Declaration(
            Variable(
                Symbol('a'),
                type=UnsignedIntType(
                    String('uint16'),
                    nbits=Integer(16)
                    ),
                value=Integer(1)
                )
            )

        assert res_type6[1] == Declaration(
            Variable(
                Symbol('b'),
                type=UnsignedIntType(
                    String('uint16'),
                    nbits=Integer(16)
                    ),
                value=Integer(5)
                )
            )

        assert res_type6[2] == Declaration(
            Variable(
                Symbol('c'),
                type=UnsignedIntType(
                    String('uint16'),
                    nbits=Integer(16)
                    ),
                value=Integer(1)
                )
            )

        assert res_type6[3] == Declaration(
            Variable(
                Symbol('d'),
                type=UnsignedIntType(
                    String('uint16'),
                    nbits=Integer(16)
                    ),
                value=Integer(5)
                )
            )

        assert res_type7[0] == Declaration(
            Variable(
                Symbol('a'),
                type=UnsignedIntType(
                    String('uint32'),
                    nbits=Integer(32)
                    ),
                value=Integer(1)
                )
            )

        assert res_type7[1] == Declaration(
            Variable(
                Symbol('b'),
                type=UnsignedIntType(
                    String('uint32'),
                    nbits=Integer(32)
                    ),
                value=Integer(5)
                )
            )

        assert res_type8[0] == Declaration(
            Variable(
                Symbol('a'),
                type=UnsignedIntType(
                    String('uint64'),
                    nbits=Integer(64)
                    ),
                value=Integer(1)
                )
            )

        assert res_type8[1] == Declaration(
            Variable(
                Symbol('b'),
                type=UnsignedIntType(
                    String('uint64'),
                    nbits=Integer(64)
                    ),
                value=Integer(5)
                )
            )

        assert res_type8[2] == Declaration(
            Variable(
                Symbol('c'),
                type=UnsignedIntType(
                    String('uint64'),
                    nbits=Integer(64)
                    ),
                value=Integer(1)
                )
            )

        assert res_type8[3] == Declaration(
            Variable(
                Symbol('d'),
                type=UnsignedIntType(
                    String('uint64'),
                    nbits=Integer(64)
                    ),
                value=Integer(5)
                )
            )


    def test_float():
        c_src1 = 'float a = 1.0;'
        c_src2 = (
            'float a = 1.25;' + '\n' +
            'float b = 2.39;' + '\n'
        )
        c_src3 = 'float x = 1, y = 2;'
        c_src4 = 'float p = 5, e = 7.89;'
        c_src5 = 'float r = true, s = false;'

        # cin.TypeKind.FLOAT
        c_src_type1 = 'float x = 1, y = 2.5;'

        # cin.TypeKind.DOUBLE
        c_src_type2 = 'double x = 1, y = 2.5;'

        # cin.TypeKind.LONGDOUBLE
        c_src_type3 = 'long double x = 1, y = 2.5;'

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()
        res5 = SymPyExpression(c_src5, 'c').return_expr()

        res_type1 = SymPyExpression(c_src_type1, 'c').return_expr()
        res_type2 = SymPyExpression(c_src_type2, 'c').return_expr()
        res_type3 = SymPyExpression(c_src_type3, 'c').return_expr()

        assert res1[0] == Declaration(
            Variable(
                Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('1.0', precision=53)
                )
            )

        assert res2[0] == Declaration(
            Variable(
                Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('1.25', precision=53)
            )
        )

        assert res2[1] == Declaration(
            Variable(
                Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.3900000000000001', precision=53)
            )
        )

        assert res3[0] == Declaration(
            Variable(
                Symbol('x'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('1.0', precision=53)
            )
        )

        assert res3[1] == Declaration(
            Variable(
                Symbol('y'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.0', precision=53)
            )
        )

        assert res4[0] == Declaration(
            Variable(
                Symbol('p'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                ),
                value=Float('5.0', precision=53)
            )
        )

        assert res4[1] == Declaration(
            Variable(
                Symbol('e'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('7.89', precision=53)
            )
        )

        assert res5[0] == Declaration(
            Variable(
                Symbol('r'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('1.0', precision=53)
            )
        )

        assert res5[1] == Declaration(
            Variable(
                Symbol('s'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('0.0', precision=53)
            )
        )

        assert res_type1[0] == Declaration(
            Variable(
                Symbol('x'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('1.0', precision=53)
                )
            )

        assert res_type1[1] == Declaration(
            Variable(
                Symbol('y'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.5', precision=53)
                )
            )
        assert res_type2[0] == Declaration(
            Variable(
                Symbol('x'),
                type=FloatType(
                    String('float64'),
                    nbits=Integer(64),
                    nmant=Integer(52),
                    nexp=Integer(11)
                    ),
                value=Float('1.0', precision=53)
                )
            )

        assert res_type2[1] == Declaration(
            Variable(
                Symbol('y'),
                type=FloatType(
                    String('float64'),
                    nbits=Integer(64),
                    nmant=Integer(52),
                    nexp=Integer(11)
                    ),
                value=Float('2.5', precision=53)
                )
            )

        assert res_type3[0] == Declaration(
            Variable(
                Symbol('x'),
                type=FloatType(
                    String('float80'),
                    nbits=Integer(80),
                    nmant=Integer(63),
                    nexp=Integer(15)
                    ),
                value=Float('1.0', precision=53)
                )
            )

        assert res_type3[1] == Declaration(
            Variable(
                Symbol('y'),
                type=FloatType(
                    String('float80'),
                    nbits=Integer(80),
                    nmant=Integer(63),
                    nexp=Integer(15)
                    ),
                value=Float('2.5', precision=53)
                )
            )


    def test_bool():
        c_src1 = (
            'bool a = true, b = false;'
        )

        c_src2 = (
            'bool a = 1, b = 0;'
        )

        c_src3 = (
            'bool a = 10, b = 20;'
        )

        c_src4 = (
            'bool a = 19.1, b = 9.0, c = 0.0;'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()

        assert res1[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res1[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res2[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true)
            )

        assert res2[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res3[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res3[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res4[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true)
            )

        assert res4[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res4[2] == Declaration(
            Variable(Symbol('c'),
                type=Type(String('bool')),
                value=false
                )
            )

    @XFAIL # this is expected to fail because of a bug in the C parser.
    def test_function():
        c_src1 = (
            'void fun1()' + '\n' +
            '{' + '\n' +
            'int a;' + '\n' +
            '}'
        )
        c_src2 = (
            'int fun2()' + '\n' +
            '{'+ '\n' +
            'int a;' + '\n' +
            'return a;' + '\n' +
            '}'
        )
        c_src3 = (
            'float fun3()' + '\n' +
            '{' + '\n' +
            'float b;' + '\n' +
            'return b;' + '\n' +
            '}'
        )
        c_src4 = (
            'float fun4()' + '\n' +
            '{}'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()

        assert res1[0] == FunctionDefinition(
            NoneToken(),
            name=String('fun1'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc'))
                    )
                )
            )
        )

        assert res2[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun2'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc'))
                    )
                ),
                Return('a')
            )
        )

        assert res3[0] == FunctionDefinition(
            FloatType(
                String('float32'),
                nbits=Integer(32),
                nmant=Integer(23),
                nexp=Integer(8)
                ),
            name=String('fun3'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('b'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                    )
                ),
                Return('b')
            )
        )

        assert res4[0] == FunctionPrototype(
            FloatType(
                String('float32'),
                nbits=Integer(32),
                nmant=Integer(23),
                nexp=Integer(8)
                ),
            name=String('fun4'),
            parameters=()
        )

    @XFAIL # this is expected to fail because of a bug in the C parser.
    def test_parameters():
        c_src1 = (
            'void fun1( int a)' + '\n' +
            '{' + '\n' +
            'int i;' + '\n' +
            '}'
        )
        c_src2 = (
            'int fun2(float x, float y)' + '\n' +
            '{'+ '\n' +
            'int a;' + '\n' +
            'return a;' + '\n' +
            '}'
        )
        c_src3 = (
            'float fun3(int p, float q, int r)' + '\n' +
            '{' + '\n' +
            'float b;' + '\n' +
            'return b;' + '\n' +
            '}'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()

        assert res1[0] == FunctionDefinition(
            NoneToken(),
            name=String('fun1'),
            parameters=(
                Variable(
                    Symbol('a'),
                    type=IntBaseType(String('intc'))
                ),
            ),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('i'),
                        type=IntBaseType(String('intc'))
                    )
                )
            )
        )

        assert res2[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun2'),
            parameters=(
                Variable(
                    Symbol('x'),
                    type=FloatType(
                        String('float32'),
                        nbits=Integer(32),
                        nmant=Integer(23),
                        nexp=Integer(8)
                        )
                ),
                Variable(
                    Symbol('y'),
                    type=FloatType(
                        String('float32'),
                        nbits=Integer(32),
                        nmant=Integer(23),
                        nexp=Integer(8)
                        )
                )
            ),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc'))
                    )
                ),
                Return('a')
            )
        )

        assert res3[0] == FunctionDefinition(
            FloatType(
                String('float32'),
                nbits=Integer(32),
                nmant=Integer(23),
                nexp=Integer(8)
                ),
            name=String('fun3'),
            parameters=(
                Variable(
                    Symbol('p'),
                    type=IntBaseType(String('intc'))
                ),
                Variable(
                    Symbol('q'),
                    type=FloatType(
                        String('float32'),
                        nbits=Integer(32),
                        nmant=Integer(23),
                        nexp=Integer(8)
                        )
                ),
                Variable(
                    Symbol('r'),
                    type=IntBaseType(String('intc'))
                )
            ),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('b'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                    )
                ),
                Return('b')
            )
        )

    @XFAIL # this is expected to fail because of a bug in the C parser.
    def test_function_call():
        c_src1 = (
            'int fun1(int x)' + '\n' +
            '{' + '\n' +
            'return x;' + '\n' +
            '}' + '\n' +
            'void caller()' + '\n' +
            '{' + '\n' +
            'int x = fun1(2);' + '\n' +
            '}'
        )

        c_src2 = (
            'int fun2(int a, int b, int c)' + '\n' +
            '{' + '\n' +
            'return a;' + '\n' +
            '}' + '\n' +
            'void caller()' + '\n' +
            '{' + '\n' +
            'int y = fun2(2, 3, 4);' + '\n' +
            '}'
        )

        c_src3 = (
            'int fun3(int a, int b, int c)' + '\n' +
            '{' + '\n' +
            'return b;' + '\n' +
            '}' + '\n' +
            'void caller()' + '\n' +
            '{' + '\n' +
            'int p;' + '\n' +
            'int q;' + '\n' +
            'int r;' + '\n' +
            'int z = fun3(p, q, r);' + '\n' +
            '}'
        )

        c_src4 = (
            'int fun4(float a, float b, int c)' + '\n' +
            '{' + '\n' +
            'return c;' + '\n' +
            '}' + '\n' +
            'void caller()' + '\n' +
            '{' + '\n' +
            'float x;' + '\n' +
            'float y;' + '\n' +
            'int z;' + '\n' +
            'int i = fun4(x, y, z)' + '\n' +
            '}'
        )

        c_src5 = (
            'int fun()' + '\n' +
            '{' + '\n' +
            'return 1;' + '\n' +
            '}' + '\n' +
            'void caller()' + '\n' +
            '{' + '\n' +
            'int a = fun()' + '\n' +
            '}'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()
        res5 = SymPyExpression(c_src5, 'c').return_expr()


        assert res1[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun1'),
            parameters=(Variable(Symbol('x'),
                type=IntBaseType(String('intc'))
                ),
            ),
            body=CodeBlock(
                Return('x')
                )
            )

        assert res1[1] == FunctionDefinition(
            NoneToken(),
            name=String('caller'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('x'),
                        value=FunctionCall(String('fun1'),
                            function_args=(
                                Integer(2),
                                )
                            )
                        )
                    )
                )
            )

        assert res2[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun2'),
            parameters=(Variable(Symbol('a'),
                type=IntBaseType(String('intc'))
                ),
            Variable(Symbol('b'),
                type=IntBaseType(String('intc'))
                ),
            Variable(Symbol('c'),
                type=IntBaseType(String('intc'))
                )
            ),
            body=CodeBlock(
                Return('a')
                )
            )

        assert res2[1] == FunctionDefinition(
            NoneToken(),
            name=String('caller'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('y'),
                        value=FunctionCall(
                            String('fun2'),
                            function_args=(
                                Integer(2),
                                Integer(3),
                                Integer(4)
                                )
                            )
                        )
                    )
                )
            )

        assert res3[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun3'),
            parameters=(
                Variable(Symbol('a'),
                    type=IntBaseType(String('intc'))
                    ),
                Variable(Symbol('b'),
                    type=IntBaseType(String('intc'))
                    ),
                Variable(Symbol('c'),
                    type=IntBaseType(String('intc'))
                    )
                ),
            body=CodeBlock(
                Return('b')
                )
            )

        assert res3[1] == FunctionDefinition(
            NoneToken(),
            name=String('caller'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('p'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('q'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('r'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('z'),
                        value=FunctionCall(
                            String('fun3'),
                            function_args=(
                                Symbol('p'),
                                Symbol('q'),
                                Symbol('r')
                                )
                            )
                        )
                    )
                )
            )

        assert res4[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun4'),
            parameters=(Variable(Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
                ),
            Variable(Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    )
                ),
            Variable(Symbol('c'),
                type=IntBaseType(String('intc'))
                )
            ),
            body=CodeBlock(
                Return('c')
                )
            )

        assert res4[1] == FunctionDefinition(
            NoneToken(),
            name=String('caller'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('x'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                        )
                    ),
                Declaration(
                    Variable(Symbol('y'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                        )
                    ),
                Declaration(
                    Variable(Symbol('z'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('i'),
                        value=FunctionCall(String('fun4'),
                            function_args=(
                                Symbol('x'),
                                Symbol('y'),
                                Symbol('z')
                                )
                            )
                        )
                    )
                )
            )

        assert res5[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('fun'),
            parameters=(),
            body=CodeBlock(
                Return('')
                )
            )

        assert res5[1] == FunctionDefinition(
            NoneToken(),
            name=String('caller'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        value=FunctionCall(String('fun'),
                            function_args=()
                            )
                        )
                    )
                )
            )


    def test_parse():
        c_src1 = (
            'int a;' + '\n' +
            'int b;' + '\n'
        )
        c_src2 = (
            'void fun1()' + '\n' +
            '{' + '\n' +
            'int a;' + '\n' +
            '}'
        )

        f1 = open('..a.h', 'w')
        f2 = open('..b.h', 'w')

        f1.write(c_src1)
        f2. write(c_src2)

        f1.close()
        f2.close()

        res1 = SymPyExpression('..a.h', 'c').return_expr()
        res2 = SymPyExpression('..b.h', 'c').return_expr()

        os.remove('..a.h')
        os.remove('..b.h')

        assert res1[0] == Declaration(
            Variable(
                Symbol('a'),
                type=IntBaseType(String('intc'))
            )
        )
        assert res1[1] == Declaration(
            Variable(
                Symbol('b'),
                type=IntBaseType(String('intc'))
            )
        )
        assert res2[0] == FunctionDefinition(
            NoneToken(),
            name=String('fun1'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc'))
                    )
                )
            )
        )


    def test_binary_operators():
        c_src1 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'a = 1;' + '\n' +
            '}'
        )
        c_src2 = (
            'void func()'+
            '{' + '\n' +
                'int a = 0;' + '\n' +
                'a = a + 1;' + '\n' +
                'a = 3*a - 10;' + '\n' +
            '}'
        )
        c_src3 = (
            'void func()'+
            '{' + '\n' +
                'int a = 10;' + '\n' +
                'a = 1 + a - 3 * 6;' + '\n' +
            '}'
        )
        c_src4 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'int b;' + '\n' +
                'a = 100;' + '\n' +
                'b = a*a + a*a + a + 19*a + 1 + 24;' + '\n' +
            '}'
        )
        c_src5 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'int b;' + '\n' +
                'int c;' + '\n' +
                'int d;' + '\n' +
                'a = 1;' + '\n' +
                'b = 2;' + '\n' +
                'c = b;' + '\n' +
                'd = ((a+b)*(a+c))*((c-d)*(a+c));' + '\n' +
            '}'
        )
        c_src6 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'int b;' + '\n' +
                'int c;' + '\n' +
                'int d;' + '\n' +
                'a = 1;' + '\n' +
                'b = 2;' + '\n' +
                'c = 3;' + '\n' +
                'd = (a*a*a*a + 3*b*b + b + b + c*d);' + '\n' +
            '}'
        )
        c_src7 = (
            'void func()'+
            '{' + '\n' +
                'float a;' + '\n' +
                'a = 1.01;' + '\n' +
            '}'
        )

        c_src8 = (
            'void func()'+
            '{' + '\n' +
                'float a;' + '\n' +
                'a = 10.0 + 2.5;' + '\n' +
            '}'
        )

        c_src9 = (
            'void func()'+
            '{' + '\n' +
                'float a;' + '\n' +
                'a = 10.0 / 2.5;' + '\n' +
            '}'
        )

        c_src10 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'a = 100 / 4;' + '\n' +
            '}'
        )

        c_src11 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'a = 20 - 100 / 4 * 5 + 10;' + '\n' +
            '}'
        )

        c_src12 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'a = (20 - 100) / 4 * (5 + 10);' + '\n' +
            '}'
        )

        c_src13 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'int b;' + '\n' +
                'float c;' + '\n' +
                'c = b/a;' + '\n' +
            '}'
        )

        c_src14 = (
            'void func()'+
            '{' + '\n' +
                'int a = 2;' + '\n' +
                'int d = 5;' + '\n' +
                'int n = 10;' + '\n' +
                'int s;' + '\n' +
                's = (a/2)*(2*a + (n-1)*d);' + '\n' +
            '}'
        )

        c_src15 = (
            'void func()'+
            '{' + '\n' +
                'int a;' + '\n' +
                'a = 1 % 2;' + '\n' +
            '}'
        )

        c_src16 = (
            'void func()'+
            '{' + '\n' +
                'int a = 2;' + '\n' +
                'int b;' + '\n' +
                'b = a % 3;' + '\n' +
            '}'
        )

        c_src17 = (
            'void func()'+
            '{' + '\n' +
                'int a = 100;' + '\n' +
                'int b = 3;' + '\n' +
                'int c;' + '\n' +
                'c = a % b;' + '\n' +
            '}'
        )

        c_src18 = (
            'void func()'+
            '{' + '\n' +
                'int a = 100;' + '\n' +
                'int b = 3;' + '\n' +
                'int mod = 1000000007;' + '\n' +
                'int c;' + '\n' +
                'c = (a + b * (100/a)) % mod;' + '\n' +
            '}'
        )

        c_src19 = (
            'void func()'+
            '{' + '\n' +
                'int a = 100;' + '\n' +
                'int b = 3;' + '\n' +
                'int mod = 1000000007;' + '\n' +
                'int c;' + '\n' +
                'c = ((a % mod + b % mod) % mod' \
                '* (a % mod - b % mod) % mod) % mod;' + '\n' +
            '}'
        )

        c_src20 = (
            'void func()'+
            '{' + '\n' +
            'bool a' + '\n' +
            'bool b;' + '\n' +
            'a = 1 == 2;' + '\n' +
            'b = 1 != 2;' + '\n' +
            '}'
        )

        c_src21 = (
            'void func()'+
            '{' + '\n' +
            'bool a;' + '\n' +
            'bool b;' + '\n' +
            'bool c;' + '\n' +
            'bool d;' + '\n' +
            'a = 1 == 2;' + '\n' +
            'b = 1 <= 2;' + '\n' +
            'c = 1 > 2;' + '\n' +
            'd = 1 >= 2;' + '\n' +
            '}'
        )

        c_src22 = (
            'void func()'+
            '{' + '\n' +
            'int a = 1;' + '\n' +
            'int b = 2;' + '\n' +

            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +
            'bool c5;' + '\n' +
            'bool c6;' + '\n' +
            'bool c7;' + '\n' +
            'bool c8;' + '\n' +

            'c1 = a == 1;' + '\n' +
            'c2 = b == 2;' + '\n' +

            'c3 = 1 != a;' + '\n' +
            'c4 = 1 != b;' + '\n' +

            'c5 = a < 0;' + '\n' +
            'c6 = b <= 10;' + '\n' +
            'c7 = a > 0;' + '\n' +
            'c8 = b >= 11;' + '\n' +
            '}'
        )

        c_src23 = (
            'void func()'+
            '{' + '\n' +
            'int a = 3;' + '\n' +
            'int b = 4;' + '\n' +

            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +
            'bool c5;' + '\n' +
            'bool c6;' + '\n' +

            'c1 = a == b;' + '\n' +
            'c2 = a != b;' + '\n' +
            'c3 = a < b;' + '\n' +
            'c4 = a <= b;' + '\n' +
            'c5 = a > b;' + '\n' +
            'c6 = a >= b;' + '\n' +
            '}'
        )

        c_src24 = (
            'void func()'+
            '{' + '\n' +
            'float a = 1.25'
            'float b = 2.5;' + '\n' +

            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +

            'c1 = a == 1.25;' + '\n' +
            'c2 = b == 2.54;' + '\n' +

            'c3 = 1.2 != a;' + '\n' +
            'c4 = 1.5 != b;' + '\n' +
            '}'
        )

        c_src25 = (
            'void func()'+
            '{' + '\n' +
            'float a = 1.25' + '\n' +
            'float b = 2.5;' + '\n' +

            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +
            'bool c5;' + '\n' +
            'bool c6;' + '\n' +

            'c1 = a == b;' + '\n' +
            'c2 = a != b;' + '\n' +
            'c3 = a < b;' + '\n' +
            'c4 = a <= b;' + '\n' +
            'c5 = a > b;' + '\n' +
            'c6 = a >= b;' + '\n' +
            '}'
        )

        c_src26 = (
            'void func()'+
            '{' + '\n' +
            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +
            'bool c5;' + '\n' +
            'bool c6;' + '\n' +

            'c1 = true == true;' + '\n' +
            'c2 = true == false;' + '\n' +
            'c3 = false == false;' + '\n' +

            'c4 = true != true;' + '\n' +
            'c5 = true != false;' + '\n' +
            'c6 = false != false;' + '\n' +
            '}'
        )

        c_src27 = (
            'void func()'+
            '{' + '\n' +
            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +
            'bool c5;' + '\n' +
            'bool c6;' + '\n' +

            'c1 = true && true;' + '\n' +
            'c2 = true && false;' + '\n' +
            'c3 = false && false;' + '\n' +

            'c4 = true || true;' + '\n' +
            'c5 = true || false;' + '\n' +
            'c6 = false || false;' + '\n' +
            '}'
        )

        c_src28 = (
            'void func()'+
            '{' + '\n' +
            'bool a;' + '\n' +
            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +

            'c1 = a && true;' + '\n' +
            'c2 = false && a;' + '\n' +

            'c3 = true || a;' + '\n' +
            'c4 = a || false;' + '\n' +
            '}'
        )

        c_src29 = (
            'void func()'+
            '{' + '\n' +
            'int a;' + '\n' +
            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +

            'c1 = a && 1;' + '\n' +
            'c2 = a && 0;' + '\n' +

            'c3 = a || 1;' + '\n' +
            'c4 = 0 || a;' + '\n' +
            '}'
        )

        c_src30 = (
            'void func()'+
            '{' + '\n' +
            'int a;' + '\n' +
            'int b;' + '\n' +
            'bool c;'+ '\n' +
            'bool d;'+ '\n' +

            'bool c1;' + '\n' +
            'bool c2;' + '\n' +
            'bool c3;' + '\n' +
            'bool c4;' + '\n' +
            'bool c5;' + '\n' +
            'bool c6;' + '\n' +

            'c1 = a && b;' + '\n' +
            'c2 = a && c;' + '\n' +
            'c3 = c && d;' + '\n' +

            'c4 = a || b;' + '\n' +
            'c5 = a || c;' + '\n' +
            'c6 = c || d;' + '\n' +
            '}'
        )

        c_src_raise1 = (
            'void func()'+
            '{' + '\n' +
            'int a;' + '\n' +
            'a = -1;' + '\n' +
            '}'
        )

        c_src_raise2 = (
            'void func()'+
            '{' + '\n' +
            'int a;' + '\n' +
            'a = -+1;' + '\n' +
            '}'
        )

        c_src_raise3 = (
            'void func()'+
            '{' + '\n' +
            'int a;' + '\n' +
            'a = 2*-2;' + '\n' +
            '}'
        )

        c_src_raise4 = (
            'void func()'+
            '{' + '\n' +
            'int a;' + '\n' +
            'a = (int)2.0;' + '\n' +
            '}'
        )

        c_src_raise5 = (
            'void func()'+
            '{' + '\n' +
            'int a=100;' + '\n' +
            'a = (a==100)?(1):(0);' + '\n' +
            '}'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()
        res5 = SymPyExpression(c_src5, 'c').return_expr()
        res6 = SymPyExpression(c_src6, 'c').return_expr()
        res7 = SymPyExpression(c_src7, 'c').return_expr()
        res8 = SymPyExpression(c_src8, 'c').return_expr()
        res9 = SymPyExpression(c_src9, 'c').return_expr()
        res10 = SymPyExpression(c_src10, 'c').return_expr()
        res11 = SymPyExpression(c_src11, 'c').return_expr()
        res12 = SymPyExpression(c_src12, 'c').return_expr()
        res13 = SymPyExpression(c_src13, 'c').return_expr()
        res14 = SymPyExpression(c_src14, 'c').return_expr()
        res15 = SymPyExpression(c_src15, 'c').return_expr()
        res16 = SymPyExpression(c_src16, 'c').return_expr()
        res17 = SymPyExpression(c_src17, 'c').return_expr()
        res18 = SymPyExpression(c_src18, 'c').return_expr()
        res19 = SymPyExpression(c_src19, 'c').return_expr()
        res20 = SymPyExpression(c_src20, 'c').return_expr()
        res21 = SymPyExpression(c_src21, 'c').return_expr()
        res22 = SymPyExpression(c_src22, 'c').return_expr()
        res23 = SymPyExpression(c_src23, 'c').return_expr()
        res24 = SymPyExpression(c_src24, 'c').return_expr()
        res25 = SymPyExpression(c_src25, 'c').return_expr()
        res26 = SymPyExpression(c_src26, 'c').return_expr()
        res27 = SymPyExpression(c_src27, 'c').return_expr()
        res28 = SymPyExpression(c_src28, 'c').return_expr()
        res29 = SymPyExpression(c_src29, 'c').return_expr()
        res30 = SymPyExpression(c_src30, 'c').return_expr()

        assert res1[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(Variable(Symbol('a')), Integer(1))
                )
            )

        assert res2[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                    type=IntBaseType(String('intc')),
                    value=Integer(0))),
                Assignment(
                    Variable(Symbol('a')),
                    Add(Symbol('a'),
                        Integer(1))
                    ),
                Assignment(Variable(Symbol('a')),
                    Add(
                        Mul(
                            Integer(3),
                            Symbol('a')),
                        Integer(-10)
                        )
                    )
                )
            )

        assert res3[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(10)
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Add(
                        Symbol('a'),
                        Integer(-17)
                        )
                    )
                )
            )

        assert res4[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Integer(100)),
                Assignment(
                    Variable(Symbol('b')),
                    Add(
                        Mul(
                            Integer(2),
                        Pow(
                            Symbol('a'),
                            Integer(2))
                        ),
                        Mul(
                            Integer(20),
                            Symbol('a')),
                        Integer(25)
                        )
                    )
                )
            )

        assert res5[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('d'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Integer(1)),
                Assignment(
                    Variable(Symbol('b')),
                    Integer(2)
                    ),
                Assignment(
                    Variable(Symbol('c')),
                    Symbol('b')),
                Assignment(
                    Variable(Symbol('d')),
                    Mul(
                        Add(
                            Symbol('a'),
                            Symbol('b')),
                        Pow(
                            Add(
                                Symbol('a'),
                                Symbol('c')
                                ),
                            Integer(2)
                            ),
                        Add(
                            Symbol('c'),
                            Mul(
                                Integer(-1),
                                Symbol('d')
                                )
                            )
                        )
                    )
                )
            )

        assert res6[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('d'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Integer(1)
                    ),
                Assignment(
                    Variable(Symbol('b')),
                    Integer(2)
                    ),
                Assignment(
                    Variable(Symbol('c')),
                    Integer(3)
                    ),
                Assignment(
                    Variable(Symbol('d')),
                    Add(
                        Pow(
                            Symbol('a'),
                            Integer(4)
                            ),
                        Mul(
                            Integer(3),
                            Pow(
                                Symbol('b'),
                                Integer(2)
                                )
                            ),
                        Mul(
                            Integer(2),
                            Symbol('b')
                            ),
                        Mul(
                            Symbol('c'),
                            Symbol('d')
                            )
                        )
                    )
                )
            )

        assert res7[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Float('1.01', precision=53)
                    )
                )
            )

        assert res8[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Float('12.5', precision=53)
                    )
                )
            )

        assert res9[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Float('4.0', precision=53)
                    )
                )
            )

        assert res10[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Integer(25)
                    )
                )
            )

        assert res11[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Integer(-95)
                    )
                )
            )

        assert res12[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Integer(-300)
                    )
                )
            )

        assert res13[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                        )
                    ),
                Assignment(
                    Variable(Symbol('c')),
                    Mul(
                        Pow(
                            Symbol('a'),
                            Integer(-1)
                            ),
                        Symbol('b')
                        )
                    )
                )
            )

        assert res14[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(2)
                        )
                    ),
                Declaration(
                    Variable(Symbol('d'),
                        type=IntBaseType(String('intc')),
                        value=Integer(5)
                        )
                    ),
                Declaration(
                    Variable(Symbol('n'),
                        type=IntBaseType(String('intc')),
                        value=Integer(10)
                        )
                    ),
                Declaration(
                    Variable(Symbol('s'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('s')),
                    Mul(
                        Rational(1, 2),
                        Symbol('a'),
                        Add(
                            Mul(
                                Integer(2),
                                Symbol('a')
                                ),
                            Mul(
                                Symbol('d'),
                                Add(
                                    Symbol('n'),
                                    Integer(-1)
                                    )
                                )
                            )
                        )
                    )
                )
            )

        assert res15[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Integer(1)
                    )
                )
            )

        assert res16[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(2)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('b')),
                    Mod(
                        Symbol('a'),
                        Integer(3)
                        )
                    )
                )
            )

        assert res17[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(100)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(3)
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c')),
                    Mod(
                        Symbol('a'),
                        Symbol('b')
                        )
                    )
                )
            )

        assert res18[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(100)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(3)
                        )
                    ),
                Declaration(
                    Variable(Symbol('mod'),
                        type=IntBaseType(String('intc')),
                        value=Integer(1000000007)
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c')),
                    Mod(
                        Add(
                            Symbol('a'),
                            Mul(
                                Integer(100),
                                Pow(
                                    Symbol('a'),
                                    Integer(-1)
                                    ),
                                Symbol('b')
                                )
                            ),
                        Symbol('mod')
                        )
                    )
                )
            )

        assert res19[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(100)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(3)
                        )
                    ),
                Declaration(
                    Variable(Symbol('mod'),
                        type=IntBaseType(String('intc')),
                        value=Integer(1000000007)
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c')),
                    Mod(
                        Mul(
                            Add(
                                Mod(
                                    Symbol('a'),
                                    Symbol('mod')
                                    ),
                                Mul(
                                    Integer(-1),
                                    Mod(
                                        Symbol('b'),
                                        Symbol('mod')
                                        )
                                    )
                                ),
                            Mod(
                                Add(
                                    Symbol('a'),
                                    Symbol('b')
                                    ),
                                Symbol('mod')
                                )
                            ),
                        Symbol('mod')
                        )
                    )
                )
            )

        assert res20[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('b')),
                    true
                    )
                )
            )

        assert res21[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('d'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('b')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('d')),
                    false
                    )
                )
            )

        assert res22[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(1)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(2)
                        )
                    ),
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c5'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c6'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c7'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c8'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    Equality(
                        Symbol('a'),
                        Integer(1)
                        )
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    Equality(
                        Symbol('b'),
                        Integer(2)
                        )
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    Unequality(
                        Integer(1),
                        Symbol('a')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    Unequality(
                        Integer(1),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c5')),
                    StrictLessThan(
                        Symbol('a'),
                        Integer(0)
                        )
                    ),
                Assignment(
                    Variable(Symbol('c6')),
                    LessThan(
                        Symbol('b'),
                        Integer(10)
                        )
                    ),
                Assignment(
                    Variable(Symbol('c7')),
                    StrictGreaterThan(
                        Symbol('a'),
                        Integer(0)
                        )
                    ),
                Assignment(
                    Variable(Symbol('c8')),
                    GreaterThan(
                        Symbol('b'),
                        Integer(11)
                        )
                    )
                )
            )

        assert res23[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(3)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(4)
                        )
                    ),
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c5'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c6'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    Equality(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    Unequality(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    StrictLessThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    LessThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c5')),
                    StrictGreaterThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c6')),
                    GreaterThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    )
                )
            )

        assert res24[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            )
                        )
                    ),
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    Equality(
                        Symbol('a'),
                        Float('1.25', precision=53)
                        )
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    Unequality(
                        Float('1.2', precision=53),
                        Symbol('a')
                        )
                    )
                )
            )


        assert res25[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            ),
                        value=Float('1.25', precision=53)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=FloatType(
                            String('float32'),
                            nbits=Integer(32),
                            nmant=Integer(23),
                            nexp=Integer(8)
                            ),
                        value=Float('2.5', precision=53)
                        )
                    ),
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool')
                            )
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c5'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c6'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    Equality(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    Unequality(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    StrictLessThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    LessThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c5')),
                    StrictGreaterThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c6')),
                    GreaterThan(
                        Symbol('a'),
                        Symbol('b')
                        )
                    )
                )
            )

        assert res26[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(), body=CodeBlock(
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c5'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c6'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('c5')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c6')),
                    false
                    )
                )
            )

        assert res27[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c5'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c6'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c5')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c6')),
                    false)
                )
            )

        assert res28[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    Symbol('a')
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    Symbol('a')
                    )
                )
            )

        assert res29[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    Symbol('a')
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    false
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    true
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    Symbol('a')
                    )
                )
            )

        assert res30[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('d'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c1'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c2'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c3'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c4'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c5'),
                        type=Type(String('bool'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('c6'),
                        type=Type(String('bool'))
                        )
                    ),
                Assignment(
                    Variable(Symbol('c1')),
                    And(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c2')),
                    And(
                        Symbol('a'),
                        Symbol('c')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c3')),
                    And(
                        Symbol('c'),
                        Symbol('d')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c4')),
                    Or(
                        Symbol('a'),
                        Symbol('b')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c5')),
                    Or(
                        Symbol('a'),
                        Symbol('c')
                        )
                    ),
                Assignment(
                    Variable(Symbol('c6')),
                    Or(
                        Symbol('c'),
                        Symbol('d')
                        )
                    )
                )
            )

        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise1, 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise2, 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise3, 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise4, 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise5, 'c'))


    @XFAIL
    def test_var_decl():
        c_src1 = (
            'int b = 100;' + '\n' +
            'int a = b;' + '\n'
        )

        c_src2 = (
            'int a = 1;' + '\n' +
            'int b = a + 1;' + '\n'
        )

        c_src3 = (
            'float a = 10.0 + 2.5;' + '\n' +
            'float b = a * 20.0;' + '\n'
        )

        c_src4 = (
            'int a = 1 + 100 - 3 * 6;' + '\n'
        )

        c_src5 = (
            'int a = (((1 + 100) * 12) - 3) * (6 - 10);' + '\n'
        )

        c_src6 = (
            'int b = 2;' + '\n' +
            'int c = 3;' + '\n' +
            'int a = b + c * 4;' + '\n'
        )

        c_src7 = (
            'int b = 1;' + '\n' +
            'int c = b + 2;' + '\n' +
            'int a = 10 * b * b * c;' + '\n'
        )

        c_src8 = (
            'void func()'+
            '{' + '\n' +
                'int a = 1;' + '\n' +
                'int b = 2;' + '\n' +
                'int temp = a;' + '\n' +
                'a = b;' + '\n' +
                'b = temp;' + '\n' +
            '}'
        )

        c_src9 = (
            'int a = 1;' + '\n' +
            'int b = 2;' + '\n' +
            'int c = a;' + '\n' +
            'int d = a + b + c;' + '\n' +
            'int e = a*a*a + 3*a*a*b + 3*a*b*b + b*b*b;' + '\n'
            'int f = (a + b + c) * (a + b - c);' + '\n' +
            'int g = (a + b + c + d)*(a + b + c + d)*(a * (b - c));'
            + '\n'
        )

        c_src10 = (
            'float a = 10.0;' + '\n' +
            'float b = 2.5;' + '\n' +
            'float c = a*a + 2*a*b + b*b;' + '\n'
        )

        c_src11 = (
            'float a = 10.0 / 2.5;' + '\n'
        )

        c_src12 = (
            'int a = 100 / 4;' + '\n'
        )

        c_src13 = (
            'int a = 20 - 100 / 4 * 5 + 10;' + '\n'
        )

        c_src14 = (
            'int a = (20 - 100) / 4 * (5 + 10);' + '\n'
        )

        c_src15 = (
            'int a = 4;' + '\n' +
            'int b = 2;' + '\n' +
            'float c = b/a;' + '\n'
        )

        c_src16 = (
            'int a = 2;' + '\n' +
            'int d = 5;' + '\n' +
            'int n = 10;' + '\n' +
            'int s = (a/2)*(2*a + (n-1)*d);' + '\n'
        )

        c_src17 = (
            'int a = 1 % 2;' + '\n'
        )

        c_src18 = (
            'int a = 2;' + '\n' +
            'int b = a % 3;' + '\n'
        )

        c_src19 = (
            'int a = 100;' + '\n' +
            'int b = 3;' + '\n' +
            'int c = a % b;' + '\n'
        )

        c_src20 = (
            'int a = 100;' + '\n' +
            'int b = 3;' + '\n' +
            'int mod = 1000000007;' + '\n' +
            'int c = (a + b * (100/a)) % mod;' + '\n'
        )

        c_src21 = (
            'int a = 100;' + '\n' +
            'int b = 3;' + '\n' +
            'int mod = 1000000007;' + '\n' +
            'int c = ((a % mod + b % mod) % mod *' \
            '(a % mod - b % mod) % mod) % mod;' + '\n'
        )

        c_src22 = (
            'bool a = 1 == 2, b = 1 != 2;'
        )

        c_src23 = (
            'bool a = 1 < 2, b = 1 <= 2, c = 1 > 2, d = 1 >= 2;'
        )

        c_src24 = (
            'int a = 1, b = 2;' + '\n' +

            'bool c1 = a == 1;' + '\n' +
            'bool c2 = b == 2;' + '\n' +

            'bool c3 = 1 != a;' + '\n' +
            'bool c4 = 1 != b;' + '\n' +

            'bool c5 = a < 0;' + '\n' +
            'bool c6 = b <= 10;' + '\n' +
            'bool c7 = a > 0;' + '\n' +
            'bool c8 = b >= 11;'

        )

        c_src25 = (
            'int a = 3, b = 4;' + '\n' +

            'bool c1 = a == b;' + '\n' +
            'bool c2 = a != b;' + '\n' +
            'bool c3 = a < b;' + '\n' +
            'bool c4 = a <= b;' + '\n' +
            'bool c5 = a > b;' + '\n' +
            'bool c6 = a >= b;'
        )

        c_src26 = (
            'float a = 1.25, b = 2.5;' + '\n' +

            'bool c1 = a == 1.25;' + '\n' +
            'bool c2 = b == 2.54;' + '\n' +

            'bool c3 = 1.2 != a;' + '\n' +
            'bool c4 = 1.5 != b;'
        )

        c_src27 = (
            'float a = 1.25, b = 2.5;' + '\n' +

            'bool c1 = a == b;' + '\n' +
            'bool c2 = a != b;' + '\n' +
            'bool c3 = a < b;' + '\n' +
            'bool c4 = a <= b;' + '\n' +
            'bool c5 = a > b;' + '\n' +
            'bool c6 = a >= b;'
        )

        c_src28 = (
            'bool c1 = true == true;' + '\n' +
            'bool c2 = true == false;' + '\n' +
            'bool c3 = false == false;' + '\n' +

            'bool c4 = true != true;' + '\n' +
            'bool c5 = true != false;' + '\n' +
            'bool c6 = false != false;'
        )

        c_src29 = (
            'bool c1 = true && true;' + '\n' +
            'bool c2 = true && false;' + '\n' +
            'bool c3 = false && false;' + '\n' +

            'bool c4 = true || true;' + '\n' +
            'bool c5 = true || false;' + '\n' +
            'bool c6 = false || false;'
        )

        c_src30 = (
            'bool a = false;' + '\n' +

            'bool c1 = a && true;' + '\n' +
            'bool c2 = false && a;' + '\n' +

            'bool c3 = true || a;' + '\n' +
            'bool c4 = a || false;'
        )

        c_src31 = (
            'int a = 1;' + '\n' +

            'bool c1 = a && 1;' + '\n' +
            'bool c2 = a && 0;' + '\n' +

            'bool c3 = a || 1;' + '\n' +
            'bool c4 = 0 || a;'
        )

        c_src32 = (
            'int a = 1, b = 0;' + '\n' +
            'bool c = false, d = true;'+ '\n' +

            'bool c1 = a && b;' + '\n' +
            'bool c2 = a && c;' + '\n' +
            'bool c3 = c && d;' + '\n' +

            'bool c4 = a || b;' + '\n' +
            'bool c5 = a || c;' + '\n' +
            'bool c6 = c || d;'
        )

        c_src_raise1 = (
            "char a = 'b';"
        )

        c_src_raise2 = (
            'int a[] = {10, 20};'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()
        res5 = SymPyExpression(c_src5, 'c').return_expr()
        res6 = SymPyExpression(c_src6, 'c').return_expr()
        res7 = SymPyExpression(c_src7, 'c').return_expr()
        res8 = SymPyExpression(c_src8, 'c').return_expr()
        res9 = SymPyExpression(c_src9, 'c').return_expr()
        res10 = SymPyExpression(c_src10, 'c').return_expr()
        res11 = SymPyExpression(c_src11, 'c').return_expr()
        res12 = SymPyExpression(c_src12, 'c').return_expr()
        res13 = SymPyExpression(c_src13, 'c').return_expr()
        res14 = SymPyExpression(c_src14, 'c').return_expr()
        res15 = SymPyExpression(c_src15, 'c').return_expr()
        res16 = SymPyExpression(c_src16, 'c').return_expr()
        res17 = SymPyExpression(c_src17, 'c').return_expr()
        res18 = SymPyExpression(c_src18, 'c').return_expr()
        res19 = SymPyExpression(c_src19, 'c').return_expr()
        res20 = SymPyExpression(c_src20, 'c').return_expr()
        res21 = SymPyExpression(c_src21, 'c').return_expr()
        res22 = SymPyExpression(c_src22, 'c').return_expr()
        res23 = SymPyExpression(c_src23, 'c').return_expr()
        res24 = SymPyExpression(c_src24, 'c').return_expr()
        res25 = SymPyExpression(c_src25, 'c').return_expr()
        res26 = SymPyExpression(c_src26, 'c').return_expr()
        res27 = SymPyExpression(c_src27, 'c').return_expr()
        res28 = SymPyExpression(c_src28, 'c').return_expr()
        res29 = SymPyExpression(c_src29, 'c').return_expr()
        res30 = SymPyExpression(c_src30, 'c').return_expr()
        res31 = SymPyExpression(c_src31, 'c').return_expr()
        res32 = SymPyExpression(c_src32, 'c').return_expr()

        assert res1[0] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(100)
                )
            )

        assert res1[1] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Symbol('b')
                )
            )

        assert res2[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res2[1] == Declaration(Variable(Symbol('b'),
            type=IntBaseType(String('intc')),
            value=Add(
                Symbol('a'),
                Integer(1)
                )
            )
        )

        assert res3[0] == Declaration(
            Variable(Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('12.5', precision=53)
                )
            )

        assert res3[1] == Declaration(
            Variable(Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Mul(
                    Float('20.0', precision=53),
                    Symbol('a')
                    )
                )
            )

        assert res4[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(83)
                )
            )

        assert res5[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(-4836)
                )
            )

        assert res6[0] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        assert res6[1] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Integer(3)
                )
            )

        assert res6[2] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Symbol('b'),
                    Mul(
                        Integer(4),
                        Symbol('c')
                        )
                    )
                )
            )

        assert res7[0] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res7[1] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Symbol('b'),
                    Integer(2)
                    )
                )
            )

        assert res7[2] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Mul(
                    Integer(10),
                    Pow(
                        Symbol('b'),
                        Integer(2)
                        ),
                    Symbol('c')
                    )
                )
            )

        assert res8[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(1)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(2)
                        )
                    ),
                Declaration(
                    Variable(Symbol('temp'),
                        type=IntBaseType(String('intc')),
                        value=Symbol('a')
                        )
                    ),
                Assignment(
                    Variable(Symbol('a')),
                    Symbol('b')
                    ),
                Assignment(
                    Variable(Symbol('b')),
                    Symbol('temp')
                    )
                )
            )

        assert res9[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res9[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        assert res9[2] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Symbol('a')
                )
            )

        assert res9[3] == Declaration(
            Variable(Symbol('d'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Symbol('a'),
                    Symbol('b'),
                    Symbol('c')
                    )
                )
            )

        assert res9[4] == Declaration(
            Variable(Symbol('e'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Pow(
                        Symbol('a'),
                        Integer(3)
                        ),
                    Mul(
                        Integer(3),
                        Pow(
                            Symbol('a'),
                            Integer(2)
                            ),
                        Symbol('b')
                        ),
                    Mul(
                        Integer(3),
                        Symbol('a'),
                        Pow(
                            Symbol('b'),
                            Integer(2)
                            )
                        ),
                    Pow(
                        Symbol('b'),
                        Integer(3)
                        )
                    )
                )
            )

        assert res9[5] == Declaration(
            Variable(Symbol('f'),
                type=IntBaseType(String('intc')),
                value=Mul(
                    Add(
                        Symbol('a'),
                        Symbol('b'),
                        Mul(
                            Integer(-1),
                            Symbol('c')
                            )
                        ),
                    Add(
                        Symbol('a'),
                        Symbol('b'),
                        Symbol('c')
                        )
                    )
                )
            )

        assert res9[6] == Declaration(
            Variable(Symbol('g'),
                type=IntBaseType(String('intc')),
                value=Mul(
                    Symbol('a'),
                    Add(
                        Symbol('b'),
                        Mul(
                            Integer(-1),
                            Symbol('c')
                            )
                        ),
                    Pow(
                        Add(
                            Symbol('a'),
                            Symbol('b'),
                            Symbol('c'),
                            Symbol('d')
                            ),
                        Integer(2)
                        )
                    )
                )
            )

        assert res10[0] == Declaration(
            Variable(Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('10.0', precision=53)
                )
            )

        assert res10[1] == Declaration(
            Variable(Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.5', precision=53)
                )
            )

        assert res10[2] == Declaration(
            Variable(Symbol('c'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Add(
                    Pow(
                        Symbol('a'),
                        Integer(2)
                        ),
                    Mul(
                        Integer(2),
                        Symbol('a'),
                        Symbol('b')
                        ),
                    Pow(
                        Symbol('b'),
                        Integer(2)
                        )
                    )
                )
            )

        assert res11[0] == Declaration(
            Variable(Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('4.0', precision=53)
                )
            )

        assert res12[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(25)
                )
            )

        assert res13[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(-95)
                )
            )

        assert res14[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(-300)
                )
            )

        assert res15[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(4)
                )
            )

        assert res15[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        assert res15[2] == Declaration(
            Variable(Symbol('c'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Mul(
                    Pow(
                        Symbol('a'),
                        Integer(-1)
                        ),
                    Symbol('b')
                    )
                )
            )

        assert res16[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        assert res16[1] == Declaration(
            Variable(Symbol('d'),
                type=IntBaseType(String('intc')),
                value=Integer(5)
                )
            )

        assert res16[2] == Declaration(
            Variable(Symbol('n'),
                type=IntBaseType(String('intc')),
                value=Integer(10)
                )
            )

        assert res16[3] == Declaration(
            Variable(Symbol('s'),
                type=IntBaseType(String('intc')),
                value=Mul(
                    Rational(1, 2),
                    Symbol('a'),
                    Add(
                        Mul(
                            Integer(2),
                            Symbol('a')
                            ),
                        Mul(
                            Symbol('d'),
                            Add(
                                Symbol('n'),
                                Integer(-1)
                                )
                            )
                        )
                    )
                )
            )

        assert res17[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res18[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        assert res18[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Mod(
                    Symbol('a'),
                    Integer(3)
                    )
                )
            )

        assert res19[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(100)
                )
            )
        assert res19[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(3)
                )
            )

        assert res19[2] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Mod(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res20[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(100)
                )
            )

        assert res20[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(3)
                )
            )

        assert res20[2] == Declaration(
            Variable(Symbol('mod'),
                type=IntBaseType(String('intc')),
                value=Integer(1000000007)
                )
            )

        assert res20[3] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Mod(
                    Add(
                        Symbol('a'),
                        Mul(
                            Integer(100),
                            Pow(
                                Symbol('a'),
                                Integer(-1)
                                ),
                            Symbol('b')
                            )
                        ),
                    Symbol('mod')
                    )
                )
            )

        assert res21[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(100)
                )
            )

        assert res21[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(3)
                )
            )

        assert res21[2] == Declaration(
            Variable(Symbol('mod'),
                type=IntBaseType(String('intc')),
                value=Integer(1000000007)
                )
            )

        assert res21[3] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Mod(
                    Mul(
                        Add(
                            Symbol('a'),
                            Mul(
                                Integer(-1),
                                Symbol('b')
                                )
                            ),
                        Add(
                            Symbol('a'),
                            Symbol('b')
                            )
                        ),
                    Symbol('mod')
                    )
                )
            )

        assert res22[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res22[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res23[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res23[1] == Declaration(
            Variable(Symbol('b'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res23[2] == Declaration(
            Variable(Symbol('c'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res23[3] == Declaration(
            Variable(Symbol('d'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res24[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res24[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        assert res24[2] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=Equality(
                    Symbol('a'),
                    Integer(1)
                    )
                )
            )

        assert res24[3] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=Equality(
                    Symbol('b'),
                    Integer(2)
                    )
                )
            )

        assert res24[4] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=Unequality(
                    Integer(1),
                    Symbol('a')
                    )
                )
            )

        assert res24[5] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=Unequality(
                    Integer(1),
                    Symbol('b')
                    )
                )
            )

        assert res24[6] == Declaration(
            Variable(Symbol('c5'),
                type=Type(String('bool')),
                value=StrictLessThan(Symbol('a'),
                    Integer(0)
                    )
                )
            )

        assert res24[7] == Declaration(
            Variable(Symbol('c6'),
                type=Type(String('bool')),
                value=LessThan(
                    Symbol('b'),
                    Integer(10)
                    )
                )
            )

        assert res24[8] == Declaration(
            Variable(Symbol('c7'),
                type=Type(String('bool')),
                value=StrictGreaterThan(
                    Symbol('a'),
                    Integer(0)
                    )
                )
            )

        assert res24[9] == Declaration(
            Variable(Symbol('c8'),
                type=Type(String('bool')),
                value=GreaterThan(
                    Symbol('b'),
                    Integer(11)
                    )
                )
            )

        assert res25[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(3)
                )
            )

        assert res25[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(4)
                )
            )

        assert res25[2] == Declaration(Variable(Symbol('c1'),
            type=Type(String('bool')),
            value=Equality(
                Symbol('a'),
                Symbol('b')
                )
            )
        )

        assert res25[3] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=Unequality(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res25[4] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=StrictLessThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res25[5] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=LessThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res25[6] == Declaration(
            Variable(Symbol('c5'),
                type=Type(String('bool')),
                value=StrictGreaterThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res25[7] == Declaration(
            Variable(Symbol('c6'),
                type=Type(String('bool')),
                value=GreaterThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res26[0] == Declaration(
            Variable(Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('1.25', precision=53)
                )
            )

        assert res26[1] == Declaration(
            Variable(Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.5', precision=53)
                )
            )

        assert res26[2] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=Equality(
                    Symbol('a'),
                    Float('1.25', precision=53)
                    )
                )
            )

        assert res26[3] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=Equality(
                    Symbol('b'),
                    Float('2.54', precision=53)
                    )
                )
            )

        assert res26[4] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=Unequality(
                    Float('1.2', precision=53),
                    Symbol('a')
                    )
                )
            )

        assert res26[5] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=Unequality(
                    Float('1.5', precision=53),
                    Symbol('b')
                    )
                )
            )

        assert res27[0] == Declaration(
            Variable(Symbol('a'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('1.25', precision=53)
                )
            )

        assert res27[1] == Declaration(
            Variable(Symbol('b'),
                type=FloatType(
                    String('float32'),
                    nbits=Integer(32),
                    nmant=Integer(23),
                    nexp=Integer(8)
                    ),
                value=Float('2.5', precision=53)
                )
            )

        assert res27[2] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=Equality(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res27[3] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=Unequality(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res27[4] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=StrictLessThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res27[5] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=LessThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res27[6] == Declaration(
            Variable(Symbol('c5'),
                type=Type(String('bool')),
                value=StrictGreaterThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res27[7] == Declaration(
            Variable(Symbol('c6'),
                type=Type(String('bool')),
                value=GreaterThan(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res28[0] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res28[1] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res28[2] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res28[3] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res28[4] == Declaration(
            Variable(Symbol('c5'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res28[5] == Declaration(
            Variable(Symbol('c6'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res29[0] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res29[1] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res29[2] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res29[3] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res29[4] == Declaration(
            Variable(Symbol('c5'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res29[5] == Declaration(
            Variable(Symbol('c6'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res30[0] == Declaration(
            Variable(Symbol('a'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res30[1] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=Symbol('a')
                )
            )

        assert res30[2] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res30[3] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res30[4] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=Symbol('a')
                )
            )

        assert res31[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res31[1] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=Symbol('a')
                )
            )

        assert res31[2] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res31[3] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res31[4] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=Symbol('a')
                )
            )

        assert res32[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res32[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(0)
                )
            )

        assert res32[2] == Declaration(
            Variable(Symbol('c'),
                type=Type(String('bool')),
                value=false
                )
            )

        assert res32[3] == Declaration(
            Variable(Symbol('d'),
                type=Type(String('bool')),
                value=true
                )
            )

        assert res32[4] == Declaration(
            Variable(Symbol('c1'),
                type=Type(String('bool')),
                value=And(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res32[5] == Declaration(
            Variable(Symbol('c2'),
                type=Type(String('bool')),
                value=And(
                    Symbol('a'),
                    Symbol('c')
                    )
                )
            )

        assert res32[6] == Declaration(
            Variable(Symbol('c3'),
                type=Type(String('bool')),
                value=And(
                    Symbol('c'),
                    Symbol('d')
                    )
                )
            )

        assert res32[7] == Declaration(
            Variable(Symbol('c4'),
                type=Type(String('bool')),
                value=Or(
                    Symbol('a'),
                    Symbol('b')
                    )
                )
            )

        assert res32[8] == Declaration(
            Variable(Symbol('c5'),
                type=Type(String('bool')),
                value=Or(
                    Symbol('a'),
                    Symbol('c')
                    )
                )
            )

        assert res32[9] == Declaration(
            Variable(Symbol('c6'),
                type=Type(String('bool')),
                value=Or(
                    Symbol('c'),
                    Symbol('d')
                    )
                )
            )

        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise1, 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise2, 'c'))


    def test_paren_expr():
        c_src1 = (
            'int a = (1);'
            'int b = (1 + 2 * 3);'
        )

        c_src2 = (
            'int a = 1, b = 2, c = 3;'
            'int d = (a);'
            'int e = (a + 1);'
            'int f = (a + b * c - d / e);'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()

        assert res1[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res1[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(7)
                )
            )

        assert res2[0] == Declaration(
            Variable(Symbol('a'),
                type=IntBaseType(String('intc')),
                value=Integer(1)
                )
            )

        assert res2[1] == Declaration(
            Variable(Symbol('b'),
                type=IntBaseType(String('intc')),
                value=Integer(2)
                )
            )

        assert res2[2] == Declaration(
            Variable(Symbol('c'),
                type=IntBaseType(String('intc')),
                value=Integer(3)
                )
            )

        assert res2[3] == Declaration(
            Variable(Symbol('d'),
                type=IntBaseType(String('intc')),
                value=Symbol('a')
                )
            )

        assert res2[4] == Declaration(
            Variable(Symbol('e'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Symbol('a'),
                    Integer(1)
                    )
                )
            )

        assert res2[5] == Declaration(
            Variable(Symbol('f'),
                type=IntBaseType(String('intc')),
                value=Add(
                    Symbol('a'),
                    Mul(
                        Symbol('b'),
                        Symbol('c')
                        ),
                    Mul(
                        Integer(-1),
                        Symbol('d'),
                        Pow(
                            Symbol('e'),
                            Integer(-1)
                            )
                        )
                    )
                )
            )


    def test_unary_operators():
        c_src1 = (
            'void func()'+
            '{' + '\n' +
                'int a = 10;' + '\n' +
                'int b = 20;' + '\n' +
                '++a;' + '\n' +
                '--b;' + '\n' +
                'a++;' + '\n' +
                'b--;' + '\n' +
            '}'
        )

        c_src2 = (
            'void func()'+
            '{' + '\n' +
                'int a = 10;' + '\n' +
                'int b = -100;' + '\n' +
                'int c = +19;' + '\n' +
                'int d = ++a;' + '\n' +
                'int e = --b;' + '\n' +
                'int f = a++;' + '\n' +
                'int g = b--;' + '\n' +
                'bool h = !false;' + '\n' +
                'bool i = !d;' + '\n' +
                'bool j = !0;' + '\n' +
                'bool k = !10.0;' + '\n' +
            '}'
        )

        c_src_raise1 = (
            'void func()'+
            '{' + '\n' +
                'int a = 10;' + '\n' +
                'int b = ~a;' + '\n' +
            '}'
        )

        c_src_raise2 = (
            'void func()'+
            '{' + '\n' +
                'int a = 10;' + '\n' +
                'int b = *&a;' + '\n' +
            '}'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()

        assert res1[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(10)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(20)
                        )
                    ),
                PreIncrement(Symbol('a')),
                PreDecrement(Symbol('b')),
                PostIncrement(Symbol('a')),
                PostDecrement(Symbol('b'))
                )
            )

        assert res2[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(10)
                        )
                    ),
                Declaration(
                    Variable(Symbol('b'),
                        type=IntBaseType(String('intc')),
                        value=Integer(-100)
                        )
                    ),
                Declaration(
                    Variable(Symbol('c'),
                        type=IntBaseType(String('intc')),
                        value=Integer(19)
                        )
                    ),
                Declaration(
                    Variable(Symbol('d'),
                        type=IntBaseType(String('intc')),
                        value=PreIncrement(Symbol('a'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('e'),
                        type=IntBaseType(String('intc')),
                        value=PreDecrement(Symbol('b'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('f'),
                        type=IntBaseType(String('intc')),
                        value=PostIncrement(Symbol('a'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('g'),
                        type=IntBaseType(String('intc')),
                        value=PostDecrement(Symbol('b'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('h'),
                        type=Type(String('bool')),
                        value=true
                        )
                    ),
                Declaration(
                    Variable(Symbol('i'),
                        type=Type(String('bool')),
                        value=Not(Symbol('d'))
                        )
                    ),
                Declaration(
                    Variable(Symbol('j'),
                        type=Type(String('bool')),
                        value=true
                        )
                    ),
                Declaration(
                    Variable(Symbol('k'),
                        type=Type(String('bool')),
                        value=false
                        )
                    )
                )
            )

        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise1, 'c'))
        raises(NotImplementedError, lambda: SymPyExpression(c_src_raise2, 'c'))


    def test_compound_assignment_operator():
        c_src = (
            'void func()'+
            '{' + '\n' +
                'int a = 100;' + '\n' +
                'a += 10;' + '\n' +
                'a -= 10;' + '\n' +
                'a *= 10;' + '\n' +
                'a /= 10;' + '\n' +
                'a %= 10;' + '\n' +
            '}'
        )

        res = SymPyExpression(c_src, 'c').return_expr()

        assert res[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('a'),
                        type=IntBaseType(String('intc')),
                        value=Integer(100)
                        )
                    ),
                AddAugmentedAssignment(
                    Variable(Symbol('a')),
                    Integer(10)
                    ),
                SubAugmentedAssignment(
                    Variable(Symbol('a')),
                    Integer(10)
                    ),
                MulAugmentedAssignment(
                    Variable(Symbol('a')),
                    Integer(10)
                    ),
                DivAugmentedAssignment(
                    Variable(Symbol('a')),
                    Integer(10)
                    ),
                ModAugmentedAssignment(
                    Variable(Symbol('a')),
                    Integer(10)
                    )
                )
            )

    @XFAIL # this is expected to fail because of a bug in the C parser.
    def test_while_stmt():
        c_src1 = (
            'void func()'+
            '{' + '\n' +
                'int i = 0;' + '\n' +
                'while(i < 10)' + '\n' +
                '{' + '\n' +
                    'i++;' + '\n' +
                '}'
            '}'
        )

        c_src2 = (
            'void func()'+
            '{' + '\n' +
                'int i = 0;' + '\n' +
                'while(i < 10)' + '\n' +
                    'i++;' + '\n' +
            '}'
        )

        c_src3 = (
            'void func()'+
            '{' + '\n' +
                'int i = 10;' + '\n' +
                'int cnt = 0;' + '\n' +
                'while(i > 0)' + '\n' +
                '{' + '\n' +
                    'i--;' + '\n' +
                    'cnt++;' + '\n' +
                '}' + '\n' +
            '}'
        )

        c_src4 = (
            'int digit_sum(int n)'+
            '{' + '\n' +
                'int sum = 0;' + '\n' +
                'while(n > 0)' + '\n' +
                '{' + '\n' +
                    'sum += (n % 10);' + '\n' +
                    'n /= 10;' + '\n' +
                '}' + '\n' +
                'return sum;' + '\n' +
            '}'
        )

        c_src5 = (
            'void func()'+
            '{' + '\n' +
                'while(1);' + '\n' +
            '}'
        )

        res1 = SymPyExpression(c_src1, 'c').return_expr()
        res2 = SymPyExpression(c_src2, 'c').return_expr()
        res3 = SymPyExpression(c_src3, 'c').return_expr()
        res4 = SymPyExpression(c_src4, 'c').return_expr()
        res5 = SymPyExpression(c_src5, 'c').return_expr()

        assert res1[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(Symbol('i'),
                        type=IntBaseType(String('intc')),
                        value=Integer(0)
                        )
                    ),
                While(
                    StrictLessThan(
                        Symbol('i'),
                        Integer(10)
                        ),
                    body=CodeBlock(
                        PostIncrement(
                            Symbol('i')
                            )
                        )
                    )
                )
            )

        assert res2[0] == res1[0]

        assert res3[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('i'),
                        type=IntBaseType(String('intc')),
                        value=Integer(10)
                        )
                    ),
                Declaration(
                    Variable(
                        Symbol('cnt'),
                        type=IntBaseType(String('intc')),
                        value=Integer(0)
                        )
                    ),
                While(
                    StrictGreaterThan(
                        Symbol('i'),
                        Integer(0)
                        ),
                    body=CodeBlock(
                        PostDecrement(
                            Symbol('i')
                            ),
                        PostIncrement(
                            Symbol('cnt')
                            )
                        )
                    )
                )
            )

        assert res4[0] == FunctionDefinition(
            IntBaseType(String('intc')),
            name=String('digit_sum'),
            parameters=(
                Variable(
                    Symbol('n'),
                    type=IntBaseType(String('intc'))
                    ),
                ),
            body=CodeBlock(
                Declaration(
                    Variable(
                        Symbol('sum'),
                        type=IntBaseType(String('intc')),
                        value=Integer(0)
                        )
                    ),
                While(
                    StrictGreaterThan(
                        Symbol('n'),
                        Integer(0)
                        ),
                    body=CodeBlock(
                        AddAugmentedAssignment(
                            Variable(
                                Symbol('sum')
                                ),
                            Mod(
                                Symbol('n'),
                                Integer(10)
                                )
                            ),
                        DivAugmentedAssignment(
                            Variable(
                                Symbol('n')
                                ),
                            Integer(10)
                            )
                        )
                    ),
                Return('sum')
                )
            )

        assert res5[0] == FunctionDefinition(
            NoneToken(),
            name=String('func'),
            parameters=(),
            body=CodeBlock(
                While(
                    Integer(1),
                    body=CodeBlock(
                        NoneToken()
                        )
                    )
                )
            )


else:
    def test_raise():
        from sympy.parsing.c.c_parser import CCodeConverter
        raises(ImportError, lambda: CCodeConverter())
        raises(ImportError, lambda: SymPyExpression(' ', mode = 'c'))
