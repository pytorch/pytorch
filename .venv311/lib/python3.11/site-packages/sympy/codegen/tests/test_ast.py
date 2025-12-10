import math
from sympy.core.containers import Tuple
from sympy.core.numbers import nan, oo, Float, Integer
from sympy.core.relational import Lt
from sympy.core.symbol import symbols, Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.testing.pytest import raises


from sympy.codegen.ast import (
    Assignment, Attribute, aug_assign, CodeBlock, For, Type, Variable, Pointer, Declaration,
    AddAugmentedAssignment, SubAugmentedAssignment, MulAugmentedAssignment,
    DivAugmentedAssignment, ModAugmentedAssignment, value_const, pointer_const,
    integer, real, complex_, int8, uint8, float16 as f16, float32 as f32,
    float64 as f64, float80 as f80, float128 as f128, complex64 as c64, complex128 as c128,
    While, Scope, String, Print, QuotedString, FunctionPrototype, FunctionDefinition, Return,
    FunctionCall, untyped, IntBaseType, intc, Node, none, NoneToken, Token, Comment
)

x, y, z, t, x0, x1, x2, a, b = symbols("x, y, z, t, x0, x1, x2, a, b")
n = symbols("n", integer=True)
A = MatrixSymbol('A', 3, 1)
mat = Matrix([1, 2, 3])
B = IndexedBase('B')
i = Idx("i", n)
A22 = MatrixSymbol('A22',2,2)
B22 = MatrixSymbol('B22',2,2)


def test_Assignment():
    # Here we just do things to show they don't error
    Assignment(x, y)
    Assignment(x, 0)
    Assignment(A, mat)
    Assignment(A[1,0], 0)
    Assignment(A[1,0], x)
    Assignment(B[i], x)
    Assignment(B[i], 0)
    a = Assignment(x, y)
    assert a.func(*a.args) == a
    assert a.op == ':='
    # Here we test things to show that they error
    # Matrix to scalar
    raises(ValueError, lambda: Assignment(B[i], A))
    raises(ValueError, lambda: Assignment(B[i], mat))
    raises(ValueError, lambda: Assignment(x, mat))
    raises(ValueError, lambda: Assignment(x, A))
    raises(ValueError, lambda: Assignment(A[1,0], mat))
    # Scalar to matrix
    raises(ValueError, lambda: Assignment(A, x))
    raises(ValueError, lambda: Assignment(A, 0))
    # Non-atomic lhs
    raises(TypeError, lambda: Assignment(mat, A))
    raises(TypeError, lambda: Assignment(0, x))
    raises(TypeError, lambda: Assignment(x*x, 1))
    raises(TypeError, lambda: Assignment(A + A, mat))
    raises(TypeError, lambda: Assignment(B, 0))


def test_AugAssign():
    # Here we just do things to show they don't error
    aug_assign(x, '+', y)
    aug_assign(x, '+', 0)
    aug_assign(A, '+', mat)
    aug_assign(A[1, 0], '+', 0)
    aug_assign(A[1, 0], '+', x)
    aug_assign(B[i], '+', x)
    aug_assign(B[i], '+', 0)

    # Check creation via aug_assign vs constructor
    for binop, cls in [
            ('+', AddAugmentedAssignment),
            ('-', SubAugmentedAssignment),
            ('*', MulAugmentedAssignment),
            ('/', DivAugmentedAssignment),
            ('%', ModAugmentedAssignment),
        ]:
        a = aug_assign(x, binop, y)
        b = cls(x, y)
        assert a.func(*a.args) == a == b
        assert a.binop == binop
        assert a.op == binop + '='

    # Here we test things to show that they error
    # Matrix to scalar
    raises(ValueError, lambda: aug_assign(B[i], '+', A))
    raises(ValueError, lambda: aug_assign(B[i], '+', mat))
    raises(ValueError, lambda: aug_assign(x, '+', mat))
    raises(ValueError, lambda: aug_assign(x, '+', A))
    raises(ValueError, lambda: aug_assign(A[1, 0], '+', mat))
    # Scalar to matrix
    raises(ValueError, lambda: aug_assign(A, '+', x))
    raises(ValueError, lambda: aug_assign(A, '+', 0))
    # Non-atomic lhs
    raises(TypeError, lambda: aug_assign(mat, '+', A))
    raises(TypeError, lambda: aug_assign(0, '+', x))
    raises(TypeError, lambda: aug_assign(x * x, '+', 1))
    raises(TypeError, lambda: aug_assign(A + A, '+', mat))
    raises(TypeError, lambda: aug_assign(B, '+', 0))


def test_Assignment_printing():
    assignment_classes = [
        Assignment,
        AddAugmentedAssignment,
        SubAugmentedAssignment,
        MulAugmentedAssignment,
        DivAugmentedAssignment,
        ModAugmentedAssignment,
    ]
    pairs = [
        (x, 2 * y + 2),
        (B[i], x),
        (A22, B22),
        (A[0, 0], x),
    ]

    for cls in assignment_classes:
        for lhs, rhs in pairs:
            a = cls(lhs, rhs)
            assert repr(a) == '%s(%s, %s)' % (cls.__name__, repr(lhs), repr(rhs))


def test_CodeBlock():
    c = CodeBlock(Assignment(x, 1), Assignment(y, x + 1))
    assert c.func(*c.args) == c

    assert c.left_hand_sides == Tuple(x, y)
    assert c.right_hand_sides == Tuple(1, x + 1)

def test_CodeBlock_topological_sort():
    assignments = [
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(t, x),
        Assignment(y, 2),
        ]

    ordered_assignments = [
        # Note that the unrelated z=1 and y=2 are kept in that order
        Assignment(z, 1),
        Assignment(y, 2),
        Assignment(x, y + z),
        Assignment(t, x),
        ]
    c1 = CodeBlock.topological_sort(assignments)
    assert c1 == CodeBlock(*ordered_assignments)

    # Cycle
    invalid_assignments = [
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(y, x),
        Assignment(y, 2),
        ]

    raises(ValueError, lambda: CodeBlock.topological_sort(invalid_assignments))

    # Free symbols
    free_assignments = [
        Assignment(x, y + z),
        Assignment(z, a * b),
        Assignment(t, x),
        Assignment(y, b + 3),
        ]

    free_assignments_ordered = [
        Assignment(z, a * b),
        Assignment(y, b + 3),
        Assignment(x, y + z),
        Assignment(t, x),
        ]

    c2 = CodeBlock.topological_sort(free_assignments)
    assert c2 == CodeBlock(*free_assignments_ordered)

def test_CodeBlock_free_symbols():
    c1 = CodeBlock(
        Assignment(x, y + z),
        Assignment(z, 1),
        Assignment(t, x),
        Assignment(y, 2),
        )
    assert c1.free_symbols == set()

    c2 = CodeBlock(
        Assignment(x, y + z),
        Assignment(z, a * b),
        Assignment(t, x),
        Assignment(y, b + 3),
    )
    assert c2.free_symbols == {a, b}

def test_CodeBlock_cse():
    c1 = CodeBlock(
        Assignment(y, 1),
        Assignment(x, sin(y)),
        Assignment(z, sin(y)),
        Assignment(t, x*z),
        )
    assert c1.cse() == CodeBlock(
        Assignment(y, 1),
        Assignment(x0, sin(y)),
        Assignment(x, x0),
        Assignment(z, x0),
        Assignment(t, x*z),
    )

    # Multiple assignments to same symbol not supported
    raises(NotImplementedError, lambda: CodeBlock(
        Assignment(x, 1),
        Assignment(y, 1), Assignment(y, 2)
    ).cse())

    # Check auto-generated symbols do not collide with existing ones
    c2 = CodeBlock(
        Assignment(x0, sin(y) + 1),
        Assignment(x1, 2 * sin(y)),
        Assignment(z, x * y),
        )
    assert c2.cse() == CodeBlock(
        Assignment(x2, sin(y)),
        Assignment(x0, x2 + 1),
        Assignment(x1, 2 * x2),
        Assignment(z, x * y),
        )


def test_CodeBlock_cse__issue_14118():
    # see https://github.com/sympy/sympy/issues/14118
    c = CodeBlock(
        Assignment(A22, Matrix([[x, sin(y)],[3, 4]])),
        Assignment(B22, Matrix([[sin(y), 2*sin(y)], [sin(y)**2, 7]]))
    )
    assert c.cse() == CodeBlock(
        Assignment(x0, sin(y)),
        Assignment(A22, Matrix([[x, x0],[3, 4]])),
        Assignment(B22, Matrix([[x0, 2*x0], [x0**2, 7]]))
    )

def test_For():
    f = For(n, Range(0, 3), (Assignment(A[n, 0], x + n), aug_assign(x, '+', y)))
    f = For(n, (1, 2, 3, 4, 5), (Assignment(A[n, 0], x + n),))
    assert f.func(*f.args) == f
    raises(TypeError, lambda: For(n, x, (x + y,)))


def test_none():
    assert none.is_Atom
    assert none == none
    class Foo(Token):
        pass
    foo = Foo()
    assert foo != none
    assert none == None
    assert none == NoneToken()
    assert none.func(*none.args) == none


def test_String():
    st = String('foobar')
    assert st.is_Atom
    assert st == String('foobar')
    assert st.text == 'foobar'
    assert st.func(**st.kwargs()) == st
    assert st.func(*st.args) == st


    class Signifier(String):
        pass

    si = Signifier('foobar')
    assert si != st
    assert si.text == st.text
    s = String('foo')
    assert str(s) == 'foo'
    assert repr(s) == "String('foo')"

def test_Comment():
    c = Comment('foobar')
    assert c.text == 'foobar'
    assert str(c) == 'foobar'

def test_Node():
    n = Node()
    assert n == Node()
    assert n.func(*n.args) == n


def test_Type():
    t = Type('MyType')
    assert len(t.args) == 1
    assert t.name == String('MyType')
    assert str(t) == 'MyType'
    assert repr(t) == "Type(String('MyType'))"
    assert Type(t) == t
    assert t.func(*t.args) == t
    t1 = Type('t1')
    t2 = Type('t2')
    assert t1 != t2
    assert t1 == t1 and t2 == t2
    t1b = Type('t1')
    assert t1 == t1b
    assert t2 != t1b


def test_Type__from_expr():
    assert Type.from_expr(i) == integer
    u = symbols('u', real=True)
    assert Type.from_expr(u) == real
    assert Type.from_expr(n) == integer
    assert Type.from_expr(3) == integer
    assert Type.from_expr(3.0) == real
    assert Type.from_expr(3+1j) == complex_
    raises(ValueError, lambda: Type.from_expr(sum))


def test_Type__cast_check__integers():
    # Rounding
    raises(ValueError, lambda: integer.cast_check(3.5))
    assert integer.cast_check('3') == 3
    assert integer.cast_check(Float('3.0000000000000000000')) == 3
    assert integer.cast_check(Float('3.0000000000000000001')) == 3  # unintuitive maybe?

    # Range
    assert int8.cast_check(127.0) == 127
    raises(ValueError, lambda: int8.cast_check(128))
    assert int8.cast_check(-128) == -128
    raises(ValueError, lambda: int8.cast_check(-129))

    assert uint8.cast_check(0) == 0
    assert uint8.cast_check(128) == 128
    raises(ValueError, lambda: uint8.cast_check(256.0))
    raises(ValueError, lambda: uint8.cast_check(-1))

def test_Attribute():
    noexcept = Attribute('noexcept')
    assert noexcept == Attribute('noexcept')
    alignas16 = Attribute('alignas', [16])
    alignas32 = Attribute('alignas', [32])
    assert alignas16 != alignas32
    assert alignas16.func(*alignas16.args) == alignas16


def test_Variable():
    v = Variable(x, type=real)
    assert v == Variable(v)
    assert v == Variable('x', type=real)
    assert v.symbol == x
    assert v.type == real
    assert value_const not in v.attrs
    assert v.func(*v.args) == v
    assert str(v) == 'Variable(x, type=real)'

    w = Variable(y, f32, attrs={value_const})
    assert w.symbol == y
    assert w.type == f32
    assert value_const in w.attrs
    assert w.func(*w.args) == w

    v_n = Variable(n, type=Type.from_expr(n))
    assert v_n.type == integer
    assert v_n.func(*v_n.args) == v_n
    v_i = Variable(i, type=Type.from_expr(n))
    assert v_i.type == integer
    assert v_i != v_n

    a_i = Variable.deduced(i)
    assert a_i.type == integer
    assert Variable.deduced(Symbol('x', real=True)).type == real
    assert a_i.func(*a_i.args) == a_i

    v_n2 = Variable.deduced(n, value=3.5, cast_check=False)
    assert v_n2.func(*v_n2.args) == v_n2
    assert abs(v_n2.value - 3.5) < 1e-15
    raises(ValueError, lambda: Variable.deduced(n, value=3.5, cast_check=True))

    v_n3 = Variable.deduced(n)
    assert v_n3.type == integer
    assert str(v_n3) == 'Variable(n, type=integer)'
    assert Variable.deduced(z, value=3).type == integer
    assert Variable.deduced(z, value=3.0).type == real
    assert Variable.deduced(z, value=3.0+1j).type == complex_


def test_Pointer():
    p = Pointer(x)
    assert p.symbol == x
    assert p.type == untyped
    assert value_const not in p.attrs
    assert pointer_const not in p.attrs
    assert p.func(*p.args) == p

    u = symbols('u', real=True)
    pu = Pointer(u, type=Type.from_expr(u), attrs={value_const, pointer_const})
    assert pu.symbol is u
    assert pu.type == real
    assert value_const in pu.attrs
    assert pointer_const in pu.attrs
    assert pu.func(*pu.args) == pu

    i = symbols('i', integer=True)
    deref = pu[i]
    assert deref.indices == (i,)


def test_Declaration():
    u = symbols('u', real=True)
    vu = Variable(u, type=Type.from_expr(u))
    assert Declaration(vu).variable.type == real
    vn = Variable(n, type=Type.from_expr(n))
    assert Declaration(vn).variable.type == integer

    # PR 19107, does not allow comparison between expressions and Basic
    # lt = StrictLessThan(vu, vn)
    # assert isinstance(lt, StrictLessThan)

    vuc = Variable(u, Type.from_expr(u), value=3.0, attrs={value_const})
    assert value_const in vuc.attrs
    assert pointer_const not in vuc.attrs
    decl = Declaration(vuc)
    assert decl.variable == vuc
    assert isinstance(decl.variable.value, Float)
    assert decl.variable.value == 3.0
    assert decl.func(*decl.args) == decl
    assert vuc.as_Declaration() == decl
    assert vuc.as_Declaration(value=None, attrs=None) == Declaration(vu)

    vy = Variable(y, type=integer, value=3)
    decl2 = Declaration(vy)
    assert decl2.variable == vy
    assert decl2.variable.value == Integer(3)

    vi = Variable(i, type=Type.from_expr(i), value=3.0)
    decl3 = Declaration(vi)
    assert decl3.variable.type == integer
    assert decl3.variable.value == 3.0

    raises(ValueError, lambda: Declaration(vi, 42))


def test_IntBaseType():
    assert intc.name == String('intc')
    assert intc.args == (intc.name,)
    assert str(IntBaseType('a').name) == 'a'


def test_FloatType():
    assert f16.dig == 3
    assert f32.dig == 6
    assert f64.dig == 15
    assert f80.dig == 18
    assert f128.dig == 33

    assert f16.decimal_dig == 5
    assert f32.decimal_dig == 9
    assert f64.decimal_dig == 17
    assert f80.decimal_dig == 21
    assert f128.decimal_dig == 36

    assert f16.max_exponent == 16
    assert f32.max_exponent == 128
    assert f64.max_exponent == 1024
    assert f80.max_exponent == 16384
    assert f128.max_exponent == 16384

    assert f16.min_exponent == -13
    assert f32.min_exponent == -125
    assert f64.min_exponent == -1021
    assert f80.min_exponent == -16381
    assert f128.min_exponent == -16381

    assert abs(f16.eps / Float('0.00097656', precision=16) - 1) < 0.1*10**-f16.dig
    assert abs(f32.eps / Float('1.1920929e-07', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.eps / Float('2.2204460492503131e-16', precision=64) - 1) < 0.1*10**-f64.dig
    assert abs(f80.eps / Float('1.08420217248550443401e-19', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.eps / Float(' 1.92592994438723585305597794258492732e-34', precision=128) - 1) < 0.1*10**-f128.dig

    assert abs(f16.max / Float('65504', precision=16) - 1) < .1*10**-f16.dig
    assert abs(f32.max / Float('3.40282347e+38', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.max / Float('1.79769313486231571e+308', precision=64) - 1) < 0.1*10**-f64.dig  # cf. np.finfo(np.float64).max
    assert abs(f80.max / Float('1.18973149535723176502e+4932', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.max / Float('1.18973149535723176508575932662800702e+4932', precision=128) - 1) < 0.1*10**-f128.dig

    # cf. np.finfo(np.float32).tiny
    assert abs(f16.tiny / Float('6.1035e-05', precision=16) - 1) < 0.1*10**-f16.dig
    assert abs(f32.tiny / Float('1.17549435e-38', precision=32) - 1) < 0.1*10**-f32.dig
    assert abs(f64.tiny / Float('2.22507385850720138e-308', precision=64) - 1) < 0.1*10**-f64.dig
    assert abs(f80.tiny / Float('3.36210314311209350626e-4932', precision=80) - 1) < 0.1*10**-f80.dig
    assert abs(f128.tiny / Float('3.3621031431120935062626778173217526e-4932', precision=128) - 1) < 0.1*10**-f128.dig

    assert f64.cast_check(0.5) == Float(0.5, 17)
    assert abs(f64.cast_check(3.7) - 3.7) < 3e-17
    assert isinstance(f64.cast_check(3), (Float, float))

    assert f64.cast_nocheck(oo) == float('inf')
    assert f64.cast_nocheck(-oo) == float('-inf')
    assert f64.cast_nocheck(float(oo)) == float('inf')
    assert f64.cast_nocheck(float(-oo)) == float('-inf')
    assert math.isnan(f64.cast_nocheck(nan))

    assert f32 != f64
    assert f64 == f64.func(*f64.args)


def test_Type__cast_check__floating_point():
    raises(ValueError, lambda: f32.cast_check(123.45678949))
    raises(ValueError, lambda: f32.cast_check(12.345678949))
    raises(ValueError, lambda: f32.cast_check(1.2345678949))
    raises(ValueError, lambda: f32.cast_check(.12345678949))
    assert abs(123.456789049 - f32.cast_check(123.456789049) - 4.9e-8) < 1e-8
    assert abs(0.12345678904 - f32.cast_check(0.12345678904) - 4e-11) < 1e-11

    dcm21 = Float('0.123456789012345670499')  # 21 decimals
    assert abs(dcm21 - f64.cast_check(dcm21) - 4.99e-19) < 1e-19

    f80.cast_check(Float('0.12345678901234567890103', precision=88))
    raises(ValueError, lambda: f80.cast_check(Float('0.12345678901234567890149', precision=88)))

    v10 = 12345.67894
    raises(ValueError, lambda: f32.cast_check(v10))
    assert abs(Float(str(v10), precision=64+8) - f64.cast_check(v10)) < v10*1e-16

    assert abs(f32.cast_check(2147483647) - 2147483650) < 1


def test_Type__cast_check__complex_floating_point():
    val9_11 = 123.456789049 + 0.123456789049j
    raises(ValueError, lambda: c64.cast_check(.12345678949 + .12345678949j))
    assert abs(val9_11 - c64.cast_check(val9_11) - 4.9e-8) < 1e-8

    dcm21 = Float('0.123456789012345670499') + 1e-20j  # 21 decimals
    assert abs(dcm21 - c128.cast_check(dcm21) - 4.99e-19) < 1e-19
    v19 = Float('0.1234567890123456749') + 1j*Float('0.1234567890123456749')
    raises(ValueError, lambda: c128.cast_check(v19))


def test_While():
    xpp = AddAugmentedAssignment(x, 1)
    whl1 = While(x < 2, [xpp])
    assert whl1.condition.args[0] == x
    assert whl1.condition.args[1] == 2
    assert whl1.condition == Lt(x, 2, evaluate=False)
    assert whl1.body.args == (xpp,)
    assert whl1.func(*whl1.args) == whl1

    cblk = CodeBlock(AddAugmentedAssignment(x, 1))
    whl2 = While(x < 2, cblk)
    assert whl1 == whl2
    assert whl1 != While(x < 3, [xpp])


def test_Scope():
    assign = Assignment(x, y)
    incr = AddAugmentedAssignment(x, 1)
    scp = Scope([assign, incr])
    cblk = CodeBlock(assign, incr)
    assert scp.body == cblk
    assert scp == Scope(cblk)
    assert scp != Scope([incr, assign])
    assert scp.func(*scp.args) == scp


def test_Print():
    fmt = "%d %.3f"
    ps = Print([n, x], fmt)
    assert str(ps.format_string) == fmt
    assert ps.print_args == Tuple(n, x)
    assert ps.args == (Tuple(n, x), QuotedString(fmt), none)
    assert ps == Print((n, x), fmt)
    assert ps != Print([x, n], fmt)
    assert ps.func(*ps.args) == ps

    ps2 = Print([n, x])
    assert ps2 == Print([n, x])
    assert ps2 != ps
    assert ps2.format_string == None


def test_FunctionPrototype_and_FunctionDefinition():
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    fp1 = FunctionPrototype(real, 'power', [vx, vn])
    assert fp1.return_type == real
    assert fp1.name == String('power')
    assert fp1.parameters == Tuple(vx, vn)
    assert fp1 == FunctionPrototype(real, 'power', [vx, vn])
    assert fp1 != FunctionPrototype(real, 'power', [vn, vx])
    assert fp1.func(*fp1.args) == fp1


    body = [Assignment(x, x**n), Return(x)]
    fd1 = FunctionDefinition(real, 'power', [vx, vn], body)
    assert fd1.return_type == real
    assert str(fd1.name) == 'power'
    assert fd1.parameters == Tuple(vx, vn)
    assert fd1.body == CodeBlock(*body)
    assert fd1 == FunctionDefinition(real, 'power', [vx, vn], body)
    assert fd1 != FunctionDefinition(real, 'power', [vx, vn], body[::-1])
    assert fd1.func(*fd1.args) == fd1

    fp2 = FunctionPrototype.from_FunctionDefinition(fd1)
    assert fp2 == fp1

    fd2 = FunctionDefinition.from_FunctionPrototype(fp1, body)
    assert fd2 == fd1


def test_Return():
    rs = Return(x)
    assert rs.args == (x,)
    assert rs == Return(x)
    assert rs != Return(y)
    assert rs.func(*rs.args) == rs


def test_FunctionCall():
    fc = FunctionCall('power', (x, 3))
    assert fc.function_args[0] == x
    assert fc.function_args[1] == 3
    assert len(fc.function_args) == 2
    assert isinstance(fc.function_args[1], Integer)
    assert fc == FunctionCall('power', (x, 3))
    assert fc != FunctionCall('power', (3, x))
    assert fc != FunctionCall('Power', (x, 3))
    assert fc.func(*fc.args) == fc

    fc2 = FunctionCall('fma', [2, 3, 4])
    assert len(fc2.function_args) == 3
    assert fc2.function_args[0] == 2
    assert fc2.function_args[1] == 3
    assert fc2.function_args[2] == 4
    assert str(fc2) in ( # not sure if QuotedString is a better default...
        'FunctionCall(fma, function_args=(2, 3, 4))',
        'FunctionCall("fma", function_args=(2, 3, 4))',
    )

def test_ast_replace():
    x = Variable('x', real)
    y = Variable('y', real)
    n = Variable('n', integer)

    pwer = FunctionDefinition(real, 'pwer', [x, n], [pow(x.symbol, n.symbol)])
    pname = pwer.name
    pcall = FunctionCall('pwer', [y, 3])

    tree1 = CodeBlock(pwer, pcall)
    assert str(tree1.args[0].name) == 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    for a, b in zip(tree1, [pwer, pcall]):
        assert a == b

    tree2 = tree1.replace(pname, String('power'))
    assert str(tree1.args[0].name) == 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    assert str(tree2.args[0].name) == 'power'
    assert str(tree2.args[1].name) == 'power'
