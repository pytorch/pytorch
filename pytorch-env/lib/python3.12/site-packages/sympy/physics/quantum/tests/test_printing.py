# -*- encoding: utf-8 -*-
"""
TODO:
* Address Issue 2251, printing of spin states
"""
from __future__ import annotations
from typing import Any

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.cg import CG, Wigner3j, Wigner6j, Wigner9j
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import CGate, CNotGate, IdentityGate, UGate, XGate
from sympy.physics.quantum.hilbert import ComplexSpace, FockSpace, HilbertSpace, L2
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.operator import Operator, OuterProduct, DifferentialOperator
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.qubit import Qubit, IntQubit
from sympy.physics.quantum.spin import Jz, J2, JzBra, JzBraCoupled, JzKet, JzKetCoupled, Rotation, WignerD
from sympy.physics.quantum.state import Bra, Ket, TimeDepBra, TimeDepKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.sho1d import RaisingOp

from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import oo
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.matrices.dense import Matrix
from sympy.sets.sets import Interval
from sympy.testing.pytest import XFAIL

# Imports used in srepr strings
from sympy.physics.quantum.spin import JzOp

from sympy.printing import srepr
from sympy.printing.pretty import pretty as xpretty
from sympy.printing.latex import latex

MutableDenseMatrix = Matrix


ENV: dict[str, Any] = {}
exec('from sympy import *', ENV)
exec('from sympy.physics.quantum import *', ENV)
exec('from sympy.physics.quantum.cg import *', ENV)
exec('from sympy.physics.quantum.spin import *', ENV)
exec('from sympy.physics.quantum.hilbert import *', ENV)
exec('from sympy.physics.quantum.qubit import *', ENV)
exec('from sympy.physics.quantum.qexpr import *', ENV)
exec('from sympy.physics.quantum.gate import *', ENV)
exec('from sympy.physics.quantum.constants import *', ENV)


def sT(expr, string):
    """
    sT := sreprTest
    from sympy/printing/tests/test_repr.py
    """
    assert srepr(expr) == string
    assert eval(string, ENV) == expr


def pretty(expr):
    """ASCII pretty-printing"""
    return xpretty(expr, use_unicode=False, wrap_line=False)


def upretty(expr):
    """Unicode pretty-printing"""
    return xpretty(expr, use_unicode=True, wrap_line=False)


def test_anticommutator():
    A = Operator('A')
    B = Operator('B')
    ac = AntiCommutator(A, B)
    ac_tall = AntiCommutator(A**2, B)
    assert str(ac) == '{A,B}'
    assert pretty(ac) == '{A,B}'
    assert upretty(ac) == '{A,B}'
    assert latex(ac) == r'\left\{A,B\right\}'
    sT(ac, "AntiCommutator(Operator(Symbol('A')),Operator(Symbol('B')))")
    assert str(ac_tall) == '{A**2,B}'
    ascii_str = \
"""\
/ 2  \\\n\
<A ,B>\n\
\\    /\
"""
    ucode_str = \
"""\
⎧ 2  ⎫\n\
⎨A ,B⎬\n\
⎩    ⎭\
"""
    assert pretty(ac_tall) == ascii_str
    assert upretty(ac_tall) == ucode_str
    assert latex(ac_tall) == r'\left\{A^{2},B\right\}'
    sT(ac_tall, "AntiCommutator(Pow(Operator(Symbol('A')), Integer(2)),Operator(Symbol('B')))")


def test_cg():
    cg = CG(1, 2, 3, 4, 5, 6)
    wigner3j = Wigner3j(1, 2, 3, 4, 5, 6)
    wigner6j = Wigner6j(1, 2, 3, 4, 5, 6)
    wigner9j = Wigner9j(1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert str(cg) == 'CG(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
 5,6    \n\
C       \n\
 1,2,3,4\
"""
    ucode_str = \
"""\
 5,6    \n\
C       \n\
 1,2,3,4\
"""
    assert pretty(cg) == ascii_str
    assert upretty(cg) == ucode_str
    assert latex(cg) == 'C^{5,6}_{1,2,3,4}'
    assert latex(cg ** 2) == R'\left(C^{5,6}_{1,2,3,4}\right)^{2}'
    sT(cg, "CG(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(wigner3j) == 'Wigner3j(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
/1  3  5\\\n\
|       |\n\
\\2  4  6/\
"""
    ucode_str = \
"""\
⎛1  3  5⎞\n\
⎜       ⎟\n\
⎝2  4  6⎠\
"""
    assert pretty(wigner3j) == ascii_str
    assert upretty(wigner3j) == ucode_str
    assert latex(wigner3j) == \
        r'\left(\begin{array}{ccc} 1 & 3 & 5 \\ 2 & 4 & 6 \end{array}\right)'
    sT(wigner3j, "Wigner3j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(wigner6j) == 'Wigner6j(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
/1  2  3\\\n\
<       >\n\
\\4  5  6/\
"""
    ucode_str = \
"""\
⎧1  2  3⎫\n\
⎨       ⎬\n\
⎩4  5  6⎭\
"""
    assert pretty(wigner6j) == ascii_str
    assert upretty(wigner6j) == ucode_str
    assert latex(wigner6j) == \
        r'\left\{\begin{array}{ccc} 1 & 2 & 3 \\ 4 & 5 & 6 \end{array}\right\}'
    sT(wigner6j, "Wigner6j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(wigner9j) == 'Wigner9j(1, 2, 3, 4, 5, 6, 7, 8, 9)'
    ascii_str = \
"""\
/1  2  3\\\n\
|       |\n\
<4  5  6>\n\
|       |\n\
\\7  8  9/\
"""
    ucode_str = \
"""\
⎧1  2  3⎫\n\
⎪       ⎪\n\
⎨4  5  6⎬\n\
⎪       ⎪\n\
⎩7  8  9⎭\
"""
    assert pretty(wigner9j) == ascii_str
    assert upretty(wigner9j) == ucode_str
    assert latex(wigner9j) == \
        r'\left\{\begin{array}{ccc} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{array}\right\}'
    sT(wigner9j, "Wigner9j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6), Integer(7), Integer(8), Integer(9))")


def test_commutator():
    A = Operator('A')
    B = Operator('B')
    c = Commutator(A, B)
    c_tall = Commutator(A**2, B)
    assert str(c) == '[A,B]'
    assert pretty(c) == '[A,B]'
    assert upretty(c) == '[A,B]'
    assert latex(c) == r'\left[A,B\right]'
    sT(c, "Commutator(Operator(Symbol('A')),Operator(Symbol('B')))")
    assert str(c_tall) == '[A**2,B]'
    ascii_str = \
"""\
[ 2  ]\n\
[A ,B]\
"""
    ucode_str = \
"""\
⎡ 2  ⎤\n\
⎣A ,B⎦\
"""
    assert pretty(c_tall) == ascii_str
    assert upretty(c_tall) == ucode_str
    assert latex(c_tall) == r'\left[A^{2},B\right]'
    sT(c_tall, "Commutator(Pow(Operator(Symbol('A')), Integer(2)),Operator(Symbol('B')))")


def test_constants():
    assert str(hbar) == 'hbar'
    assert pretty(hbar) == 'hbar'
    assert upretty(hbar) == 'ℏ'
    assert latex(hbar) == r'\hbar'
    sT(hbar, "HBar()")


def test_dagger():
    x = symbols('x')
    expr = Dagger(x)
    assert str(expr) == 'Dagger(x)'
    ascii_str = \
"""\
 +\n\
x \
"""
    ucode_str = \
"""\
 †\n\
x \
"""
    assert pretty(expr) == ascii_str
    assert upretty(expr) == ucode_str
    assert latex(expr) == r'x^{\dagger}'
    sT(expr, "Dagger(Symbol('x'))")


@XFAIL
def test_gate_failing():
    a, b, c, d = symbols('a,b,c,d')
    uMat = Matrix([[a, b], [c, d]])
    g = UGate((0,), uMat)
    assert str(g) == 'U(0)'


def test_gate():
    a, b, c, d = symbols('a,b,c,d')
    uMat = Matrix([[a, b], [c, d]])
    q = Qubit(1, 0, 1, 0, 1)
    g1 = IdentityGate(2)
    g2 = CGate((3, 0), XGate(1))
    g3 = CNotGate(1, 0)
    g4 = UGate((0,), uMat)
    assert str(g1) == '1(2)'
    assert pretty(g1) == '1 \n 2'
    assert upretty(g1) == '1 \n 2'
    assert latex(g1) == r'1_{2}'
    sT(g1, "IdentityGate(Integer(2))")
    assert str(g1*q) == '1(2)*|10101>'
    ascii_str = \
"""\
1 *|10101>\n\
 2        \
"""
    ucode_str = \
"""\
1 ⋅❘10101⟩\n\
 2        \
"""
    assert pretty(g1*q) == ascii_str
    assert upretty(g1*q) == ucode_str
    assert latex(g1*q) == r'1_{2} {\left|10101\right\rangle }'
    sT(g1*q, "Mul(IdentityGate(Integer(2)), Qubit(Integer(1),Integer(0),Integer(1),Integer(0),Integer(1)))")
    assert str(g2) == 'C((3,0),X(1))'
    ascii_str = \
"""\
C   /X \\\n\
 3,0\\ 1/\
"""
    ucode_str = \
"""\
C   ⎛X ⎞\n\
 3,0⎝ 1⎠\
"""
    assert pretty(g2) == ascii_str
    assert upretty(g2) == ucode_str
    assert latex(g2) == r'C_{3,0}{\left(X_{1}\right)}'
    sT(g2, "CGate(Tuple(Integer(3), Integer(0)),XGate(Integer(1)))")
    assert str(g3) == 'CNOT(1,0)'
    ascii_str = \
"""\
CNOT   \n\
    1,0\
"""
    ucode_str = \
"""\
CNOT   \n\
    1,0\
"""
    assert pretty(g3) == ascii_str
    assert upretty(g3) == ucode_str
    assert latex(g3) == r'\text{CNOT}_{1,0}'
    sT(g3, "CNotGate(Integer(1),Integer(0))")
    ascii_str = \
"""\
U \n\
 0\
"""
    ucode_str = \
"""\
U \n\
 0\
"""
    assert str(g4) == \
"""\
U((0,),Matrix([\n\
[a, b],\n\
[c, d]]))\
"""
    assert pretty(g4) == ascii_str
    assert upretty(g4) == ucode_str
    assert latex(g4) == r'U_{0}'
    sT(g4, "UGate(Tuple(Integer(0)),ImmutableDenseMatrix([[Symbol('a'), Symbol('b')], [Symbol('c'), Symbol('d')]]))")


def test_hilbert():
    h1 = HilbertSpace()
    h2 = ComplexSpace(2)
    h3 = FockSpace()
    h4 = L2(Interval(0, oo))
    assert str(h1) == 'H'
    assert pretty(h1) == 'H'
    assert upretty(h1) == 'H'
    assert latex(h1) == r'\mathcal{H}'
    sT(h1, "HilbertSpace()")
    assert str(h2) == 'C(2)'
    ascii_str = \
"""\
 2\n\
C \
"""
    ucode_str = \
"""\
 2\n\
C \
"""
    assert pretty(h2) == ascii_str
    assert upretty(h2) == ucode_str
    assert latex(h2) == r'\mathcal{C}^{2}'
    sT(h2, "ComplexSpace(Integer(2))")
    assert str(h3) == 'F'
    assert pretty(h3) == 'F'
    assert upretty(h3) == 'F'
    assert latex(h3) == r'\mathcal{F}'
    sT(h3, "FockSpace()")
    assert str(h4) == 'L2(Interval(0, oo))'
    ascii_str = \
"""\
 2\n\
L \
"""
    ucode_str = \
"""\
 2\n\
L \
"""
    assert pretty(h4) == ascii_str
    assert upretty(h4) == ucode_str
    assert latex(h4) == r'{\mathcal{L}^2}\left( \left[0, \infty\right) \right)'
    sT(h4, "L2(Interval(Integer(0), oo, false, true))")
    assert str(h1 + h2) == 'H+C(2)'
    ascii_str = \
"""\
     2\n\
H + C \
"""
    ucode_str = \
"""\
     2\n\
H ⊕ C \
"""
    assert pretty(h1 + h2) == ascii_str
    assert upretty(h1 + h2) == ucode_str
    assert latex(h1 + h2)
    sT(h1 + h2, "DirectSumHilbertSpace(HilbertSpace(),ComplexSpace(Integer(2)))")
    assert str(h1*h2) == "H*C(2)"
    ascii_str = \
"""\
     2\n\
H x C \
"""
    ucode_str = \
"""\
     2\n\
H ⨂ C \
"""
    assert pretty(h1*h2) == ascii_str
    assert upretty(h1*h2) == ucode_str
    assert latex(h1*h2)
    sT(h1*h2,
       "TensorProductHilbertSpace(HilbertSpace(),ComplexSpace(Integer(2)))")
    assert str(h1**2) == 'H**2'
    ascii_str = \
"""\
 x2\n\
H  \
"""
    ucode_str = \
"""\
 ⨂2\n\
H  \
"""
    assert pretty(h1**2) == ascii_str
    assert upretty(h1**2) == ucode_str
    assert latex(h1**2) == r'{\mathcal{H}}^{\otimes 2}'
    sT(h1**2, "TensorPowerHilbertSpace(HilbertSpace(),Integer(2))")


def test_innerproduct():
    x = symbols('x')
    ip1 = InnerProduct(Bra(), Ket())
    ip2 = InnerProduct(TimeDepBra(), TimeDepKet())
    ip3 = InnerProduct(JzBra(1, 1), JzKet(1, 1))
    ip4 = InnerProduct(JzBraCoupled(1, 1, (1, 1)), JzKetCoupled(1, 1, (1, 1)))
    ip_tall1 = InnerProduct(Bra(x/2), Ket(x/2))
    ip_tall2 = InnerProduct(Bra(x), Ket(x/2))
    ip_tall3 = InnerProduct(Bra(x/2), Ket(x))
    assert str(ip1) == '<psi|psi>'
    assert pretty(ip1) == '<psi|psi>'
    assert upretty(ip1) == '⟨ψ❘ψ⟩'
    assert latex(
        ip1) == r'\left\langle \psi \right. {\left|\psi\right\rangle }'
    sT(ip1, "InnerProduct(Bra(Symbol('psi')),Ket(Symbol('psi')))")
    assert str(ip2) == '<psi;t|psi;t>'
    assert pretty(ip2) == '<psi;t|psi;t>'
    assert upretty(ip2) == '⟨ψ;t❘ψ;t⟩'
    assert latex(ip2) == \
        r'\left\langle \psi;t \right. {\left|\psi;t\right\rangle }'
    sT(ip2, "InnerProduct(TimeDepBra(Symbol('psi'),Symbol('t')),TimeDepKet(Symbol('psi'),Symbol('t')))")
    assert str(ip3) == "<1,1|1,1>"
    assert pretty(ip3) == '<1,1|1,1>'
    assert upretty(ip3) == '⟨1,1❘1,1⟩'
    assert latex(ip3) == r'\left\langle 1,1 \right. {\left|1,1\right\rangle }'
    sT(ip3, "InnerProduct(JzBra(Integer(1),Integer(1)),JzKet(Integer(1),Integer(1)))")
    assert str(ip4) == "<1,1,j1=1,j2=1|1,1,j1=1,j2=1>"
    assert pretty(ip4) == '<1,1,j1=1,j2=1|1,1,j1=1,j2=1>'
    assert upretty(ip4) == '⟨1,1,j₁=1,j₂=1❘1,1,j₁=1,j₂=1⟩'
    assert latex(ip4) == \
        r'\left\langle 1,1,j_{1}=1,j_{2}=1 \right. {\left|1,1,j_{1}=1,j_{2}=1\right\rangle }'
    sT(ip4, "InnerProduct(JzBraCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))),JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))))")
    assert str(ip_tall1) == '<x/2|x/2>'
    ascii_str = \
"""\
 / | \\ \n\
/ x|x \\\n\
\\ -|- /\n\
 \\2|2/ \
"""
    ucode_str = \
"""\
 ╱ │ ╲ \n\
╱ x│x ╲\n\
╲ ─│─ ╱\n\
 ╲2│2╱ \
"""
    assert pretty(ip_tall1) == ascii_str
    assert upretty(ip_tall1) == ucode_str
    assert latex(ip_tall1) == \
        r'\left\langle \frac{x}{2} \right. {\left|\frac{x}{2}\right\rangle }'
    sT(ip_tall1, "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Mul(Rational(1, 2), Symbol('x'))))")
    assert str(ip_tall2) == '<x|x/2>'
    ascii_str = \
"""\
 / | \\ \n\
/  |x \\\n\
\\ x|- /\n\
 \\ |2/ \
"""
    ucode_str = \
"""\
 ╱ │ ╲ \n\
╱  │x ╲\n\
╲ x│─ ╱\n\
 ╲ │2╱ \
"""
    assert pretty(ip_tall2) == ascii_str
    assert upretty(ip_tall2) == ucode_str
    assert latex(ip_tall2) == \
        r'\left\langle x \right. {\left|\frac{x}{2}\right\rangle }'
    sT(ip_tall2,
       "InnerProduct(Bra(Symbol('x')),Ket(Mul(Rational(1, 2), Symbol('x'))))")
    assert str(ip_tall3) == '<x/2|x>'
    ascii_str = \
"""\
 / | \\ \n\
/ x|  \\\n\
\\ -|x /\n\
 \\2| / \
"""
    ucode_str = \
"""\
 ╱ │ ╲ \n\
╱ x│  ╲\n\
╲ ─│x ╱\n\
 ╲2│ ╱ \
"""
    assert pretty(ip_tall3) == ascii_str
    assert upretty(ip_tall3) == ucode_str
    assert latex(ip_tall3) == \
        r'\left\langle \frac{x}{2} \right. {\left|x\right\rangle }'
    sT(ip_tall3,
       "InnerProduct(Bra(Mul(Rational(1, 2), Symbol('x'))),Ket(Symbol('x')))")


def test_operator():
    a = Operator('A')
    b = Operator('B', Symbol('t'), S.Half)
    inv = a.inv()
    f = Function('f')
    x = symbols('x')
    d = DifferentialOperator(Derivative(f(x), x), f(x))
    op = OuterProduct(Ket(), Bra())
    assert str(a) == 'A'
    assert pretty(a) == 'A'
    assert upretty(a) == 'A'
    assert latex(a) == 'A'
    sT(a, "Operator(Symbol('A'))")
    assert str(inv) == 'A**(-1)'
    ascii_str = \
"""\
 -1\n\
A  \
"""
    ucode_str = \
"""\
 -1\n\
A  \
"""
    assert pretty(inv) == ascii_str
    assert upretty(inv) == ucode_str
    assert latex(inv) == r'A^{-1}'
    sT(inv, "Pow(Operator(Symbol('A')), Integer(-1))")
    assert str(d) == 'DifferentialOperator(Derivative(f(x), x),f(x))'
    ascii_str = \
"""\
                    /d            \\\n\
DifferentialOperator|--(f(x)),f(x)|\n\
                    \\dx           /\
"""
    ucode_str = \
"""\
                    ⎛d            ⎞\n\
DifferentialOperator⎜──(f(x)),f(x)⎟\n\
                    ⎝dx           ⎠\
"""
    assert pretty(d) == ascii_str
    assert upretty(d) == ucode_str
    assert latex(d) == \
        r'DifferentialOperator\left(\frac{d}{d x} f{\left(x \right)},f{\left(x \right)}\right)'
    sT(d, "DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))")
    assert str(b) == 'Operator(B,t,1/2)'
    assert pretty(b) == 'Operator(B,t,1/2)'
    assert upretty(b) == 'Operator(B,t,1/2)'
    assert latex(b) == r'Operator\left(B,t,\frac{1}{2}\right)'
    sT(b, "Operator(Symbol('B'),Symbol('t'),Rational(1, 2))")
    assert str(op) == '|psi><psi|'
    assert pretty(op) == '|psi><psi|'
    assert upretty(op) == '❘ψ⟩⟨ψ❘'
    assert latex(op) == r'{\left|\psi\right\rangle }{\left\langle \psi\right|}'
    sT(op, "OuterProduct(Ket(Symbol('psi')),Bra(Symbol('psi')))")


def test_qexpr():
    q = QExpr('q')
    assert str(q) == 'q'
    assert pretty(q) == 'q'
    assert upretty(q) == 'q'
    assert latex(q) == r'q'
    sT(q, "QExpr(Symbol('q'))")


def test_qubit():
    q1 = Qubit('0101')
    q2 = IntQubit(8)
    assert str(q1) == '|0101>'
    assert pretty(q1) == '|0101>'
    assert upretty(q1) == '❘0101⟩'
    assert latex(q1) == r'{\left|0101\right\rangle }'
    sT(q1, "Qubit(Integer(0),Integer(1),Integer(0),Integer(1))")
    assert str(q2) == '|8>'
    assert pretty(q2) == '|8>'
    assert upretty(q2) == '❘8⟩'
    assert latex(q2) == r'{\left|8\right\rangle }'
    sT(q2, "IntQubit(8)")


def test_spin():
    lz = JzOp('L')
    ket = JzKet(1, 0)
    bra = JzBra(1, 0)
    cket = JzKetCoupled(1, 0, (1, 2))
    cbra = JzBraCoupled(1, 0, (1, 2))
    cket_big = JzKetCoupled(1, 0, (1, 2, 3))
    cbra_big = JzBraCoupled(1, 0, (1, 2, 3))
    rot = Rotation(1, 2, 3)
    bigd = WignerD(1, 2, 3, 4, 5, 6)
    smalld = WignerD(1, 2, 3, 0, 4, 0)
    assert str(lz) == 'Lz'
    ascii_str = \
"""\
L \n\
 z\
"""
    ucode_str = \
"""\
L \n\
 z\
"""
    assert pretty(lz) == ascii_str
    assert upretty(lz) == ucode_str
    assert latex(lz) == 'L_z'
    sT(lz, "JzOp(Symbol('L'))")
    assert str(J2) == 'J2'
    ascii_str = \
"""\
 2\n\
J \
"""
    ucode_str = \
"""\
 2\n\
J \
"""
    assert pretty(J2) == ascii_str
    assert upretty(J2) == ucode_str
    assert latex(J2) == r'J^2'
    sT(J2, "J2Op(Symbol('J'))")
    assert str(Jz) == 'Jz'
    ascii_str = \
"""\
J \n\
 z\
"""
    ucode_str = \
"""\
J \n\
 z\
"""
    assert pretty(Jz) == ascii_str
    assert upretty(Jz) == ucode_str
    assert latex(Jz) == 'J_z'
    sT(Jz, "JzOp(Symbol('J'))")
    assert str(ket) == '|1,0>'
    assert pretty(ket) == '|1,0>'
    assert upretty(ket) == '❘1,0⟩'
    assert latex(ket) == r'{\left|1,0\right\rangle }'
    sT(ket, "JzKet(Integer(1),Integer(0))")
    assert str(bra) == '<1,0|'
    assert pretty(bra) == '<1,0|'
    assert upretty(bra) == '⟨1,0❘'
    assert latex(bra) == r'{\left\langle 1,0\right|}'
    sT(bra, "JzBra(Integer(1),Integer(0))")
    assert str(cket) == '|1,0,j1=1,j2=2>'
    assert pretty(cket) == '|1,0,j1=1,j2=2>'
    assert upretty(cket) == '❘1,0,j₁=1,j₂=2⟩'
    assert latex(cket) == r'{\left|1,0,j_{1}=1,j_{2}=2\right\rangle }'
    sT(cket, "JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))")
    assert str(cbra) == '<1,0,j1=1,j2=2|'
    assert pretty(cbra) == '<1,0,j1=1,j2=2|'
    assert upretty(cbra) == '⟨1,0,j₁=1,j₂=2❘'
    assert latex(cbra) == r'{\left\langle 1,0,j_{1}=1,j_{2}=2\right|}'
    sT(cbra, "JzBraCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))")
    assert str(cket_big) == '|1,0,j1=1,j2=2,j3=3,j(1,2)=3>'
    # TODO: Fix non-unicode pretty printing
    # i.e. j1,2 -> j(1,2)
    assert pretty(cket_big) == '|1,0,j1=1,j2=2,j3=3,j1,2=3>'
    assert upretty(cket_big) == '❘1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3⟩'
    assert latex(cket_big) == \
        r'{\left|1,0,j_{1}=1,j_{2}=2,j_{3}=3,j_{1,2}=3\right\rangle }'
    sT(cket_big, "JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2), Integer(3)),Tuple(Tuple(Integer(1), Integer(2), Integer(3)), Tuple(Integer(1), Integer(3), Integer(1))))")
    assert str(cbra_big) == '<1,0,j1=1,j2=2,j3=3,j(1,2)=3|'
    assert pretty(cbra_big) == '<1,0,j1=1,j2=2,j3=3,j1,2=3|'
    assert upretty(cbra_big) == '⟨1,0,j₁=1,j₂=2,j₃=3,j₁,₂=3❘'
    assert latex(cbra_big) == \
        r'{\left\langle 1,0,j_{1}=1,j_{2}=2,j_{3}=3,j_{1,2}=3\right|}'
    sT(cbra_big, "JzBraCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(2), Integer(3)),Tuple(Tuple(Integer(1), Integer(2), Integer(3)), Tuple(Integer(1), Integer(3), Integer(1))))")
    assert str(rot) == 'R(1,2,3)'
    assert pretty(rot) == 'R (1,2,3)'
    assert upretty(rot) == 'ℛ (1,2,3)'
    assert latex(rot) == r'\mathcal{R}\left(1,2,3\right)'
    sT(rot, "Rotation(Integer(1),Integer(2),Integer(3))")
    assert str(bigd) == 'WignerD(1, 2, 3, 4, 5, 6)'
    ascii_str = \
"""\
 1         \n\
D   (4,5,6)\n\
 2,3       \
"""
    ucode_str = \
"""\
 1         \n\
D   (4,5,6)\n\
 2,3       \
"""
    assert pretty(bigd) == ascii_str
    assert upretty(bigd) == ucode_str
    assert latex(bigd) == r'D^{1}_{2,3}\left(4,5,6\right)'
    sT(bigd, "WignerD(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6))")
    assert str(smalld) == 'WignerD(1, 2, 3, 0, 4, 0)'
    ascii_str = \
"""\
 1     \n\
d   (4)\n\
 2,3   \
"""
    ucode_str = \
"""\
 1     \n\
d   (4)\n\
 2,3   \
"""
    assert pretty(smalld) == ascii_str
    assert upretty(smalld) == ucode_str
    assert latex(smalld) == r'd^{1}_{2,3}\left(4\right)'
    sT(smalld, "WignerD(Integer(1), Integer(2), Integer(3), Integer(0), Integer(4), Integer(0))")


def test_state():
    x = symbols('x')
    bra = Bra()
    ket = Ket()
    bra_tall = Bra(x/2)
    ket_tall = Ket(x/2)
    tbra = TimeDepBra()
    tket = TimeDepKet()
    assert str(bra) == '<psi|'
    assert pretty(bra) == '<psi|'
    assert upretty(bra) == '⟨ψ❘'
    assert latex(bra) == r'{\left\langle \psi\right|}'
    sT(bra, "Bra(Symbol('psi'))")
    assert str(ket) == '|psi>'
    assert pretty(ket) == '|psi>'
    assert upretty(ket) == '❘ψ⟩'
    assert latex(ket) == r'{\left|\psi\right\rangle }'
    sT(ket, "Ket(Symbol('psi'))")
    assert str(bra_tall) == '<x/2|'
    ascii_str = \
"""\
 / |\n\
/ x|\n\
\\ -|\n\
 \\2|\
"""
    ucode_str = \
"""\
 ╱ │\n\
╱ x│\n\
╲ ─│\n\
 ╲2│\
"""
    assert pretty(bra_tall) == ascii_str
    assert upretty(bra_tall) == ucode_str
    assert latex(bra_tall) == r'{\left\langle \frac{x}{2}\right|}'
    sT(bra_tall, "Bra(Mul(Rational(1, 2), Symbol('x')))")
    assert str(ket_tall) == '|x/2>'
    ascii_str = \
"""\
| \\ \n\
|x \\\n\
|- /\n\
|2/ \
"""
    ucode_str = \
"""\
│ ╲ \n\
│x ╲\n\
│─ ╱\n\
│2╱ \
"""
    assert pretty(ket_tall) == ascii_str
    assert upretty(ket_tall) == ucode_str
    assert latex(ket_tall) == r'{\left|\frac{x}{2}\right\rangle }'
    sT(ket_tall, "Ket(Mul(Rational(1, 2), Symbol('x')))")
    assert str(tbra) == '<psi;t|'
    assert pretty(tbra) == '<psi;t|'
    assert upretty(tbra) == '⟨ψ;t❘'
    assert latex(tbra) == r'{\left\langle \psi;t\right|}'
    sT(tbra, "TimeDepBra(Symbol('psi'),Symbol('t'))")
    assert str(tket) == '|psi;t>'
    assert pretty(tket) == '|psi;t>'
    assert upretty(tket) == '❘ψ;t⟩'
    assert latex(tket) == r'{\left|\psi;t\right\rangle }'
    sT(tket, "TimeDepKet(Symbol('psi'),Symbol('t'))")


def test_tensorproduct():
    tp = TensorProduct(JzKet(1, 1), JzKet(1, 0))
    assert str(tp) == '|1,1>x|1,0>'
    assert pretty(tp) == '|1,1>x |1,0>'
    assert upretty(tp) == '❘1,1⟩⨂ ❘1,0⟩'
    assert latex(tp) == \
        r'{{\left|1,1\right\rangle }}\otimes {{\left|1,0\right\rangle }}'
    sT(tp, "TensorProduct(JzKet(Integer(1),Integer(1)), JzKet(Integer(1),Integer(0)))")


def test_big_expr():
    f = Function('f')
    x = symbols('x')
    e1 = Dagger(AntiCommutator(Operator('A') + Operator('B'), Pow(DifferentialOperator(Derivative(f(x), x), f(x)), 3))*TensorProduct(Jz**2, Operator('A') + Operator('B')))*(JzBra(1, 0) + JzBra(1, 1))*(JzKet(0, 0) + JzKet(1, -1))
    e2 = Commutator(Jz**2, Operator('A') + Operator('B'))*AntiCommutator(Dagger(Operator('C')*Operator('D')), Operator('E').inv()**2)*Dagger(Commutator(Jz, J2))
    e3 = Wigner3j(1, 2, 3, 4, 5, 6)*TensorProduct(Commutator(Operator('A') + Dagger(Operator('B')), Operator('C') + Operator('D')), Jz - J2)*Dagger(OuterProduct(Dagger(JzBra(1, 1)), JzBra(1, 0)))*TensorProduct(JzKetCoupled(1, 1, (1, 1)) + JzKetCoupled(1, 0, (1, 1)), JzKetCoupled(1, -1, (1, 1)))
    e4 = (ComplexSpace(1)*ComplexSpace(2) + FockSpace()**2)*(L2(Interval(
        0, oo)) + HilbertSpace())
    assert str(e1) == '(Jz**2)x(Dagger(A) + Dagger(B))*{Dagger(DifferentialOperator(Derivative(f(x), x),f(x)))**3,Dagger(A) + Dagger(B)}*(<1,0| + <1,1|)*(|0,0> + |1,-1>)'
    ascii_str = \
"""\
                 /                                      3        \\                                 \n\
                 |/                                   +\\         |                                 \n\
    2  / +    +\\ <|                    /d            \\ |   +    +>                                 \n\
/J \\ x \\A  + B /*||DifferentialOperator|--(f(x)),f(x)| | ,A  + B |*(<1,0| + <1,1|)*(|0,0> + |1,-1>)\n\
\\ z/             \\\\                    \\dx           / /         /                                 \
"""
    ucode_str = \
"""\
                 ⎧                                      3        ⎫                                 \n\
                 ⎪⎛                                   †⎞         ⎪                                 \n\
    2  ⎛ †    †⎞ ⎨⎜                    ⎛d            ⎞ ⎟   †    †⎬                                 \n\
⎛J ⎞ ⨂ ⎝A  + B ⎠⋅⎪⎜DifferentialOperator⎜──(f(x)),f(x)⎟ ⎟ ,A  + B ⎪⋅(⟨1,0❘ + ⟨1,1❘)⋅(❘0,0⟩ + ❘1,-1⟩)\n\
⎝ z⎠             ⎩⎝                    ⎝dx           ⎠ ⎠         ⎭                                 \
"""
    assert pretty(e1) == ascii_str
    assert upretty(e1) == ucode_str
    assert latex(e1) == \
        r'{J_z^{2}}\otimes \left({A^{\dagger} + B^{\dagger}}\right) \left\{\left(DifferentialOperator\left(\frac{d}{d x} f{\left(x \right)},f{\left(x \right)}\right)^{\dagger}\right)^{3},A^{\dagger} + B^{\dagger}\right\} \left({\left\langle 1,0\right|} + {\left\langle 1,1\right|}\right) \left({\left|0,0\right\rangle } + {\left|1,-1\right\rangle }\right)'
    sT(e1, "Mul(TensorProduct(Pow(JzOp(Symbol('J')), Integer(2)), Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), AntiCommutator(Pow(Dagger(DifferentialOperator(Derivative(Function('f')(Symbol('x')), Tuple(Symbol('x'), Integer(1))),Function('f')(Symbol('x')))), Integer(3)),Add(Dagger(Operator(Symbol('A'))), Dagger(Operator(Symbol('B'))))), Add(JzBra(Integer(1),Integer(0)), JzBra(Integer(1),Integer(1))), Add(JzKet(Integer(0),Integer(0)), JzKet(Integer(1),Integer(-1))))")
    assert str(e2) == '[Jz**2,A + B]*{E**(-2),Dagger(D)*Dagger(C)}*[J2,Jz]'
    ascii_str = \
"""\
[    2      ] / -2  +  +\\ [ 2   ]\n\
[/J \\ ,A + B]*<E  ,D *C >*[J ,J ]\n\
[\\ z/       ] \\         / [    z]\
"""
    ucode_str = \
"""\
⎡    2      ⎤ ⎧ -2  †  †⎫ ⎡ 2   ⎤\n\
⎢⎛J ⎞ ,A + B⎥⋅⎨E  ,D ⋅C ⎬⋅⎢J ,J ⎥\n\
⎣⎝ z⎠       ⎦ ⎩         ⎭ ⎣    z⎦\
"""
    assert pretty(e2) == ascii_str
    assert upretty(e2) == ucode_str
    assert latex(e2) == \
        r'\left[J_z^{2},A + B\right] \left\{E^{-2},D^{\dagger} C^{\dagger}\right\} \left[J^2,J_z\right]'
    sT(e2, "Mul(Commutator(Pow(JzOp(Symbol('J')), Integer(2)),Add(Operator(Symbol('A')), Operator(Symbol('B')))), AntiCommutator(Pow(Operator(Symbol('E')), Integer(-2)),Mul(Dagger(Operator(Symbol('D'))), Dagger(Operator(Symbol('C'))))), Commutator(J2Op(Symbol('J')),JzOp(Symbol('J'))))")
    assert str(e3) == \
        "Wigner3j(1, 2, 3, 4, 5, 6)*[Dagger(B) + A,C + D]x(-J2 + Jz)*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x|1,-1,j1=1,j2=1>"
    ascii_str = \
"""\
          [ +          ]  /   2     \\                                                                 \n\
/1  3  5\\*[B  + A,C + D]x |- J  + J |*|1,0><1,1|*(|1,0,j1=1,j2=1> + |1,1,j1=1,j2=1>)x |1,-1,j1=1,j2=1>\n\
|       |                 \\        z/                                                                 \n\
\\2  4  6/                                                                                             \
"""
    ucode_str = \
"""\
          ⎡ †          ⎤  ⎛   2     ⎞                                                                 \n\
⎛1  3  5⎞⋅⎣B  + A,C + D⎦⨂ ⎜- J  + J ⎟⋅❘1,0⟩⟨1,1❘⋅(❘1,0,j₁=1,j₂=1⟩ + ❘1,1,j₁=1,j₂=1⟩)⨂ ❘1,-1,j₁=1,j₂=1⟩\n\
⎜       ⎟                 ⎝        z⎠                                                                 \n\
⎝2  4  6⎠                                                                                             \
"""
    assert pretty(e3) == ascii_str
    assert upretty(e3) == ucode_str
    assert latex(e3) == \
        r'\left(\begin{array}{ccc} 1 & 3 & 5 \\ 2 & 4 & 6 \end{array}\right) {\left[B^{\dagger} + A,C + D\right]}\otimes \left({- J^2 + J_z}\right) {\left|1,0\right\rangle }{\left\langle 1,1\right|} \left({{\left|1,0,j_{1}=1,j_{2}=1\right\rangle } + {\left|1,1,j_{1}=1,j_{2}=1\right\rangle }}\right)\otimes {{\left|1,-1,j_{1}=1,j_{2}=1\right\rangle }}'
    sT(e3, "Mul(Wigner3j(Integer(1), Integer(2), Integer(3), Integer(4), Integer(5), Integer(6)), TensorProduct(Commutator(Add(Dagger(Operator(Symbol('B'))), Operator(Symbol('A'))),Add(Operator(Symbol('C')), Operator(Symbol('D')))), Add(Mul(Integer(-1), J2Op(Symbol('J'))), JzOp(Symbol('J')))), OuterProduct(JzKet(Integer(1),Integer(0)),JzBra(Integer(1),Integer(1))), TensorProduct(Add(JzKetCoupled(Integer(1),Integer(0),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1)))), JzKetCoupled(Integer(1),Integer(1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))), JzKetCoupled(Integer(1),Integer(-1),Tuple(Integer(1), Integer(1)),Tuple(Tuple(Integer(1), Integer(2), Integer(1))))))")
    assert str(e4) == '(C(1)*C(2)+F**2)*(L2(Interval(0, oo))+H)'
    ascii_str = \
"""\
// 1    2\\    x2\\   / 2    \\\n\
\\\\C  x C / + F  / x \\L  + H/\
"""
    ucode_str = \
"""\
⎛⎛ 1    2⎞    ⨂2⎞   ⎛ 2    ⎞\n\
⎝⎝C  ⨂ C ⎠ ⊕ F  ⎠ ⨂ ⎝L  ⊕ H⎠\
"""
    assert pretty(e4) == ascii_str
    assert upretty(e4) == ucode_str
    assert latex(e4) == \
        r'\left(\left(\mathcal{C}^{1}\otimes \mathcal{C}^{2}\right)\oplus {\mathcal{F}}^{\otimes 2}\right)\otimes \left({\mathcal{L}^2}\left( \left[0, \infty\right) \right)\oplus \mathcal{H}\right)'
    sT(e4, "TensorProductHilbertSpace((DirectSumHilbertSpace(TensorProductHilbertSpace(ComplexSpace(Integer(1)),ComplexSpace(Integer(2))),TensorPowerHilbertSpace(FockSpace(),Integer(2)))),(DirectSumHilbertSpace(L2(Interval(Integer(0), oo, false, true)),HilbertSpace())))")


def _test_sho1d():
    ad = RaisingOp('a')
    assert pretty(ad) == ' \N{DAGGER}\na '
    assert latex(ad) == 'a^{\\dagger}'
