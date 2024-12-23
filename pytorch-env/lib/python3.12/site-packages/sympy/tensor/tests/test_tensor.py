from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
    get_symmetric_group_sgs, TensorIndex, tensor_mul, TensAdd, \
    riemann_cyclic_replace, riemann_cyclic, TensMul, tensor_heads, \
    TensorManager, TensExpr, TensorHead, canon_bp, \
    tensorhead, tensorsymmetry, TensorType, substitute_indices, \
    WildTensorIndex, WildTensorHead, _WildTensExpr
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag

def _is_equal(arg1, arg2):
    if isinstance(arg1, TensExpr):
        return arg1.equals(arg2)
    elif isinstance(arg2, TensExpr):
        return arg2.equals(arg1)
    return arg1 == arg2


#################### Tests from tensor_can.py #######################
def test_canonicalize_no_slot_sym():
    # A_d0 * B^d0; T_c = A^d0*B_d0
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, d0, d1 = tensor_indices('a,b,d0,d1', Lorentz)
    A, B = tensor_heads('A,B', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(-d0)*B(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*B(-L_0)'

    # A^a * B^b;  T_c = T
    t = A(a)*B(b)
    tc = t.canon_bp()
    assert tc == t
    # B^b * A^a
    t1 = B(b)*A(a)
    tc = t1.canon_bp()
    assert str(tc) == 'A(a)*B(b)'

    # A symmetric
    # A^{b}_{d0}*A^{d0, a}; T_c = A^{a d0}*A{b}_{d0}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(b, -d0)*A(d0, a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0)*A(b, -L_0)'

    # A^{d1}_{d0}*B^d0*C_d1
    # T_c = A^{d0 d1}*B_d0*C_d1
    B, C = tensor_heads('B,C', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(d1, -d0)*B(d0)*C(-d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_0)*C(-L_1)'

    # A without symmetry
    # A^{d1}_{d0}*B^d0*C_d1 ord=[d0,-d0,d1,-d1]; g = [2,1,0,3,4,5]
    # T_c = A^{d0 d1}*B_d1*C_d0; can = [0,2,3,1,4,5]
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0)*B(d0)*C(-d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_1)*C(-L_0)'

    # A, B without symmetry
    # A^{d1}_{d0}*B_{d1}^{d0}
    # T_c = A^{d0 d1}*B_{d0 d1}
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0)*B(-d1, d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_0, -L_1)'
    # A_{d0}^{d1}*B_{d1}^{d0}
    # T_c = A^{d0 d1}*B_{d1 d0}
    t = A(-d0, d1)*B(-d1, d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-L_1, -L_0)'

    # A, B, C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b}
    # T_c=A^{d0 d1}*B_{a d1}*C_{d0 b}
    C = TensorHead('C', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_1)*C(-L_0, -b)'

    # A symmetric, B and C without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b}
    # T_c = A^{d0 d1}*B_{a d0}*C_{d1 b}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-L_1, -b)'

    # A and C symmetric, B without symmetry
    # A^{d1 d0}*B_{a d0}*C_{d1 b} ord=[a,b,d0,-d0,d1,-d1]
    # T_c = A^{d0 d1}*B_{a d0}*C_{b d1}
    C = TensorHead('C', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d1, d0)*B(-a, -d0)*C(-d1, -b)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1)*B(-a, -L_0)*C(-b, -L_1)'

def test_canonicalize_no_dummies():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a, b, c, d', Lorentz)

    # A commuting
    # A^c A^b A^a
    # T_c = A^a A^b A^c
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(c)*A(b)*A(a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a)*A(b)*A(c)'

    # A anticommuting
    # A^c A^b A^a
    # T_c = -A^a A^b A^c
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    t = A(c)*A(b)*A(a)
    tc = t.canon_bp()
    assert str(tc) == '-A(a)*A(b)*A(c)'

    # A commuting and symmetric
    # A^{b,d}*A^{c,a}
    # T_c = A^{a c}*A^{b d}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(b, d)*A(c, a)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, c)*A(b, d)'

    # A anticommuting and symmetric
    # A^{b,d}*A^{c,a}
    # T_c = -A^{a c}*A^{b d}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2), 1)
    t = A(b, d)*A(c, a)
    tc = t.canon_bp()
    assert str(tc) == '-A(a, c)*A(b, d)'

    # A^{c,a}*A^{b,d}
    # T_c = A^{a c}*A^{b d}
    t = A(c, a)*A(b, d)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, c)*A(b, d)'

def test_tensorhead_construction_without_symmetry():
    L = TensorIndexType('Lorentz')
    A1 = TensorHead('A', [L, L])
    A2 = TensorHead('A', [L, L], TensorSymmetry.no_symmetry(2))
    assert A1 == A2
    A3 = TensorHead('A', [L, L], TensorSymmetry.fully_symmetric(2))  # Symmetric
    assert A1 != A3

def test_no_metric_symmetry():
    # no metric symmetry; A no symmetry
    # A^d1_d0 * A^d0_d1
    # T_c = A^d0_d1 * A^d1_d0
    Lorentz = TensorIndexType('Lorentz', dummy_name='L', metric_symmetry=0)
    d0, d1, d2, d3 = tensor_indices('d:4', Lorentz)
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
    t = A(d1, -d0)*A(d0, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)'

    # A^d1_d2 * A^d0_d3 * A^d2_d1 * A^d3_d0
    # T_c = A^d0_d1 * A^d1_d0 * A^d2_d3 * A^d3_d2
    t = A(d1, -d2)*A(d0, -d3)*A(d2, -d1)*A(d3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_0)*A(L_2, -L_3)*A(L_3, -L_2)'

    # A^d0_d2 * A^d1_d3 * A^d3_d0 * A^d2_d1
    # T_c = A^d0_d1 * A^d1_d2 * A^d2_d3 * A^d3_d0
    t = A(d0, -d1)*A(d1, -d2)*A(d2, -d3)*A(d3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, -L_1)*A(L_1, -L_2)*A(L_2, -L_3)*A(L_3, -L_0)'

def test_canonicalize1():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Lorentz)

    # A_d0*A^d0; ord = [d0,-d0]
    # T_c = A^d0*A_d0
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1))
    t = A(-d0)*A(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*A(-L_0)'

    # A commuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0
    # T_c = A^d0*A_d0*A^d1*A_d1*A^d2*A_d2
    t = A(-d0)*A(-d1)*A(-d2)*A(d2)*A(d1)*A(d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0)*A(-L_0)*A(L_1)*A(-L_1)*A(L_2)*A(-L_2)'

    # A anticommuting
    # A_d0*A_d1*A_d2*A^d2*A^d1*A^d0
    # T_c 0
    A = TensorHead('A', [Lorentz], TensorSymmetry.no_symmetry(1), 1)
    t = A(-d0)*A(-d1)*A(-d2)*A(d2)*A(d1)*A(d0)
    tc = t.canon_bp()
    assert tc == 0

    # A commuting symmetric
    # A^{d0 b}*A^a_d1*A^d1_d0
    # T_c = A^{a d0}*A^{b d1}*A_{d0 d1}
    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d0, b)*A(a, -d1)*A(d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0)*A(b, L_1)*A(-L_0, -L_1)'

    # A, B commuting symmetric
    # A^{d0 b}*A^d1_d0*B^a_d1
    # T_c = A^{b d0}*A_d0^d1*B^a_d1
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(d0, b)*A(d1, -d0)*B(a, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'A(b, L_0)*A(-L_0, L_1)*B(a, -L_1)'

    # A commuting symmetric
    # A^{d1 d0 b}*A^{a}_{d1 d0}; ord=[a,b, d0,-d0,d1,-d1]
    # T_c = A^{a d0 d1}*A^{b}_{d0 d1}
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3))
    t = A(d1, d0, b)*A(a, -d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a, L_0, L_1)*A(b, -L_0, -L_1)'

    # A^{d3 d0 d2}*A^a0_{d1 d2}*A^d1_d3^a1*A^{a2 a3}_d0
    # T_c = A^{a0 d0 d1}*A^a1_d0^d2*A^{a2 a3 d3}*A_{d1 d2 d3}
    t = A(d3, d0, d2)*A(a0, -d1, -d2)*A(d1, -d3, a1)*A(a2, a3, -d0)
    tc = t.canon_bp()
    assert str(tc) == 'A(a0, L_0, L_1)*A(a1, -L_0, L_2)*A(a2, a3, L_3)*A(-L_1, -L_2, -L_3)'

    # A commuting symmetric, B antisymmetric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # in this esxample and in the next three,
    # renaming dummy indices and using symmetry of A,
    # T = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    # can = 0
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3))
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert tc == 0

    # A anticommuting symmetric, B antisymmetric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == 'A(L_0, L_1, L_2)*A(-L_0, -L_1, L_3)*B(-L_2, -L_3)'

    # A anticommuting symmetric, B antisymmetric commuting, antisymmetric metric
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = -A^{d0 d1 d2} * A_{d0 d1}^d3 * B_{d2 d3}
    Spinor = TensorIndexType('Spinor', dummy_name='S', metric_symmetry=-1)
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Spinor)
    A = TensorHead('A', [Spinor]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Spinor]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == '-A(S_0, S_1, S_2)*A(-S_0, -S_1, S_3)*B(-S_2, -S_3)'

    # A anticommuting symmetric, B antisymmetric anticommuting,
    # no metric symmetry
    # A^{d0 d1 d2} * A_{d2 d3 d1} * B_d0^d3
    # T_c = A^{d0 d1 d2} * A_{d0 d1 d3} * B_d2^d3
    Mat = TensorIndexType('Mat', metric_symmetry=0, dummy_name='M')
    a, a0, a1, a2, a3, b, d0, d1, d2, d3 = \
        tensor_indices('a,a0,a1,a2,a3,b,d0,d1,d2,d3', Mat)
    A = TensorHead('A', [Mat]*3, TensorSymmetry.fully_symmetric(3), 1)
    B = TensorHead('B', [Mat]*2, TensorSymmetry.fully_symmetric(-2))
    t = A(d0, d1, d2)*A(-d2, -d3, -d1)*B(-d0, d3)
    tc = t.canon_bp()
    assert str(tc) == 'A(M_0, M_1, M_2)*A(-M_0, -M_1, -M_3)*B(-M_2, M_3)'

    # Gamma anticommuting
    # Gamma_{mu nu} * gamma^rho * Gamma^{nu mu alpha}
    # T_c = -Gamma^{mu nu} * gamma^rho * Gamma_{alpha mu nu}
    alpha, beta, gamma, mu, nu, rho = \
        tensor_indices('alpha,beta,gamma,mu,nu,rho', Lorentz)
    Gamma = TensorHead('Gamma', [Lorentz],
                       TensorSymmetry.fully_symmetric(1), 2)
    Gamma2 = TensorHead('Gamma', [Lorentz]*2,
                        TensorSymmetry.fully_symmetric(-2), 2)
    Gamma3 = TensorHead('Gamma', [Lorentz]*3,
                        TensorSymmetry.fully_symmetric(-3), 2)
    t = Gamma2(-mu, -nu)*Gamma(rho)*Gamma3(nu, mu, alpha)
    tc = t.canon_bp()
    assert str(tc) == '-Gamma(L_0, L_1)*Gamma(rho)*Gamma(alpha, -L_0, -L_1)'

    # Gamma_{mu nu} * Gamma^{gamma beta} * gamma_rho * Gamma^{nu mu alpha}
    # T_c = Gamma^{mu nu} * Gamma^{beta gamma} * gamma_rho * Gamma^alpha_{mu nu}
    t = Gamma2(mu, nu)*Gamma2(beta, gamma)*Gamma(-rho)*Gamma3(alpha, -mu, -nu)
    tc = t.canon_bp()
    assert str(tc) == 'Gamma(L_0, L_1)*Gamma(beta, gamma)*Gamma(-rho)*Gamma(alpha, -L_0, -L_1)'

    # f^a_{b,c} antisymmetric in b,c; A_mu^a no symmetry
    # f^c_{d a} * f_{c e b} * A_mu^d * A_nu^a * A^{nu e} * A^{mu b}
    # g = [8,11,5, 9,13,7, 1,10, 3,4, 2,12, 0,6, 14,15]
    # T_c = -f^{a b c} * f_a^{d e} * A^mu_b * A_{mu d} * A^nu_c * A_{nu e}
    Flavor = TensorIndexType('Flavor', dummy_name='F')
    a, b, c, d, e, ff = tensor_indices('a,b,c,d,e,f', Flavor)
    mu, nu = tensor_indices('mu,nu', Lorentz)
    f = TensorHead('f', [Flavor]*3, TensorSymmetry.direct_product(1, -2))
    A = TensorHead('A', [Lorentz, Flavor], TensorSymmetry.no_symmetry(2))
    t = f(c, -d, -a)*f(-c, -e, -b)*A(-mu, d)*A(-nu, a)*A(nu, e)*A(mu, b)
    tc = t.canon_bp()
    assert str(tc) == '-f(F_0, F_1, F_2)*f(-F_0, F_3, F_4)*A(L_0, -F_1)*A(-L_0, -F_3)*A(L_1, -F_2)*A(-L_1, -F_4)'


def test_bug_correction_tensor_indices():
    # to make sure that tensor_indices does not return a list if creating
    # only one index:
    A = TensorIndexType("A")
    i = tensor_indices('i', A)
    assert not isinstance(i, (tuple, list))
    assert isinstance(i, TensorIndex)


def test_riemann_invariants():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11 = \
        tensor_indices('d0:12', Lorentz)
    # R^{d0 d1}_{d1 d0}; ord = [d0,-d0,d1,-d1]
    # T_c = -R^{d0 d1}_{d0 d1}
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(d0, d1, -d1, -d0)
    tc = t.canon_bp()
    assert str(tc) == '-R(L_0, L_1, -L_0, -L_1)'

    # R_d11^d1_d0^d5 * R^{d6 d4 d0}_d5 * R_{d7 d2 d8 d9} *
    # R_{d10 d3 d6 d4} * R^{d2 d7 d11}_d1 * R^{d8 d9 d3 d10}
    # can = [0,2,4,6, 1,3,8,10, 5,7,12,14, 9,11,16,18, 13,15,20,22,
    #        17,19,21<F10,23, 24,25]
    # T_c = R^{d0 d1 d2 d3} * R_{d0 d1}^{d4 d5} * R_{d2 d3}^{d6 d7} *
    # R_{d4 d5}^{d8 d9} * R_{d6 d7}^{d10 d11} * R_{d8 d9 d10 d11}

    t = R(-d11,d1,-d0,d5)*R(d6,d4,d0,-d5)*R(-d7,-d2,-d8,-d9)* \
        R(-d10,-d3,-d6,-d4)*R(d2,d7,d11,-d1)*R(d8,d9,d3,d10)
    tc = t.canon_bp()
    assert str(tc) == 'R(L_0, L_1, L_2, L_3)*R(-L_0, -L_1, L_4, L_5)*R(-L_2, -L_3, L_6, L_7)*R(-L_4, -L_5, L_8, L_9)*R(-L_6, -L_7, L_10, L_11)*R(-L_8, -L_9, -L_10, -L_11)'

def test_riemann_products():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    d0, d1, d2, d3, d4, d5, d6 = tensor_indices('d0:7', Lorentz)
    a0, a1, a2, a3, a4, a5 = tensor_indices('a0:6', Lorentz)
    a, b = tensor_indices('a,b', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    # R^{a b d0}_d0 = 0
    t = R(a, b, d0, -d0)
    tc = t.canon_bp()
    assert tc == 0

    # R^{d0 b a}_d0
    # T_c = -R^{a d0 b}_d0
    t = R(d0, b, a, -d0)
    tc = t.canon_bp()
    assert str(tc) == '-R(a, L_0, b, -L_0)'

    # R^d1_d2^b_d0 * R^{d0 a}_d1^d2; ord=[a,b,d0,-d0,d1,-d1,d2,-d2]
    # T_c = -R^{a d0 d1 d2}* R^b_{d0 d1 d2}
    t = R(d1, -d2, b, -d0)*R(d0, a, -d1, d2)
    tc = t.canon_bp()
    assert str(tc) == '-R(a, L_0, L_1, L_2)*R(b, -L_0, -L_1, -L_2)'

    # A symmetric commuting
    # R^{d6 d5}_d2^d1 * R^{d4 d0 d2 d3} * A_{d6 d0} A_{d3 d1} * A_{d4 d5}
    # g = [12,10,5,2, 8,0,4,6, 13,1, 7,3, 9,11,14,15]
    # T_c = -R^{d0 d1 d2 d3} * R_d0^{d4 d5 d6} * A_{d1 d4}*A_{d2 d5}*A_{d3 d6}
    V = TensorHead('V', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = R(d6, d5, -d2, d1)*R(d4, d0, d2, d3)*V(-d6, -d0)*V(-d3, -d1)*V(-d4, -d5)
    tc = t.canon_bp()
    assert str(tc) == '-R(L_0, L_1, L_2, L_3)*R(-L_0, L_4, L_5, L_6)*V(-L_1, -L_4)*V(-L_2, -L_5)*V(-L_3, -L_6)'

    # R^{d2 a0 a2 d0} * R^d1_d2^{a1 a3} * R^{a4 a5}_{d0 d1}
    # T_c = R^{a0 d0 a2 d1}*R^{a1 a3}_d0^d2*R^{a4 a5}_{d1 d2}
    t = R(d2, a0, a2, d0)*R(d1, -d2, a1, a3)*R(a4, a5, -d0, -d1)
    tc = t.canon_bp()
    assert str(tc) == 'R(a0, L_0, a2, L_1)*R(a1, a3, -L_0, L_2)*R(a4, a5, -L_1, -L_2)'
######################################################################


def test_canonicalize2():
    D = Symbol('D')
    Eucl = TensorIndexType('Eucl', metric_symmetry=1, dim=D, dummy_name='E')
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14 = \
        tensor_indices('i0:15', Eucl)
    A = TensorHead('A', [Eucl]*3, TensorSymmetry.fully_symmetric(-3))

    # two examples from Cvitanovic, Group Theory page 59
    # of identities for antisymmetric tensors of rank 3
    # contracted according to the Kuratowski graph  eq.(6.59)
    t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i3,i7,i5)*A(-i2,-i5,i6)*A(-i4,-i6,i8)
    t1 = t.canon_bp()
    assert t1 == 0

    # eq.(6.60)
    #t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i2,i5,i6)*A(-i3,i7,i8)*A(-i6,-i7,i9)*
    #    A(-i8,i10,i13)*A(-i5,-i10,i11)*A(-i4,-i11,i12)*A(-i3,-i12,i14)
    t = A(i0,i1,i2)*A(-i1,i3,i4)*A(-i2,i5,i6)*A(-i3,i7,i8)*A(-i6,-i7,i9)*\
        A(-i8,i10,i13)*A(-i5,-i10,i11)*A(-i4,-i11,i12)*A(-i9,-i12,i14)
    t1 = t.canon_bp()
    assert t1 == 0

def test_canonicalize3():
    D = Symbol('D')
    Spinor = TensorIndexType('Spinor', dim=D, metric_symmetry=-1, dummy_name='S')
    a0,a1,a2,a3,a4 = tensor_indices('a0:5', Spinor)
    chi, psi = tensor_heads('chi,psi', [Spinor], TensorSymmetry.no_symmetry(1), 1)

    t = chi(a1)*psi(a0)
    t1 = t.canon_bp()
    assert t1 == t

    t = psi(a1)*chi(a0)
    t1 = t.canon_bp()
    assert t1 == -chi(a0)*psi(a1)


def test_TensorIndexType():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', metric_name='g', metric_symmetry=1,
                              dim=D, dummy_name='L')
    m0, m1, m2, m3, m4 = tensor_indices('m0:5', Lorentz)
    sym2 = TensorSymmetry.fully_symmetric(2)
    sym2n = TensorSymmetry(*get_symmetric_group_sgs(2))
    assert sym2 == sym2n
    g = Lorentz.metric
    assert str(g) == 'g(Lorentz,Lorentz)'
    assert Lorentz.eps_dim == Lorentz.dim

    TSpace = TensorIndexType('TSpace', dummy_name = 'TSpace')
    i0, i1 = tensor_indices('i0 i1', TSpace)
    g = TSpace.metric
    A = TensorHead('A', [TSpace]*2, sym2)
    assert str(A(i0,-i0).canon_bp()) == 'A(TSpace_0, -TSpace_0)'

def test_indices():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    assert a.tensor_index_type == Lorentz
    assert a != -a
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(a,b)*B(-b,c)
    indices = t.get_indices()
    L_0 = TensorIndex('L_0', Lorentz)
    assert indices == [a, L_0, -L_0, c]
    raises(ValueError, lambda: tensor_indices(3, Lorentz))
    raises(ValueError, lambda: A(a,b,c))

    A = TensorHead('A', [Lorentz, Lorentz])
    assert A('a', 'b') == A(TensorIndex('a', Lorentz),
                            TensorIndex('b', Lorentz))
    assert A('a', '-b') == A(TensorIndex('a', Lorentz),
                             TensorIndex('b', Lorentz, is_up=False))
    assert A('a', TensorIndex('b', Lorentz)) == A(TensorIndex('a', Lorentz),
                                                  TensorIndex('b', Lorentz))

def test_TensorSymmetry():
    assert TensorSymmetry.fully_symmetric(2) == \
        TensorSymmetry(get_symmetric_group_sgs(2))
    assert TensorSymmetry.fully_symmetric(-3) == \
        TensorSymmetry(get_symmetric_group_sgs(3, True))
    assert TensorSymmetry.direct_product(-4) == \
        TensorSymmetry.fully_symmetric(-4)
    assert TensorSymmetry.fully_symmetric(-1) == \
        TensorSymmetry.fully_symmetric(1)
    assert TensorSymmetry.direct_product(1, -1, 1) == \
        TensorSymmetry.no_symmetry(3)
    assert TensorSymmetry(get_symmetric_group_sgs(2)) == \
        TensorSymmetry(*get_symmetric_group_sgs(2))
    # TODO: add check for *get_symmetric_group_sgs(0)
    sym = TensorSymmetry.fully_symmetric(-3)
    assert sym.rank == 3
    assert sym.base == Tuple(0, 1)
    assert sym.generators == Tuple(Permutation(0, 1)(3, 4), Permutation(1, 2)(3, 4))

def test_TensExpr():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    g = Lorentz.metric
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    raises(ValueError, lambda: g(c, d)/g(a, b))
    raises(ValueError, lambda: S.One/g(a, b))
    raises(ValueError, lambda: (A(c, d) + g(c, d))/g(a, b))
    raises(ValueError, lambda: S.One/(A(c, d) + g(c, d)))
    raises(ValueError, lambda: A(a, b) + A(a, c))

    #t = A(a, b) + B(a, b) # assigned to t for below
    #raises(NotImplementedError, lambda: TensExpr.__mul__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__add__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__radd__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__sub__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__rsub__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__truediv__(t, 'a'))
    #raises(NotImplementedError, lambda: TensExpr.__rtruediv__(t, 'a'))
    with warns_deprecated_sympy():
        # DO NOT REMOVE THIS AFTER DEPRECATION REMOVED:
        raises(ValueError, lambda: A(a, b)**2)
    raises(NotImplementedError, lambda: 2**A(a, b))
    raises(NotImplementedError, lambda: abs(A(a, b)))

def test_TensorHead():
    # simple example of algebraic expression
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    A = TensorHead('A', [Lorentz]*2)
    assert A.name == 'A'
    assert A.index_types == [Lorentz, Lorentz]
    assert A.rank == 2
    assert A.symmetry == TensorSymmetry.no_symmetry(2)
    assert A.comm == 0


def test_add1():
    assert TensAdd().args == ()
    assert TensAdd().doit() == 0
    # simple example of algebraic expression
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a,b,d0,d1,i,j,k = tensor_indices('a,b,d0,d1,i,j,k', Lorentz)
    # A, B symmetric
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t1 = A(b, -d0)*B(d0, a)
    assert TensAdd(t1).equals(t1)
    t2a = B(d0, a) + A(d0, a)
    t2 = A(b, -d0)*t2a
    assert str(t2) == 'A(b, -L_0)*(A(L_0, a) + B(L_0, a))'
    t2 = t2.expand()
    assert str(t2) == 'A(b, -L_0)*A(L_0, a) + A(b, -L_0)*B(L_0, a)'
    t2 = t2.canon_bp()
    assert str(t2) == 'A(a, L_0)*A(b, -L_0) + A(b, L_0)*B(a, -L_0)'
    t2b = t2 + t1
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + A(b, -L_0)*B(L_0, a) + A(b, L_0)*B(a, -L_0)'
    t2b = t2b.canon_bp()
    assert str(t2b) == 'A(a, L_0)*A(b, -L_0) + 2*A(b, L_0)*B(a, -L_0)'
    p, q, r = tensor_heads('p,q,r', [Lorentz])
    t = q(d0)*2
    assert str(t) == '2*q(d0)'
    t = 2*q(d0)
    assert str(t) == '2*q(d0)'
    t1 = p(d0) + 2*q(d0)
    assert str(t1) == '2*q(d0) + p(d0)'
    t2 = p(-d0) + 2*q(-d0)
    assert str(t2) == '2*q(-d0) + p(-d0)'
    t1 = p(d0)
    t3 = t1*t2
    assert str(t3) == 'p(L_0)*(2*q(-L_0) + p(-L_0))'
    t3 = t3.expand()
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    t3 = t2*t1
    t3 = t3.expand()
    assert str(t3) == 'p(-L_0)*p(L_0) + 2*q(-L_0)*p(L_0)'
    t3 = t3.canon_bp()
    assert str(t3) == 'p(L_0)*p(-L_0) + 2*p(L_0)*q(-L_0)'
    t1 = p(d0) + 2*q(d0)
    t3 = t1*t2
    t3 = t3.canon_bp()
    assert str(t3) == 'p(L_0)*p(-L_0) + 4*p(L_0)*q(-L_0) + 4*q(L_0)*q(-L_0)'
    t1 = p(d0) - 2*q(d0)
    assert str(t1) == '-2*q(d0) + p(d0)'
    t2 = p(-d0) + 2*q(-d0)
    t3 = t1*t2
    t3 = t3.canon_bp()
    assert t3 == p(d0)*p(-d0) - 4*q(d0)*q(-d0)
    t = p(i)*p(j)*(p(k) + q(k)) + p(i)*(p(j) + q(j))*(p(k) - 3*q(k))
    t = t.canon_bp()
    assert t == 2*p(i)*p(j)*p(k) - 2*p(i)*p(j)*q(k) + p(i)*p(k)*q(j) - 3*p(i)*q(j)*q(k)
    t1 = (p(i) + q(i) + 2*r(i))*(p(j) - q(j))
    t2 = (p(j) + q(j) + 2*r(j))*(p(i) - q(i))
    t = t1 + t2
    t = t.canon_bp()
    assert t == 2*p(i)*p(j) + 2*p(i)*r(j) + 2*p(j)*r(i) - 2*q(i)*q(j) - 2*q(i)*r(j) - 2*q(j)*r(i)
    t = p(i)*q(j)/2
    assert 2*t == p(i)*q(j)
    t = (p(i) + q(i))/2
    assert 2*t == p(i) + q(i)

    t = S.One - p(i)*p(-i)
    t = t.canon_bp()
    tz1 = t + p(-j)*p(j)
    assert tz1 != 1
    tz1 = tz1.canon_bp()
    assert tz1.equals(1)
    t = S.One + p(i)*p(-i)
    assert (t - p(-j)*p(j)).canon_bp().equals(1)

    t = A(a, b) + B(a, b)
    assert t.rank == 2
    t1 = t - A(a, b) - B(a, b)
    assert t1 == 0
    t = 1 - (A(a, -a) + B(a, -a))
    t1 = 1 + (A(a, -a) + B(a, -a))
    assert (t + t1).expand().equals(2)
    t2 = 1 + A(a, -a)
    assert t1 != t2
    assert t2 != TensMul.from_data(0, [], [], [])

    #Test whether TensAdd.doit chokes on subterms that are zero.
    assert TensAdd(p(a), TensMul(0, p(a)) ).doit() == p(a)

def test_special_eq_ne():
    # test special equality cases:
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, d0, d1, i, j, k = tensor_indices('a,b,d0,d1,i,j,k', Lorentz)
    # A, B symmetric
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    p, q, r = tensor_heads('p,q,r', [Lorentz])

    t = 0*A(a, b)
    assert _is_equal(t, 0)
    assert _is_equal(t, S.Zero)

    assert p(i) != A(a, b)
    assert A(a, -a) != A(a, b)
    assert 0*(A(a, b) + B(a, b)) == 0
    assert 0*(A(a, b) + B(a, b)) is S.Zero

    assert 3*(A(a, b) - A(a, b)) is S.Zero

    assert p(i) + q(i) != A(a, b)
    assert p(i) + q(i) != A(a, b) + B(a, b)

    assert p(i) - p(i) == 0
    assert p(i) - p(i) is S.Zero

    assert _is_equal(A(a, b), A(b, a))

def test_add2():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m, n, p, q = tensor_indices('m,n,p,q', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    A = TensorHead('A', [Lorentz]*3, TensorSymmetry.fully_symmetric(-3))
    t1 = 2*R(m, n, p, q) - R(m, q, n, p) + R(m, p, n, q)
    t2 = t1*A(-n, -p, -q)
    t2 = t2.canon_bp()
    assert t2 == 0
    t1 = Rational(2, 3)*R(m,n,p,q) - Rational(1, 3)*R(m,q,n,p) + Rational(1, 3)*R(m,p,n,q)
    t2 = t1*A(-n, -p, -q)
    t2 = t2.canon_bp()
    assert t2 == 0
    t = A(m, -m, n) + A(n, p, -p)
    t = t.canon_bp()
    assert t == 0

def test_add3():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i0, i1 = tensor_indices('i0:2', Lorentz)
    E, px, py, pz = symbols('E px py pz')
    A = TensorHead('A', [Lorentz])
    B = TensorHead('B', [Lorentz])

    expr1 = A(i0)*A(-i0) - (E**2 - px**2 - py**2 - pz**2)
    assert expr1.args == (-E**2, px**2, py**2, pz**2, A(i0)*A(-i0))

    expr2 = E**2 - px**2 - py**2 - pz**2 - A(i0)*A(-i0)
    assert expr2.args == (E**2, -px**2, -py**2, -pz**2, -A(i0)*A(-i0))

    expr3 = A(i0)*A(-i0) - E**2 + px**2 + py**2 + pz**2
    assert expr3.args == (-E**2, px**2, py**2, pz**2, A(i0)*A(-i0))

    expr4 = B(i1)*B(-i1) + 2*E**2 - 2*px**2 - 2*py**2 - 2*pz**2 - A(i0)*A(-i0)
    assert expr4.args == (2*E**2, -2*px**2, -2*py**2, -2*pz**2, B(i1)*B(-i1), -A(i0)*A(-i0))


def test_mul():
    from sympy.abc import x
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    t = TensMul.from_data(S.One, [], [], [])
    assert str(t) == '1'
    A, B = tensor_heads('A B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = (1 + x)*A(a, b)
    assert str(t) == '(x + 1)*A(a, b)'
    assert t.index_types == [Lorentz, Lorentz]
    assert t.rank == 2
    assert t.dum == []
    assert t.coeff == 1 + x
    assert sorted(t.free) == [(a, 0), (b, 1)]
    assert t.components == [A]

    ts = A(a, b)
    assert str(ts) == 'A(a, b)'
    assert ts.index_types == [Lorentz, Lorentz]
    assert ts.rank == 2
    assert ts.dum == []
    assert ts.coeff == 1
    assert sorted(ts.free) == [(a, 0), (b, 1)]
    assert ts.components == [A]

    t = A(-b, a)*B(-a, c)*A(-c, d)
    t1 = tensor_mul(*t.split())
    assert t == t1
    assert tensor_mul(*[]) == TensMul.from_data(S.One, [], [], [])

    t = TensMul.from_data(1, [], [], [])
    C = TensorHead('C', [])
    assert str(C()) == 'C'
    assert str(t) == '1'
    assert t == 1
    raises(ValueError, lambda: A(a, b)*A(a, c))

def test_substitute_indices():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))

    p = TensorHead('p', [Lorentz])
    t = p(i)
    t1 = t.substitute_indices((j, k))
    assert t1 == t
    t1 = t.substitute_indices((i, j))
    assert t1 == p(j)
    t1 = t.substitute_indices((i, -j))
    assert t1 == p(-j)
    t1 = t.substitute_indices((-i, j))
    assert t1 == p(-j)
    t1 = t.substitute_indices((-i, -j))
    assert t1 == p(j)
    t = A(m, n)
    t1 = t.substitute_indices((m, i), (n, -i))
    assert t1 == A(n, -n)
    t1 = substitute_indices(t, (m, i), (n, -i))
    assert t1 == A(n, -n)

    t = A(i, k)*B(-k, -j)
    t1 = t.substitute_indices((i, j), (j, k))
    t1a = A(j, l)*B(-l, -k)
    assert t1 == t1a
    t1 = substitute_indices(t, (i, j), (j, k))
    assert t1 == t1a

    t = A(i, j) + B(i, j)
    t1 = t.substitute_indices((j, -i))
    t1a = A(i, -i) + B(i, -i)
    assert t1 == t1a
    t1 = substitute_indices(t, (j, -i))
    assert t1 == t1a

def test_riemann_cyclic_replace():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m0, m1, m2, m3 = tensor_indices('m:4', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(m0, m2, m1, m3)
    t1 = riemann_cyclic_replace(t)
    t1a = Rational(-1, 3)*R(m0, m3, m2, m1) + Rational(1, 3)*R(m0, m1, m2, m3) + Rational(2, 3)*R(m0, m2, m1, m3)
    assert t1 == t1a

def test_riemann_cyclic():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(i,j,k,l) + R(i,l,j,k) + R(i,k,l,j) - \
        R(i,j,l,k) - R(i,l,k,j) - R(i,k,j,l)
    t2 = t*R(-i,-j,-k,-l)
    t3 = riemann_cyclic(t2)
    assert t3 == 0
    t = R(i,j,k,l)*(R(-i,-j,-k,-l) - 2*R(-i,-k,-j,-l))
    t1 = riemann_cyclic(t)
    assert t1 == 0
    t = R(i,j,k,l)
    t1 = riemann_cyclic(t)
    assert t1 == Rational(-1, 3)*R(i, l, j, k) + Rational(1, 3)*R(i, k, j, l) + Rational(2, 3)*R(i, j, k, l)

    t = R(i,j,k,l)*R(-k,-l,m,n)*(R(-m,-n,-i,-j) + 2*R(-m,-j,-n,-i))
    t1 = riemann_cyclic(t)
    assert t1 == 0

@XFAIL
def test_div():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m0, m1, m2, m3 = tensor_indices('m0:4', Lorentz)
    R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    t = R(m0,m1,-m1,m3)
    t1 = t/S(4)
    assert str(t1) == '(1/4)*R(m0, L_0, -L_0, m3)'
    t = t.canon_bp()
    assert not t1._is_canon_bp
    t1 = t*4
    assert t1._is_canon_bp
    t1 = t1/4
    assert t1._is_canon_bp

def test_contract_metric1():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    g = Lorentz.metric
    p = TensorHead('p', [Lorentz])
    t = g(a, b)*p(-b)
    t1 = t.contract_metric(g)
    assert t1 == p(a)
    A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))

    # case with g with all free indices
    t1 = A(a,b)*B(-b,c)*g(d, e)
    t2 = t1.contract_metric(g)
    assert t1 == t2

    # case of g(d, -d)
    t1 = A(a,b)*B(-b,c)*g(-d, d)
    t2 = t1.contract_metric(g)
    assert t2 == D*A(a, d)*B(-d, c)

    # g with one free index
    t1 = A(a,b)*B(-b,-c)*g(c, d)
    t2 = t1.contract_metric(g)
    assert t2 == A(a, c)*B(-c, d)

    # g with both indices contracted with another tensor
    t1 = A(a,b)*B(-b,-c)*g(c, -a)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a, b)*B(-b, -a))

    t1 = A(a,b)*B(-b,-c)*g(c, d)*g(-a, -d)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a,b)*B(-b,-a))

    t1 = A(a,b)*g(-a,-b)
    t2 = t1.contract_metric(g)
    assert _is_equal(t2, A(a, -a))
    assert not t2.free
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    a, b = tensor_indices('a,b', Lorentz)
    g = Lorentz.metric
    assert _is_equal(g(a, -a).contract_metric(g), Lorentz.dim) # no dim

def test_contract_metric2():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    a, b, c, d, e, L_0 = tensor_indices('a,b,c,d,e,L_0', Lorentz)
    g = Lorentz.metric
    p, q = tensor_heads('p,q', [Lorentz])

    t1 = g(a,b)*p(c)*p(-c)
    t2 = 3*g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    assert t == 3*D*p(a)*p(-a)*q(b)*q(-b)
    t1 = g(a,b)*p(c)*p(-c)
    t2 = 3*q(-a)*q(-b)
    t = t1*t2
    t = t.contract_metric(g)
    t = t.canon_bp()
    assert t == 3*p(a)*p(-a)*q(b)*q(-b)

    t1 = 2*g(a,b)*p(c)*p(-c)
    t2 = - 3*g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    t = 6*g(a,b)*g(-a,-b)*p(c)*p(-c)*q(d)*q(-d)
    t = t.contract_metric(g)

    t1 = 2*g(a,b)*p(c)*p(-c)
    t2 = q(-a)*q(-b) + 3*g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    assert t == (2 + 6*D)*p(a)*p(-a)*q(b)*q(-b)

    t1 = p(a)*p(b) + p(a)*q(b) + 2*g(a,b)*p(c)*p(-c)
    t2 = q(-a)*q(-b) - g(-a,-b)*q(c)*q(-c)
    t = t1*t2
    t = t.contract_metric(g)
    t1 = (1 - 2*D)*p(a)*p(-a)*q(b)*q(-b) + p(a)*q(-a)*p(b)*q(-b)
    assert canon_bp(t - t1) == 0

    t = g(a,b)*g(c,d)*g(-b,-c)
    t1 = t.contract_metric(g)
    assert t1 == g(a, d)

    t1 = g(a,b)*g(c,d) + g(a,c)*g(b,d) + g(a,d)*g(b,c)
    t2 = t1.substitute_indices((a,-a),(b,-b),(c,-c),(d,-d))
    t = t1*t2
    t = t.contract_metric(g)
    assert t.equals(3*D**2 + 6*D)

    t = 2*p(a)*g(b,-b)
    t1 = t.contract_metric(g)
    assert t1.equals(2*D*p(a))

    t = 2*p(a)*g(b,-a)
    t1 = t.contract_metric(g)
    assert t1 == 2*p(b)

    M = Symbol('M')
    t = (p(a)*p(b) + g(a, b)*M**2)*g(-a, -b) - D*M**2
    t1 = t.contract_metric(g)
    assert t1 == p(a)*p(-a)

    A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
    t = A(a, b)*p(L_0)*g(-a, -b)
    t1 = t.contract_metric(g)
    assert str(t1) == 'A(L_1, -L_1)*p(L_0)' or str(t1) == 'A(-L_1, L_1)*p(L_0)'

def test_metric_contract3():
    D = Symbol('D')
    Spinor = TensorIndexType('Spinor', dim=D, metric_symmetry=-1, dummy_name='S')
    a0, a1, a2, a3, a4 = tensor_indices('a0:5', Spinor)
    C = Spinor.metric
    chi, psi = tensor_heads('chi,psi', [Spinor], TensorSymmetry.no_symmetry(1), 1)
    B = TensorHead('B', [Spinor]*2, TensorSymmetry.no_symmetry(2))

    t = C(a0,-a0)
    t1 = t.contract_metric(C)
    assert t1.equals(-D)

    t = C(-a0,a0)
    t1 = t.contract_metric(C)
    assert t1.equals(D)

    t = C(a0,a1)*C(-a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(D)

    t = C(a1,a0)*C(-a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(-D)

    t = C(-a0,a1)*C(a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(-D)

    t = C(a1,-a0)*C(a0,-a1)
    t1 = t.contract_metric(C)
    assert t1.equals(D)

    t = C(a0,a1)*B(-a1,-a0)
    t1 = t.contract_metric(C)
    t1 = t1.canon_bp()
    assert _is_equal(t1, B(a0,-a0))

    t = C(a1,a0)*B(-a1,-a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(a0,-a0))

    t = C(a0,-a1)*B(a1,-a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(a0,-a0))

    t = C(-a0,a1)*B(-a1,a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(a0,-a0))

    t = C(-a0,-a1)*B(a1,a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, B(a0,-a0))

    t = C(-a1, a0)*B(a1,-a0)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, B(a0,-a0))

    t = C(a0,a1)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, psi(a0))

    t = C(a1,a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -psi(a0))

    t = C(a0,a1)*chi(-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -chi(a1)*psi(-a1))

    t = C(a1,a0)*chi(-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, chi(a1)*psi(-a1))

    t = C(-a1,a0)*chi(-a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, chi(-a1)*psi(a1))

    t = C(a0,-a1)*chi(-a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -chi(-a1)*psi(a1))

    t = C(-a0,-a1)*chi(a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, chi(-a1)*psi(a1))

    t = C(-a1,-a0)*chi(a0)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -chi(-a1)*psi(a1))

    t = C(-a1,-a0)*B(a0,a2)*psi(a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, -B(-a1,a2)*psi(a1))

    t = C(a1,a0)*B(-a2,-a0)*psi(-a1)
    t1 = t.contract_metric(C)
    assert _is_equal(t1, B(-a2,a1)*psi(-a1))


def test_contract_metric4():
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices("p q r", R3)
    delta = R3.delta
    eps = R3.epsilon
    K = TensorHead("K", [R3])

    #Check whether contract_metric chokes on an expandable expression which becomes zero on canonicalization (issue #24354)
    expr = eps(p,q,r)*( K(-p)*K(-q) + delta(-p,-q) )
    assert expr.contract_metric(delta) == 0


def test_epsilon():
    Lorentz = TensorIndexType('Lorentz', dim=4, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    epsilon = Lorentz.epsilon
    p, q, r, s = tensor_heads('p,q,r,s', [Lorentz])

    t = epsilon(b,a,c,d)
    t1 = t.canon_bp()
    assert t1 == -epsilon(a,b,c,d)

    t = epsilon(c,b,d,a)
    t1 = t.canon_bp()
    assert t1 == epsilon(a,b,c,d)

    t = epsilon(c,a,d,b)
    t1 = t.canon_bp()
    assert t1 == -epsilon(a,b,c,d)

    t = epsilon(a,b,c,d)*p(-a)*q(-b)
    t1 = t.canon_bp()
    assert t1 == epsilon(c,d,a,b)*p(-a)*q(-b)

    t = epsilon(c,b,d,a)*p(-a)*q(-b)
    t1 = t.canon_bp()
    assert t1 == epsilon(c,d,a,b)*p(-a)*q(-b)

    t = epsilon(c,a,d,b)*p(-a)*q(-b)
    t1 = t.canon_bp()
    assert t1 == -epsilon(c,d,a,b)*p(-a)*q(-b)

    t = epsilon(c,a,d,b)*p(-a)*p(-b)
    t1 = t.canon_bp()
    assert t1 == 0

    t = epsilon(c,a,d,b)*p(-a)*q(-b) + epsilon(a,b,c,d)*p(-b)*q(-a)
    t1 = t.canon_bp()
    assert t1 == -2*epsilon(c,d,a,b)*p(-a)*q(-b)

    # Test that epsilon can be create with a SymPy integer:
    Lorentz = TensorIndexType('Lorentz', dim=Integer(4), dummy_name='L')
    epsilon = Lorentz.epsilon
    assert isinstance(epsilon, TensorHead)

def test_contract_delta1():
    # see Group Theory by Cvitanovic page 9
    n = Symbol('n')
    Color = TensorIndexType('Color', dim=n, dummy_name='C')
    a, b, c, d, e, f = tensor_indices('a,b,c,d,e,f', Color)
    delta = Color.delta

    def idn(a, b, d, c):
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a,c)*delta(d,b)

    def T(a, b, d, c):
        assert a.is_up and d.is_up
        assert not (b.is_up or c.is_up)
        return delta(a,b)*delta(d,c)

    def P1(a, b, c, d):
        return idn(a,b,c,d) - 1/n*T(a,b,c,d)

    def P2(a, b, c, d):
        return 1/n*T(a,b,c,d)

    t = P1(a, -b, e, -f)*P1(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert canon_bp(t1 - P1(a, -b, d, -c)) == 0

    t = P2(a, -b, e, -f)*P2(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert t1 == P2(a, -b, d, -c)

    t = P1(a, -b, e, -f)*P2(f, -e, d, -c)
    t1 = t.contract_delta(delta)
    assert t1 == 0

    t = P1(a, -b, b, -a)
    t1 = t.contract_delta(delta)
    assert t1.equals(n**2 - 1)

def test_fun():
    with warns_deprecated_sympy():
        D = Symbol('D')
        Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
        a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
        g = Lorentz.metric

        p, q = tensor_heads('p q', [Lorentz])
        t = q(c)*p(a)*q(b) + g(a,b)*g(c,d)*q(-d)
        assert t(a,b,c) == t
        assert canon_bp(t - t(b,a,c) - q(c)*p(a)*q(b) + q(c)*p(b)*q(a)) == 0
        assert t(b,c,d) == q(d)*p(b)*q(c) + g(b,c)*g(d,e)*q(-e)
        t1 = t.substitute_indices((a,b),(b,a))
        assert canon_bp(t1 - q(c)*p(b)*q(a) - g(a,b)*g(c,d)*q(-d)) == 0

        # check that g_{a b; c} = 0
        # example taken from  L. Brewin
        # "A brief introduction to Cadabra" arxiv:0903.2085
        # dg_{a b c} = \partial_{a} g_{b c} is symmetric in b, c
        dg = TensorHead('dg', [Lorentz]*3, TensorSymmetry.direct_product(1, 2))
        # gamma^a_{b c} is the Christoffel symbol
        gamma = S.Half*g(a,d)*(dg(-b,-d,-c) + dg(-c,-b,-d) - dg(-d,-b,-c))
        # t = g_{a b; c}
        t = dg(-c,-a,-b) - g(-a,-d)*gamma(d,-b,-c) - g(-b,-d)*gamma(d,-a,-c)
        t = t.contract_metric(g)
        assert t == 0
        t = q(c)*p(a)*q(b)
        assert t(b,c,d) == q(d)*p(b)*q(c)

def test_TensorManager():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    LorentzH = TensorIndexType('LorentzH', dummy_name='LH')
    i, j = tensor_indices('i,j', Lorentz)
    ih, jh = tensor_indices('ih,jh', LorentzH)
    p, q = tensor_heads('p q', [Lorentz])
    ph, qh = tensor_heads('ph qh', [LorentzH])

    Gsymbol = Symbol('Gsymbol')
    GHsymbol = Symbol('GHsymbol')
    TensorManager.set_comm(Gsymbol, GHsymbol, 0)
    G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), Gsymbol)
    assert TensorManager._comm_i2symbol[G.comm] == Gsymbol
    GH = TensorHead('GH', [LorentzH], TensorSymmetry.no_symmetry(1), GHsymbol)
    ps = G(i)*p(-i)
    psh = GH(ih)*ph(-ih)
    t = ps + psh
    t1 = t*t
    assert canon_bp(t1 - ps*ps - 2*ps*psh - psh*psh) == 0
    qs = G(i)*q(-i)
    qsh = GH(ih)*qh(-ih)
    assert _is_equal(ps*qsh, qsh*ps)
    assert not _is_equal(ps*qs, qs*ps)
    n = TensorManager.comm_symbols2i(Gsymbol)
    assert TensorManager.comm_i2symbol(n) == Gsymbol

    assert GHsymbol in TensorManager._comm_symbols2i
    raises(ValueError, lambda: TensorManager.set_comm(GHsymbol, 1, 2))
    TensorManager.set_comms((Gsymbol,GHsymbol,0),(Gsymbol,1,1))
    assert TensorManager.get_comm(n, 1) == TensorManager.get_comm(1, n) == 1
    TensorManager.clear()
    assert TensorManager.comm == [{0:0, 1:0, 2:0}, {0:0, 1:1, 2:None}, {0:0, 1:None}]
    assert GHsymbol not in TensorManager._comm_symbols2i
    nh = TensorManager.comm_symbols2i(GHsymbol)
    assert TensorManager.comm_i2symbol(nh) == GHsymbol
    assert GHsymbol in TensorManager._comm_symbols2i


def test_hash():
    D = Symbol('D')
    Lorentz = TensorIndexType('Lorentz', dim=D, dummy_name='L')
    a, b, c, d, e = tensor_indices('a,b,c,d,e', Lorentz)
    g = Lorentz.metric

    p, q = tensor_heads('p q', [Lorentz])
    p_type = p.args[1]
    t1 = p(a)*q(b)
    t2 = p(a)*p(b)
    assert hash(t1) != hash(t2)
    t3 = p(a)*p(b) + g(a,b)
    t4 = p(a)*p(b) - g(a,b)
    assert hash(t3) != hash(t4)

    assert a.func(*a.args) == a
    assert Lorentz.func(*Lorentz.args) == Lorentz
    assert g.func(*g.args) == g
    assert p.func(*p.args) == p
    assert p_type.func(*p_type.args) == p_type
    assert p(a).func(*(p(a)).args) == p(a)
    assert t1.func(*t1.args) == t1
    assert t2.func(*t2.args) == t2
    assert t3.func(*t3.args) == t3
    assert t4.func(*t4.args) == t4

    assert hash(a.func(*a.args)) == hash(a)
    assert hash(Lorentz.func(*Lorentz.args)) == hash(Lorentz)
    assert hash(g.func(*g.args)) == hash(g)
    assert hash(p.func(*p.args)) == hash(p)
    assert hash(p_type.func(*p_type.args)) == hash(p_type)
    assert hash(p(a).func(*(p(a)).args)) == hash(p(a))
    assert hash(t1.func(*t1.args)) == hash(t1)
    assert hash(t2.func(*t2.args)) == hash(t2)
    assert hash(t3.func(*t3.args)) == hash(t3)
    assert hash(t4.func(*t4.args)) == hash(t4)

    def check_all(obj):
        return all(isinstance(_, Basic) for _ in obj.args)

    assert check_all(a)
    assert check_all(Lorentz)
    assert check_all(g)
    assert check_all(p)
    assert check_all(p_type)
    assert check_all(p(a))
    assert check_all(t1)
    assert check_all(t2)
    assert check_all(t3)
    assert check_all(t4)

    tsymmetry = TensorSymmetry.direct_product(-2, 1, 3)

    assert tsymmetry.func(*tsymmetry.args) == tsymmetry
    assert hash(tsymmetry.func(*tsymmetry.args)) == hash(tsymmetry)
    assert check_all(tsymmetry)


### TEST VALUED TENSORS ###


def _get_valued_base_test_variables():
    minkowski = Matrix((
        (1, 0, 0, 0),
        (0, -1, 0, 0),
        (0, 0, -1, 0),
        (0, 0, 0, -1),
    ))
    Lorentz = TensorIndexType('Lorentz', dim=4)
    Lorentz.data = minkowski

    i0, i1, i2, i3, i4 = tensor_indices('i0:5', Lorentz)

    E, px, py, pz = symbols('E px py pz')
    A = TensorHead('A', [Lorentz])
    A.data = [E, px, py, pz]
    B = TensorHead('B', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
    B.data = range(4)
    AB = TensorHead("AB", [Lorentz]*2)
    AB.data = minkowski

    ba_matrix = Matrix((
        (1, 2, 3, 4),
        (5, 6, 7, 8),
        (9, 0, -1, -2),
        (-3, -4, -5, -6),
    ))

    BA = TensorHead("BA", [Lorentz]*2)
    BA.data = ba_matrix

    # Let's test the diagonal metric, with inverted Minkowski metric:
    LorentzD = TensorIndexType('LorentzD')
    LorentzD.data = [-1, 1, 1, 1]
    mu0, mu1, mu2 = tensor_indices('mu0:3', LorentzD)
    C = TensorHead('C', [LorentzD])
    C.data = [E, px, py, pz]

    ### non-diagonal metric ###
    ndm_matrix = (
        (1, 1, 0,),
        (1, 0, 1),
        (0, 1, 0,),
    )
    ndm = TensorIndexType("ndm")
    ndm.data = ndm_matrix
    n0, n1, n2 = tensor_indices('n0:3', ndm)
    NA = TensorHead('NA', [ndm])
    NA.data = range(10, 13)
    NB = TensorHead('NB', [ndm]*2)
    NB.data = [[i+j for j in range(10, 13)] for i in range(10, 13)]
    NC = TensorHead('NC', [ndm]*3)
    NC.data = [[[i+j+k for k in range(4, 7)] for j in range(1, 4)] for i in range(2, 5)]

    return (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
            n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4)


def test_valued_tensor_iter():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        list_BA = [Array([1, 2, 3, 4]), Array([5, 6, 7, 8]), Array([9, 0, -1, -2]), Array([-3, -4, -5, -6])]
        # iteration on VTensorHead
        assert list(A) == [E, px, py, pz]
        assert list(ba_matrix) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6]
        assert list(BA) == list_BA

        # iteration on VTensMul
        assert list(A(i1)) == [E, px, py, pz]
        assert list(BA(i1, i2)) == list_BA
        assert list(3 * BA(i1, i2)) == [3 * i for i in list_BA]
        assert list(-5 * BA(i1, i2)) == [-5 * i for i in list_BA]

        # iteration on VTensAdd
        # A(i1) + A(i1)
        assert list(A(i1) + A(i1)) == [2*E, 2*px, 2*py, 2*pz]
        assert BA(i1, i2) - BA(i1, i2) == 0
        assert list(BA(i1, i2) - 2 * BA(i1, i2)) == [-i for i in list_BA]


def test_valued_tensor_covariant_contravariant_elements():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert A(-i0)[0] == A(i0)[0]
        assert A(-i0)[1] == -A(i0)[1]

        assert AB(i0, i1)[1, 1] == -1
        assert AB(i0, -i1)[1, 1] == 1
        assert AB(-i0, -i1)[1, 1] == -1
        assert AB(-i0, i1)[1, 1] == 1


def test_valued_tensor_get_matrix():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        matab = AB(i0, i1).get_matrix()
        assert matab == Matrix([
                                [1,  0,  0,  0],
                                [0, -1,  0,  0],
                                [0,  0, -1,  0],
                                [0,  0,  0, -1],
                                ])
        # when alternating contravariant/covariant with [1, -1, -1, -1] metric
        # it becomes the identity matrix:
        assert AB(i0, -i1).get_matrix() == eye(4)

        # covariant and contravariant forms:
        assert A(i0).get_matrix() == Matrix([E, px, py, pz])
        assert A(-i0).get_matrix() == Matrix([E, -px, -py, -pz])

def test_valued_tensor_contraction():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert (A(i0) * A(-i0)).data == E ** 2 - px ** 2 - py ** 2 - pz ** 2
        assert (A(i0) * A(-i0)).data == A ** 2
        assert (A(i0) * A(-i0)).data == A(i0) ** 2
        assert (A(i0) * B(-i0)).data == -px - 2 * py - 3 * pz

        for i in range(4):
            for j in range(4):
                assert (A(i0) * B(-i1))[i, j] == [E, px, py, pz][i] * [0, -1, -2, -3][j]

        # test contraction on the alternative Minkowski metric: [-1, 1, 1, 1]
        assert (C(mu0) * C(-mu0)).data == -E ** 2 + px ** 2 + py ** 2 + pz ** 2

        contrexp = A(i0) * AB(i1, -i0)
        assert A(i0).rank == 1
        assert AB(i1, -i0).rank == 2
        assert contrexp.rank == 1
        for i in range(4):
            assert contrexp[i] == [E, px, py, pz][i]

def test_valued_tensor_self_contraction():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert AB(i0, -i0).data == 4
        assert BA(i0, -i0).data == 2


def test_valued_tensor_pow():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        assert C**2 == -E**2 + px**2 + py**2 + pz**2
        assert C**1 == sqrt(-E**2 + px**2 + py**2 + pz**2)
        assert C(mu0)**2 == C**2
        assert C(mu0)**1 == C**1


def test_valued_tensor_expressions():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        x1, x2, x3 = symbols('x1:4')

        # test coefficient in contraction:
        rank2coeff = x1 * A(i3) * B(i2)
        assert rank2coeff[1, 1] == x1 * px
        assert rank2coeff[3, 3] == 3 * pz * x1
        coeff_expr = ((x1 * A(i4)) * (B(-i4) / x2)).data

        assert coeff_expr.expand() == -px*x1/x2 - 2*py*x1/x2 - 3*pz*x1/x2

        add_expr = A(i0) + B(i0)

        assert add_expr[0] == E
        assert add_expr[1] == px + 1
        assert add_expr[2] == py + 2
        assert add_expr[3] == pz + 3

        sub_expr = A(i0) - B(i0)

        assert sub_expr[0] == E
        assert sub_expr[1] == px - 1
        assert sub_expr[2] == py - 2
        assert sub_expr[3] == pz - 3

        assert (add_expr * B(-i0)).data == -px - 2*py - 3*pz - 14

        expr1 = x1*A(i0) + x2*B(i0)
        expr2 = expr1 * B(i1) * (-4)
        expr3 = expr2 + 3*x3*AB(i0, i1)
        expr4 = expr3 / 2
        assert expr4 * 2 == expr3
        expr5 = (expr4 * BA(-i1, -i0))

        assert expr5.data.expand() == 28*E*x1 + 12*px*x1 + 20*py*x1 + 28*pz*x1 + 136*x2 + 3*x3


def test_valued_tensor_add_scalar():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # one scalar summand after the contracted tensor
        expr1 = A(i0)*A(-i0) - (E**2 - px**2 - py**2 - pz**2)
        assert expr1.data == 0

        # multiple scalar summands in front of the contracted tensor
        expr2 = E**2 - px**2 - py**2 - pz**2 - A(i0)*A(-i0)
        assert expr2.data == 0

        # multiple scalar summands after the contracted tensor
        expr3 =  A(i0)*A(-i0) - E**2 + px**2 + py**2 + pz**2
        assert expr3.data == 0

        # multiple scalar summands and multiple tensors
        expr4 = C(mu0)*C(-mu0) + 2*E**2 - 2*px**2 - 2*py**2 - 2*pz**2 - A(i0)*A(-i0)
        assert expr4.data == 0

def test_noncommuting_components():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        euclid = TensorIndexType('Euclidean')
        euclid.data = [1, 1]
        i1, i2, i3 = tensor_indices('i1:4', euclid)

        a, b, c, d = symbols('a b c d', commutative=False)
        V1 = TensorHead('V1', [euclid]*2)
        V1.data = [[a, b], (c, d)]
        V2 = TensorHead('V2', [euclid]*2)
        V2.data = [[a, c], [b, d]]

        vtp = V1(i1, i2) * V2(-i2, -i1)

        assert vtp.data == a**2 + b**2 + c**2 + d**2
        assert vtp.data != a**2 + 2*b*c + d**2

        vtp2 = V1(i1, i2)*V1(-i2, -i1)

        assert vtp2.data == a**2 + b*c + c*b + d**2
        assert vtp2.data != a**2 + 2*b*c + d**2

        Vc = (b * V1(i1, -i1)).data
        assert Vc.expand() == b * a + b * d


def test_valued_non_diagonal_metric():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        mmatrix = Matrix(ndm_matrix)
        assert (NA(n0)*NA(-n0)).data == (NA(n0).get_matrix().T * mmatrix * NA(n0).get_matrix())[0, 0]


def test_valued_assign_numpy_ndarray():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # this is needed to make sure that a numpy.ndarray can be assigned to a
        # tensor.
        arr = [E+1, px-1, py, pz]
        A.data = Array(arr)
        for i in range(4):
                assert A(i0).data[i] == arr[i]

        qx, qy, qz = symbols('qx qy qz')
        A(-i0).data = Array([E, qx, qy, qz])
        for i in range(4):
            assert A(i0).data[i] == [E, -qx, -qy, -qz][i]
            assert A.data[i] == [E, -qx, -qy, -qz][i]

        # test on multi-indexed tensors.
        random_4x4_data = [[(i**3-3*i**2)%(j+7) for i in range(4)] for j in range(4)]
        AB(-i0, -i1).data = random_4x4_data

        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)*(-1 if j else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if j else 1)
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j]

        AB(-i0, i1).data = random_4x4_data
        for i in range(4):
            for j in range(4):
                assert AB(i0, i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)
                assert AB(-i0, i1).data[i, j] == random_4x4_data[i][j]
                assert AB(i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if i else 1)*(-1 if j else 1)
                assert AB(-i0, -i1).data[i, j] == random_4x4_data[i][j]*(-1 if j else 1)


def test_valued_metric_inverse():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        # let's assign some fancy matrix, just to verify it:
        # (this has no physical sense, it's just testing sympy);
        # it is symmetrical:
        md = [[2, 2, 2, 1], [2, 3, 1, 0], [2, 1, 2, 3], [1, 0, 3, 2]]
        Lorentz.data = md
        m = Matrix(md)
        metric = Lorentz.metric
        minv = m.inv()

        meye = eye(4)

        # the Kronecker Delta:
        KD = Lorentz.get_kronecker_delta()

        for i in range(4):
            for j in range(4):
                assert metric(i0, i1).data[i, j] == m[i, j]
                assert metric(-i0, -i1).data[i, j] == minv[i, j]
                assert metric(i0, -i1).data[i, j] == meye[i, j]
                assert metric(-i0, i1).data[i, j] == meye[i, j]
                assert metric(i0, i1)[i, j] == m[i, j]
                assert metric(-i0, -i1)[i, j] == minv[i, j]
                assert metric(i0, -i1)[i, j] == meye[i, j]
                assert metric(-i0, i1)[i, j] == meye[i, j]

                assert KD(i0, -i1)[i, j] == meye[i, j]


def test_valued_canon_bp_swapaxes():
    with warns_deprecated_sympy():
        (A, B, AB, BA, C, Lorentz, E, px, py, pz, LorentzD, mu0, mu1, mu2, ndm, n0, n1,
         n2, NA, NB, NC, minkowski, ba_matrix, ndm_matrix, i0, i1, i2, i3, i4) = _get_valued_base_test_variables()

        e1 = A(i1)*A(i0)
        e2 = e1.canon_bp()
        assert e2 == A(i0)*A(i1)
        for i in range(4):
            for j in range(4):
                assert e1[i, j] == e2[j, i]
        o1 = B(i2)*A(i1)*B(i0)
        o2 = o1.canon_bp()
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    assert o1[i, j, k] == o2[j, i, k]


def test_valued_components_with_wrong_symmetry():
    with warns_deprecated_sympy():
        IT = TensorIndexType('IT', dim=3)
        i0, i1, i2, i3 = tensor_indices('i0:4', IT)
        IT.data = [1, 1, 1]
        A_nosym = TensorHead('A', [IT]*2)
        A_sym = TensorHead('A', [IT]*2, TensorSymmetry.fully_symmetric(2))
        A_antisym = TensorHead('A', [IT]*2, TensorSymmetry.fully_symmetric(-2))

        mat_nosym = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        mat_sym = mat_nosym + mat_nosym.T
        mat_antisym = mat_nosym - mat_nosym.T

        A_nosym.data = mat_nosym
        A_nosym.data = mat_sym
        A_nosym.data = mat_antisym

        def assign(A, dat):
            A.data = dat

        A_sym.data = mat_sym
        raises(ValueError, lambda: assign(A_sym, mat_nosym))
        raises(ValueError, lambda: assign(A_sym, mat_antisym))

        A_antisym.data = mat_antisym
        raises(ValueError, lambda: assign(A_antisym, mat_sym))
        raises(ValueError, lambda: assign(A_antisym, mat_nosym))

        A_sym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        A_antisym.data = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

def test_issue_10972_TensMul_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='i', dim=2)
        Lorentz.data = [-1, 1]

        mu, nu, alpha, beta = tensor_indices('\\mu, \\nu, \\alpha, \\beta',
                                             Lorentz)

        u = TensorHead('u', [Lorentz])
        u.data = [1, 0]

        F = TensorHead('F', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        F.data = [[0, 1],
                  [-1, 0]]

        mul_1 = F(mu, alpha) * u(-alpha) * F(nu, beta) * u(-beta)
        assert (mul_1.data == Array([[0, 0], [0, 1]]))

        mul_2 = F(mu, alpha) * F(nu, beta) * u(-alpha) * u(-beta)
        assert (mul_2.data == mul_1.data)

        assert ((mul_1 + mul_1).data == 2 * mul_1.data)


def test_TensMul_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='L', dim=4)
        Lorentz.data = [-1, 1, 1, 1]

        mu, nu, alpha, beta = tensor_indices('\\mu, \\nu, \\alpha, \\beta',
                                             Lorentz)

        u = TensorHead('u', [Lorentz])
        u.data = [1, 0, 0, 0]

        F = TensorHead('F', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
        F.data = [
            [0, Ex, Ey, Ez],
            [-Ex, 0, Bz, -By],
            [-Ey, -Bz, 0, Bx],
            [-Ez, By, -Bx, 0]]

        E = F(mu, nu) * u(-nu)

        assert ((E(mu) * E(nu)).data ==
                Array([[0, 0, 0, 0],
                             [0, Ex ** 2, Ex * Ey, Ex * Ez],
                             [0, Ex * Ey, Ey ** 2, Ey * Ez],
                             [0, Ex * Ez, Ey * Ez, Ez ** 2]])
                )

        assert ((E(mu) * E(nu)).canon_bp().data == (E(mu) * E(nu)).data)

        assert ((F(mu, alpha) * F(beta, nu) * u(-alpha) * u(-beta)).data ==
                - (E(mu) * E(nu)).data
                )
        assert ((F(alpha, mu) * F(beta, nu) * u(-alpha) * u(-beta)).data ==
                (E(mu) * E(nu)).data
                )

        g = TensorHead('g', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        g.data = Lorentz.data

        # tensor 'perp' is orthogonal to vector 'u'
        perp = u(mu) * u(nu) + g(mu, nu)

        mul_1 = u(-mu) * perp(mu, nu)
        assert (mul_1.data == Array([0, 0, 0, 0]))

        mul_2 = u(-mu) * perp(mu, alpha) * perp(nu, beta)
        assert (mul_2.data == Array.zeros(4, 4, 4))

        Fperp = perp(mu, alpha) * perp(nu, beta) * F(-alpha, -beta)
        assert (Fperp.data[0, :] == Array([0, 0, 0, 0]))
        assert (Fperp.data[:, 0] == Array([0, 0, 0, 0]))

        mul_3 = u(-mu) * Fperp(mu, nu)
        assert (mul_3.data == Array([0, 0, 0, 0]))

        # Test the deleter
        del g.data

def test_issue_11020_TensAdd_data():
    with warns_deprecated_sympy():
        Lorentz = TensorIndexType('Lorentz', metric_symmetry=1, dummy_name='i', dim=2)
        Lorentz.data = [-1, 1]

        a, b, c, d = tensor_indices('a, b, c, d', Lorentz)
        i0, i1 = tensor_indices('i_0:2', Lorentz)

        # metric tensor
        g = TensorHead('g', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        g.data = Lorentz.data

        u = TensorHead('u', [Lorentz])
        u.data = [1, 0]

        add_1 = g(b, c) * g(d, i0) * u(-i0) - g(b, c) * u(d)
        assert (add_1.data == Array.zeros(2, 2, 2))
        # Now let us replace index `d` with `a`:
        add_2 = g(b, c) * g(a, i0) * u(-i0) - g(b, c) * u(a)
        assert (add_2.data == Array.zeros(2, 2, 2))

        # some more tests
        # perp is tensor orthogonal to u^\mu
        perp = u(a) * u(b) + g(a, b)
        mul_1 = u(-a) * perp(a, b)
        assert (mul_1.data == Array([0, 0]))

        mul_2 = u(-c) * perp(c, a) * perp(d, b)
        assert (mul_2.data == Array.zeros(2, 2, 2))


def test_index_iteration():
    L = TensorIndexType("Lorentz", dummy_name="L")
    i0, i1, i2, i3, i4 = tensor_indices('i0:5', L)
    L0 = tensor_indices('L_0', L)
    L1 = tensor_indices('L_1', L)
    A = TensorHead("A", [L, L])
    B = TensorHead("B", [L, L], TensorSymmetry.fully_symmetric(2))

    e1 = A(i0,i2)
    e2 = A(i0,-i0)
    e3 = A(i0,i1)*B(i2,i3)
    e4 = A(i0,i1)*B(i2,-i1)
    e5 = A(i0,i1)*B(-i0,-i1)
    e6 = e1 + e4

    assert list(e1._iterate_free_indices) == [(i0, (1, 0)), (i2, (1, 1))]
    assert list(e1._iterate_dummy_indices) == []
    assert list(e1._iterate_indices) == [(i0, (1, 0)), (i2, (1, 1))]

    assert list(e2._iterate_free_indices) == []
    assert list(e2._iterate_dummy_indices) == [(L0, (1, 0)), (-L0, (1, 1))]
    assert list(e2._iterate_indices) == [(L0, (1, 0)), (-L0, (1, 1))]

    assert list(e3._iterate_free_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]
    assert list(e3._iterate_dummy_indices) == []
    assert list(e3._iterate_indices) == [(i0, (0, 1, 0)), (i1, (0, 1, 1)), (i2, (1, 1, 0)), (i3, (1, 1, 1))]

    assert list(e4._iterate_free_indices) == [(i0, (0, 1, 0)), (i2, (1, 1, 0))]
    assert list(e4._iterate_dummy_indices) == [(L0, (0, 1, 1)), (-L0, (1, 1, 1))]
    assert list(e4._iterate_indices) == [(i0, (0, 1, 0)), (L0, (0, 1, 1)), (i2, (1, 1, 0)), (-L0, (1, 1, 1))]

    assert list(e5._iterate_free_indices) == []
    assert list(e5._iterate_dummy_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]
    assert list(e5._iterate_indices) == [(L0, (0, 1, 0)), (L1, (0, 1, 1)), (-L0, (1, 1, 0)), (-L1, (1, 1, 1))]

    assert list(e6._iterate_free_indices) == [(i0, (0, 0, 1, 0)), (i2, (0, 1, 1, 0)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]
    assert list(e6._iterate_dummy_indices) == [(L0, (0, 0, 1, 1)), (-L0, (0, 1, 1, 1))]
    assert list(e6._iterate_indices) == [(i0, (0, 0, 1, 0)), (L0, (0, 0, 1, 1)), (i2, (0, 1, 1, 0)), (-L0, (0, 1, 1, 1)), (i0, (1, 1, 0)), (i2, (1, 1, 1))]

    assert e1.get_indices() == [i0, i2]
    assert e1.get_free_indices() == [i0, i2]
    assert e2.get_indices() == [L0, -L0]
    assert e2.get_free_indices() == []
    assert e3.get_indices() == [i0, i1, i2, i3]
    assert e3.get_free_indices() == [i0, i1, i2, i3]
    assert e4.get_indices() == [i0, L0, i2, -L0]
    assert e4.get_free_indices() == [i0, i2]
    assert e5.get_indices() == [L0, L1, -L0, -L1]
    assert e5.get_free_indices() == []


def test_tensor_expand():
    L = TensorIndexType("L")

    i, j, k = tensor_indices("i j k", L)
    L_0 = TensorIndex("L_0", L)

    A, B, C, D = tensor_heads("A B C D", [L])

    assert isinstance(Add(A(i), B(i)), TensAdd)
    assert isinstance(expand(A(i)+B(i)), TensAdd)

    expr = A(i)*(A(-i)+B(-i))
    assert expr.args == (A(L_0), A(-L_0) + B(-L_0))
    assert expr != A(i)*A(-i) + A(i)*B(-i)
    assert expr.expand() == A(i)*A(-i) + A(i)*B(-i)
    assert str(expr) == "A(L_0)*(A(-L_0) + B(-L_0))"

    expr = A(i)*A(j) + A(i)*B(j)
    assert str(expr) == "A(i)*A(j) + A(i)*B(j)"

    expr = A(-i)*(A(i)*A(j) + A(i)*B(j)*C(k)*C(-k))
    assert expr != A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)*C(k)*C(-k)
    assert expr.expand() == A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)*C(k)*C(-k)
    assert str(expr) == "A(-L_0)*(A(L_0)*A(j) + A(L_0)*B(j)*C(L_1)*C(-L_1))"
    assert str(expr.canon_bp()) == 'A(j)*A(L_0)*A(-L_0) + A(L_0)*A(-L_0)*B(j)*C(L_1)*C(-L_1)'

    expr = A(-i)*(2*A(i)*A(j) + A(i)*B(j))
    assert expr.expand() == 2*A(-i)*A(i)*A(j) + A(-i)*A(i)*B(j)

    expr = 2*A(i)*A(-i)
    assert expr.coeff == 2

    expr = A(i)*(B(j)*C(k) + C(j)*(A(k) + D(k)))
    assert str(expr) == "A(i)*(B(j)*C(k) + C(j)*(A(k) + D(k)))"
    assert str(expr.expand()) == "A(i)*B(j)*C(k) + A(i)*C(j)*A(k) + A(i)*C(j)*D(k)"

    assert isinstance(TensMul(3), TensMul)
    tm = TensMul(3).doit()
    assert tm == 3
    assert isinstance(tm, Integer)

    p1 = B(j)*B(-j) + B(j)*C(-j)
    p2 = C(-i)*p1
    p3 = A(i)*p2
    assert p3.expand() == A(i)*C(-i)*B(j)*B(-j) + A(i)*C(-i)*B(j)*C(-j)

    expr = A(i)*(B(-i) + C(-i)*(B(j)*B(-j) + B(j)*C(-j)))
    assert expr.expand() == A(i)*B(-i) + A(i)*C(-i)*B(j)*B(-j) + A(i)*C(-i)*B(j)*C(-j)

    expr = C(-i)*(B(j)*B(-j) + B(j)*C(-j))
    assert expr.expand() == C(-i)*B(j)*B(-j) + C(-i)*B(j)*C(-j)


def test_tensor_alternative_construction():
    L = TensorIndexType("L")
    i0, i1, i2, i3 = tensor_indices('i0:4', L)
    A = TensorHead("A", [L])
    x, y = symbols("x y")

    assert A(i0) == A(Symbol("i0"))
    assert A(-i0) == A(-Symbol("i0"))
    raises(TypeError, lambda: A(x+y))
    raises(ValueError, lambda: A(2*x))


def test_tensor_replacement():
    L = TensorIndexType("L")
    L2 = TensorIndexType("L2", dim=2)
    i, j, k, l = tensor_indices("i j k l", L)
    A, B, C, D = tensor_heads("A B C D", [L])
    H = TensorHead("H", [L, L])
    K = TensorHead("K", [L]*4)

    expr = H(i, j)
    repl = {H(i,-j): [[1,2],[3,4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, j], Array([[1, -2], [3, -4]]))

    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [Symbol("i"), -Symbol("j")]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [-2, -4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [-2, 4]])
    assert expr.replace_with_arrays(repl, [-j, i]) == Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [-j, -i]) == Array([[1, -3], [2, -4]])
    # Test stability of optional parameter 'indices'
    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])

    expr = H(i,j)
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))

    assert expr.replace_with_arrays(repl) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [2, -4]])
    assert expr.replace_with_arrays(repl, [-j, i]) == Array([[1, 3], [-2, -4]])
    assert expr.replace_with_arrays(repl, [-j, -i]) == Array([[1, -3], [-2, 4]])

    # Not the same indices:
    expr = H(i,k)
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, k], Array([[1, 2], [3, 4]]))

    expr = A(i)*A(-i)
    repl = {A(i): [1,2], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([], -3)
    assert expr.replace_with_arrays(repl, []) == -3

    expr = K(i, j, -j, k)*A(-i)*A(-k)
    repl = {A(i): [1, 2], K(i,j,k,l): Array([1]*2**4).reshape(2,2,2,2), L: diag(1, -1)}
    assert expr._extract_data(repl)

    expr = H(j, k)
    repl = {H(i,j): [[1,2],[3,4]], L: diag(1, -1)}
    raises(ValueError, lambda: expr._extract_data(repl))

    expr = A(i)
    repl = {B(i): [1, 2]}
    raises(ValueError, lambda: expr._extract_data(repl))

    expr = A(i)
    repl = {A(i): [[1, 2], [3, 4]]}
    raises(ValueError, lambda: expr._extract_data(repl))

    # TensAdd:
    expr = A(k)*H(i, j) + B(k)*H(i, j)
    repl = {A(k): [1], B(k): [1], H(i, j): [[1, 2],[3,4]], L:diag(1,1)}
    assert expr._extract_data(repl) == ([k, i, j], Array([[[2, 4], [6, 8]]]))
    assert expr.replace_with_arrays(repl, [k, i, j]) == Array([[[2, 4], [6, 8]]])
    assert expr.replace_with_arrays(repl, [k, j, i]) == Array([[[2, 6], [4, 8]]])

    expr = A(k)*A(-k) + 100
    repl = {A(k): [2, 3], L: diag(1, 1)}
    assert expr.replace_with_arrays(repl, []) == 113

    ## Symmetrization:
    expr = H(i, j) + H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[2, 5], [5, 8]]))
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[2, 5], [5, 8]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[2, 5], [5, 8]])

    ## Anti-symmetrization:
    expr = H(i, j) - H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[0, -1], [1, 0]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[0, 1], [-1, 0]])

    # Tensors with contractions in replacements:
    expr = K(i, j, k, -k)
    repl = {K(i, j, k, -k): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))

    expr = H(i, -i)
    repl = {H(i, -i): 42}
    assert expr._extract_data(repl) == ([], 42)

    expr = H(i, -i)
    repl = {
        H(-i, -j): Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
        L: Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]),
    }
    assert expr._extract_data(repl) == ([], 4)

    # Replace with array, raise exception if indices are not compatible:
    expr = A(i)*A(j)
    repl = {A(i): [1, 2]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [j]))

    # Raise exception if array dimension is not compatible:
    expr = A(i)
    repl = {A(i): [[1, 2]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [i]))

    # TensorIndexType with dimension, wrong dimension in replacement array:
    u1, u2, u3 = tensor_indices("u1:4", L2)
    U = TensorHead("U", [L2])
    expr = U(u1)*U(-u2)
    repl = {U(u1): [[1]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [u1, -u2]))


def test_rewrite_tensor_to_Indexed():
    L = TensorIndexType("L", dim=4)
    A = TensorHead("A", [L]*4)
    B = TensorHead("B", [L])

    i0, i1, i2, i3 = symbols("i0:4")
    L_0, L_1 = symbols("L_0:2")

    a1 = A(i0, i1, i2, i3)
    assert a1.rewrite(Indexed) == Indexed(Symbol("A"), i0, i1, i2, i3)

    a2 = A(i0, -i0, i2, i3)
    assert a2.rewrite(Indexed) == Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3))

    a3 = a2 + A(i2, i3, i0, -i0)
    assert a3.rewrite(Indexed) == \
        Sum(Indexed(Symbol("A"), L_0, L_0, i2, i3), (L_0, 0, 3)) +\
        Sum(Indexed(Symbol("A"), i2, i3, L_0, L_0), (L_0, 0, 3))

    b1 = B(-i0)*a1
    assert b1.rewrite(Indexed) == Sum(Indexed(Symbol("B"), L_0)*Indexed(Symbol("A"), L_0, i1, i2, i3), (L_0, 0, 3))

    b2 = B(-i3)*a2
    assert b2.rewrite(Indexed) == Sum(Indexed(Symbol("B"), L_1)*Indexed(Symbol("A"), L_0, L_0, i2, L_1), (L_0, 0, 3), (L_1, 0, 3))

def test_tensor_matching():
    """
    Test match and replace with the pattern being a WildTensor or a WildTensorIndex
    """
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices("p q r", R3)
    a,b,c = symbols("a b c", cls = WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
    g = WildTensorIndex("g", R3)
    delta = R3.delta
    eps = R3.epsilon
    K = TensorHead("K", [R3])
    V = TensorHead("V", [R3])
    A = TensorHead("A", [R3, R3])
    W = WildTensorHead('W', unordered_indices=True)
    U = WildTensorHead('U')

    assert a.matches(q) == {a:q}
    assert a.matches(-q) == {a:-q}
    assert g.matches(-q) == None
    assert g.matches(q) == {g:q}
    assert eps(p,-a,a).matches( eps(p,q,r) ) == None
    assert eps(p,-b,a).matches( eps(p,q,r) ) == {a: r, -b: q}
    assert eps(p,-q,r).replace(eps(a,b,c), 1) == 1
    assert W().matches( K(p)*V(q) ) == {W(): K(p)*V(q)}
    assert W(a).matches( K(p) ) == {a:p, W(a).head: _WildTensExpr(K(p))}
    assert W(a,p).matches( K(p)*V(q) ) == {a:q, W(a,p).head: _WildTensExpr(K(p)*V(q))}
    assert W(p,q).matches( K(p)*V(q) ) == {W(p,q).head: _WildTensExpr(K(p)*V(q))}
    assert W(p,q).matches( A(q,p) ) == {W(p,q).head: _WildTensExpr(A(q, p))}
    assert U(p,q).matches( A(q,p) ) == None
    assert ( K(q)*K(p) ).replace( W(q,p), 1) == 1

    #Some tests for matching without Wild
    assert delta(p,q).matches(delta(q,p)) == {}
    assert eps(p,q,r).matches(eps(q,p,r)) is None
    assert eps(p,q,r).matches(eps(q,r,p)) == {}

def test_TensMul_subs():
    """
    Test subs and xreplace in TensMul. See bug #24337
    """
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices("p q r", R3)
    K = TensorHead("K", [R3])
    V = TensorHead("V", [R3])
    A = TensorHead("A", [R3, R3])
    C0 = TensorIndex(R3.dummy_name + "_0", R3, True)

    assert ( K(p)*V(r)*K(-p) ).subs({V(r): K(q)*K(-q)}) == K(p)*K(q)*K(-q)*K(-p)
    assert ( K(p)*V(r)*K(-p) ).xreplace({V(r): K(q)*K(-q)}) == K(p)*K(q)*K(-q)*K(-p)
    assert ( K(p)*V(r) ).xreplace({p: C0, V(r): K(q)*K(-q)}) == K(C0)*K(q)*K(-q)
    assert ( K(p)*A(q,-q)*K(-p) ).doit() == K(p)*A(q,-q)*K(-p)


def test_tensorsymmetry():
    with warns_deprecated_sympy():
        tensorsymmetry([1]*2)

def test_tensorhead():
    with warns_deprecated_sympy():
        tensorhead('A', [])

def test_TensorType():
    with warns_deprecated_sympy():
        sym2 = TensorSymmetry.fully_symmetric(2)
        Lorentz = TensorIndexType('Lorentz')
        S2 = TensorType([Lorentz]*2, sym2)
        assert isinstance(S2, TensorType)

def test_dummy_fmt():
    with warns_deprecated_sympy():
        TensorIndexType('Lorentz', dummy_fmt='L')

def test_postprocessor():
    """
    Test if substituting a Tensor into a Mul or Add automatically converts it
    to TensMul or TensAdd respectively. See github issue #25051
    """
    R3 = TensorIndexType('R3', dim=3)
    i = tensor_indices("i", R3)
    K = TensorHead("K", [R3])
    x,y,z = symbols("x y z")

    assert isinstance((x*2).xreplace({x: K(i)}), TensMul)
    assert isinstance((x+2).xreplace({x: K(i)*K(-i)}), TensAdd)

    assert isinstance((x*2).subs({x: K(i)}), TensMul)
    assert isinstance((x+2).subs({x: K(i)*K(-i)}), TensAdd)

    assert isinstance((x*2).replace(x, K(i)), TensMul)
    assert isinstance((x+2).replace(x, K(i)*K(-i)), TensAdd)
