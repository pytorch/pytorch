from sympy.matrices.dense import eye, Matrix
from sympy.tensor.tensor import tensor_indices, TensorHead, tensor_heads, \
    TensExpr, canon_bp
from sympy.physics.hep.gamma_matrices import GammaMatrix as G, LorentzIndex, \
    kahane_simplify, gamma_trace, _simplify_single_line, simplify_gamma_expression
from sympy import Symbol


def _is_tensor_eq(arg1, arg2):
    arg1 = canon_bp(arg1)
    arg2 = canon_bp(arg2)
    if isinstance(arg1, TensExpr):
        return arg1.equals(arg2)
    elif isinstance(arg2, TensExpr):
        return arg2.equals(arg1)
    return arg1 == arg2

def execute_gamma_simplify_tests_for_function(tfunc, D):
    """
    Perform tests to check if sfunc is able to simplify gamma matrix expressions.

    Parameters
    ==========

    `sfunc`     a function to simplify a `TIDS`, shall return the simplified `TIDS`.
    `D`         the number of dimension (in most cases `D=4`).

    """

    mu, nu, rho, sigma = tensor_indices("mu, nu, rho, sigma", LorentzIndex)
    a1, a2, a3, a4, a5, a6 = tensor_indices("a1:7", LorentzIndex)
    mu11, mu12, mu21, mu31, mu32, mu41, mu51, mu52 = tensor_indices("mu11, mu12, mu21, mu31, mu32, mu41, mu51, mu52", LorentzIndex)
    mu61, mu71, mu72 = tensor_indices("mu61, mu71, mu72", LorentzIndex)
    m0, m1, m2, m3, m4, m5, m6 = tensor_indices("m0:7", LorentzIndex)

    def g(xx, yy):
        return (G(xx)*G(yy) + G(yy)*G(xx))/2

    # Some examples taken from Kahane's paper, 4 dim only:
    if D == 4:
        t = (G(a1)*G(mu11)*G(a2)*G(mu21)*G(-a1)*G(mu31)*G(-a2))
        assert _is_tensor_eq(tfunc(t), -4*G(mu11)*G(mu31)*G(mu21) - 4*G(mu31)*G(mu11)*G(mu21))

        t = (G(a1)*G(mu11)*G(mu12)*\
                              G(a2)*G(mu21)*\
                              G(a3)*G(mu31)*G(mu32)*\
                              G(a4)*G(mu41)*\
                              G(-a2)*G(mu51)*G(mu52)*\
                              G(-a1)*G(mu61)*\
                              G(-a3)*G(mu71)*G(mu72)*\
                              G(-a4))
        assert _is_tensor_eq(tfunc(t), \
            16*G(mu31)*G(mu32)*G(mu72)*G(mu71)*G(mu11)*G(mu52)*G(mu51)*G(mu12)*G(mu61)*G(mu21)*G(mu41) + 16*G(mu31)*G(mu32)*G(mu72)*G(mu71)*G(mu12)*G(mu51)*G(mu52)*G(mu11)*G(mu61)*G(mu21)*G(mu41) + 16*G(mu71)*G(mu72)*G(mu32)*G(mu31)*G(mu11)*G(mu52)*G(mu51)*G(mu12)*G(mu61)*G(mu21)*G(mu41) + 16*G(mu71)*G(mu72)*G(mu32)*G(mu31)*G(mu12)*G(mu51)*G(mu52)*G(mu11)*G(mu61)*G(mu21)*G(mu41))

    # Fully Lorentz-contracted expressions, these return scalars:

    def add_delta(ne):
        return ne * eye(4)  # DiracSpinorIndex.delta(DiracSpinorIndex.auto_left, -DiracSpinorIndex.auto_right)

    t = (G(mu)*G(-mu))
    ts = add_delta(D)
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(mu)*G(nu)*G(-mu)*G(-nu))
    ts = add_delta(2*D - D**2)  # -8
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(mu)*G(nu)*G(-nu)*G(-mu))
    ts = add_delta(D**2)  # 16
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(mu)*G(nu)*G(-rho)*G(-nu)*G(-mu)*G(rho))
    ts = add_delta(4*D - 4*D**2 + D**3)  # 16
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(mu)*G(nu)*G(rho)*G(-rho)*G(-nu)*G(-mu))
    ts = add_delta(D**3)  # 64
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(a1)*G(a2)*G(a3)*G(a4)*G(-a3)*G(-a1)*G(-a2)*G(-a4))
    ts = add_delta(-8*D + 16*D**2 - 8*D**3 + D**4)  # -32
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(-mu)*G(-nu)*G(-rho)*G(-sigma)*G(nu)*G(mu)*G(sigma)*G(rho))
    ts = add_delta(-16*D + 24*D**2 - 8*D**3 + D**4)  # 64
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(-mu)*G(nu)*G(-rho)*G(sigma)*G(rho)*G(-nu)*G(mu)*G(-sigma))
    ts = add_delta(8*D - 12*D**2 + 6*D**3 - D**4)  # -32
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(a1)*G(a2)*G(a3)*G(a4)*G(a5)*G(-a3)*G(-a2)*G(-a1)*G(-a5)*G(-a4))
    ts = add_delta(64*D - 112*D**2 + 60*D**3 - 12*D**4 + D**5)  # 256
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(a1)*G(a2)*G(a3)*G(a4)*G(a5)*G(-a3)*G(-a1)*G(-a2)*G(-a4)*G(-a5))
    ts = add_delta(64*D - 120*D**2 + 72*D**3 - 16*D**4 + D**5)  # -128
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(a1)*G(a2)*G(a3)*G(a4)*G(a5)*G(a6)*G(-a3)*G(-a2)*G(-a1)*G(-a6)*G(-a5)*G(-a4))
    ts = add_delta(416*D - 816*D**2 + 528*D**3 - 144*D**4 + 18*D**5 - D**6)  # -128
    assert _is_tensor_eq(tfunc(t), ts)

    t = (G(a1)*G(a2)*G(a3)*G(a4)*G(a5)*G(a6)*G(-a2)*G(-a3)*G(-a1)*G(-a6)*G(-a4)*G(-a5))
    ts = add_delta(416*D - 848*D**2 + 584*D**3 - 172*D**4 + 22*D**5 - D**6)  # -128
    assert _is_tensor_eq(tfunc(t), ts)

    # Expressions with free indices:

    t = (G(mu)*G(nu)*G(rho)*G(sigma)*G(-mu))
    assert _is_tensor_eq(tfunc(t), (-2*G(sigma)*G(rho)*G(nu) + (4-D)*G(nu)*G(rho)*G(sigma)))

    t = (G(mu)*G(nu)*G(-mu))
    assert _is_tensor_eq(tfunc(t), (2-D)*G(nu))

    t = (G(mu)*G(nu)*G(rho)*G(-mu))
    assert _is_tensor_eq(tfunc(t), 2*G(nu)*G(rho) + 2*G(rho)*G(nu) - (4-D)*G(nu)*G(rho))

    t = 2*G(m2)*G(m0)*G(m1)*G(-m0)*G(-m1)
    st = tfunc(t)
    assert _is_tensor_eq(st, (D*(-2*D + 4))*G(m2))

    t = G(m2)*G(m0)*G(m1)*G(-m0)*G(-m2)
    st = tfunc(t)
    assert _is_tensor_eq(st, ((-D + 2)**2)*G(m1))

    t = G(m0)*G(m1)*G(m2)*G(m3)*G(-m1)
    st = tfunc(t)
    assert _is_tensor_eq(st, (D - 4)*G(m0)*G(m2)*G(m3) + 4*G(m0)*g(m2, m3))

    t = G(m0)*G(m1)*G(m2)*G(m3)*G(-m1)*G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, ((D - 4)**2)*G(m2)*G(m3) + (8*D - 16)*g(m2, m3))

    t = G(m2)*G(m0)*G(m1)*G(-m2)*G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, ((-D + 2)*(D - 4) + 4)*G(m1))

    t = G(m3)*G(m1)*G(m0)*G(m2)*G(-m3)*G(-m0)*G(-m2)
    st = tfunc(t)
    assert _is_tensor_eq(st, (-4*D + (-D + 2)**2*(D - 4) + 8)*G(m1))

    t = 2*G(m0)*G(m1)*G(m2)*G(m3)*G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, ((-2*D + 8)*G(m1)*G(m2)*G(m3) - 4*G(m3)*G(m2)*G(m1)))

    t = G(m5)*G(m0)*G(m1)*G(m4)*G(m2)*G(-m4)*G(m3)*G(-m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, (((-D + 2)*(-D + 4))*G(m5)*G(m1)*G(m2)*G(m3) + (2*D - 4)*G(m5)*G(m3)*G(m2)*G(m1)))

    t = -G(m0)*G(m1)*G(m2)*G(m3)*G(-m0)*G(m4)
    st = tfunc(t)
    assert _is_tensor_eq(st, ((D - 4)*G(m1)*G(m2)*G(m3)*G(m4) + 2*G(m3)*G(m2)*G(m1)*G(m4)))

    t = G(-m5)*G(m0)*G(m1)*G(m2)*G(m3)*G(m4)*G(-m0)*G(m5)
    st = tfunc(t)

    result1 = ((-D + 4)**2 + 4)*G(m1)*G(m2)*G(m3)*G(m4) +\
        (4*D - 16)*G(m3)*G(m2)*G(m1)*G(m4) + (4*D - 16)*G(m4)*G(m1)*G(m2)*G(m3)\
        + 4*G(m2)*G(m1)*G(m4)*G(m3) + 4*G(m3)*G(m4)*G(m1)*G(m2) +\
        4*G(m4)*G(m3)*G(m2)*G(m1)

    # Kahane's algorithm yields this result, which is equivalent to `result1`
    # in four dimensions, but is not automatically recognized as equal:
    result2 = 8*G(m1)*G(m2)*G(m3)*G(m4) + 8*G(m4)*G(m3)*G(m2)*G(m1)

    if D == 4:
        assert _is_tensor_eq(st, (result1)) or _is_tensor_eq(st, (result2))
    else:
        assert _is_tensor_eq(st, (result1))

    # and a few very simple cases, with no contracted indices:

    t = G(m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, t)

    t = -7*G(m0)
    st = tfunc(t)
    assert _is_tensor_eq(st, t)

    t = 224*G(m0)*G(m1)*G(-m2)*G(m3)
    st = tfunc(t)
    assert _is_tensor_eq(st, t)


def test_kahane_algorithm():
    # Wrap this function to convert to and from TIDS:

    def tfunc(e):
        return _simplify_single_line(e)

    execute_gamma_simplify_tests_for_function(tfunc, D=4)


def test_kahane_simplify1():
    i0,i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15 = tensor_indices('i0:16', LorentzIndex)
    mu, nu, rho, sigma = tensor_indices("mu, nu, rho, sigma", LorentzIndex)
    D = 4
    t = G(i0)*G(i1)
    r = kahane_simplify(t)
    assert r.equals(t)

    t = G(i0)*G(i1)*G(-i0)
    r = kahane_simplify(t)
    assert r.equals(-2*G(i1))
    t = G(i0)*G(i1)*G(-i0)
    r = kahane_simplify(t)
    assert r.equals(-2*G(i1))

    t = G(i0)*G(i1)
    r = kahane_simplify(t)
    assert r.equals(t)
    t = G(i0)*G(i1)
    r = kahane_simplify(t)
    assert r.equals(t)
    t = G(i0)*G(-i0)
    r = kahane_simplify(t)
    assert r.equals(4*eye(4))
    t = G(i0)*G(-i0)
    r = kahane_simplify(t)
    assert r.equals(4*eye(4))
    t = G(i0)*G(-i0)
    r = kahane_simplify(t)
    assert r.equals(4*eye(4))
    t = G(i0)*G(i1)*G(-i0)
    r = kahane_simplify(t)
    assert r.equals(-2*G(i1))
    t = G(i0)*G(i1)*G(-i0)*G(-i1)
    r = kahane_simplify(t)
    assert r.equals((2*D - D**2)*eye(4))
    t = G(i0)*G(i1)*G(-i0)*G(-i1)
    r = kahane_simplify(t)
    assert r.equals((2*D - D**2)*eye(4))
    t = G(i0)*G(-i0)*G(i1)*G(-i1)
    r = kahane_simplify(t)
    assert r.equals(16*eye(4))
    t = (G(mu)*G(nu)*G(-nu)*G(-mu))
    r = kahane_simplify(t)
    assert r.equals(D**2*eye(4))
    t = (G(mu)*G(nu)*G(-nu)*G(-mu))
    r = kahane_simplify(t)
    assert r.equals(D**2*eye(4))
    t = (G(mu)*G(nu)*G(-nu)*G(-mu))
    r = kahane_simplify(t)
    assert r.equals(D**2*eye(4))
    t = (G(mu)*G(nu)*G(-rho)*G(-nu)*G(-mu)*G(rho))
    r = kahane_simplify(t)
    assert r.equals((4*D - 4*D**2 + D**3)*eye(4))
    t = (G(-mu)*G(-nu)*G(-rho)*G(-sigma)*G(nu)*G(mu)*G(sigma)*G(rho))
    r = kahane_simplify(t)
    assert r.equals((-16*D + 24*D**2 - 8*D**3 + D**4)*eye(4))
    t = (G(-mu)*G(nu)*G(-rho)*G(sigma)*G(rho)*G(-nu)*G(mu)*G(-sigma))
    r = kahane_simplify(t)
    assert r.equals((8*D - 12*D**2 + 6*D**3 - D**4)*eye(4))

    # Expressions with free indices:
    t = (G(mu)*G(nu)*G(rho)*G(sigma)*G(-mu))
    r = kahane_simplify(t)
    assert r.equals(-2*G(sigma)*G(rho)*G(nu))
    t = (G(mu)*G(-mu)*G(rho)*G(sigma))
    r = kahane_simplify(t)
    assert r.equals(4*G(rho)*G(sigma))
    t = (G(rho)*G(sigma)*G(mu)*G(-mu))
    r = kahane_simplify(t)
    assert r.equals(4*G(rho)*G(sigma))

def test_gamma_matrix_class():
    i, j, k = tensor_indices('i,j,k', LorentzIndex)

    # define another type of TensorHead to see if exprs are correctly handled:
    A = TensorHead('A', [LorentzIndex])

    t = A(k)*G(i)*G(-i)
    ts = simplify_gamma_expression(t)
    assert _is_tensor_eq(ts, Matrix([
        [4, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 4, 0],
        [0, 0, 0, 4]])*A(k))

    t = G(i)*A(k)*G(j)
    ts = simplify_gamma_expression(t)
    assert _is_tensor_eq(ts, A(k)*G(i)*G(j))

    execute_gamma_simplify_tests_for_function(simplify_gamma_expression, D=4)


def test_gamma_matrix_trace():
    g = LorentzIndex.metric

    m0, m1, m2, m3, m4, m5, m6 = tensor_indices('m0:7', LorentzIndex)
    n0, n1, n2, n3, n4, n5 = tensor_indices('n0:6', LorentzIndex)

    # working in D=4 dimensions
    D = 4

    # traces of odd number of gamma matrices are zero:
    t = G(m0)
    t1 = gamma_trace(t)
    assert t1.equals(0)

    t = G(m0)*G(m1)*G(m2)
    t1 = gamma_trace(t)
    assert t1.equals(0)

    t = G(m0)*G(m1)*G(-m0)
    t1 = gamma_trace(t)
    assert t1.equals(0)

    t = G(m0)*G(m1)*G(m2)*G(m3)*G(m4)
    t1 = gamma_trace(t)
    assert t1.equals(0)

    # traces without internal contractions:
    t = G(m0)*G(m1)
    t1 = gamma_trace(t)
    assert _is_tensor_eq(t1, 4*g(m0, m1))

    t = G(m0)*G(m1)*G(m2)*G(m3)
    t1 = gamma_trace(t)
    t2 = -4*g(m0, m2)*g(m1, m3) + 4*g(m0, m1)*g(m2, m3) + 4*g(m0, m3)*g(m1, m2)
    assert _is_tensor_eq(t1, t2)

    t = G(m0)*G(m1)*G(m2)*G(m3)*G(m4)*G(m5)
    t1 = gamma_trace(t)
    t2 = t1*g(-m0, -m5)
    t2 = t2.contract_metric(g)
    assert _is_tensor_eq(t2, D*gamma_trace(G(m1)*G(m2)*G(m3)*G(m4)))

    # traces of expressions with internal contractions:
    t = G(m0)*G(-m0)
    t1 = gamma_trace(t)
    assert t1.equals(4*D)

    t = G(m0)*G(m1)*G(-m0)*G(-m1)
    t1 = gamma_trace(t)
    assert t1.equals(8*D - 4*D**2)

    t = G(m0)*G(m1)*G(m2)*G(m3)*G(m4)*G(-m0)
    t1 = gamma_trace(t)
    t2 = (-4*D)*g(m1, m3)*g(m2, m4) + (4*D)*g(m1, m2)*g(m3, m4) + \
                 (4*D)*g(m1, m4)*g(m2, m3)
    assert _is_tensor_eq(t1, t2)

    t = G(-m5)*G(m0)*G(m1)*G(m2)*G(m3)*G(m4)*G(-m0)*G(m5)
    t1 = gamma_trace(t)
    t2 = (32*D + 4*(-D + 4)**2 - 64)*(g(m1, m2)*g(m3, m4) - \
            g(m1, m3)*g(m2, m4) + g(m1, m4)*g(m2, m3))
    assert _is_tensor_eq(t1, t2)

    t = G(m0)*G(m1)*G(-m0)*G(m3)
    t1 = gamma_trace(t)
    assert t1.equals((-4*D + 8)*g(m1, m3))

#    p, q = S1('p,q')
#    ps = p(m0)*G(-m0)
#    qs = q(m0)*G(-m0)
#    t = ps*qs*ps*qs
#    t1 = gamma_trace(t)
#    assert t1 == 8*p(m0)*q(-m0)*p(m1)*q(-m1) - 4*p(m0)*p(-m0)*q(m1)*q(-m1)

    t = G(m0)*G(m1)*G(m2)*G(m3)*G(m4)*G(m5)*G(-m0)*G(-m1)*G(-m2)*G(-m3)*G(-m4)*G(-m5)
    t1 = gamma_trace(t)
    assert t1.equals(-4*D**6 + 120*D**5 - 1040*D**4 + 3360*D**3 - 4480*D**2 + 2048*D)

    t = G(m0)*G(m1)*G(n1)*G(m2)*G(n2)*G(m3)*G(m4)*G(-n2)*G(-n1)*G(-m0)*G(-m1)*G(-m2)*G(-m3)*G(-m4)
    t1 = gamma_trace(t)
    tresu = -7168*D + 16768*D**2 - 14400*D**3 + 5920*D**4 - 1232*D**5 + 120*D**6 - 4*D**7
    assert t1.equals(tresu)

    # checked with Mathematica
    # In[1]:= <<Tracer.m
    # In[2]:= Spur[l];
    # In[3]:= GammaTrace[l, {m0},{m1},{n1},{m2},{n2},{m3},{m4},{n3},{n4},{m0},{m1},{m2},{m3},{m4}]
    t = G(m0)*G(m1)*G(n1)*G(m2)*G(n2)*G(m3)*G(m4)*G(n3)*G(n4)*G(-m0)*G(-m1)*G(-m2)*G(-m3)*G(-m4)
    t1 = gamma_trace(t)
#    t1 = t1.expand_coeff()
    c1 = -4*D**5 + 120*D**4 - 1200*D**3 + 5280*D**2 - 10560*D + 7808
    c2 = -4*D**5 + 88*D**4 - 560*D**3 + 1440*D**2 - 1600*D + 640
    assert _is_tensor_eq(t1, c1*g(n1, n4)*g(n2, n3) + c2*g(n1, n2)*g(n3, n4) + \
            (-c1)*g(n1, n3)*g(n2, n4))

    p, q = tensor_heads('p,q', [LorentzIndex])
    ps = p(m0)*G(-m0)
    qs = q(m0)*G(-m0)
    p2 = p(m0)*p(-m0)
    q2 = q(m0)*q(-m0)
    pq = p(m0)*q(-m0)
    t = ps*qs*ps*qs
    r = gamma_trace(t)
    assert _is_tensor_eq(r, 8*pq*pq - 4*p2*q2)
    t = ps*qs*ps*qs*ps*qs
    r = gamma_trace(t)
    assert _is_tensor_eq(r, -12*p2*pq*q2 + 16*pq*pq*pq)
    t = ps*qs*ps*qs*ps*qs*ps*qs
    r = gamma_trace(t)
    assert _is_tensor_eq(r, -32*pq*pq*p2*q2 + 32*pq*pq*pq*pq + 4*p2*p2*q2*q2)

    t = 4*p(m1)*p(m0)*p(-m0)*q(-m1)*q(m2)*q(-m2)
    assert _is_tensor_eq(gamma_trace(t), t)
    t = ps*ps*ps*ps*ps*ps*ps*ps
    r = gamma_trace(t)
    assert r.equals(4*p2*p2*p2*p2)


def test_bug_13636():
    """Test issue 13636 regarding handling traces of sums of products
    of GammaMatrix mixed with other factors."""
    pi, ki, pf = tensor_heads("pi, ki, pf", [LorentzIndex])
    i0, i1, i2, i3, i4 = tensor_indices("i0:5", LorentzIndex)
    x = Symbol("x")
    pis = pi(i2) * G(-i2)
    kis = ki(i3) * G(-i3)
    pfs = pf(i4) * G(-i4)

    a = pfs * G(i0) * kis * G(i1) * pis * G(-i1) * kis * G(-i0)
    b = pfs * G(i0) * kis * G(i1) * pis * x * G(-i0) * pi(-i1)
    ta = gamma_trace(a)
    tb = gamma_trace(b)
    t_a_plus_b = gamma_trace(a + b)
    assert ta == 4 * (
        -4 * ki(i0) * ki(-i0) * pf(i1) * pi(-i1)
        + 8 * ki(i0) * ki(i1) * pf(-i0) * pi(-i1)
    )
    assert tb == -8 * x * ki(i0) * pf(-i0) * pi(i1) * pi(-i1)
    assert t_a_plus_b == ta + tb
