"""
Algorithms for solving Parametric Risch Differential Equations.

The methods used for solving Parametric Risch Differential Equations parallel
those for solving Risch Differential Equations.  See the outline in the
docstring of rde.py for more information.

The Parametric Risch Differential Equation problem is, given f, g1, ..., gm in
K(t), to determine if there exist y in K(t) and c1, ..., cm in Const(K) such
that Dy + f*y == Sum(ci*gi, (i, 1, m)), and to find such y and ci if they exist.

For the algorithms here G is a list of tuples of factions of the terms on the
right hand side of the equation (i.e., gi in k(t)), and Q is a list of terms on
the right hand side of the equation (i.e., qi in k[t]).  See the docstring of
each function for more information.
"""
import itertools
from functools import reduce

from sympy.core.intfunc import ilcm
from sympy.core import Dummy, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
    bound_degree)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
    residue_reduce, splitfactor, residue_reduce_derivation, DecrementLevel,
    recognize_log_derivative)
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve

zeros = Matrix.zeros
eye = Matrix.eye


def prde_normal_denom(fa, fd, G, DE):
    """
    Parametric Risch Differential Equation - Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g1, ..., gm in k(t) with f weakly
    normalized with respect to t, return the tuple (a, b, G, h) such that
    a, h in k[t], b in k<t>, G = [g1, ..., gm] in k(t)^m, and for any solution
    c1, ..., cm in Const(k) and y in k(t) of Dy + f*y == Sum(ci*gi, (i, 1, m)),
    q == y*h in k<t> satisfies a*Dq + b*q == Sum(ci*Gi, (i, 1, m)).
    """
    dn, ds = splitfactor(fd, DE)
    Gas, Gds = list(zip(*G))
    gd = reduce(lambda i, j: i.lcm(j), Gds, Poly(1, DE.t))
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    G = [(c*A).cancel(D, include=True) for A, D in G]

    return (a, (ba, bd), G, h)

def real_imag(ba, bd, gen):
    """
    Helper function, to get the real and imaginary part of a rational function
    evaluated at sqrt(-1) without actually evaluating it at sqrt(-1).

    Explanation
    ===========

    Separates the even and odd power terms by checking the degree of terms wrt
    mod 4. Returns a tuple (ba[0], ba[1], bd) where ba[0] is real part
    of the numerator ba[1] is the imaginary part and bd is the denominator
    of the rational function.
    """
    bd = bd.as_poly(gen).as_dict()
    ba = ba.as_poly(gen).as_dict()
    denom_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in bd.items()]
    denom_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in bd.items()]
    bd_real = sum(r for r in denom_real)
    bd_imag = sum(r for r in denom_imag)
    num_real = [value if key[0] % 4 == 0 else -value if key[0] % 4 == 2 else 0 for key, value in ba.items()]
    num_imag = [value if key[0] % 4 == 1 else -value if key[0] % 4 == 3 else 0 for key, value in ba.items()]
    ba_real = sum(r for r in num_real)
    ba_imag = sum(r for r in num_imag)
    ba = ((ba_real*bd_real + ba_imag*bd_imag).as_poly(gen), (ba_imag*bd_real - ba_real*bd_imag).as_poly(gen))
    bd = (bd_real*bd_real + bd_imag*bd_imag).as_poly(gen)
    return (ba[0], ba[1], bd)


def prde_special_denom(a, ba, bd, G, DE, case='auto'):
    """
    Parametric Risch Differential Equation - Special part of the denominator.

    Explanation
    ===========

    Case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the hyperexponential
    (resp. hypertangent) case, given a derivation D on k[t] and a in k[t],
    b in k<t>, and g1, ..., gm in k(t) with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the tuple (A, B, GG, h) such that A, B, h in
    k[t], GG = [gg1, ..., ggm] in k(t)^m, and for any solution c1, ..., cm in
    Const(k) and q in k<t> of a*Dq + b*q == Sum(ci*gi, (i, 1, m)), r == q*h in
    k[t] satisfies A*Dr + B*r == Sum(ci*ggi, (i, 1, m)).

    For case == 'primitive', k<t> == k[t], so it returns (a, b, G, 1) in this
    case.
    """
    # TODO: Merge this with the very similar special_denom() in rde.py
    if case == 'auto':
        case = DE.case

    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t**2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.quo(bd)
        return (a, B, G, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base'}, not %s." % case)

    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = min(order_at(Ga, p, DE.t) - order_at(Gd, p, DE.t) for Ga, Gd in G)
    n = min(0, nc - min(0, nb))
    if not nb:
        # Possible cancellation.
        if case == 'exp':
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    if Q == 1:
                        n = min(n, m)

        elif case == 'tan':
            dcoeff = DE.d.quo(Poly(DE.t**2 + 1, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                betaa, alphaa, alphad =  real_imag(ba, bd*a, DE.t)
                betad = alphad
                etaa, etad = frac_in(dcoeff, DE.t)
                if recognize_log_derivative(Poly(2, DE.t)*betaa, betad, DE):
                    A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                    B = parametric_log_deriv(betaa, betad, etaa, etad, DE)
                    if A is not None and B is not None:
                        Q, s, z = A
                        # TODO: Add test
                        if Q == 1:
                            n = min(n, s/2)

    N = max(0, -nb)
    pN = p**N
    pn = p**-n  # This is 1/h

    A = a*pN
    B = ba*pN.quo(bd) + Poly(n, DE.t)*a*derivation(p, DE).quo(p)*pN
    G = [(Ga*pN*pn).cancel(Gd, include=True) for Ga, Gd in G]
    h = pn

    # (a*p**N, (b + n*a*Dp/p)*p**N, g1*p**(N - n), ..., gm*p**(N - n), p**-n)
    return (A, B, G, h)


def prde_linear_constraints(a, b, G, DE):
    """
    Parametric Risch Differential Equation - Generate linear constraints on the constants.

    Explanation
    ===========

    Given a derivation D on k[t], a, b, in k[t] with gcd(a, b) == 1, and
    G = [g1, ..., gm] in k(t)^m, return Q = [q1, ..., qm] in k[t]^m and a
    matrix M with entries in k(t) such that for any solution c1, ..., cm in
    Const(k) and p in k[t] of a*Dp + b*p == Sum(ci*gi, (i, 1, m)),
    (c1, ..., cm) is a solution of Mx == 0, and p and the ci satisfy
    a*Dp + b*p == Sum(ci*qi, (i, 1, m)).

    Because M has entries in k(t), and because Matrix does not play well with
    Poly, M will be a Matrix of Basic expressions.
    """
    m = len(G)

    Gns, Gds = list(zip(*G))
    d = reduce(lambda i, j: i.lcm(j), Gds)
    d = Poly(d, field=True)
    Q = [(ga*(d).quo(gd)).div(d) for ga, gd in G]

    if not all(ri.is_zero for _, ri in Q):
        N = max(ri.degree(DE.t) for _, ri in Q)
        M = Matrix(N + 1, m, lambda i, j: Q[j][1].nth(i), DE.t)
    else:
        M = Matrix(0, m, [], DE.t)  # No constraints, return the empty matrix.

    qs, _ = list(zip(*Q))
    return (qs, M)

def poly_linear_constraints(p, d):
    """
    Given p = [p1, ..., pm] in k[t]^m and d in k[t], return
    q = [q1, ..., qm] in k[t]^m and a matrix M with entries in k such
    that Sum(ci*pi, (i, 1, m)), for c1, ..., cm in k, is divisible
    by d if and only if (c1, ..., cm) is a solution of Mx = 0, in
    which case the quotient is Sum(ci*qi, (i, 1, m)).
    """
    m = len(p)
    q, r = zip(*[pi.div(d) for pi in p])

    if not all(ri.is_zero for ri in r):
        n = max(ri.degree() for ri in r)
        M = Matrix(n + 1, m, lambda i, j: r[j].nth(i), d.gens)
    else:
        M = Matrix(0, m, [], d.gens)  # No constraints.

    return q, M

def constant_system(A, u, DE):
    """
    Generate a system for the constant solutions.

    Explanation
    ===========

    Given a differential field (K, D) with constant field C = Const(K), a Matrix
    A, and a vector (Matrix) u with coefficients in K, returns the tuple
    (B, v, s), where B is a Matrix with coefficients in C and v is a vector
    (Matrix) such that either v has coefficients in C, in which case s is True
    and the solutions in C of Ax == u are exactly all the solutions of Bx == v,
    or v has a non-constant coefficient, in which case s is False Ax == u has no
    constant solution.

    This algorithm is used both in solving parametric problems and in
    determining if an element a of K is a derivative of an element of K or the
    logarithmic derivative of a K-radical using the structure theorem approach.

    Because Poly does not play well with Matrix yet, this algorithm assumes that
    all matrix entries are Basic expressions.
    """
    if not A:
        return A, u
    Au = A.row_join(u)
    Au, _ = Au.rref()
    # Warning: This will NOT return correct results if cancel() cannot reduce
    # an identically zero expression to 0.  The danger is that we might
    # incorrectly prove that an integral is nonelementary (such as
    # risch_integrate(exp((sin(x)**2 + cos(x)**2 - 1)*x**2), x).
    # But this is a limitation in computer algebra in general, and implicit
    # in the correctness of the Risch Algorithm is the computability of the
    # constant field (actually, this same correctness problem exists in any
    # algorithm that uses rref()).
    #
    # We therefore limit ourselves to constant fields that are computable
    # via the cancel() function, in order to prevent a speed bottleneck from
    # calling some more complex simplification function (rational function
    # coefficients will fall into this class).  Furthermore, (I believe) this
    # problem will only crop up if the integral explicitly contains an
    # expression in the constant field that is identically zero, but cannot
    # be reduced to such by cancel().  Therefore, a careful user can avoid this
    # problem entirely by being careful with the sorts of expressions that
    # appear in his integrand in the variables other than the integration
    # variable (the structure theorems should be able to completely decide these
    # problems in the integration variable).

    A, u = Au[:, :-1], Au[:, -1]

    D = lambda x: derivation(x, DE, basic=True)

    for j, i in itertools.product(range(A.cols), range(A.rows)):
        if A[i, j].expr.has(*DE.T):
            # This assumes that const(F(t0, ..., tn) == const(K) == F
            Ri = A[i, :]
            # Rm+1; m = A.rows
            DAij = D(A[i, j])
            Rm1 = Ri.applyfunc(lambda x: D(x) / DAij)
            um1 = D(u[i]) / DAij

            Aj = A[:, j]
            A = A - Aj * Rm1
            u = u - Aj * um1

            A = A.col_join(Rm1)
            u = u.col_join(Matrix([um1], u.gens))

    return (A, u)


def prde_spde(a, b, Q, n, DE):
    """
    Special Polynomial Differential Equation algorithm: Parametric Version.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n, and a, b, q1, ..., qm in k[t]
    with deg(a) > 0 and gcd(a, b) == 1, return (A, B, Q, R, n1), with
    Qq = [q1, ..., qm] and R = [r1, ..., rm], such that for any solution
    c1, ..., cm in Const(k) and q in k[t] of degree at most n of
    a*Dq + b*q == Sum(ci*gi, (i, 1, m)), p = (q - Sum(ci*ri, (i, 1, m)))/a has
    degree at most n1 and satisfies A*Dp + B*p == Sum(ci*qi, (i, 1, m))
    """
    R, Z = list(zip(*[gcdex_diophantine(b, a, qi) for qi in Q]))

    A = a
    B = b + derivation(a, DE)
    Qq = [zi - derivation(ri, DE) for ri, zi in zip(R, Z)]
    R = list(R)
    n1 = n - a.degree(DE.t)

    return (A, B, Qq, R, n1)


def prde_no_cancel_b_large(b, Q, n, DE):
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], n in ZZ, and b, q1, ..., qm in k[t] with
    b != 0 and either D == d/dt or deg(b) > max(0, deg(D) - 1), returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k) such that
    if c1, ..., cm in Const(k) and q in k[t] satisfy deg(q) <= n and
    Dq + b*q == Sum(ci*qi, (i, 1, m)), then q = Sum(dj*hj, (j, 1, r)), where
    d1, ..., dr in Const(k) and A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0.
    """
    db = b.degree(DE.t)
    m = len(Q)
    H = [Poly(0, DE.t)]*m

    for N, i in itertools.product(range(n, -1, -1), range(m)):  # [n, ..., 0]
        si = Q[i].nth(N + db)/b.LC()
        sitn = Poly(si*DE.t**N, DE.t)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    if all(qi.is_zero for qi in Q):
        dc = -1
    else:
        dc = max(qi.degree(DE.t) for qi in Q)
    M = Matrix(dc + 1, m, lambda i, j: Q[j].nth(i), DE.t)
    A, u = constant_system(M, zeros(dc + 1, 1, DE.t), DE)
    c = eye(m, DE.t)
    A = A.row_join(zeros(A.rows, m, DE.t)).col_join(c.row_join(-c))

    return (H, A)


def prde_no_cancel_b_small(b, Q, n, DE):
    """
    Parametric Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], n in ZZ, and b, q1, ..., qm in k[t] with
    deg(b) < deg(D) - 1 and either D == d/dt or deg(D) >= 2, returns
    h1, ..., hr in k[t] and a matrix A with coefficients in Const(k) such that
    if c1, ..., cm in Const(k) and q in k[t] satisfy deg(q) <= n and
    Dq + b*q == Sum(ci*qi, (i, 1, m)) then q = Sum(dj*hj, (j, 1, r)) where
    d1, ..., dr in Const(k) and A*Matrix([[c1, ..., cm, d1, ..., dr]]).T == 0.
    """
    m = len(Q)
    H = [Poly(0, DE.t)]*m

    for N, i in itertools.product(range(n, 0, -1), range(m)):  # [n, ..., 1]
        si = Q[i].nth(N + DE.d.degree(DE.t) - 1)/(N*DE.d.LC())
        sitn = Poly(si*DE.t**N, DE.t)
        H[i] = H[i] + sitn
        Q[i] = Q[i] - derivation(sitn, DE) - b*sitn

    if b.degree(DE.t) > 0:
        for i in range(m):
            si = Poly(Q[i].nth(b.degree(DE.t))/b.LC(), DE.t)
            H[i] = H[i] + si
            Q[i] = Q[i] - derivation(si, DE) - b*si
        if all(qi.is_zero for qi in Q):
            dc = -1
        else:
            dc = max(qi.degree(DE.t) for qi in Q)
        M = Matrix(dc + 1, m, lambda i, j: Q[j].nth(i), DE.t)
        A, u = constant_system(M, zeros(dc + 1, 1, DE.t), DE)
        c = eye(m, DE.t)
        A = A.row_join(zeros(A.rows, m, DE.t)).col_join(c.row_join(-c))
        return (H, A)

    # else: b is in k, deg(qi) < deg(Dt)

    t = DE.t
    if DE.case != 'base':
        with DecrementLevel(DE):
            t0 = DE.t  # k = k0(t0)
            ba, bd = frac_in(b, t0, field=True)
            Q0 = [frac_in(qi.TC(), t0, field=True) for qi in Q]
            f, B = param_rischDE(ba, bd, Q0, DE)

            # f = [f1, ..., fr] in k^r and B is a matrix with
            # m + r columns and entries in Const(k) = Const(k0)
            # such that Dy0 + b*y0 = Sum(ci*qi, (i, 1, m)) has
            # a solution y0 in k with c1, ..., cm in Const(k)
            # if and only y0 = Sum(dj*fj, (j, 1, r)) where
            # d1, ..., dr ar in Const(k) and
            # B*Matrix([c1, ..., cm, d1, ..., dr]) == 0.

        # Transform fractions (fa, fd) in f into constant
        # polynomials fa/fd in k[t].
        # (Is there a better way?)
        f = [Poly(fa.as_expr()/fd.as_expr(), t, field=True)
             for fa, fd in f]
        B = Matrix.from_Matrix(B.to_Matrix(), t)
    else:
        # Base case. Dy == 0 for all y in k and b == 0.
        # Dy + b*y = Sum(ci*qi) is solvable if and only if
        # Sum(ci*qi) == 0 in which case the solutions are
        # y = d1*f1 for f1 = 1 and any d1 in Const(k) = k.

        f = [Poly(1, t, field=True)]  # r = 1
        B = Matrix([[qi.TC() for qi in Q] + [S.Zero]], DE.t)
        # The condition for solvability is
        # B*Matrix([c1, ..., cm, d1]) == 0
        # There are no constraints on d1.

    # Coefficients of t^j (j > 0) in Sum(ci*qi) must be zero.
    d = max(qi.degree(DE.t) for qi in Q)
    if d > 0:
        M = Matrix(d, m, lambda i, j: Q[j].nth(i + 1), DE.t)
        A, _ = constant_system(M, zeros(d, 1, DE.t), DE)
    else:
        # No constraints on the hj.
        A = Matrix(0, m, [], DE.t)

    # Solutions of the original equation are
    #    y = Sum(dj*fj, (j, 1, r) + Sum(ei*hi, (i, 1, m)),
    # where  ei == ci  (i = 1, ..., m),  when
    # A*Matrix([c1, ..., cm]) == 0 and
    # B*Matrix([c1, ..., cm, d1, ..., dr]) == 0

    # Build combined constraint matrix with m + r + m columns.

    r = len(f)
    I = eye(m, DE.t)
    A = A.row_join(zeros(A.rows, r + m, DE.t))
    B = B.row_join(zeros(B.rows, m, DE.t))
    C = I.row_join(zeros(m, r, DE.t)).row_join(-I)

    return f + H, A.col_join(B).col_join(C)


def prde_cancel_liouvillian(b, Q, n, DE):
    """
    Pg, 237.
    """
    H = []

    # Why use DecrementLevel? Below line answers that:
    # Assuming that we can solve such problems over 'k' (not k[t])
    if DE.case == 'primitive':
        with DecrementLevel(DE):
            ba, bd = frac_in(b, DE.t, field=True)

    for i in range(n, -1, -1):
        if DE.case == 'exp': # this re-checking can be avoided
            with DecrementLevel(DE):
                ba, bd = frac_in(b + (i*(derivation(DE.t, DE)/DE.t)).as_poly(b.gens),
                                DE.t, field=True)
        with DecrementLevel(DE):
            Qy = [frac_in(q.nth(i), DE.t, field=True) for q in Q]
            fi, Ai = param_rischDE(ba, bd, Qy, DE)
        fi = [Poly(fa.as_expr()/fd.as_expr(), DE.t, field=True)
                for fa, fd in fi]
        Ai = Ai.set_gens(DE.t)

        ri = len(fi)

        if i == n:
            M = Ai
        else:
            M = Ai.col_join(M.row_join(zeros(M.rows, ri, DE.t)))

        Fi, hi = [None]*ri, [None]*ri

        # from eq. on top of p.238 (unnumbered)
        for j in range(ri):
            hji = fi[j] * (DE.t**i).as_poly(fi[j].gens)
            hi[j] = hji
            # building up Sum(djn*(D(fjn*t^n) - b*fjnt^n))
            Fi[j] = -(derivation(hji, DE) - b*hji)

        H += hi
        # in the next loop instead of Q it has
        # to be Q + Fi taking its place
        Q = Q + Fi

    return (H, M)


def param_poly_rischDE(a, b, q, n, DE):
    """Polynomial solutions of a parametric Risch differential equation.

    Explanation
    ===========

    Given a derivation D in k[t], a, b in k[t] relatively prime, and q
    = [q1, ..., qm] in k[t]^m, return h = [h1, ..., hr] in k[t]^r and
    a matrix A with m + r columns and entries in Const(k) such that
    a*Dp + b*p = Sum(ci*qi, (i, 1, m)) has a solution p of degree <= n
    in k[t] with c1, ..., cm in Const(k) if and only if p = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.
    """
    m = len(q)
    if n < 0:
        # Only the trivial zero solution is possible.
        # Find relations between the qi.
        if all(qi.is_zero for qi in q):
            return [], zeros(1, m, DE.t)  # No constraints.

        N = max(qi.degree(DE.t) for qi in q)
        M = Matrix(N + 1, m, lambda i, j: q[j].nth(i), DE.t)
        A, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)

        return [], A

    if a.is_ground:
        # Normalization: a = 1.
        a = a.LC()
        b, q = b.to_field().exquo_ground(a), [qi.to_field().exquo_ground(a) for qi in q]

        if not b.is_zero and (DE.case == 'base' or
                b.degree() > max(0, DE.d.degree() - 1)):
            return prde_no_cancel_b_large(b, q, n, DE)

        elif ((b.is_zero or b.degree() < DE.d.degree() - 1)
                and (DE.case == 'base' or DE.d.degree() >= 2)):
            return prde_no_cancel_b_small(b, q, n, DE)

        elif (DE.d.degree() >= 2 and
              b.degree() == DE.d.degree() - 1 and
              n > -b.as_poly().LC()/DE.d.as_poly().LC()):
            raise NotImplementedError("prde_no_cancel_b_equal() is "
                "not yet implemented.")

        else:
            # Liouvillian cases
            if DE.case in ('primitive', 'exp'):
                return prde_cancel_liouvillian(b, q, n, DE)
            else:
                raise NotImplementedError("non-linear and hypertangent "
                        "cases have not yet been implemented")

    # else: deg(a) > 0

    # Iterate SPDE as long as possible cumulating coefficient
    # and terms for the recovery of original solutions.
    alpha, beta = a.one, [a.zero]*m
    while n >= 0:  # and a, b relatively prime
        a, b, q, r, n = prde_spde(a, b, q, n, DE)
        beta = [betai + alpha*ri for betai, ri in zip(beta, r)]
        alpha *= a
        # Solutions p of a*Dp + b*p = Sum(ci*qi) correspond to
        # solutions alpha*p + Sum(ci*betai) of the initial equation.
        d = a.gcd(b)
        if not d.is_ground:
            break

    # a*Dp + b*p = Sum(ci*qi) may have a polynomial solution
    # only if the sum is divisible by d.

    qq, M = poly_linear_constraints(q, d)
    # qq = [qq1, ..., qqm] where qqi = qi.quo(d).
    # M is a matrix with m columns an entries in k.
    # Sum(fi*qi, (i, 1, m)), where f1, ..., fm are elements of k, is
    # divisible by d if and only if M*Matrix([f1, ..., fm]) == 0,
    # in which case the quotient is Sum(fi*qqi).

    A, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)
    # A is a matrix with m columns and entries in Const(k).
    # Sum(ci*qqi) is Sum(ci*qi).quo(d), and the remainder is zero
    # for c1, ..., cm in Const(k) if and only if
    # A*Matrix([c1, ...,cm]) == 0.

    V = A.nullspace()
    # V = [v1, ..., vu] where each vj is a column matrix with
    # entries aj1, ..., ajm in Const(k).
    # Sum(aji*qi) is divisible by d with exact quotient Sum(aji*qqi).
    # Sum(ci*qi) is divisible by d if and only if ci = Sum(dj*aji)
    # (i = 1, ..., m) for some d1, ..., du in Const(k).
    # In that case, solutions of
    #     a*Dp + b*p = Sum(ci*qi) = Sum(dj*Sum(aji*qi))
    # are the same as those of
    #     (a/d)*Dp + (b/d)*p = Sum(dj*rj)
    # where rj = Sum(aji*qqi).

    if not V:  # No non-trivial solution.
        return [], eye(m, DE.t)  # Could return A, but this has
                                 # the minimum number of rows.

    Mqq = Matrix([qq])  # A single row.
    r = [(Mqq*vj)[0] for vj in V]  # [r1, ..., ru]

    # Solutions of (a/d)*Dp + (b/d)*p = Sum(dj*rj) correspond to
    # solutions alpha*p + Sum(Sum(dj*aji)*betai) of the initial
    # equation. These are equal to alpha*p + Sum(dj*fj) where
    # fj = Sum(aji*betai).
    Mbeta = Matrix([beta])
    f = [(Mbeta*vj)[0] for vj in V]  # [f1, ..., fu]

    #
    # Solve the reduced equation recursively.
    #
    g, B = param_poly_rischDE(a.quo(d), b.quo(d), r, n, DE)

    # g = [g1, ..., gv] in k[t]^v and and B is a matrix with u + v
    # columns and entries in Const(k) such that
    # (a/d)*Dp + (b/d)*p = Sum(dj*rj) has a solution p of degree <= n
    # in k[t] if and only if p = Sum(ek*gk) where e1, ..., ev are in
    # Const(k) and B*Matrix([d1, ..., du, e1, ..., ev]) == 0.
    # The solutions of the original equation are then
    # Sum(dj*fj, (j, 1, u)) + alpha*Sum(ek*gk, (k, 1, v)).

    # Collect solution components.
    h = f + [alpha*gk for gk in g]

    # Build combined relation matrix.
    A = -eye(m, DE.t)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(g), DE.t))
    A = A.col_join(zeros(B.rows, m, DE.t).row_join(B))

    return h, A


def param_rischDE(fa, fd, G, DE):
    """
    Solve a Parametric Risch Differential Equation: Dy + f*y == Sum(ci*Gi, (i, 1, m)).

    Explanation
    ===========

    Given a derivation D in k(t), f in k(t), and G
    = [G1, ..., Gm] in k(t)^m, return h = [h1, ..., hr] in k(t)^r and
    a matrix A with m + r columns and entries in Const(k) such that
    Dy + f*y = Sum(ci*Gi, (i, 1, m)) has a solution y
    in k(t) with c1, ..., cm in Const(k) if and only if y = Sum(dj*hj,
    (j, 1, r)) where d1, ..., dr are in Const(k) and (c1, ..., cm,
    d1, ..., dr) is a solution of Ax == 0.

    Elements of k(t) are tuples (a, d) with a and d in k[t].
    """
    m = len(G)
    q, (fa, fd) = weak_normalizer(fa, fd, DE)
    # Solutions of the weakly normalized equation Dz + f*z = q*Sum(ci*Gi)
    # correspond to solutions y = z/q of the original equation.
    gamma = q
    G = [(q*ga).cancel(gd, include=True) for ga, gd in G]

    a, (ba, bd), G, hn = prde_normal_denom(fa, fd, G, DE)
    # Solutions q in k<t> of  a*Dq + b*q = Sum(ci*Gi) correspond
    # to solutions z = q/hn of the weakly normalized equation.
    gamma *= hn

    A, B, G, hs = prde_special_denom(a, ba, bd, G, DE)
    # Solutions p in k[t] of  A*Dp + B*p = Sum(ci*Gi) correspond
    # to solutions q = p/hs of the previous equation.
    gamma *= hs

    g = A.gcd(B)
    a, b, g = A.quo(g), B.quo(g), [gia.cancel(gid*g, include=True) for
        gia, gid in G]

    # a*Dp + b*p = Sum(ci*gi)  may have a polynomial solution
    # only if the sum is in k[t].

    q, M = prde_linear_constraints(a, b, g, DE)

    # q = [q1, ..., qm] where qi in k[t] is the polynomial component
    # of the partial fraction expansion of gi.
    # M is a matrix with m columns and entries in k.
    # Sum(fi*gi, (i, 1, m)), where f1, ..., fm are elements of k,
    # is a polynomial if and only if M*Matrix([f1, ..., fm]) == 0,
    # in which case the sum is equal to Sum(fi*qi).

    M, _ = constant_system(M, zeros(M.rows, 1, DE.t), DE)
    # M is a matrix with m columns and entries in Const(k).
    # Sum(ci*gi) is in k[t] for c1, ..., cm in Const(k)
    # if and only if M*Matrix([c1, ..., cm]) == 0,
    # in which case the sum is Sum(ci*qi).

    ## Reduce number of constants at this point

    V = M.nullspace()
    # V = [v1, ..., vu] where each vj is a column matrix with
    # entries aj1, ..., ajm in Const(k).
    # Sum(aji*gi) is in k[t] and equal to Sum(aji*qi) (j = 1, ..., u).
    # Sum(ci*gi) is in k[t] if and only is ci = Sum(dj*aji)
    # (i = 1, ..., m) for some d1, ..., du in Const(k).
    # In that case,
    #     Sum(ci*gi) = Sum(ci*qi) = Sum(dj*Sum(aji*qi)) = Sum(dj*rj)
    # where rj = Sum(aji*qi) (j = 1, ..., u) in k[t].

    if not V:  # No non-trivial solution
        return [], eye(m, DE.t)

    Mq = Matrix([q])  # A single row.
    r = [(Mq*vj)[0] for vj in V]  # [r1, ..., ru]

    # Solutions of a*Dp + b*p = Sum(dj*rj) correspond to solutions
    # y = p/gamma of the initial equation with ci = Sum(dj*aji).

    try:
        # We try n=5. At least for prde_spde, it will always
        # terminate no matter what n is.
        n = bound_degree(a, b, r, DE, parametric=True)
    except NotImplementedError:
        # A temporary bound is set. Eventually, it will be removed.
        # the currently added test case takes large time
        # even with n=5, and much longer with large n's.
        n = 5

    h, B = param_poly_rischDE(a, b, r, n, DE)

    # h = [h1, ..., hv] in k[t]^v and and B is a matrix with u + v
    # columns and entries in Const(k) such that
    # a*Dp + b*p = Sum(dj*rj) has a solution p of degree <= n
    # in k[t] if and only if p = Sum(ek*hk) where e1, ..., ev are in
    # Const(k) and B*Matrix([d1, ..., du, e1, ..., ev]) == 0.
    # The solutions of the original equation for ci = Sum(dj*aji)
    # (i = 1, ..., m) are then y = Sum(ek*hk, (k, 1, v))/gamma.

    ## Build combined relation matrix with m + u + v columns.

    A = -eye(m, DE.t)
    for vj in V:
        A = A.row_join(vj)
    A = A.row_join(zeros(m, len(h), DE.t))
    A = A.col_join(zeros(B.rows, m, DE.t).row_join(B))

    ## Eliminate d1, ..., du.

    W = A.nullspace()

    # W = [w1, ..., wt] where each wl is a column matrix with
    # entries blk (k = 1, ..., m + u + v) in Const(k).
    # The vectors (bl1, ..., blm) generate the space of those
    # constant families (c1, ..., cm) for which a solution of
    # the equation Dy + f*y == Sum(ci*Gi) exists. They generate
    # the space and form a basis except possibly when Dy + f*y == 0
    # is solvable in k(t}. The corresponding solutions are
    # y = Sum(blk'*hk, (k, 1, v))/gamma, where k' = k + m + u.

    v = len(h)
    shape = (len(W), m+v)
    elements = [wl[:m] + wl[-v:] for wl in W] # excise dj's.
    items = [e for row in elements for e in row]

    # Need to set the shape in case W is empty
    M = Matrix(*shape, items, DE.t)
    N = M.nullspace()

    # N = [n1, ..., ns] where the ni in Const(k)^(m + v) are column
    # vectors generating the space of linear relations between
    # c1, ..., cm, e1, ..., ev.

    C = Matrix([ni[:] for ni in N], DE.t)  # rows n1, ..., ns.

    return [hk.cancel(gamma, include=True) for hk in h], C


def limited_integrate_reduce(fa, fd, G, DE):
    """
    Simpler version of step 1 & 2 for the limited integration problem.

    Explanation
    ===========

    Given a derivation D on k(t) and f, g1, ..., gn in k(t), return
    (a, b, h, N, g, V) such that a, b, h in k[t], N is a non-negative integer,
    g in k(t), V == [v1, ..., vm] in k(t)^m, and for any solution v in k(t),
    c1, ..., cm in C of f == Dv + Sum(ci*wi, (i, 1, m)), p = v*h is in k<t>, and
    p and the ci satisfy a*Dp + b*p == g + Sum(ci*vi, (i, 1, m)).  Furthermore,
    if S1irr == Sirr, then p is in k[t], and if t is nonlinear or Liouvillian
    over k, then deg(p) <= N.

    So that the special part is always computed, this function calls the more
    general prde_special_denom() automatically if it cannot determine that
    S1irr == Sirr.  Furthermore, it will automatically call bound_degree() when
    t is linear and non-Liouvillian, which for the transcendental case, implies
    that Dt == a*t + b with for some a, b in k*.
    """
    dn, ds = splitfactor(fd, DE)
    E = [splitfactor(gd, DE) for _, gd in G]
    En, Es = list(zip(*E))
    c = reduce(lambda i, j: i.lcm(j), (dn,) + En)  # lcm(dn, en1, ..., enm)
    hn = c.gcd(c.diff(DE.t))
    a = hn
    b = -derivation(hn, DE)
    N = 0

    # These are the cases where we know that S1irr = Sirr, but there could be
    # others, and this algorithm will need to be extended to handle them.
    if DE.case in ('base', 'primitive', 'exp', 'tan'):
        hs = reduce(lambda i, j: i.lcm(j), (ds,) + Es)  # lcm(ds, es1, ..., esm)
        a = hn*hs
        b -= (hn*derivation(hs, DE)).quo(hs)
        mu = min(order_at_oo(fa, fd, DE.t), min(order_at_oo(ga, gd, DE.t) for
            ga, gd in G))
        # So far, all the above are also nonlinear or Liouvillian, but if this
        # changes, then this will need to be updated to call bound_degree()
        # as per the docstring of this function (DE.case == 'other_linear').
        N = hn.degree(DE.t) + hs.degree(DE.t) + max(0, 1 - DE.d.degree(DE.t) - mu)
    else:
        # TODO: implement this
        raise NotImplementedError

    V = [(-a*hn*ga).cancel(gd, include=True) for ga, gd in G]
    return (a, b, a, N, (a*hn*fa).cancel(fd, include=True), V)


def limited_integrate(fa, fd, G, DE):
    """
    Solves the limited integration problem:  f = Dv + Sum(ci*wi, (i, 1, n))
    """
    fa, fd = fa*Poly(1/fd.LC(), DE.t), fd.monic()
    # interpreting limited integration problem as a
    # parametric Risch DE problem
    Fa = Poly(0, DE.t)
    Fd = Poly(1, DE.t)
    G = [(fa, fd)] + G
    h, A = param_rischDE(Fa, Fd, G, DE)
    V = A.nullspace()
    V = [v for v in V if v[0] != 0]
    if not V:
        return None
    else:
        # we can take any vector from V, we take V[0]
        c0 = V[0][0]
        # v = [-1, c1, ..., cm, d1, ..., dr]
        v = V[0]/(-c0)
        r = len(h)
        m = len(v) - r - 1
        C = list(v[1: m + 1])
        y = -sum(v[m + 1 + i]*h[i][0].as_expr()/h[i][1].as_expr() \
                for i in range(r))
        y_num, y_den = y.as_numer_denom()
        Ya, Yd = Poly(y_num, DE.t), Poly(y_den, DE.t)
        Y = Ya*Poly(1/Yd.LC(), DE.t), Yd.monic()
        return Y, C


def parametric_log_deriv_heu(fa, fd, wa, wd, DE, c1=None):
    """
    Parametric logarithmic derivative heuristic.

    Explanation
    ===========

    Given a derivation D on k[t], f in k(t), and a hyperexponential monomial
    theta over k(t), raises either NotImplementedError, in which case the
    heuristic failed, or returns None, in which case it has proven that no
    solution exists, or returns a solution (n, m, v) of the equation
    n*f == Dv/v + m*Dtheta/theta, with v in k(t)* and n, m in ZZ with n != 0.

    If this heuristic fails, the structure theorem approach will need to be
    used.

    The argument w == Dtheta/theta
    """
    # TODO: finish writing this and write tests
    c1 = c1 or Dummy('c1')

    p, a = fa.div(fd)
    q, b = wa.div(wd)

    B = max(0, derivation(DE.t, DE).degree(DE.t) - 1)
    C = max(p.degree(DE.t), q.degree(DE.t))

    if q.degree(DE.t) > B:
        eqs = [p.nth(i) - c1*q.nth(i) for i in range(B + 1, C + 1)]
        s = solve(eqs, c1)
        if not s or not s[c1].is_Rational:
            # deg(q) > B, no solution for c.
            return None

        M, N = s[c1].as_numer_denom()
        M_poly = M.as_poly(q.gens)
        N_poly = N.as_poly(q.gens)

        nfmwa = N_poly*fa*wd - M_poly*wa*fd
        nfmwd = fd*wd
        Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE, 'auto')
        if Qv is None:
            # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
            return None

        Q, v = Qv

        if Q.is_zero or v.is_zero:
            return None

        return (Q*N, Q*M, v)

    if p.degree(DE.t) > B:
        return None

    c = lcm(fd.as_poly(DE.t).LC(), wd.as_poly(DE.t).LC())
    l = fd.monic().lcm(wd.monic())*Poly(c, DE.t)
    ln, ls = splitfactor(l, DE)
    z = ls*ln.gcd(ln.diff(DE.t))

    if not z.has(DE.t):
        # TODO: We treat this as 'no solution', until the structure
        # theorem version of parametric_log_deriv is implemented.
        return None

    u1, r1 = (fa*l.quo(fd)).div(z)  # (l*f).div(z)
    u2, r2 = (wa*l.quo(wd)).div(z)  # (l*w).div(z)

    eqs = [r1.nth(i) - c1*r2.nth(i) for i in range(z.degree(DE.t))]
    s = solve(eqs, c1)
    if not s or not s[c1].is_Rational:
        # deg(q) <= B, no solution for c.
        return None

    M, N = s[c1].as_numer_denom()

    nfmwa = N.as_poly(DE.t)*fa*wd - M.as_poly(DE.t)*wa*fd
    nfmwd = fd*wd
    Qv = is_log_deriv_k_t_radical_in_field(nfmwa, nfmwd, DE)
    if Qv is None:
        # (N*f - M*w) is not the logarithmic derivative of a k(t)-radical.
        return None

    Q, v = Qv

    if Q.is_zero or v.is_zero:
        return None

    return (Q*N, Q*M, v)


def parametric_log_deriv(fa, fd, wa, wd, DE):
    # TODO: Write the full algorithm using the structure theorems.
#    try:
    A = parametric_log_deriv_heu(fa, fd, wa, wd, DE)
#    except NotImplementedError:
        # Heuristic failed, we have to use the full method.
        # TODO: This could be implemented more efficiently.
        # It isn't too worrisome, because the heuristic handles most difficult
        # cases.
    return A


def is_deriv_k(fa, fd, DE):
    r"""
    Checks if Df/f is the derivative of an element of k(t).

    Explanation
    ===========

    a in k(t) is the derivative of an element of k(t) if there exists b in k(t)
    such that a = Db.  Either returns (ans, u), such that Df/f == Du, or None,
    which means that Df/f is not the derivative of an element of k(t).  ans is
    a list of tuples such that Add(*[i*j for i, j in ans]) == u.  This is useful
    for seeing exactly which elements of k(t) produce u.

    This function uses the structure theorem approach, which says that for any
    f in K, Df/f is the derivative of a element of K if and only if there are ri
    in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i      Df
            /     i     i     /     i   ---   =  --.
            ---               ---        t        f
         i in L            i in E         i
               K/C(x)            K/C(x)


    Where C = Const(K), L_K/C(x) = { i in {1, ..., n} such that t_i is
    transcendental over C(x)(t_1, ..., t_i-1) and Dt_i = Da_i/a_i, for some a_i
    in C(x)(t_1, ..., t_i-1)* } (i.e., the set of all indices of logarithmic
    monomials of K over C(x)), and E_K/C(x) = { i in {1, ..., n} such that t_i
    is transcendental over C(x)(t_1, ..., t_i-1) and Dt_i/t_i = Da_i, for some
    a_i in C(x)(t_1, ..., t_i-1) } (i.e., the set of all indices of
    hyperexponential monomials of K over C(x)).  If K is an elementary extension
    over C(x), then the cardinality of L_K/C(x) U E_K/C(x) is exactly the
    transcendence degree of K over C(x).  Furthermore, because Const_D(K) ==
    Const_D(C(x)) == C, deg(Dt_i) == 1 when t_i is in E_K/C(x) and
    deg(Dt_i) == 0 when t_i is in L_K/C(x), implying in particular that E_K/C(x)
    and L_K/C(x) are disjoint.

    The sets L_K/C(x) and E_K/C(x) must, by their nature, be computed
    recursively using this same function.  Therefore, it is required to pass
    them as indices to D (or T).  E_args are the arguments of the
    hyperexponentials indexed by E_K (i.e., if i is in E_K, then T[i] ==
    exp(E_args[i])).  This is needed to compute the final answer u such that
    Df/f == Du.

    log(f) will be the same as u up to a additive constant.  This is because
    they will both behave the same as monomials. For example, both log(x) and
    log(2*x) == log(x) + log(2) satisfy Dt == 1/x, because log(2) is constant.
    Therefore, the term const is returned.  const is such that
    log(const) + f == u.  This is calculated by dividing the arguments of one
    logarithm from the other.  Therefore, it is necessary to pass the arguments
    of the logarithmic terms in L_args.

    To handle the case where we are given Df/f, not f, use is_deriv_k_in_field().

    See also
    ========
    is_log_deriv_k_t_radical_in_field, is_log_deriv_k_t_radical

    """
    # Compute Df/f
    dfa, dfd = (fd*derivation(fa, DE) - fa*derivation(fd, DE)), fd*fa
    dfa, dfd = dfa.cancel(dfd, include=True)

    # Our assumption here is that each monomial is recursively transcendental
    if len(DE.exts) != len(DE.D):
        if [i for i in DE.cases if i == 'tan'] or \
                ({i for i in DE.cases if i == 'primitive'} -
                        set(DE.indices('log'))):
            raise NotImplementedError("Real version of the structure "
                "theorems with hypertangent support is not yet implemented.")

        # TODO: What should really be done in this case?
        raise NotImplementedError("Nonelementary extensions not supported "
            "in the structure theorems.")

    E_part = [DE.D[i].quo(Poly(DE.T[i], DE.T[i])).as_expr() for i in DE.indices('exp')]
    L_part = [DE.D[i].as_expr() for i in DE.indices('log')]

    # The expression dfa/dfd might not be polynomial in any of its symbols so we
    # use a Dummy as the generator for PolyMatrix.
    dum = Dummy()
    lhs = Matrix([E_part + L_part], dum)
    rhs = Matrix([dfa.as_expr()/dfd.as_expr()], dum)

    A, u = constant_system(lhs, rhs, DE)

    u = u.to_Matrix()  # Poly to Expr

    if not A or not all(derivation(i, DE, basic=True).is_zero for i in u):
        # If the elements of u are not all constant
        # Note: See comment in constant_system

        # Also note: derivation(basic=True) calls cancel()
        return None
    else:
        if not all(i.is_Rational for i in u):
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        else:
            terms = ([DE.extargs[i] for i in DE.indices('exp')] +
                    [DE.T[i] for i in DE.indices('log')])
            ans = list(zip(terms, u))
            result = Add(*[Mul(i, j) for i, j in ans])
            argterms = ([DE.T[i] for i in DE.indices('exp')] +
                    [DE.extargs[i] for i in DE.indices('log')])
            l = []
            ld = []
            for i, j in zip(argterms, u):
                # We need to get around things like sqrt(x**2) != x
                # and also sqrt(x**2 + 2*x + 1) != x + 1
                # Issue 10798: i need not be a polynomial
                i, d = i.as_numer_denom()
                icoeff, iterms = sqf_list(i)
                l.append(Mul(*([Pow(icoeff, j)] + [Pow(b, e*j) for b, e in iterms])))
                dcoeff, dterms = sqf_list(d)
                ld.append(Mul(*([Pow(dcoeff, j)] + [Pow(b, e*j) for b, e in dterms])))
            const = cancel(fa.as_expr()/fd.as_expr()/Mul(*l)*Mul(*ld))

            return (ans, result, const)


def is_log_deriv_k_t_radical(fa, fd, DE, Df=True):
    r"""
    Checks if Df is the logarithmic derivative of a k(t)-radical.

    Explanation
    ===========

    b in k(t) can be written as the logarithmic derivative of a k(t) radical if
    there exist n in ZZ and u in k(t) with n, u != 0 such that n*b == Du/u.
    Either returns (ans, u, n, const) or None, which means that Df cannot be
    written as the logarithmic derivative of a k(t)-radical.  ans is a list of
    tuples such that Mul(*[i**j for i, j in ans]) == u.  This is useful for
    seeing exactly what elements of k(t) produce u.

    This function uses the structure theorem approach, which says that for any
    f in K, Df is the logarithmic derivative of a K-radical if and only if there
    are ri in QQ such that::

            ---               ---       Dt
            \    r  * Dt   +  \    r  *   i
            /     i     i     /     i   ---   =  Df.
            ---               ---        t
         i in L            i in E         i
               K/C(x)            K/C(x)


    Where C = Const(K), L_K/C(x) = { i in {1, ..., n} such that t_i is
    transcendental over C(x)(t_1, ..., t_i-1) and Dt_i = Da_i/a_i, for some a_i
    in C(x)(t_1, ..., t_i-1)* } (i.e., the set of all indices of logarithmic
    monomials of K over C(x)), and E_K/C(x) = { i in {1, ..., n} such that t_i
    is transcendental over C(x)(t_1, ..., t_i-1) and Dt_i/t_i = Da_i, for some
    a_i in C(x)(t_1, ..., t_i-1) } (i.e., the set of all indices of
    hyperexponential monomials of K over C(x)).  If K is an elementary extension
    over C(x), then the cardinality of L_K/C(x) U E_K/C(x) is exactly the
    transcendence degree of K over C(x).  Furthermore, because Const_D(K) ==
    Const_D(C(x)) == C, deg(Dt_i) == 1 when t_i is in E_K/C(x) and
    deg(Dt_i) == 0 when t_i is in L_K/C(x), implying in particular that E_K/C(x)
    and L_K/C(x) are disjoint.

    The sets L_K/C(x) and E_K/C(x) must, by their nature, be computed
    recursively using this same function.  Therefore, it is required to pass
    them as indices to D (or T).  L_args are the arguments of the logarithms
    indexed by L_K (i.e., if i is in L_K, then T[i] == log(L_args[i])).  This is
    needed to compute the final answer u such that n*f == Du/u.

    exp(f) will be the same as u up to a multiplicative constant.  This is
    because they will both behave the same as monomials.  For example, both
    exp(x) and exp(x + 1) == E*exp(x) satisfy Dt == t. Therefore, the term const
    is returned.  const is such that exp(const)*f == u.  This is calculated by
    subtracting the arguments of one exponential from the other.  Therefore, it
    is necessary to pass the arguments of the exponential terms in E_args.

    To handle the case where we are given Df, not f, use
    is_log_deriv_k_t_radical_in_field().

    See also
    ========

    is_log_deriv_k_t_radical_in_field, is_deriv_k

    """
    if Df:
        dfa, dfd = (fd*derivation(fa, DE) - fa*derivation(fd, DE)).cancel(fd**2,
            include=True)
    else:
        dfa, dfd = fa, fd

    # Our assumption here is that each monomial is recursively transcendental
    if len(DE.exts) != len(DE.D):
        if [i for i in DE.cases if i == 'tan'] or \
                ({i for i in DE.cases if i == 'primitive'} -
                        set(DE.indices('log'))):
            raise NotImplementedError("Real version of the structure "
                "theorems with hypertangent support is not yet implemented.")

        # TODO: What should really be done in this case?
        raise NotImplementedError("Nonelementary extensions not supported "
            "in the structure theorems.")

    E_part = [DE.D[i].quo(Poly(DE.T[i], DE.T[i])).as_expr() for i in DE.indices('exp')]
    L_part = [DE.D[i].as_expr() for i in DE.indices('log')]

    # The expression dfa/dfd might not be polynomial in any of its symbols so we
    # use a Dummy as the generator for PolyMatrix.
    dum = Dummy()
    lhs = Matrix([E_part + L_part], dum)
    rhs = Matrix([dfa.as_expr()/dfd.as_expr()], dum)

    A, u = constant_system(lhs, rhs, DE)

    u = u.to_Matrix()  # Poly to Expr

    if not A or not all(derivation(i, DE, basic=True).is_zero for i in u):
        # If the elements of u are not all constant
        # Note: See comment in constant_system

        # Also note: derivation(basic=True) calls cancel()
        return None
    else:
        if not all(i.is_Rational for i in u):
            # TODO: But maybe we can tell if they're not rational, like
            # log(2)/log(3). Also, there should be an option to continue
            # anyway, even if the result might potentially be wrong.
            raise NotImplementedError("Cannot work with non-rational "
                "coefficients in this case.")
        else:
            n = S.One*reduce(ilcm, [i.as_numer_denom()[1] for i in u])
            u *= n
            terms = ([DE.T[i] for i in DE.indices('exp')] +
                    [DE.extargs[i] for i in DE.indices('log')])
            ans = list(zip(terms, u))
            result = Mul(*[Pow(i, j) for i, j in ans])

            # exp(f) will be the same as result up to a multiplicative
            # constant.  We now find the log of that constant.
            argterms = ([DE.extargs[i] for i in DE.indices('exp')] +
                    [DE.T[i] for i in DE.indices('log')])
            const = cancel(fa.as_expr()/fd.as_expr() -
                Add(*[Mul(i, j/n) for i, j in zip(argterms, u)]))

            return (ans, result, n, const)


def is_log_deriv_k_t_radical_in_field(fa, fd, DE, case='auto', z=None):
    """
    Checks if f can be written as the logarithmic derivative of a k(t)-radical.

    Explanation
    ===========

    It differs from is_log_deriv_k_t_radical(fa, fd, DE, Df=False)
    for any given fa, fd, DE in that it finds the solution in the
    given field not in some (possibly unspecified extension) and
    "in_field" with the function name is used to indicate that.

    f in k(t) can be written as the logarithmic derivative of a k(t) radical if
    there exist n in ZZ and u in k(t) with n, u != 0 such that n*f == Du/u.
    Either returns (n, u) or None, which means that f cannot be written as the
    logarithmic derivative of a k(t)-radical.

    case is one of {'primitive', 'exp', 'tan', 'auto'} for the primitive,
    hyperexponential, and hypertangent cases, respectively.  If case is 'auto',
    it will attempt to determine the type of the derivation automatically.

    See also
    ========
    is_log_deriv_k_t_radical, is_deriv_k

    """
    fa, fd = fa.cancel(fd, include=True)

    # f must be simple
    n, s = splitfactor(fd, DE)
    if not s.is_one:
        pass

    z = z or Dummy('z')
    H, b = residue_reduce(fa, fd, DE, z=z)
    if not b:
        # I will have to verify, but I believe that the answer should be
        # None in this case. This should never happen for the
        # functions given when solving the parametric logarithmic
        # derivative problem when integration elementary functions (see
        # Bronstein's book, page 255), so most likely this indicates a bug.
        return None

    roots = [(i, i.real_roots()) for i, _ in H]
    if not all(len(j) == i.degree() and all(k.is_Rational for k in j) for
               i, j in roots):
        # If f is the logarithmic derivative of a k(t)-radical, then all the
        # roots of the resultant must be rational numbers.
        return None

    # [(a, i), ...], where i*log(a) is a term in the log-part of the integral
    # of f
    respolys, residues = list(zip(*roots)) or [[], []]
    # Note: this might be empty, but everything below should work find in that
    # case (it should be the same as if it were [[1, 1]])
    residueterms = [(H[j][1].subs(z, i), i) for j in range(len(H)) for
        i in residues[j]]

    # TODO: finish writing this and write tests

    p = cancel(fa.as_expr()/fd.as_expr() - residue_reduce_derivation(H, DE, z))

    p = p.as_poly(DE.t)
    if p is None:
        # f - Dg will be in k[t] if f is the logarithmic derivative of a k(t)-radical
        return None

    if p.degree(DE.t) >= max(1, DE.d.degree(DE.t)):
        return None

    if case == 'auto':
        case = DE.case

    if case == 'exp':
        wa, wd = derivation(DE.t, DE).cancel(Poly(DE.t, DE.t), include=True)
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t, cancel=True)
            wa, wd = frac_in((wa, wd), DE.t)
            A = parametric_log_deriv(pa, pd, wa, wd, DE)
        if A is None:
            return None
        n, e, u = A
        u *= DE.t**e

    elif case == 'primitive':
        with DecrementLevel(DE):
            pa, pd = frac_in(p, DE.t)
            A = is_log_deriv_k_t_radical_in_field(pa, pd, DE, case='auto')
        if A is None:
            return None
        n, u = A

    elif case == 'base':
        # TODO: we can use more efficient residue reduction from ratint()
        if not fd.is_sqf or fa.degree() >= fd.degree():
            # f is the logarithmic derivative in the base case if and only if
            # f = fa/fd, fd is square-free, deg(fa) < deg(fd), and
            # gcd(fa, fd) == 1.  The last condition is handled by cancel() above.
            return None
        # Note: if residueterms = [], returns (1, 1)
        # f had better be 0 in that case.
        n = S.One*reduce(ilcm, [i.as_numer_denom()[1] for _, i in residueterms], 1)
        u = Mul(*[Pow(i, j*n) for i, j in residueterms])
        return (n, u)

    elif case == 'tan':
        raise NotImplementedError("The hypertangent case is "
        "not yet implemented for is_log_deriv_k_t_radical_in_field()")

    elif case in ('other_linear', 'other_nonlinear'):
        # XXX: If these are supported by the structure theorems, change to NotImplementedError.
        raise ValueError("The %s case is not supported in this function." % case)

    else:
        raise ValueError("case must be one of {'primitive', 'exp', 'tan', "
        "'base', 'auto'}, not %s" % case)

    common_denom = S.One*reduce(ilcm, [i.as_numer_denom()[1] for i in [j for _, j in
        residueterms]] + [n], 1)
    residueterms = [(i, j*common_denom) for i, j in residueterms]
    m = common_denom//n
    if common_denom != n*m:  # Verify exact division
        raise ValueError("Inexact division")
    u = cancel(u**m*Mul(*[Pow(i, j) for i, j in residueterms]))

    return (common_denom, u)
