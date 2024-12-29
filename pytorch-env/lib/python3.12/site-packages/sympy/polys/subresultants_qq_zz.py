"""
This module contains functions for the computation
of Euclidean, (generalized) Sturmian, (modified) subresultant
polynomial remainder sequences (prs's) of two polynomials;
included are also three functions for the computation of the
resultant of two polynomials.

Except for the function res_z(), which computes the resultant
of two polynomials, the pseudo-remainder function prem()
of sympy is _not_ used by any of the functions in the module.

Instead of prem() we use the function

rem_z().

Included is also the function quo_z().

An explanation of why we avoid prem() can be found in the
references stated in the docstring of rem_z().

1. Theoretical background:
==========================
Consider the polynomials f, g in Z[x] of degrees deg(f) = n and
deg(g) = m with n >= m.

Definition 1:
=============
The sign sequence of a polynomial remainder sequence (prs) is the
sequence of signs of the leading coefficients of its polynomials.

Sign sequences can be computed with the function:

sign_seq(poly_seq, x)

Definition 2:
=============
A polynomial remainder sequence (prs) is called complete if the
degree difference between any two consecutive polynomials is 1;
otherwise, it called incomplete.

It is understood that f, g belong to the sequences mentioned in
the two definitions above.

1A. Euclidean and subresultant prs's:
=====================================
The subresultant prs of f, g is a sequence of polynomials in Z[x]
analogous to the Euclidean prs, the sequence obtained by applying
on f, g Euclid's algorithm for polynomial greatest common divisors
(gcd) in Q[x].

The subresultant prs differs from the Euclidean prs in that the
coefficients of each polynomial in the former sequence are determinants
--- also referred to as subresultants --- of appropriately selected
sub-matrices of sylvester1(f, g, x), Sylvester's matrix of 1840 of
dimensions (n + m) * (n + m).

Recall that the determinant of sylvester1(f, g, x) itself is
called the resultant of f, g and serves as a criterion of whether
the two polynomials have common roots or not.

In SymPy the resultant is computed with the function
resultant(f, g, x). This function does _not_ evaluate the
determinant of sylvester(f, g, x, 1); instead, it returns
the last member of the subresultant prs of f, g, multiplied
(if needed) by an appropriate power of -1; see the caveat below.

In this module we use three functions to compute the
resultant of f, g:
a) res(f, g, x) computes the resultant by evaluating
the determinant of sylvester(f, g, x, 1);
b) res_q(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Q[x] with the function rem();
c) res_z(f, g, x) computes the resultant recursively, by
performing polynomial divisions in Z[x] with the function prem().

Caveat: If Df = degree(f, x) and Dg = degree(g, x), then:

resultant(f, g, x) = (-1)**(Df*Dg) * resultant(g, f, x).

For complete prs's the sign sequence of the Euclidean prs of f, g
is identical to the sign sequence of the subresultant prs of f, g
and the coefficients of one sequence  are easily computed from the
coefficients of the  other.

For incomplete prs's the polynomials in the subresultant prs, generally
differ in sign from those of the Euclidean prs, and --- unlike the
case of complete prs's --- it is not at all obvious how to compute
the coefficients of one sequence from the coefficients of the  other.

1B. Sturmian and modified subresultant prs's:
=============================================
For the same polynomials f, g in Z[x] mentioned above, their ``modified''
subresultant prs is a sequence of polynomials similar to the Sturmian
prs, the sequence obtained by applying in Q[x] Sturm's algorithm on f, g.

The two sequences differ in that the coefficients of each polynomial
in the modified subresultant prs are the determinants --- also referred
to as modified subresultants --- of appropriately selected  sub-matrices
of sylvester2(f, g, x), Sylvester's matrix of 1853 of dimensions 2n x 2n.

The determinant of sylvester2 itself is called the modified resultant
of f, g and it also can serve as a criterion of whether the two
polynomials have common roots or not.

For complete prs's the  sign sequence of the Sturmian prs of f, g is
identical to the sign sequence of the modified subresultant prs of
f, g and the coefficients of one sequence  are easily computed from
the coefficients of the other.

For incomplete prs's the polynomials in the modified subresultant prs,
generally differ in sign from those of the Sturmian prs, and --- unlike
the case of complete prs's --- it is not at all obvious how to compute
the coefficients of one sequence from the coefficients of the  other.

As Sylvester pointed out, the coefficients of the polynomial remainders
obtained as (modified) subresultants are the smallest possible without
introducing rationals and without computing (integer) greatest common
divisors.

1C. On terminology:
===================
Whence the terminology? Well generalized Sturmian prs's are
``modifications'' of Euclidean prs's; the hint came from the title
of the Pell-Gordon paper of 1917.

In the literature one also encounters the name ``non signed'' and
``signed'' prs for Euclidean and Sturmian prs respectively.

Likewise ``non signed'' and ``signed'' subresultant prs for
subresultant and modified subresultant prs respectively.

2. Functions in the module:
===========================
No function utilizes SymPy's function prem().

2A. Matrices:
=============
The functions sylvester(f, g, x, method=1) and
sylvester(f, g, x, method=2) compute either Sylvester matrix.
They can be used to compute (modified) subresultant prs's by
direct determinant evaluation.

The function bezout(f, g, x, method='prs') provides a matrix of
smaller dimensions than either Sylvester matrix. It is the function
of choice for computing (modified) subresultant prs's by direct
determinant evaluation.

sylvester(f, g, x, method=1)
sylvester(f, g, x, method=2)
bezout(f, g, x, method='prs')

The following identity holds:

bezout(f, g, x, method='prs') =
backward_eye(deg(f))*bezout(f, g, x, method='bz')*backward_eye(deg(f))

2B. Subresultant and modified subresultant prs's by
===================================================
determinant evaluations:
=======================
We use the Sylvester matrices of 1840 and 1853 to
compute, respectively, subresultant and modified
subresultant polynomial remainder sequences. However,
for large matrices this approach takes a lot of time.

Instead of utilizing the Sylvester matrices, we can
employ the Bezout matrix which is of smaller dimensions.

subresultants_sylv(f, g, x)
modified_subresultants_sylv(f, g, x)
subresultants_bezout(f, g, x)
modified_subresultants_bezout(f, g, x)

2C. Subresultant prs's by ONE determinant evaluation:
=====================================================
All three functions in this section evaluate one determinant
per remainder polynomial; this is the determinant of an
appropriately selected sub-matrix of sylvester1(f, g, x),
Sylvester's matrix of 1840.

To compute the remainder polynomials the function
subresultants_rem(f, g, x) employs rem(f, g, x).
By contrast, the other two functions implement Van Vleck's ideas
of 1900 and compute the remainder polynomials by trinagularizing
sylvester2(f, g, x), Sylvester's matrix of 1853.


subresultants_rem(f, g, x)
subresultants_vv(f, g, x)
subresultants_vv_2(f, g, x).

2E. Euclidean, Sturmian prs's in Q[x]:
======================================
euclid_q(f, g, x)
sturm_q(f, g, x)

2F. Euclidean, Sturmian and (modified) subresultant prs's P-G:
==============================================================
All functions in this section are based on the Pell-Gordon (P-G)
theorem of 1917.
Computations are done in Q[x], employing the function rem(f, g, x)
for the computation of the remainder polynomials.

euclid_pg(f, g, x)
sturm pg(f, g, x)
subresultants_pg(f, g, x)
modified_subresultants_pg(f, g, x)

2G. Euclidean, Sturmian and (modified) subresultant prs's A-M-V:
================================================================
All functions in this section are based on the Akritas-Malaschonok-
Vigklas (A-M-V) theorem of 2015.
Computations are done in Z[x], employing the function rem_z(f, g, x)
for the computation of the remainder polynomials.

euclid_amv(f, g, x)
sturm_amv(f, g, x)
subresultants_amv(f, g, x)
modified_subresultants_amv(f, g, x)

2Ga. Exception:
===============
subresultants_amv_q(f, g, x)

This function employs rem(f, g, x) for the computation of
the remainder polynomials, despite the fact that it implements
the A-M-V Theorem.

It is included in our module in order to show that theorems P-G
and A-M-V can be implemented utilizing either the function
rem(f, g, x) or the function rem_z(f, g, x).

For clearly historical reasons --- since the Collins-Brown-Traub
coefficients-reduction factor beta_i was not available in 1917 ---
we have implemented the Pell-Gordon theorem with the function
rem(f, g, x) and the A-M-V Theorem  with the function rem_z(f, g, x).

2H. Resultants:
===============
res(f, g, x)
res_q(f, g, x)
res_z(f, g, x)
"""


from sympy.concrete.summations import summation
from sympy.core.function import expand
from sympy.core.numbers import nan
from sympy.core.singleton import S
from sympy.core.symbol import Dummy as var
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import eye, Matrix, zeros
from sympy.printing.pretty.pretty import pretty_print as pprint
from sympy.simplify.simplify import simplify
from sympy.polys.domains import QQ
from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
from sympy.polys.polyerrors import PolynomialError


def sylvester(f, g, x, method = 1):
    '''
      The input polynomials f, g are in Z[x] or in Q[x]. Let m = degree(f, x),
      n = degree(g, x) and mx = max(m, n).

      a. If method = 1 (default), computes sylvester1, Sylvester's matrix of 1840
          of dimension (m + n) x (m + n). The determinants of properly chosen
          submatrices of this matrix (a.k.a. subresultants) can be
          used to compute the coefficients of the Euclidean PRS of f, g.

      b. If method = 2, computes sylvester2, Sylvester's matrix of 1853
          of dimension (2*mx) x (2*mx). The determinants of properly chosen
          submatrices of this matrix (a.k.a. ``modified'' subresultants) can be
          used to compute the coefficients of the Sturmian PRS of f, g.

      Applications of these Matrices can be found in the references below.
      Especially, for applications of sylvester2, see the first reference!!

      References
      ==========
      1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
      by Van Vleck Regarding Sturm Sequences. Serdica Journal of Computing,
      Vol. 7, No 4, 101-134, 2013.

      2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
      and Modified Subresultant Polynomial Remainder Sequences.''
      Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    '''
    # obtain degrees of polys
    m, n = degree( Poly(f, x), x), degree( Poly(g, x), x)

    # Special cases:
    # A:: case m = n < 0 (i.e. both polys are 0)
    if m == n and n < 0:
        return Matrix([])

    # B:: case m = n = 0  (i.e. both polys are constants)
    if m == n and n == 0:
        return Matrix([])

    # C:: m == 0 and n < 0 or m < 0 and n == 0
    # (i.e. one poly is constant and the other is 0)
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])

    # D:: m >= 1 and n < 0 or m < 0 and n >=1
    # (i.e. one poly is of degree >=1 and the other is 0)
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])

    fp = Poly(f, x).all_coeffs()
    gp = Poly(g, x).all_coeffs()

    # Sylvester's matrix of 1840 (default; a.k.a. sylvester1)
    if method <= 1:
        M = zeros(m + n)
        k = 0
        for i in range(n):
            j = k
            for coeff in fp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        k = 0
        for i in range(n, m + n):
            j = k
            for coeff in gp:
                M[i, j] = coeff
                j = j + 1
            k = k + 1
        return M

    # Sylvester's matrix of 1853 (a.k.a sylvester2)
    if method >= 2:
        if len(fp) < len(gp):
            h = []
            for i in range(len(gp) - len(fp)):
                h.append(0)
            fp[ : 0] = h
        else:
            h = []
            for i in range(len(fp) - len(gp)):
                h.append(0)
            gp[ : 0] = h
        mx = max(m, n)
        dim = 2*mx
        M = zeros( dim )
        k = 0
        for i in range( mx ):
            j = k
            for coeff in fp:
                M[2*i, j] = coeff
                j = j + 1
            j = k
            for coeff in gp:
                M[2*i + 1, j] = coeff
                j = j + 1
            k = k + 1
        return M

def process_matrix_output(poly_seq, x):
    """
    poly_seq is a polynomial remainder sequence computed either by
    (modified_)subresultants_bezout or by (modified_)subresultants_sylv.

    This function removes from poly_seq all zero polynomials as well
    as all those whose degree is equal to the degree of a preceding
    polynomial in poly_seq, as we scan it from left to right.

    """
    L = poly_seq[:]  # get a copy of the input sequence
    d = degree(L[1], x)
    i = 2
    while i < len(L):
        d_i = degree(L[i], x)
        if d_i < 0:          # zero poly
            L.remove(L[i])
            i = i - 1
        if d == d_i:         # poly degree equals degree of previous poly
            L.remove(L[i])
            i = i - 1
        if d_i >= 0:
            d = d_i
        i = i + 1

    return L

def subresultants_sylv(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed
    that deg(f) >= deg(g).

    Computes the subresultant polynomial remainder sequence (prs)
    of f, g by evaluating determinants of appropriately selected
    submatrices of sylvester(f, g, x, 1). The dimensions of the
    latter are (deg(f) + deg(g)) x (deg(f) + deg(g)).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of sylvester(f, g, x, 1).

    If the subresultant prs is complete, then the output coincides
    with the Euclidean sequence of the polynomials f, g.

    References:
    ===========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """

    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list

    # form matrix sylvester(f, g, x, 1)
    S = sylvester(f, g, x, 1)

    # pick appropriate submatrices of S
    # and form subresultant polys
    j = m - 1

    while j > 0:
        Sp = S[:, :]  # copy of S
        # delete last j rows of coeffs of g
        for ind in range(m + n - j, m + n):
            Sp.row_del(m + n - j)
        # delete last j rows of coeffs of f
        for ind in range(m - j, m):
            Sp.row_del(m - j)

        # evaluate determinants and form coefficients list
        coeff_L, k, l = [], Sp.rows, 0
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1

        # form poly and append to SP_L
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1

    # j = 0
    SR_L.append(S.det())

    return process_matrix_output(SR_L, x)

def modified_subresultants_sylv(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x]. It is assumed
    that deg(f) >= deg(g).

    Computes the modified subresultant polynomial remainder sequence (prs)
    of f, g by evaluating determinants of appropriately selected
    submatrices of sylvester(f, g, x, 2). The dimensions of the
    latter are (2*deg(f)) x (2*deg(f)).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of sylvester(f, g, x, 2).

    If the modified subresultant prs is complete, then the output coincides
    with the Sturmian sequence of the polynomials f, g.

    References:
    ===========
    1. A. G. Akritas,G.I. Malaschonok and P.S. Vigklas:
    Sturm Sequences and Modified Subresultant Polynomial Remainder
    Sequences. Serdica Journal of Computing, Vol. 8, No 1, 29--46, 2014.

    """

    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # modified subresultant list

    # form matrix sylvester(f, g, x, 2)
    S = sylvester(f, g, x, 2)

    # pick appropriate submatrices of S
    # and form modified subresultant polys
    j = m - 1

    while j > 0:
        # delete last 2*j rows of pairs of coeffs of f, g
        Sp = S[0:2*n - 2*j, :]  # copy of first 2*n - 2*j rows of S

        # evaluate determinants and form coefficients list
        coeff_L, k, l = [], Sp.rows, 0
        while l <= j:
            coeff_L.append(Sp[:, 0:k].det())
            Sp.col_swap(k - 1, k + l)
            l += 1

        # form poly and append to SP_L
        SR_L.append(Poly(coeff_L, x).as_expr())
        j -= 1

    # j = 0
    SR_L.append(S.det())

    return process_matrix_output(SR_L, x)

def res(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed by evaluating
    the determinant of the matrix sylvester(f, g, x, 1).

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.

    """
    if f == 0 or g == 0:
         raise PolynomialError("The resultant of %s and %s is not defined" % (f, g))
    else:
        return sylvester(f, g, x, 1).det()

def res_q(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed recursively
    by polynomial divisions in Q[x], using the function rem.
    See Cohen's book p. 281.

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.
    """
    m = degree(f, x)
    n = degree(g, x)
    if m < n:
        return (-1)**(m*n) * res_q(g, f, x)
    elif n == 0:  # g is a constant
        return g**m
    else:
        r = rem(f, g, x)
        if r == 0:
            return 0
        else:
            s = degree(r, x)
            l = LC(g, x)
            return (-1)**(m*n) * l**(m-s)*res_q(g, r, x)

def res_z(f, g, x):
    """
    The input polynomials f, g are in Z[x] or in Q[x].

    The output is the resultant of f, g computed recursively
    by polynomial divisions in Z[x], using the function prem().
    See Cohen's book p. 283.

    References:
    ===========
    1. J. S. Cohen: Computer Algebra and Symbolic Computation
     - Mathematical Methods. A. K. Peters, 2003.
    """
    m = degree(f, x)
    n = degree(g, x)
    if m < n:
        return (-1)**(m*n) * res_z(g, f, x)
    elif n == 0:  # g is a constant
        return g**m
    else:
        r = prem(f, g, x)
        if r == 0:
            return 0
        else:
            delta = m - n + 1
            w = (-1)**(m*n) * res_z(g, r, x)
            s = degree(r, x)
            l = LC(g, x)
            k = delta * n - m + s
            return quo(w, l**k, x)

def sign_seq(poly_seq, x):
    """
    Given a sequence of polynomials poly_seq, it returns
    the sequence of signs of the leading coefficients of
    the polynomials in poly_seq.

    """
    return [sign(LC(poly_seq[i], x)) for i in range(len(poly_seq))]

def bezout(p, q, x, method='bz'):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. Let
    mx = max(degree(p, x), degree(q, x)).

    The default option bezout(p, q, x, method='bz') returns Bezout's
    symmetric matrix of p and q, of dimensions (mx) x (mx). The
    determinant of this matrix is equal to the determinant of sylvester2,
    Sylvester's matrix of 1853, whose dimensions are (2*mx) x (2*mx);
    however the subresultants of these two matrices may differ.

    The other option, bezout(p, q, x, 'prs'), is of interest to us
    in this module because it returns a matrix equivalent to sylvester2.
    In this case all subresultants of the two matrices are identical.

    Both the subresultant polynomial remainder sequence (prs) and
    the modified subresultant prs of p and q can be computed by
    evaluating determinants of appropriately selected submatrices of
    bezout(p, q, x, 'prs') --- one determinant per coefficient of the
    remainder polynomials.

    The matrices bezout(p, q, x, 'bz') and bezout(p, q, x, 'prs')
    are related by the formula

    bezout(p, q, x, 'prs') =
    backward_eye(deg(p)) * bezout(p, q, x, 'bz') * backward_eye(deg(p)),

    where backward_eye() is the backward identity function.

    References
    ==========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    # obtain degrees of polys
    m, n = degree( Poly(p, x), x), degree( Poly(q, x), x)

    # Special cases:
    # A:: case m = n < 0 (i.e. both polys are 0)
    if m == n and n < 0:
        return Matrix([])

    # B:: case m = n = 0  (i.e. both polys are constants)
    if m == n and n == 0:
        return Matrix([])

    # C:: m == 0 and n < 0 or m < 0 and n == 0
    # (i.e. one poly is constant and the other is 0)
    if m == 0 and n < 0:
        return Matrix([])
    elif m < 0 and n == 0:
        return Matrix([])

    # D:: m >= 1 and n < 0 or m < 0 and n >=1
    # (i.e. one poly is of degree >=1 and the other is 0)
    if m >= 1 and n < 0:
        return Matrix([0])
    elif m < 0 and n >= 1:
        return Matrix([0])

    y = var('y')

    # expr is 0 when x = y
    expr = p * q.subs({x:y}) - p.subs({x:y}) * q

    # hence expr is exactly divisible by x - y
    poly = Poly( quo(expr, x-y), x, y)

    # form Bezout matrix and store them in B as indicated to get
    # the LC coefficient of each poly either in the first position
    # of each row (method='prs') or in the last (method='bz').
    mx = max(m, n)
    B = zeros(mx)
    for i in range(mx):
        for j in range(mx):
            if method == 'prs':
                B[mx - 1 - i, mx - 1 - j] = poly.nth(i, j)
            else:
                B[i, j] = poly.nth(i, j)
    return B

def backward_eye(n):
    '''
    Returns the backward identity matrix of dimensions n x n.

    Needed to "turn" the Bezout matrices
    so that the leading coefficients are first.
    See docstring of the function bezout(p, q, x, method='bz').
    '''
    M = eye(n)  # identity matrix of order n

    for i in range(int(M.rows / 2)):
        M.row_swap(0 + i, M.rows - 1 - i)

    return M

def subresultants_bezout(p, q, x):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant polynomial remainder sequence
    of p, q by evaluating determinants of appropriately selected
    submatrices of bezout(p, q, x, 'prs'). The dimensions of the
    latter are deg(p) x deg(p).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of bezout(p, q, x, 'prs').

    bezout(p, q, x, 'prs) is used instead of sylvester(p, q, x, 1),
    Sylvester's matrix of 1840, because the dimensions of the latter
    are (deg(p) + deg(q)) x (deg(p) + deg(q)).

    If the subresultant prs is complete, then the output coincides
    with the Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    f, g = p, q
    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list
    F = LC(f, x)**(degF - degG)

    # form the bezout matrix
    B = bezout(f, g, x, 'prs')

    # pick appropriate submatrices of B
    # and form subresultant polys
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        k, coeff_L = j - 1, []
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1

        # apply Theorem 2.1 in the paper by Toca & Vega 2004
        # to get correct signs
        SR_L.append(int((-1)**(j*(j-1)/2)) * (Poly(coeff_L, x) / F).as_expr())
        j = j + 1

    return process_matrix_output(SR_L, x)

def modified_subresultants_bezout(p, q, x):
    """
    The input polynomials p, q are in Z[x] or in Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the modified subresultant polynomial remainder sequence
    of p, q by evaluating determinants of appropriately selected
    submatrices of bezout(p, q, x, 'prs'). The dimensions of the
    latter are deg(p) x deg(p).

    Each coefficient is computed by evaluating the determinant of the
    corresponding submatrix of bezout(p, q, x, 'prs').

    bezout(p, q, x, 'prs') is used instead of sylvester(p, q, x, 2),
    Sylvester's matrix of 1853, because the dimensions of the latter
    are 2*deg(p) x 2*deg(p).

    If the modified subresultant prs is complete, and LC( p ) > 0, the output
    coincides with the (generalized) Sturm's sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    2. G.M.Diaz-Toca,L.Gonzalez-Vega: Various New Expressions for Subresultants
    and Their Applications. Appl. Algebra in Engin., Communic. and Comp.,
    Vol. 15, 233-266, 2004.


    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    f, g = p, q
    n = degF = degree(f, x)
    m = degG = degree(g, x)

    # make sure proper degrees
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, degF, degG, f, g = m, n, degG, degF, g, f
    if n > 0 and m == 0:
        return [f, g]

    SR_L = [f, g]      # subresultant list

    # form the bezout matrix
    B = bezout(f, g, x, 'prs')

    # pick appropriate submatrices of B
    # and form subresultant polys
    if degF > degG:
        j = 2
    if degF == degG:
        j = 1
    while j <= degF:
        M = B[0:j, :]
        k, coeff_L = j - 1, []
        while k <= degF - 1:
            coeff_L.append(M[:, 0:j].det())
            if k < degF - 1:
                M.col_swap(j - 1, k + 1)
            k = k + 1

        ## Theorem 2.1 in the paper by Toca & Vega 2004 is _not needed_
        ## in this case since
        ## the bezout matrix is equivalent to sylvester2
        SR_L.append(( Poly(coeff_L, x)).as_expr())
        j = j + 1

    return process_matrix_output(SR_L, x)

def sturm_pg(p, q, x, method=0):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].
    If q = diff(p, x, 1) it is the usual Sturm sequence.

    A. If method == 0, default, the remainder coefficients of the sequence
       are (in absolute value) ``modified'' subresultants, which for non-monic
       polynomials are greater than the coefficients of the corresponding
       subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).

    B. If method == 1, the remainder coefficients of the sequence are (in
       absolute value) subresultants, which for non-monic polynomials are
       smaller than the coefficients of the corresponding ``modified''
       subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).

    If the Sturm sequence is complete, method=0 and LC( p ) > 0, the coefficients
    of the polynomials in the sequence are ``modified'' subresultants.
    That is, they are  determinants of appropriately selected submatrices of
    sylvester2, Sylvester's matrix of 1853. In this case the Sturm sequence
    coincides with the ``modified'' subresultant prs, of the polynomials
    p, q.

    If the Sturm sequence is incomplete and method=0 then the signs of the
    coefficients of the polynomials in the sequence may differ from the signs
    of the coefficients of the corresponding polynomials in the ``modified''
    subresultant prs; however, the absolute values are the same.

    To compute the coefficients, no determinant evaluation takes place. Instead,
    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);
    the coefficients of the remainders computed this way become (``modified'')
    subresultants with the help of the Pell-Gordon Theorem of 1917.
    See also the function euclid_pg(p, q, x).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # make sure LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:
        flag = 1
        p = -p
        q = -q

    # initialize
    lcf = LC(p, x)**(d0 - d1)               # lcf * subr = modified subr
    a0, a1 = p, q                           # the input polys
    sturm_seq = [a0, a1]                    # the output list
    del0 = d0 - d1                          # degree difference
    rho1 =  LC(a1, x)                       # leading coeff of a1
    exp_deg = d1 - 1                        # expected degree of a2
    a2 = - rem(a0, a1, domain=QQ)           # first remainder
    rho2 =  LC(a2,x)                        # leading coeff of a2
    d2 =  degree(a2, x)                     # actual degree of a2
    deg_diff_new = exp_deg - d2             # expected - actual degree
    del1 = d1 - d2                          # degree difference

    # mul_fac is the factor by which a2 is multiplied to
    # get integer coefficients
    mul_fac_old = rho1**(del0 + del1 - deg_diff_new)

    # append accordingly
    if method == 0:
        sturm_seq.append( simplify(lcf * a2 *  Abs(mul_fac_old)))
    else:
        sturm_seq.append( simplify( a2 *  Abs(mul_fac_old)))

    # main loop
    deg_diff_old = deg_diff_new
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        del0 = del1                          # update degree difference
        exp_deg = d1 - 1                     # new expected degree
        a2 = - rem(a0, a1, domain=QQ)        # new remainder
        rho3 =  LC(a2, x)                    # leading coeff of a2
        d2 =  degree(a2, x)                  # actual degree of a2
        deg_diff_new = exp_deg - d2          # expected - actual degree
        del1 = d1 - d2                       # degree difference

        # take into consideration the power
        # rho1**deg_diff_old that was "left out"
        expo_old = deg_diff_old               # rho1 raised to this power
        expo_new = del0 + del1 - deg_diff_new # rho2 raised to this power

        # update variables and append
        mul_fac_new = rho2**(expo_new) * rho1**(expo_old) * mul_fac_old
        deg_diff_old, mul_fac_old = deg_diff_new, mul_fac_new
        rho1, rho2 = rho2, rho3
        if method == 0:
            sturm_seq.append( simplify(lcf * a2 *  Abs(mul_fac_old)))
        else:
            sturm_seq.append( simplify( a2 *  Abs(mul_fac_old)))

    if flag:          # change the sign of the sequence
        sturm_seq = [-i for i in sturm_seq]

    # gcd is of degree > 0 ?
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)

    return sturm_seq

def sturm_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Q[x].
    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).

    The coefficients of the polynomials in the Sturm sequence can be uniquely
    determined from the corresponding coefficients of the polynomials found
    either in:

        (a) the ``modified'' subresultant prs, (references 1, 2)

    or in

        (b) the subresultant prs (reference 3).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2 Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # make sure LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:
        flag = 1
        p = -p
        q = -q

    # initialize
    a0, a1 = p, q                           # the input polys
    sturm_seq = [a0, a1]                    # the output list
    a2 = -rem(a0, a1, domain=QQ)            # first remainder
    d2 =  degree(a2, x)                     # degree of a2
    sturm_seq.append( a2 )

    # main loop
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        a2 = -rem(a0, a1, domain=QQ)         # new remainder
        d2 =  degree(a2, x)                  # actual degree of a2
        sturm_seq.append( a2 )

    if flag:          # change the sign of the sequence
        sturm_seq = [-i for i in sturm_seq]

    # gcd is of degree > 0 ?
    m = len(sturm_seq)
    if sturm_seq[m - 1] == nan or sturm_seq[m - 1] == 0:
        sturm_seq.pop(m - 1)

    return sturm_seq

def sturm_amv(p, q, x, method=0):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the (generalized) Sturm sequence of p and q in Z[x] or Q[x].
    If q = diff(p, x, 1) it is the usual Sturm sequence.

    A. If method == 0, default, the remainder coefficients of the
       sequence are (in absolute value) ``modified'' subresultants, which
       for non-monic polynomials are greater than the coefficients of the
       corresponding subresultants by the factor Abs(LC(p)**( deg(p)- deg(q))).

    B. If method == 1, the remainder coefficients of the sequence are (in
       absolute value) subresultants, which for non-monic polynomials are
       smaller than the coefficients of the corresponding ``modified''
       subresultants by the factor Abs( LC(p)**( deg(p)- deg(q)) ).

    If the Sturm sequence is complete, method=0 and LC( p ) > 0, then the
    coefficients of the polynomials in the sequence are ``modified'' subresultants.
    That is, they are  determinants of appropriately selected submatrices of
    sylvester2, Sylvester's matrix of 1853. In this case the Sturm sequence
    coincides with the ``modified'' subresultant prs, of the polynomials
    p, q.

    If the Sturm sequence is incomplete and method=0 then the signs of the
    coefficients of the polynomials in the sequence may differ from the signs
    of the coefficients of the corresponding polynomials in the ``modified''
    subresultant prs; however, the absolute values are the same.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, we first compute the euclidean sequence  of p and q using
    euclid_amv(p, q, x) and then: (a) change the signs of the remainders in the
    Euclidean sequence according to the pattern "-, -, +, +, -, -, +, +,..."
    (see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference)
    and (b) if method=0, assuming deg(p) > deg(q), we multiply the remainder
    coefficients of the Euclidean sequence times the factor
    Abs( LC(p)**( deg(p)- deg(q)) ) to make them modified subresultants.
    See also the function sturm_pg(p, q, x).

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.'' Serdica
    Journal of Computing 9(2) (2015), 123-138.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # compute the euclidean sequence
    prs = euclid_amv(p, q, x)

    # defensive
    if prs == [] or len(prs) == 2:
        return prs

    # the coefficients in prs are subresultants and hence are smaller
    # than the corresponding subresultants by the factor
    # Abs( LC(prs[0])**( deg(prs[0]) - deg(prs[1])) ); Theorem 2, 2nd reference.
    lcf = Abs( LC(prs[0])**( degree(prs[0], x) - degree(prs[1], x) ) )

    # the signs of the first two polys in the sequence stay the same
    sturm_seq = [prs[0], prs[1]]

    # change the signs according to "-, -, +, +, -, -, +, +,..."
    # and multiply times lcf if needed
    flag = 0
    m = len(prs)
    i = 2
    while i <= m-1:
        if  flag == 0:
            sturm_seq.append( - prs[i] )
            i = i + 1
            if i == m:
                break
            sturm_seq.append( - prs[i] )
            i = i + 1
            flag = 1
        elif flag == 1:
            sturm_seq.append( prs[i] )
            i = i + 1
            if i == m:
                break
            sturm_seq.append( prs[i] )
            i = i + 1
            flag = 0

    # subresultants or modified subresultants?
    if method == 0 and lcf > 1:
        aux_seq = [sturm_seq[0], sturm_seq[1]]
        for i in range(2, m):
            aux_seq.append(simplify(sturm_seq[i] * lcf ))
        sturm_seq = aux_seq

    return sturm_seq

def euclid_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the Euclidean sequence of p and q in Z[x] or Q[x].

    If the Euclidean sequence is complete the coefficients of the polynomials
    in the sequence are subresultants. That is, they are  determinants of
    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
    In this case the Euclidean sequence coincides with the subresultant prs
    of the polynomials p, q.

    If the Euclidean sequence is incomplete the signs of the coefficients of the
    polynomials in the sequence may differ from the signs of the coefficients of
    the corresponding polynomials in the subresultant prs; however, the absolute
    values are the same.

    To compute the Euclidean sequence, no determinant evaluation takes place.
    We first compute the (generalized) Sturm sequence  of p and q using
    sturm_pg(p, q, x, 1), in which case the coefficients are (in absolute value)
    equal to subresultants. Then we change the signs of the remainders in the
    Sturm sequence according to the pattern "-, -, +, +, -, -, +, +,..." ;
    see Lemma 1 in the 1st reference or Theorem 3 in the 2nd reference as well as
    the function sturm_pg(p, q, x).

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.'' Serdica
    Journal of Computing 9(2) (2015), 123-138.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    Remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.
    """
    # compute the sturmian sequence using the Pell-Gordon (or AMV) theorem
    # with the coefficients in the prs being (in absolute value) subresultants
    prs = sturm_pg(p, q, x, 1)  ## any other method would do

    # defensive
    if prs == [] or len(prs) == 2:
        return prs

    # the signs of the first two polys in the sequence stay the same
    euclid_seq = [prs[0], prs[1]]

    # change the signs according to "-, -, +, +, -, -, +, +,..."
    flag = 0
    m = len(prs)
    i = 2
    while i <= m-1:
        if  flag == 0:
            euclid_seq.append(- prs[i] )
            i = i + 1
            if i == m:
                break
            euclid_seq.append(- prs[i] )
            i = i + 1
            flag = 1
        elif flag == 1:
            euclid_seq.append(prs[i] )
            i = i + 1
            if i == m:
                break
            euclid_seq.append(prs[i] )
            i = i + 1
            flag = 0

    return euclid_seq

def euclid_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the Euclidean sequence of p and q in Q[x].
    Polynomial divisions in Q[x] are performed, using the function rem(p, q, x).

    The coefficients of the polynomials in the Euclidean sequence can be uniquely
    determined from the corresponding coefficients of the polynomials found
    either in:

        (a) the ``modified'' subresultant polynomial remainder sequence,
    (references 1, 2)

    or in

        (b) the subresultant polynomial remainder sequence (references 3).

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # make sure LC(p) > 0
    flag = 0
    if  LC(p,x) < 0:
        flag = 1
        p = -p
        q = -q

    # initialize
    a0, a1 = p, q                           # the input polys
    euclid_seq = [a0, a1]                   # the output list
    a2 = rem(a0, a1, domain=QQ)             # first remainder
    d2 =  degree(a2, x)                     # degree of a2
    euclid_seq.append( a2 )

    # main loop
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        a2 = rem(a0, a1, domain=QQ)          # new remainder
        d2 =  degree(a2, x)                  # actual degree of a2
        euclid_seq.append( a2 )

    if flag:          # change the sign of the sequence
        euclid_seq = [-i for i in euclid_seq]

    # gcd is of degree > 0 ?
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)

    return euclid_seq

def euclid_amv(f, g, x):
    """
    f, g are polynomials in Z[x] or Q[x]. It is assumed
    that degree(f, x) >= degree(g, x).

    Computes the Euclidean sequence of p and q in Z[x] or Q[x].

    If the Euclidean sequence is complete the coefficients of the polynomials
    in the sequence are subresultants. That is, they are  determinants of
    appropriately selected submatrices of sylvester1, Sylvester's matrix of 1840.
    In this case the Euclidean sequence coincides with the subresultant prs,
    of the polynomials p, q.

    If the Euclidean sequence is incomplete the signs of the coefficients of the
    polynomials in the sequence may differ from the signs of the coefficients of
    the corresponding polynomials in the subresultant prs; however, the absolute
    values are the same.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Z[x] or Q[x] are performed, using
    the function rem_z(f, g, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Collins-Brown-Traub formula for coefficient reduction.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    # make sure proper degrees
    d0 =  degree(f, x)
    d1 =  degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        d0, d1 = d1, d0
        f, g = g, f
    if d0 > 0 and d1 == 0:
        return [f, g]

    # initialize
    a0 = f
    a1 = g
    euclid_seq = [a0, a1]
    deg_dif_p1, c = degree(a0, x) - degree(a1, x) + 1, -1

    # compute the first polynomial of the prs
    i = 1
    a2 = rem_z(a0, a1, x) / Abs( (-1)**deg_dif_p1 )     # first remainder
    euclid_seq.append( a2 )
    d2 =  degree(a2, x)                              # actual degree of a2

    # main loop
    while d2 >= 1:
        a0, a1, d0, d1 = a1, a2, d1, d2       # update polys and degrees
        i += 1
        sigma0 = -LC(a0)
        c = (sigma0**(deg_dif_p1 - 1)) / (c**(deg_dif_p1 - 2))
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs( (c**(deg_dif_p1 - 1)) * sigma0 )
        euclid_seq.append( a2 )
        d2 =  degree(a2, x)                   # actual degree of a2

    # gcd is of degree > 0 ?
    m = len(euclid_seq)
    if euclid_seq[m - 1] == nan or euclid_seq[m - 1] == 0:
        euclid_seq.pop(m - 1)

    return euclid_seq

def modified_subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the ``modified'' subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    ``modified'' subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester2, Sylvester's matrix of 1853.

    To compute the coefficients, no determinant evaluation takes place. Instead,
    polynomial divisions in Q[x] are performed, using the function rem(p, q, x);
    the coefficients of the remainders computed this way become ``modified''
    subresultants with the help of the Pell-Gordon Theorem of 1917.

    If the ``modified'' subresultant prs is complete, and LC( p ) > 0, it coincides
    with the (generalized) Sturm sequence of the polynomials p, q.

    References
    ==========
    1. Pell A. J., R. L. Gordon. The Modified Remainders Obtained in Finding
    the Highest Common Factor of Two Polynomials. Annals of MatheMatics,
    Second Series, 18 (1917), No. 4, 188-193.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p,x)
    d1 =  degree(q,x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p,q]

    # initialize
    k =  var('k')                               # index in summation formula
    u_list = []                                 # of elements (-1)**u_i
    subres_l = [p, q]                           # mod. subr. prs output list
    a0, a1 = p, q                               # the input polys
    del0 = d0 - d1                              # degree difference
    degdif = del0                               # save it
    rho_1 = LC(a0)                              # lead. coeff (a0)

    # Initialize Pell-Gordon variables
    rho_list_minus_1 =  sign( LC(a0, x))      # sign of LC(a0)
    rho1 =  LC(a1, x)                         # leading coeff of a1
    rho_list = [ sign(rho1)]                  # of signs
    p_list = [del0]                           # of degree differences
    u =  summation(k, (k, 1, p_list[0]))      # value of u
    u_list.append(u)                          # of u values
    v = sum(p_list)                           # v value

    # first remainder
    exp_deg = d1 - 1                          # expected degree of a2
    a2 = - rem(a0, a1, domain=QQ)             # first remainder
    rho2 =  LC(a2, x)                         # leading coeff of a2
    d2 =  degree(a2, x)                       # actual degree of a2
    deg_diff_new = exp_deg - d2               # expected - actual degree
    del1 = d1 - d2                            # degree difference

    # mul_fac is the factor by which a2 is multiplied to
    # get integer coefficients
    mul_fac_old = rho1**(del0 + del1 - deg_diff_new)

    # update Pell-Gordon variables
    p_list.append(1 + deg_diff_new)              # deg_diff_new is 0 for complete seq

    # apply Pell-Gordon formula (7) in second reference
    num = 1                                     # numerator of fraction
    for u in u_list:
        num *= (-1)**u
    num = num * (-1)**v

    # denominator depends on complete / incomplete seq
    if deg_diff_new == 0:                        # complete seq
        den = 1
        for k in range(len(rho_list)):
            den *= rho_list[k]**(p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
    else:                                       # incomplete seq
        den = 1
        for k in range(len(rho_list)-1):
            den *= rho_list[k]**(p_list[k] + p_list[k + 1])
        den = den * rho_list_minus_1
        expo = (p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new)
        den = den * rho_list[len(rho_list) - 1]**expo

    # the sign of the determinant depends on sg(num / den)
    if  sign(num / den) > 0:
        subres_l.append( simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )
    else:
        subres_l.append(- simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )

    # update Pell-Gordon variables
    k = var('k')
    rho_list.append( sign(rho2))
    u =  summation(k, (k, 1, p_list[len(p_list) - 1]))
    u_list.append(u)
    v = sum(p_list)
    deg_diff_old=deg_diff_new

    # main loop
    while d2 > 0:
        a0, a1, d0, d1 = a1, a2, d1, d2      # update polys and degrees
        del0 = del1                          # update degree difference
        exp_deg = d1 - 1                     # new expected degree
        a2 = - rem(a0, a1, domain=QQ)        # new remainder
        rho3 =  LC(a2, x)                    # leading coeff of a2
        d2 =  degree(a2, x)                  # actual degree of a2
        deg_diff_new = exp_deg - d2          # expected - actual degree
        del1 = d1 - d2                       # degree difference

        # take into consideration the power
        # rho1**deg_diff_old that was "left out"
        expo_old = deg_diff_old               # rho1 raised to this power
        expo_new = del0 + del1 - deg_diff_new # rho2 raised to this power

        mul_fac_new = rho2**(expo_new) * rho1**(expo_old) * mul_fac_old

        # update variables
        deg_diff_old, mul_fac_old = deg_diff_new, mul_fac_new
        rho1, rho2 = rho2, rho3

        # update Pell-Gordon variables
        p_list.append(1 + deg_diff_new)       # deg_diff_new is 0 for complete seq

        # apply Pell-Gordon formula (7) in second reference
        num = 1                              # numerator
        for u in u_list:
            num *= (-1)**u
        num = num * (-1)**v

        # denominator depends on complete / incomplete seq
        if deg_diff_new == 0:                 # complete seq
            den = 1
            for k in range(len(rho_list)):
                den *= rho_list[k]**(p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
        else:                                # incomplete seq
            den = 1
            for k in range(len(rho_list)-1):
                den *= rho_list[k]**(p_list[k] + p_list[k + 1])
            den = den * rho_list_minus_1
            expo = (p_list[len(rho_list) - 1] + p_list[len(rho_list)] - deg_diff_new)
            den = den * rho_list[len(rho_list) - 1]**expo

        # the sign of the determinant depends on sg(num / den)
        if  sign(num / den) > 0:
            subres_l.append( simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )
        else:
            subres_l.append(- simplify(rho_1**degdif*a2* Abs(mul_fac_old) ) )

        # update Pell-Gordon variables
        k = var('k')
        rho_list.append( sign(rho2))
        u =  summation(k, (k, 1, p_list[len(p_list) - 1]))
        u_list.append(u)
        v = sum(p_list)

    # gcd is of degree > 0 ?
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    # LC( p ) < 0
    m = len(subres_l)   # list may be shorter now due to deg(gcd ) > 0
    if LC( p ) < 0:
        aux_seq = [subres_l[0], subres_l[1]]
        for i in range(2, m):
            aux_seq.append(simplify(subres_l[i] * (-1) ))
        subres_l = aux_seq

    return  subres_l

def subresultants_pg(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x], from
    the modified subresultant prs of p and q.

    The coefficients of the polynomials in these two sequences differ only
    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in
    Theorem 2 of the reference.

    The coefficients of the polynomials in the output sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials."
    Serdica Journal of Computing 9(2) (2015), 123-138.

    """
    # compute the modified subresultant prs
    lst = modified_subresultants_pg(p,q,x)  ## any other method would do

    # defensive
    if lst == [] or len(lst) == 2:
        return lst

    # the coefficients in lst are modified subresultants and, hence, are
    # greater than those of the corresponding subresultants by the factor
    # LC(lst[0])**( deg(lst[0]) - deg(lst[1])); see Theorem 2 in reference.
    lcf = LC(lst[0])**( degree(lst[0], x) - degree(lst[1], x) )

    # Initialize the subresultant prs list
    subr_seq = [lst[0], lst[1]]

    # compute the degree sequences m_i and j_i of Theorem 2 in reference.
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m-1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]

    # compute the AMV factors of Theorem 2 in reference.
    fact = [(-1)**( j*(j-1)/S(2) ) for j in j_seq]

    # shortened list without the first two polys
    lst_s = lst[2:]

    # poly lst_s[k] is multiplied times fact[k], divided by lcf
    # and appended to the subresultant prs list
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append(-lst_s[k] / lcf)
        else:
            subr_seq.append(lst_s[k] / lcf)

    return subr_seq

def subresultants_amv_q(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Q[x] are performed, using the
    function rem(p, q, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Akritas-Malaschonok-Vigklas Theorem of 2015.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    d0 =  degree(p, x)
    d1 =  degree(q, x)
    if d0 == 0 and d1 == 0:
        return [p, q]
    if d1 > d0:
        d0, d1 = d1, d0
        p, q = q, p
    if d0 > 0 and d1 == 0:
        return [p, q]

    # initialize
    i, s = 0, 0                              # counters for remainders & odd elements
    p_odd_index_sum = 0                      # contains the sum of p_1, p_3, etc
    subres_l = [p, q]                        # subresultant prs output list
    a0, a1 = p, q                            # the input polys
    sigma1 =  LC(a1, x)                      # leading coeff of a1
    p0 = d0 - d1                             # degree difference
    if p0 % 2 == 1:
        s += 1
    phi = floor( (s + 1) / 2 )
    mul_fac = 1
    d2 = d1

    # main loop
    while d2 > 0:
        i += 1
        a2 = rem(a0, a1, domain= QQ)          # new remainder
        if i == 1:
            sigma2 =  LC(a2, x)
        else:
            sigma3 =  LC(a2, x)
            sigma1, sigma2 = sigma2, sigma3
        d2 =  degree(a2, x)
        p1 = d1 - d2
        psi = i + phi + p_odd_index_sum

        # new mul_fac
        mul_fac = sigma1**(p0 + 1) * mul_fac

        ## compute the sign of the first fraction in formula (9) of the paper
        # numerator
        num = (-1)**psi
        # denominator
        den = sign(mul_fac)

        # the sign of the determinant depends on sign( num / den ) != 0
        if  sign(num / den) > 0:
            subres_l.append( simplify(expand(a2* Abs(mul_fac))))
        else:
            subres_l.append(- simplify(expand(a2* Abs(mul_fac))))

        ## bring into mul_fac the missing power of sigma if there was a degree gap
        if p1 - 1 > 0:
            mul_fac = mul_fac * sigma1**(p1 - 1)

       # update AMV variables
        a0, a1, d0, d1 = a1, a2, d1, d2
        p0 = p1
        if p0 % 2 ==1:
            s += 1
        phi = floor( (s + 1) / 2 )
        if i%2 == 1:
            p_odd_index_sum += p0             # p_i has odd index

    # gcd is of degree > 0 ?
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    return  subres_l

def compute_sign(base, expo):
    '''
    base != 0 and expo >= 0 are integers;

    returns the sign of base**expo without
    evaluating the power itself!
    '''
    sb = sign(base)
    if sb == 1:
        return 1
    pe = expo % 2
    if pe == 0:
        return -sb
    else:
        return sb

def rem_z(p, q, x):
    '''
    Intended mainly for p, q polynomials in Z[x] so that,
    on dividing p by q, the remainder will also be in Z[x]. (However,
    it also works fine for polynomials in Q[x].) It is assumed
    that degree(p, x) >= degree(q, x).

    It premultiplies p by the _absolute_ value of the leading coefficient
    of q, raised to the power deg(p) - deg(q) + 1 and then performs
    polynomial division in Q[x], using the function rem(p, q, x).

    By contrast the function prem(p, q, x) does _not_ use the absolute
    value of the leading coefficient of q.
    This results not only in ``messing up the signs'' of the Euclidean and
    Sturmian prs's as mentioned in the second reference,
    but also in violation of the main results of the first and third
    references --- Theorem 4 and Theorem 1 respectively. Theorems 4 and 1
    establish a one-to-one correspondence between the Euclidean and the
    Sturmian prs of p, q, on one hand, and the subresultant prs of p, q,
    on the other.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials.''
    Serdica Journal of Computing, 9(2) (2015), 123-138.

    2. https://planetMath.org/sturmstheorem

    3. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result on
    the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    '''
    if (p.as_poly().is_univariate and q.as_poly().is_univariate and
            p.as_poly().gens == q.as_poly().gens):
        delta = (degree(p, x) - degree(q, x) + 1)
        return rem(Abs(LC(q, x))**delta  *  p, q, x)
    else:
        return prem(p, q, x)

def quo_z(p, q, x):
    """
    Intended mainly for p, q polynomials in Z[x] so that,
    on dividing p by q, the quotient will also be in Z[x]. (However,
    it also works fine for polynomials in Q[x].) It is assumed
    that degree(p, x) >= degree(q, x).

    It premultiplies p by the _absolute_ value of the leading coefficient
    of q, raised to the power deg(p) - deg(q) + 1 and then performs
    polynomial division in Q[x], using the function quo(p, q, x).

    By contrast the function pquo(p, q, x) does _not_ use the absolute
    value of the leading coefficient of q.

    See also function rem_z(p, q, x) for additional comments and references.

    """
    if (p.as_poly().is_univariate and q.as_poly().is_univariate and
            p.as_poly().gens == q.as_poly().gens):
        delta = (degree(p, x) - degree(q, x) + 1)
        return quo(Abs(LC(q, x))**delta  *  p, q, x)
    else:
        return pquo(p, q, x)

def subresultants_amv(f, g, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(f, x) >= degree(g, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients, no determinant evaluation takes place.
    Instead, polynomial divisions in Z[x] or Q[x] are performed, using
    the function rem_z(p, q, x);  the coefficients of the remainders
    computed this way become subresultants with the help of the
    Akritas-Malaschonok-Vigklas Theorem of 2015 and the Collins-Brown-
    Traub formula for coefficient reduction.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``A Basic Result
    on the Theory of Subresultants.'' Serdica Journal of Computing 10 (2016), No.1, 31-48.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Subresultant Polynomial
    remainder Sequences Obtained by Polynomial Divisions in Q[x] or in Z[x].''
    Serdica Journal of Computing 10 (2016), No.3-4, 197-217.

    """
    # make sure neither f nor g is 0
    if f == 0 or g == 0:
        return [f, g]

    # make sure proper degrees
    d0 =  degree(f, x)
    d1 =  degree(g, x)
    if d0 == 0 and d1 == 0:
        return [f, g]
    if d1 > d0:
        d0, d1 = d1, d0
        f, g = g, f
    if d0 > 0 and d1 == 0:
        return [f, g]

    # initialize
    a0 = f
    a1 = g
    subres_l = [a0, a1]
    deg_dif_p1, c = degree(a0, x) - degree(a1, x) + 1, -1

    # initialize AMV variables
    sigma1 =  LC(a1, x)                      # leading coeff of a1
    i, s = 0, 0                              # counters for remainders & odd elements
    p_odd_index_sum = 0                      # contains the sum of p_1, p_3, etc
    p0 = deg_dif_p1 - 1
    if p0 % 2 == 1:
        s += 1
    phi = floor( (s + 1) / 2 )

    # compute the first polynomial of the prs
    i += 1
    a2 = rem_z(a0, a1, x) / Abs( (-1)**deg_dif_p1 )     # first remainder
    sigma2 =  LC(a2, x)                       # leading coeff of a2
    d2 =  degree(a2, x)                       # actual degree of a2
    p1 = d1 - d2                              # degree difference

    # sgn_den is the factor, the denominator 1st fraction of (9),
    # by which a2 is multiplied to get integer coefficients
    sgn_den = compute_sign( sigma1, p0 + 1 )

    ## compute sign of the 1st fraction in formula (9) of the paper
    # numerator
    psi = i + phi + p_odd_index_sum
    num = (-1)**psi
    # denominator
    den = sgn_den

    # the sign of the determinant depends on sign(num / den) != 0
    if  sign(num / den) > 0:
        subres_l.append( a2 )
    else:
        subres_l.append( -a2 )

    # update AMV variable
    if p1 % 2 == 1:
        s += 1

    # bring in the missing power of sigma if there was gap
    if p1 - 1 > 0:
        sgn_den = sgn_den * compute_sign( sigma1, p1 - 1 )

    # main loop
    while d2 >= 1:
        phi = floor( (s + 1) / 2 )
        if i%2 == 1:
            p_odd_index_sum += p1             # p_i has odd index
        a0, a1, d0, d1 = a1, a2, d1, d2       # update polys and degrees
        p0 = p1                               # update degree difference
        i += 1
        sigma0 = -LC(a0)
        c = (sigma0**(deg_dif_p1 - 1)) / (c**(deg_dif_p1 - 2))
        deg_dif_p1 = degree(a0, x) - d2 + 1
        a2 = rem_z(a0, a1, x) / Abs( (c**(deg_dif_p1 - 1)) * sigma0 )
        sigma3 =  LC(a2, x)                   # leading coeff of a2
        d2 =  degree(a2, x)                   # actual degree of a2
        p1 = d1 - d2                          # degree difference
        psi = i + phi + p_odd_index_sum

        # update variables
        sigma1, sigma2 = sigma2, sigma3

        # new sgn_den
        sgn_den = compute_sign( sigma1, p0 + 1 ) * sgn_den

        # compute the sign of the first fraction in formula (9) of the paper
        # numerator
        num = (-1)**psi
        # denominator
        den = sgn_den

        # the sign of the determinant depends on sign( num / den ) != 0
        if  sign(num / den) > 0:
            subres_l.append( a2 )
        else:
            subres_l.append( -a2 )

        # update AMV variable
        if p1 % 2 ==1:
            s += 1

        # bring in the missing power of sigma if there was gap
        if p1 - 1 > 0:
            sgn_den = sgn_den * compute_sign( sigma1, p1 - 1 )

    # gcd is of degree > 0 ?
    m = len(subres_l)
    if subres_l[m - 1] == nan or subres_l[m - 1] == 0:
        subres_l.pop(m - 1)

    return subres_l

def modified_subresultants_amv(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the modified subresultant prs of p and q in Z[x] or Q[x],
    from the subresultant prs of p and q.
    The coefficients of the polynomials in the two sequences differ only
    in sign and the factor LC(p)**( deg(p)- deg(q)) as stated in
    Theorem 2 of the reference.

    The coefficients of the polynomials in the output sequence are
    modified subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester2, Sylvester's matrix of 1853.

    If the modified subresultant prs is complete, and LC( p ) > 0, it coincides
    with the (generalized) Sturm's sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: "On the Remainders
    Obtained in Finding the Greatest Common Divisor of Two Polynomials."
    Serdica Journal of Computing, Serdica Journal of Computing, 9(2) (2015), 123-138.

    """
    # compute the subresultant prs
    lst = subresultants_amv(p,q,x)     ## any other method would do

    # defensive
    if lst == [] or len(lst) == 2:
        return lst

    # the coefficients in lst are subresultants and, hence, smaller than those
    # of the corresponding modified subresultants by the factor
    # LC(lst[0])**( deg(lst[0]) - deg(lst[1])); see Theorem 2.
    lcf = LC(lst[0])**( degree(lst[0], x) - degree(lst[1], x) )

    # Initialize the modified subresultant prs list
    subr_seq = [lst[0], lst[1]]

    # compute the degree sequences m_i and j_i of Theorem 2
    deg_seq = [degree(Poly(poly, x), x) for poly in lst]
    deg = deg_seq[0]
    deg_seq_s = deg_seq[1:-1]
    m_seq = [m-1 for m in deg_seq_s]
    j_seq = [deg - m for m in m_seq]

    # compute the AMV factors of Theorem 2
    fact = [(-1)**( j*(j-1)/S(2) ) for j in j_seq]

    # shortened list without the first two polys
    lst_s = lst[2:]

    # poly lst_s[k] is multiplied times fact[k] and times lcf
    # and appended to the subresultant prs list
    m = len(fact)
    for k in range(m):
        if sign(fact[k]) == -1:
            subr_seq.append( simplify(-lst_s[k] * lcf) )
        else:
            subr_seq.append( simplify(lst_s[k] * lcf) )

    return subr_seq

def correct_sign(deg_f, deg_g, s1, rdel, cdel):
    """
    Used in various subresultant prs algorithms.

    Evaluates the determinant, (a.k.a. subresultant) of a properly selected
    submatrix of s1, Sylvester's matrix of 1840, to get the correct sign
    and value of the leading coefficient of a given polynomial remainder.

    deg_f, deg_g are the degrees of the original polynomials p, q for which the
    matrix s1 = sylvester(p, q, x, 1) was constructed.

    rdel denotes the expected degree of the remainder; it is the number of
    rows to be deleted from each group of rows in s1 as described in the
    reference below.

    cdel denotes the expected degree minus the actual degree of the remainder;
    it is the number of columns to be deleted --- starting with the last column
    forming the square matrix --- from the matrix resulting after the row deletions.

    References
    ==========
    Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``Sturm Sequences
    and Modified Subresultant Polynomial Remainder Sequences.''
    Serdica Journal of Computing, Vol. 8, No 1, 29-46, 2014.

    """
    M = s1[:, :]   # copy of matrix s1

    # eliminate rdel rows from the first deg_g rows
    for i in range(M.rows - deg_f - 1, M.rows - deg_f - rdel - 1, -1):
        M.row_del(i)

    # eliminate rdel rows from the last deg_f rows
    for i in range(M.rows - 1, M.rows - rdel - 1, -1):
        M.row_del(i)

    # eliminate cdel columns
    for i in range(cdel):
        M.col_del(M.rows - 1)

    # define submatrix
    Md = M[:, 0: M.rows]

    return Md.det()

def subresultants_rem(p, q, x):
    """
    p, q are polynomials in Z[x] or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p and q in Z[x] or Q[x];
    the coefficients of the polynomials in the sequence are
    subresultants. That is, they are  determinants of appropriately
    selected submatrices of sylvester1, Sylvester's matrix of 1840.

    To compute the coefficients polynomial divisions in Q[x] are
    performed, using the function rem(p, q, x). The coefficients
    of the remainders computed this way become subresultants by evaluating
    one subresultant per remainder --- that of the leading coefficient.
    This way we obtain the correct sign and value of the leading coefficient
    of the remainder and we easily ``force'' the rest of the coefficients
    to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # initialize
    s1 = sylvester(f, g, x, 1)
    sr_list = [f, g]      # subresultant list

    # main loop
    while deg_g > 0:
        r = rem(p, q, x)
        d = degree(r, x)
        if d < 0:
            return sr_list

        # make coefficients subresultants evaluating ONE determinant
        exp_deg = deg_g - 1          # expected degree
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        r = simplify((r / LC(r, x)) * sign_value)

        # append poly with subresultant coeffs
        sr_list.append(r)

        # update degrees and polys
        deg_f, deg_g = deg_g, d
        p, q = q, r

    # gcd is of degree > 0 ?
    m = len(sr_list)
    if sr_list[m - 1] == nan or sr_list[m - 1] == 0:
        sr_list.pop(m - 1)

    return sr_list

def pivot(M, i, j):
    '''
    M is a matrix, and M[i, j] specifies the pivot element.

    All elements below M[i, j], in the j-th column, will
    be zeroed, if they are not already 0, according to
    Dodgson-Bareiss' integer preserving transformations.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    '''
    ma = M[:, :] # copy of matrix M
    rs = ma.rows # No. of rows
    cs = ma.cols # No. of cols
    for r in range(i+1, rs):
        if ma[r, j] != 0:
            for c in range(j + 1, cs):
                ma[r, c] = ma[i, j] * ma[r, c] - ma[i, c] * ma[r, j]
            ma[r, j] = 0
    return ma

def rotate_r(L, k):
    '''
    Rotates right by k. L is a row of a matrix or a list.

    '''
    ll = list(L)
    if ll == []:
        return []
    for i in range(k):
        el = ll.pop(len(ll) - 1)
        ll.insert(0, el)
    return ll if isinstance(L, list) else Matrix([ll])

def rotate_l(L, k):
    '''
    Rotates left by k. L is a row of a matrix or a list.

    '''
    ll = list(L)
    if ll == []:
        return []
    for i in range(k):
        el = ll.pop(0)
        ll.insert(len(ll) - 1, el)
    return ll if isinstance(L, list) else Matrix([ll])

def row2poly(row, deg, x):
    '''
    Converts the row of a matrix to a poly of degree deg and variable x.
    Some entries at the beginning and/or at the end of the row may be zero.

    '''
    k = 0
    poly = []
    leng = len(row)

    # find the beginning of the poly ; i.e. the first
    # non-zero element of the row
    while row[k] == 0:
        k = k + 1

    # append the next deg + 1 elements to poly
    for j in range( deg + 1):
        if k + j <= leng:
            poly.append(row[k + j])

    return Poly(poly, x)

def create_ma(deg_f, deg_g, row1, row2, col_num):
    '''
    Creates a ``small'' matrix M to be triangularized.

    deg_f, deg_g are the degrees of the divident and of the
    divisor polynomials respectively, deg_g > deg_f.

    The coefficients of the divident poly are the elements
    in row2 and those of the divisor poly are the elements
    in row1.

    col_num defines the number of columns of the matrix M.

    '''
    if deg_g - deg_f >= 1:
        print('Reverse degrees')
        return

    m = zeros(deg_f - deg_g + 2, col_num)

    for i in range(deg_f - deg_g + 1):
        m[i, :] = rotate_r(row1, i)
    m[deg_f - deg_g + 1, :] = row2

    return m

def find_degree(M, deg_f):
    '''
    Finds the degree of the poly corresponding (after triangularization)
    to the _last_ row of the ``small'' matrix M, created by create_ma().

    deg_f is the degree of the divident poly.
    If _last_ row is all 0's returns None.

    '''
    j = deg_f
    for i in range(0, M.cols):
        if M[M.rows - 1, i] == 0:
            j = j - 1
        else:
            return j if j >= 0 else 0

def final_touches(s2, r, deg_g):
    """
    s2 is sylvester2, r is the row pointer in s2,
    deg_g is the degree of the poly last inserted in s2.

    After a gcd of degree > 0 has been found with Van Vleck's
    method, and was inserted into s2, if its last term is not
    in the last column of s2, then it is inserted as many
    times as needed, rotated right by one each time, until
    the condition is met.

    """
    R = s2.row(r-1)

    # find the first non zero term
    for i in range(s2.cols):
        if R[0,i] == 0:
            continue
        else:
            break

    # missing rows until last term is in last column
    mr = s2.cols - (i + deg_g + 1)

    # insert them by replacing the existing entries in the row
    i = 0
    while mr != 0 and r + i < s2.rows :
        s2[r + i, : ] = rotate_r(R, i + 1)
        i += 1
        mr -= 1

    return s2

def subresultants_vv(p, q, x, method = 0):
    """
    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p, q by triangularizing,
    in Z[x] or in Q[x], all the smaller matrices encountered in the
    process of triangularizing sylvester2, Sylvester's matrix of 1853;
    see references 1 and 2 for Van Vleck's method. With each remainder,
    sylvester2 gets updated and is prepared to be printed if requested.

    If sylvester2 has small dimensions and you want to see the final,
    triangularized matrix use this version with method=1; otherwise,
    use either this version with method=0 (default) or the faster version,
    subresultants_vv_2(p, q, x), where sylvester2 is used implicitly.

    Sylvester's matrix sylvester1  is also used to compute one
    subresultant per remainder; namely, that of the leading
    coefficient, in order to obtain the correct sign and to
    force the remainder coefficients to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    If the final, triangularized matrix s2 is printed, then:
        (a) if deg(p) - deg(q) > 1 or deg( gcd(p, q) ) > 0, several
            of the last rows in s2 will remain unprocessed;
        (b) if deg(p) - deg(q) == 0, p will not appear in the final matrix.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    3. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # initialize
    s1 = sylvester(f, g, x, 1)
    s2 = sylvester(f, g, x, 2)
    sr_list = [f, g]
    col_num = 2 * n         # columns in s2

    # make two rows (row0, row1) of poly coefficients
    row0 = Poly(f, x, domain = QQ).all_coeffs()
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)
    row0 = Matrix([row0])
    row1 = Poly(g,x, domain = QQ).all_coeffs()
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)
    row1 = Matrix([row1])

    # row pointer for deg_f - deg_g == 1; may be reset below
    r = 2

    # modify first rows of s2 matrix depending on poly degrees
    if deg_f - deg_g > 1:
        r = 1
        # replacing the existing entries in the rows of s2,
        # insert row0 (deg_f - deg_g - 1) times, rotated each time
        for i in range(deg_f - deg_g - 1):
            s2[r + i, : ] = rotate_r(row0, i + 1)
        r = r + deg_f - deg_g - 1
        # insert row1 (deg_f - deg_g) times, rotated each time
        for i in range(deg_f - deg_g):
            s2[r + i, : ] = rotate_r(row1, r + i)
        r = r + deg_f - deg_g

    if deg_f - deg_g == 0:
        r = 0

    # main loop
    while deg_g > 0:
        # create a small matrix M, and triangularize it;
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        # will need only the first and last rows of M
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]

        # treat last row of M as poly; find its degree
        d = find_degree(M, deg_f)
        if d is None:
            break
        exp_deg = deg_g - 1

        # evaluate one determinant & make coefficients subresultants
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        temp2 = LC(poly, x)
        poly = simplify((poly / temp2) * sign_value)

        # update s2 by inserting first row of M as needed
        row0 = M[0, :]
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row0, r + i)
        r = r + deg_g - d

        # update s2 by inserting last row of M as needed
        row1 = rotate_l(M[M.rows - 1, :], deg_f - d)
        row1 = (row1 / temp2) * sign_value
        for i in range(deg_g - d):
            s2[r + i, :] = rotate_r(row1, r + i)
        r = r + deg_g - d

        # update degrees
        deg_f, deg_g = deg_g, d

        # append poly with subresultant coeffs
        sr_list.append(poly)

    # final touches to print the s2 matrix
    if method != 0 and s2.rows > 2:
        s2 = final_touches(s2, r, deg_g)
        pprint(s2)
    elif method != 0 and s2.rows == 2:
        s2[1, :] = rotate_r(s2.row(1), 1)
        pprint(s2)

    return sr_list

def subresultants_vv_2(p, q, x):
    """
    p, q are polynomials in Z[x] (intended) or Q[x]. It is assumed
    that degree(p, x) >= degree(q, x).

    Computes the subresultant prs of p, q by triangularizing,
    in Z[x] or in Q[x], all the smaller matrices encountered in the
    process of triangularizing sylvester2, Sylvester's matrix of 1853;
    see references 1 and 2 for Van Vleck's method.

    If the sylvester2 matrix has big dimensions use this version,
    where sylvester2 is used implicitly. If you want to see the final,
    triangularized matrix sylvester2, then use the first version,
    subresultants_vv(p, q, x, 1).

    sylvester1, Sylvester's matrix of 1840, is also used to compute
    one subresultant per remainder; namely, that of the leading
    coefficient, in order to obtain the correct sign and to
    ``force'' the remainder coefficients to become subresultants.

    If the subresultant prs is complete, then it coincides with the
    Euclidean sequence of the polynomials p, q.

    References
    ==========
    1. Akritas, A. G.: ``A new method for computing polynomial greatest
    common divisors and polynomial remainder sequences.''
    Numerische MatheMatik 52, 119-127, 1988.

    2. Akritas, A. G., G.I. Malaschonok and P.S. Vigklas: ``On a Theorem
    by Van Vleck Regarding Sturm Sequences.''
    Serdica Journal of Computing, 7, No 4, 101-134, 2013.

    3. Akritas, A. G.:``Three New Methods for Computing Subresultant
    Polynomial Remainder Sequences (PRS's).'' Serdica Journal of Computing 9(1) (2015), 1-26.

    """
    # make sure neither p nor q is 0
    if p == 0 or q == 0:
        return [p, q]

    # make sure proper degrees
    f, g = p, q
    n = deg_f = degree(f, x)
    m = deg_g = degree(g, x)
    if n == 0 and m == 0:
        return [f, g]
    if n < m:
        n, m, deg_f, deg_g, f, g = m, n, deg_g, deg_f, g, f
    if n > 0 and m == 0:
        return [f, g]

    # initialize
    s1 = sylvester(f, g, x, 1)
    sr_list = [f, g]      # subresultant list
    col_num = 2 * n       # columns in sylvester2

    # make two rows (row0, row1) of poly coefficients
    row0 = Poly(f, x, domain = QQ).all_coeffs()
    leng0 = len(row0)
    for i in range(col_num - leng0):
        row0.append(0)
    row0 = Matrix([row0])
    row1 = Poly(g,x, domain = QQ).all_coeffs()
    leng1 = len(row1)
    for i in range(col_num - leng1):
        row1.append(0)
    row1 = Matrix([row1])

    # main loop
    while deg_g > 0:
        # create a small matrix M, and triangularize it
        M = create_ma(deg_f, deg_g, row1, row0, col_num)
        for i in range(deg_f - deg_g + 1):
            M1 = pivot(M, i, i)
            M = M1[:, :]

        # treat last row of M as poly; find its degree
        d = find_degree(M, deg_f)
        if d is None:
            return sr_list
        exp_deg = deg_g - 1

        # evaluate one determinant & make coefficients subresultants
        sign_value = correct_sign(n, m, s1, exp_deg, exp_deg - d)
        poly = row2poly(M[M.rows - 1, :], d, x)
        poly = simplify((poly / LC(poly, x)) * sign_value)

        # append poly with subresultant coeffs
        sr_list.append(poly)

        # update degrees and rows
        deg_f, deg_g = deg_g, d
        row0 = row1
        row1 = Poly(poly, x, domain = QQ).all_coeffs()
        leng1 = len(row1)
        for i in range(col_num - leng1):
            row1.append(0)
        row1 = Matrix([row1])

    return sr_list
