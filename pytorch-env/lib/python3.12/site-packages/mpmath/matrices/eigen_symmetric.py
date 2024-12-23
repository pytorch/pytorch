#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################
#     module for the symmetric eigenvalue problem
#       Copyright 2013 Timo Hartmann (thartmann15 at gmail.com)
#
# todo:
#  - implement balancing
#
##################################################################################################

"""
The symmetric eigenvalue problem.
---------------------------------

This file contains routines for the symmetric eigenvalue problem.

high level routines:

  eigsy : real symmetric (ordinary) eigenvalue problem
  eighe : complex hermitian (ordinary) eigenvalue problem
  eigh  : unified interface for eigsy and eighe
  svd_r : singular value decomposition for real matrices
  svd_c : singular value decomposition for complex matrices
  svd   : unified interface for svd_r and svd_c


low level routines:

  r_sy_tridiag : reduction of real symmetric matrix to real symmetric tridiagonal matrix
  c_he_tridiag_0 : reduction of complex hermitian matrix to real symmetric tridiagonal matrix
  c_he_tridiag_1 : auxiliary routine to c_he_tridiag_0
  c_he_tridiag_2 : auxiliary routine to c_he_tridiag_0
  tridiag_eigen : solves the real symmetric tridiagonal matrix eigenvalue problem
  svd_r_raw : raw singular value decomposition for real matrices
  svd_c_raw : raw singular value decomposition for complex matrices
"""

from ..libmp.backend import xrange
from .eigen import defun


def r_sy_tridiag(ctx, A, D, E, calc_ev = True):
    """
    This routine transforms a real symmetric matrix A to a real symmetric
    tridiagonal matrix T using an orthogonal similarity transformation:
          Q' * A * Q = T     (here ' denotes the matrix transpose).
    The orthogonal matrix Q is build up from Householder reflectors.

    parameters:
      A         (input/output) On input, A contains the real symmetric matrix of
                dimension (n,n). On output, if calc_ev is true, A contains the
                orthogonal matrix Q, otherwise A is destroyed.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      calc_ev   (input) If calc_ev is true, this routine explicitly calculates the
                orthogonal matrix Q which is then returned in A. If calc_ev is
                false, Q is not explicitly calculated resulting in a shorter run time.

    This routine is a python translation of the fortran routine tred2.f in the
    software library EISPACK (see netlib.org) which itself is based on the algol
    procedure tred2 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """

    # note : the vector v of the i-th houshoulder reflector is stored in a[(i+1):,i]
    #        whereas v/<v,v> is stored in a[i,(i+1):]

    n = A.rows
    for i in xrange(n - 1, 0, -1):
        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(A[k,i])

        scale_inv = 0
        if scale != 0:
            scale_inv = 1/scale

        # sadly there are floating point numbers not equal to zero whose reciprocal is infinity

        if i == 1 or scale == 0 or ctx.isinf(scale_inv):
            E[i] = A[i-1,i]        # nothing to do
            D[i] = 0
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[k,i] *= scale_inv
            H += A[k,i] * A[k,i]

        F = A[i-1,i]
        G = ctx.sqrt(H)
        if F > 0:
            G = -G
        E[i] = scale * G
        H -= F * G
        A[i-1,i] = F - G
        F = 0

        # apply housholder transformation

        for j in xrange(0, i):
            if calc_ev:
                A[i,j] = A[j,i] / H

            G = 0                  # calculate A*U
            for k in xrange(0, j + 1):
                G += A[k,j] * A[k,i]
            for k in xrange(j + 1, i):
                G += A[j,k] * A[k,i]

            E[j] = G / H           # calculate P
            F += E[j] * A[j,i]

        HH = F / (2 * H)

        for j in xrange(0, i):     # calculate reduced A
            F = A[j,i]
            G = E[j] - HH * F      # calculate Q
            E[j] = G

            for k in xrange(0, j + 1):
                A[k,j] -= F * E[k] + G * A[k,i]

        D[i] = H

    for i in xrange(1, n):         # better for compatibility
        E[i-1] = E[i]
    E[n-1] = 0

    if calc_ev:
        D[0] = 0
        for i in xrange(0, n):
            if D[i] != 0:
                for j in xrange(0, i):     # accumulate transformation matrices
                    G = 0
                    for k in xrange(0, i):
                        G += A[i,k] * A[k,j]
                    for k in xrange(0, i):
                        A[k,j] -= G * A[k,i]

            D[i] = A[i,i]
            A[i,i] = 1

            for j in xrange(0, i):
                A[j,i] = A[i,j] = 0
    else:
        for i in xrange(0, n):
            D[i] = A[i,i]





def c_he_tridiag_0(ctx, A, D, E, T):
    """
    This routine transforms a complex hermitian matrix A to a real symmetric
    tridiagonal matrix T using an unitary similarity transformation:
          Q' * A * Q = T     (here ' denotes the hermitian matrix transpose,
                              i.e. transposition und conjugation).
    The unitary matrix Q is build up from Householder reflectors and
    an unitary diagonal matrix.

    parameters:
      A         (input/output) On input, A contains the complex hermitian matrix
                of dimension (n,n). On output, A contains the unitary matrix Q
                in compressed form.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix.

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      T         (output) complex array of length n, contains a unitary diagonal
                matrix.

    This routine is a python translation (in slightly modified form) of the fortran
    routine htridi.f in the software library EISPACK (see netlib.org) which itself
    is a complex version of the algol procedure tred1 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """

    n = A.rows
    T[n-1] = 1
    for i in xrange(n - 1, 0, -1):

        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(ctx.re(A[k,i])) + abs(ctx.im(A[k,i]))

        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale

        # sadly there are floating point numbers not equal to zero whose reciprocal is infinity

        if scale == 0 or ctx.isinf(scale_inv):
            E[i] = 0
            D[i] = 0
            T[i-1] = 1
            continue

        if i == 1:
            F = A[i-1,i]
            f = abs(F)
            E[i] = f
            D[i] = 0
            if f != 0:
                T[i-1] = T[i] * F / f
            else:
                T[i-1] = T[i]
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[k,i] *= scale_inv
            rr = ctx.re(A[k,i])
            ii = ctx.im(A[k,i])
            H += rr * rr + ii * ii

        F = A[i-1,i]
        f = abs(F)
        G = ctx.sqrt(H)
        H += G * f
        E[i] = scale * G
        if f != 0:
            F = F / f
            TZ = - T[i] * F              # T[i-1]=-T[i]*F, but we need T[i-1] as temporary storage
            G *= F
        else:
            TZ = -T[i]                   # T[i-1]=-T[i]
        A[i-1,i] += G
        F = 0

        # apply housholder transformation

        for j in xrange(0, i):
            A[i,j] = A[j,i] / H

            G = 0                        # calculate A*U
            for k in xrange(0, j + 1):
                G += ctx.conj(A[k,j]) * A[k,i]
            for k in xrange(j + 1, i):
                G += A[j,k] * A[k,i]

            T[j] = G / H                 # calculate P
            F += ctx.conj(T[j]) * A[j,i]

        HH = F / (2 * H)

        for j in xrange(0, i):           # calculate reduced A
            F = A[j,i]
            G = T[j] - HH * F            # calculate Q
            T[j] = G

            for k in xrange(0, j + 1):
                A[k,j] -= ctx.conj(F) * T[k] + ctx.conj(G) * A[k,i]
                # as we use the lower left part for storage
                # we have to use the transpose of the normal formula

        T[i-1] = TZ
        D[i] = H

    for i in xrange(1, n):                # better for compatibility
        E[i-1] = E[i]
    E[n-1] = 0

    D[0] = 0
    for i in xrange(0, n):
        zw = D[i]
        D[i] = ctx.re(A[i,i])
        A[i,i] = zw







def c_he_tridiag_1(ctx, A, T):
    """
    This routine forms the unitary matrix Q described in c_he_tridiag_0.

    parameters:
      A    (input/output) On input, A is the same matrix as delivered by
           c_he_tridiag_0. On output, A is set to Q.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

    """

    n = A.rows

    for i in xrange(0, n):
        if A[i,i] != 0:
            for j in xrange(0, i):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i,k]) * A[k,j]
                for k in xrange(0, i):
                    A[k,j] -= G * A[k,i]

        A[i,i] = 1

        for j in xrange(0, i):
            A[j,i] = A[i,j] = 0

    for i in xrange(0, n):
        for k in xrange(0, n):
            A[i,k] *= T[k]




def c_he_tridiag_2(ctx, A, T, B):
    """
    This routine applied the unitary matrix Q described in c_he_tridiag_0
    onto the the matrix B, i.e. it forms Q*B.

    parameters:
      A    (input) On input, A is the same matrix as delivered by c_he_tridiag_0.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

      B    (input/output) On input, B is a complex matrix. On output B is replaced
           by Q*B.

    This routine is a python translation of the fortran routine htribk.f in the
    software library EISPACK (see netlib.org). See c_he_tridiag_0 for more
    references.
    """

    n = A.rows

    for i in xrange(0, n):
        for k in xrange(0, n):
            B[k,i] *= T[k]

    for i in xrange(0, n):
        if A[i,i] != 0:
            for j in xrange(0, n):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i,k]) * B[k,j]
                for k in xrange(0, i):
                    B[k,j] -= G * A[k,i]





def tridiag_eigen(ctx, d, e, z = False):
    """
    This subroutine find the eigenvalues and the first components of the
    eigenvectors of a real symmetric tridiagonal matrix using the implicit
    QL method.

    parameters:

      d (input/output) real array of length n. on input, d contains the diagonal
        elements of the input matrix. on output, d contains the eigenvalues in
        ascending order.

      e (input) real array of length n. on input, e contains the offdiagonal
        elements of the input matrix in e[0:(n-1)]. On output, e has been
        destroyed.

      z (input/output) If z is equal to False, no eigenvectors will be computed.
        Otherwise on input z should have the format z[0:m,0:n] (i.e. a real or
        complex matrix of dimension (m,n) ). On output this matrix will be
        multiplied by the matrix of the eigenvectors (i.e. the columns of this
        matrix are the eigenvectors): z --> z*EV
        That means if z[i,j]={1 if j==j; 0 otherwise} on input, then on output
        z will contain the first m components of the eigenvectors. That means
        if m is equal to n, the i-th eigenvector will be z[:,i].

    This routine is a python translation (in slightly modified form) of the
    fortran routine imtql2.f in the software library EISPACK (see netlib.org)
    which itself is based on the algol procudure imtql2 desribed in:
     - num. math. 12, p. 377-383(1968) by matrin and wilkinson
     - modified in num. math. 15, p. 450(1970) by dubrulle
     - handbook for auto. comp., vol. II-linear algebra, p. 241-248 (1971)
    See also the routine gaussq.f in netlog.org or acm algorithm 726.
    """

    n = len(d)
    e[n-1] = 0
    iterlim = 2 * ctx.dps

    for l in xrange(n):
        j = 0
        while 1:
            m = l
            while 1:
                # look for a small subdiagonal element
                if m + 1 == n:
                    break
                if abs(e[m]) <= ctx.eps * (abs(d[m]) + abs(d[m + 1])):
                    break
                m = m + 1
            if m == l:
                break

            if j >= iterlim:
                raise RuntimeError("tridiag_eigen: no convergence to an eigenvalue after %d iterations" % iterlim)

            j += 1

            # form shift

            p = d[l]
            g = (d[l + 1] - p) / (2 * e[l])
            r = ctx.hypot(g, 1)

            if g < 0:
                s = g - r
            else:
                s = g + r

            g = d[m] - p + e[l] / s

            s, c, p = 1, 1, 0

            for i in xrange(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) > abs(g):             # this here is a slight improvement also used in gaussq.f or acm algorithm 726.
                    c = g / f
                    r = ctx.hypot(c, 1)
                    e[i + 1] = f * r
                    s = 1 / r
                    c = c * s
                else:
                    s = f / g
                    r = ctx.hypot(s, 1)
                    e[i + 1] = g * r
                    c = 1 / r
                    s = s * c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                if not isinstance(z, bool):
                    # calculate eigenvectors
                    for w in xrange(z.rows):
                        f = z[w,i+1]
                        z[w,i+1] = s * z[w,i] + c * f
                        z[w,i  ] = c * z[w,i] - s * f

            d[l] = d[l] - p
            e[l] = g
            e[m] = 0

    for ii in xrange(1, n):
        # sort eigenvalues and eigenvectors (bubble-sort)
        i = ii - 1
        k = i
        p = d[i]
        for j in xrange(ii, n):
            if d[j] >= p:
                continue
            k = j
            p = d[k]
        if k == i:
            continue
        d[k] = d[i]
        d[i] = p

        if not isinstance(z, bool):
            for w in xrange(z.rows):
                p = z[w,i]
                z[w,i] = z[w,k]
                z[w,k] = p

########################################################################################

@defun
def eigsy(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    This routine solves the (ordinary) eigenvalue problem for a real symmetric
    square matrix A. Given A, an orthogonal matrix Q is calculated which
    diagonalizes A:

          Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) is a diagonal matrix whose diagonal is E.
    ' denotes the transpose.

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

          A Q[:,i] = E[i] Q[:,i]


    input:

      A: real matrix of format (n,n) which is symmetric
         (i.e. A=A' or A[i,j]=A[j,i])

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: orthogonal matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

          E          if eigvals_only is true
         (E, Q)      if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, 2], [2, 0]])
      >>> E = mp.eigsy(A, eigvals_only = True)
      >>> print(E)
      [-1.0]
      [ 4.0]

      >>> A = mp.matrix([[1, 2], [2, 3]])
      >>> E, Q = mp.eigsy(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eighe, eigh, eig
    """

    if not overwrite_a:
        A = A.copy()

    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)

    if eigvals_only:
        r_sy_tridiag(ctx, A, d, e, calc_ev = False)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        r_sy_tridiag(ctx, A, d, e, calc_ev = True)
        tridiag_eigen(ctx, d, e, A)
        return (d, A)


@defun
def eighe(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    This routine solves the (ordinary) eigenvalue problem for a complex
    hermitian square matrix A. Given A, an unitary matrix Q is calculated which
    diagonalizes A:

        Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) a is diagonal matrix whose diagonal is E.
    ' denotes the hermitian transpose (i.e. ordinary transposition and
    complex conjugation).

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

        A Q[:,i] = E[i] Q[:,i]


    input:

      A: complex matrix of format (n,n) which is hermitian
         (i.e. A=A' or A[i,j]=conj(A[j,i]))

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: unitary matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

           E         if eigvals_only is true
          (E, Q)     if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[1, -3 - 1j], [-3 + 1j, -2]])
      >>> E = mp.eighe(A, eigvals_only = True)
      >>> print(E)
      [-4.0]
      [ 3.0]

      >>> A = mp.matrix([[1, 2 + 5j], [2 - 5j, 3]])
      >>> E, Q = mp.eighe(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eigsy, eigh, eig
    """

    if not overwrite_a:
        A = A.copy()

    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)
    t = ctx.zeros(A.rows, 1)

    if eigvals_only:
        c_he_tridiag_0(ctx, A, d, e, t)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        c_he_tridiag_0(ctx, A, d, e, t)
        B = ctx.eye(A.rows)
        tridiag_eigen(ctx, d, e, B)
        c_he_tridiag_2(ctx, A, t, B)
        return (d, B)

@defun
def eigh(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    "eigh" is a unified interface for "eigsy" and "eighe". Depending on
    whether A is real or complex the appropriate function is called.

    This routine solves the (ordinary) eigenvalue problem for a real symmetric
    or complex hermitian square matrix A. Given A, an orthogonal (A real) or
    unitary (A complex) matrix Q is calculated which diagonalizes A:

        Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) a is diagonal matrix whose diagonal is E.
    ' denotes the hermitian transpose (i.e. ordinary transposition and
    complex conjugation).

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

        A Q[:,i] = E[i] Q[:,i]

    input:

      A: a real or complex square matrix of format (n,n) which is symmetric
         (i.e. A[i,j]=A[j,i]) or hermitian (i.e. A[i,j]=conj(A[j,i])).

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: an orthogonal or unitary matrix of format (n,n). contains the
         eigenvectors of A as columns.

    return value:

          E         if eigvals_only is true
         (E, Q)     if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, 2], [2, 0]])
      >>> E = mp.eigh(A, eigvals_only = True)
      >>> print(E)
      [-1.0]
      [ 4.0]

      >>> A = mp.matrix([[1, 2], [2, 3]])
      >>> E, Q = mp.eigh(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

      >>> A = mp.matrix([[1, 2 + 5j], [2 - 5j, 3]])
      >>> E, Q = mp.eigh(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eigsy, eighe, eig
    """

    iscomplex = any(type(x) is ctx.mpc for x in A)

    if iscomplex:
        return ctx.eighe(A, eigvals_only = eigvals_only, overwrite_a = overwrite_a)
    else:
        return ctx.eigsy(A, eigvals_only = eigvals_only, overwrite_a = overwrite_a)


@defun
def gauss_quadrature(ctx, n, qtype = "legendre", alpha = 0, beta = 0):
    """
    This routine calulates gaussian quadrature rules for different
    families of orthogonal polynomials. Let (a, b) be an interval,
    W(x) a positive weight function and n a positive integer.
    Then the purpose of this routine is to calculate pairs (x_k, w_k)
    for k=0, 1, 2, ... (n-1) which give

      int(W(x) * F(x), x = a..b) = sum(w_k * F(x_k),k = 0..(n-1))

    exact for all polynomials F(x) of degree (strictly) less than 2*n. For all
    integrable functions F(x) the sum is a (more or less) good approximation to
    the integral. The x_k are called nodes (which are the zeros of the
    related orthogonal polynomials) and the w_k are called the weights.

    parameters
       n        (input) The degree of the quadrature rule, i.e. its number of
                nodes.

       qtype    (input) The family of orthogonal polynmomials for which to
                compute the quadrature rule. See the list below.

       alpha    (input) real number, used as parameter for some orthogonal
                polynomials

       beta     (input) real number, used as parameter for some orthogonal
                polynomials.

    return value

      (X, W)    a pair of two real arrays where x_k = X[k] and w_k = W[k].


    orthogonal polynomials:

      qtype           polynomial
      -----           ----------

      "legendre"      Legendre polynomials, W(x)=1 on the interval (-1, +1)
      "legendre01"    shifted Legendre polynomials, W(x)=1 on the interval (0, +1)
      "hermite"       Hermite polynomials, W(x)=exp(-x*x) on (-infinity,+infinity)
      "laguerre"      Laguerre polynomials, W(x)=exp(-x) on (0,+infinity)
      "glaguerre"     generalized Laguerre polynomials, W(x)=exp(-x)*x**alpha
                      on (0, +infinity)
      "chebyshev1"    Chebyshev polynomials of the first kind, W(x)=1/sqrt(1-x*x)
                      on (-1, +1)
      "chebyshev2"    Chebyshev polynomials of the second kind, W(x)=sqrt(1-x*x)
                      on (-1, +1)
      "jacobi"        Jacobi polynomials, W(x)=(1-x)**alpha * (1+x)**beta on (-1, +1)
                      with alpha>-1 and beta>-1

    examples:
      >>> from mpmath import mp
      >>> f = lambda x: x**8 + 2 * x**6 - 3 * x**4 + 5 * x**2 - 7
      >>> X, W = mp.gauss_quadrature(5, "hermite")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = mp.sqrt(mp.pi) * 57 / 16
      >>> C = mp.quad(lambda x: mp.exp(- x * x) * f(x), [-mp.inf, +mp.inf])
      >>> mp.nprint((mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10)))
      (0.0, 0.0)

      >>> f = lambda x: x**5 - 2 * x**4 + 3 * x**3 - 5 * x**2 + 7 * x - 11
      >>> X, W = mp.gauss_quadrature(3, "laguerre")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = 76
      >>> C = mp.quad(lambda x: mp.exp(-x) * f(x), [0, +mp.inf])
      >>> mp.nprint(mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10))
      .0

      # orthogonality of the chebyshev polynomials:
      >>> f = lambda x: mp.chebyt(3, x) * mp.chebyt(2, x)
      >>> X, W = mp.gauss_quadrature(3, "chebyshev1")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> print(mp.chop(A, tol = 1e-10))
      0.0

    references:
      - golub and welsch, "calculations of gaussian quadrature rules", mathematics of
        computation 23, p. 221-230 (1969)
      - golub, "some modified matrix eigenvalue problems", siam review 15, p. 318-334 (1973)
      - stroud and secrest, "gaussian quadrature formulas", prentice-hall (1966)

    See also the routine gaussq.f in netlog.org or ACM Transactions on
    Mathematical Software algorithm 726.
    """

    d = ctx.zeros(n, 1)
    e = ctx.zeros(n, 1)
    z = ctx.zeros(1, n)

    z[0,0] = 1

    if qtype == "legendre":
        # legendre on the range -1 +1 , abramowitz, table 25.4, p.916
        w = 2
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j * j / (4 * j * j - ctx.mpf(1)))
    elif qtype == "legendre01":
        # legendre shifted to 0 1        , abramowitz, table 25.8, p.921
        w = 1
        for i in xrange(n):
            d[i] = 1 / ctx.mpf(2)
            j = i + 1
            e[i] = ctx.sqrt(j * j / (16 * j * j - ctx.mpf(4)))
    elif qtype == "hermite":
        # hermite on the range -inf +inf , abramowitz, table 25.10,p.924
        w = ctx.sqrt(ctx.pi)
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j / ctx.mpf(2))
    elif qtype == "laguerre":
        # laguerre on the range 0 +inf , abramowitz, table 25.9, p. 923
        w = 1
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1
            e[i] = j
    elif qtype=="chebyshev1":
        # chebyshev polynimials of the first kind
        w = ctx.pi
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
        e[0] = ctx.sqrt(1 / ctx.mpf(2))
    elif qtype == "chebyshev2":
        # chebyshev polynimials of the second kind
        w = ctx.pi / 2
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
    elif qtype == "glaguerre":
        # generalized laguerre on the range 0 +inf
        w = ctx.gamma(1 + alpha)
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1 + alpha
            e[i] = ctx.sqrt(j * (j + alpha))
    elif qtype == "jacobi":
        # jacobi polynomials
        alpha = ctx.mpf(alpha)
        beta = ctx.mpf(beta)
        ab = alpha + beta
        abi = ab + 2
        w = (2**(ab+1)) * ctx.gamma(alpha + 1) * ctx.gamma(beta + 1) / ctx.gamma(abi)
        d[0] = (beta - alpha) / abi
        e[0] = ctx.sqrt(4 * (1 + alpha) * (1 + beta) / ((abi + 1) * (abi * abi)))
        a2b2 = beta * beta - alpha * alpha
        for i in xrange(1, n):
            j = i + 1
            abi = 2 * j + ab
            d[i] = a2b2 / ((abi - 2) * abi)
            e[i] = ctx.sqrt(4 * j * (j + alpha) * (j + beta) * (j + ab) / ((abi * abi - 1) * abi * abi))
    elif isinstance(qtype, str):
        raise ValueError("unknown quadrature rule \"%s\"" % qtype)
    elif not isinstance(qtype, str):
        w = qtype(d, e)
    else:
        assert 0

    tridiag_eigen(ctx, d, e, z)

    for i in xrange(len(z)):
        z[i] *= z[i]

    z = z.transpose()
    return (d, w * z)

##################################################################################################
##################################################################################################
##################################################################################################

def svd_r_raw(ctx, A, V = False, calc_u = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal matrices U and V are calculated such that

                    A = U S V

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    The diagonal elements of S are the singular values of A, i.e. the
    squareroots of the eigenvalues of A' A or A A'. Here ' denotes the transpose.
    Householder bidiagonalization and a variant of the QR algorithm is used.

    overview of the matrices :

      A : m*n       A gets replaced by U
      U : m*n       U replaces A. If n>m then only the first m*m block of U is
                    non-zero. column-orthogonal: U' U = B
                    here B is a n*n matrix whose first min(m,n) diagonal
                    elements are 1 and all other elements are zero.
      S : n*n       diagonal matrix, only the diagonal elements are stored in
                    the array S. only the first min(m,n) diagonal elements are non-zero.
      V : n*n       orthogonal: V V' = V' V = 1

    parameters:
      A        (input/output) On input, A contains a real matrix of shape m*n.
               On output, if calc_u is true A contains the column-orthogonal
               matrix U; otherwise A is simply used as workspace and thus destroyed.

      V        (input/output) if false, the matrix V is not calculated. otherwise
               V must be a matrix of shape n*n.

      calc_u   (input) If true, the matrix U is calculated and replaces A.
               if false, U is not calculated and A is simply destroyed

    return value:
      S        an array of length n containing the singular values of A sorted by
               decreasing magnitude. only the first min(m,n) elements are non-zero.

    This routine is a python translation of the fortran routine svd.f in the
    software library EISPACK (see netlib.org) which itself is based on the
    algol procedure svd described in:
      - num. math. 14, 403-420(1970) by golub and reinsch.
      - wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).

    """

    m, n = A.rows, A.cols

    S = ctx.zeros(n, 1)

    # work is a temporary array of size n
    work = ctx.zeros(n, 1)

    g = scale = anorm = 0
    maxits = 3 * ctx.dps

    for i in xrange(n):     # householder reduction to bidiagonal form
        work[i] = scale*g
        g = s = scale = 0
        if i < m:
            for k in xrange(i, m):
                scale += ctx.fabs(A[k,i])
            if scale != 0:
                for k in xrange(i, m):
                    A[k,i] /= scale
                    s += A[k,i] * A[k,i]
                f = A[i,i]
                g = -ctx.sqrt(s)
                if f < 0:
                    g = -g
                h = f * g - s
                A[i,i] = f - g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i, m):
                        s += A[k,i] * A[k,j]
                    f = s / h
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for k in xrange(i,m):
                    A[k,i] *= scale

        S[i] = scale * g
        g = s = scale = 0

        if i < m and i != n - 1:
            for k in xrange(i+1, n):
                scale += ctx.fabs(A[i,k])
            if scale:
                for k in xrange(i+1, n):
                    A[i,k] /= scale
                    s += A[i,k] * A[i,k]
                f = A[i,i+1]
                g = -ctx.sqrt(s)
                if f < 0:
                    g = -g
                h = f * g - s
                A[i,i+1] = f - g

                for k in xrange(i+1, n):
                    work[k] = A[i,k] / h

                for j in xrange(i+1, m):
                    s = 0
                    for k in xrange(i+1, n):
                        s += A[j,k] * A[i,k]
                    for k in xrange(i+1, n):
                        A[j,k] += s * work[k]

                for k in xrange(i+1, n):
                    A[i,k] *= scale

        anorm = max(anorm, ctx.fabs(S[i]) + ctx.fabs(work[i]))

    if not isinstance(V, bool):
        for i in xrange(n-2, -1, -1):     # accumulation of right hand transformations
            V[i+1,i+1] = 1

            if work[i+1] != 0:
                for j in xrange(i+1, n):
                    V[i,j] = (A[i,j] / A[i,i+1]) / work[i+1]
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, n):
                        s += A[i,k] * V[j,k]
                    for k in xrange(i+1, n):
                        V[j,k] += s * V[i,k]

            for j in xrange(i+1, n):
                V[j,i] = V[i,j] = 0

        V[0,0] = 1

    if m<n : minnm = m
    else   : minnm = n

    if calc_u:
        for i in xrange(minnm-1, -1, -1): # accumulation of left hand transformations
            g = S[i]
            for j in xrange(i+1, n):
                A[i,j] = 0
            if g != 0:
                g = 1 / g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, m):
                        s += A[k,i] * A[k,j]
                    f = (s / A[i,i]) * g
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for j in xrange(i, m):
                    A[j,i] *= g
            else:
                for j in xrange(i, m):
                    A[j,i] = 0
            A[i,i] += 1

    for k in xrange(n - 1, -1, -1):
        # diagonalization of the bidiagonal form:
        #   loop over singular values, and over allowed itations

        its = 0
        while 1:
            its += 1
            flag = True

            for l in xrange(k, -1, -1):
                nm = l-1

                if ctx.fabs(work[l]) + anorm == anorm:
                    flag = False
                    break

                if ctx.fabs(S[nm]) + anorm == anorm:
                    break

            if flag:
                c = 0
                s = 1
                for i in xrange(l, k + 1):
                    f = s * work[i]
                    work[i] *= c
                    if ctx.fabs(f) + anorm == anorm:
                        break
                    g = S[i]
                    h = ctx.hypot(f, g)
                    S[i] = h
                    h = 1 / h
                    c = g * h
                    s = - f * h

                    if calc_u:
                        for j in xrange(m):
                            y = A[j,nm]
                            z = A[j,i]
                            A[j,nm] = y * c + z * s
                            A[j,i]  = z * c - y * s

            z = S[k]

            if l == k:               # convergence
                if z < 0:            # singular value is made nonnegative
                    S[k] = -z
                    if not isinstance(V, bool):
                        for j in xrange(n):
                            V[k,j] = -V[k,j]
                break

            if its >= maxits:
                raise RuntimeError("svd: no convergence to an eigenvalue after %d iterations" % its)

            x = S[l]         # shift from bottom 2 by 2 minor
            nm = k-1
            y = S[nm]
            g = work[nm]
            h = work[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h))/(2 * h * y)
            g = ctx.hypot(f, 1)
            if f >= 0: f = ((x - z) * (x + z) + h * ((y / (f + g)) - h)) / x
            else:      f = ((x - z) * (x + z) + h * ((y / (f - g)) - h)) / x

            c = s = 1         # next qt transformation

            for j in xrange(l, nm + 1):
                g = work[j+1]
                y = S[j+1]
                h = s * g
                g = c * g
                z = ctx.hypot(f, h)
                work[j] = z
                c = f / z
                s = h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                if not isinstance(V, bool):
                    for jj in xrange(n):
                        x = V[j  ,jj]
                        z = V[j+1,jj]
                        V[j    ,jj]= x * c + z * s
                        V[j+1  ,jj]= z * c - x * s
                z = ctx.hypot(f, h)
                S[j] = z
                if z != 0:            # rotation can be arbitray if z=0
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                x = c * y - s * g

                if calc_u:
                    for jj in xrange(m):
                        y = A[jj,j  ]
                        z = A[jj,j+1]
                        A[jj,j    ] = y * c + z * s
                        A[jj,j+1  ] = z * c - y * s

            work[l] = 0
            work[k] = f
            S[k] = x

    ##########################

    # Sort singular values into decreasing order (bubble-sort)

    for i in xrange(n):
        imax = i
        s = ctx.fabs(S[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = ctx.fabs(S[j])
            if c > s:
                s = c
                imax = j

        if imax != i:
            # swap singular values

            z = S[i]
            S[i] = S[imax]
            S[imax] = z

            if calc_u:
                for j in xrange(m):
                    z = A[j,i]
                    A[j,i] = A[j,imax]
                    A[j,imax] = z

            if not isinstance(V, bool):
                for j in xrange(n):
                    z = V[i,j]
                    V[i,j] = V[imax,j]
                    V[imax,j] = z

    return S

#######################

def svd_c_raw(ctx, A, V = False, calc_u = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

                    A = U S V

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    The diagonal elements of S are the singular values of A, i.e. the
    squareroots of the eigenvalues of A' A or A A'. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation). Householder bidiagonalization
    and a variant of the QR algorithm is used.

    overview of the matrices :

      A : m*n       A gets replaced by U
      U : m*n       U replaces A. If n>m then only the first m*m block of U is
                    non-zero. column-unitary: U' U = B
                    here B is a n*n matrix whose first min(m,n) diagonal
                    elements are 1 and all other elements are zero.
      S : n*n       diagonal matrix, only the diagonal elements are stored in
                    the array S. only the first min(m,n) diagonal elements are non-zero.
      V : n*n       unitary: V V' = V' V = 1

    parameters:
      A        (input/output) On input, A contains a complex matrix of shape m*n.
               On output, if calc_u is true A contains the column-unitary
               matrix U; otherwise A is simply used as workspace and thus destroyed.

      V        (input/output) if false, the matrix V is not calculated. otherwise
               V must be a matrix of shape n*n.

      calc_u   (input) If true, the matrix U is calculated and replaces A.
               if false, U is not calculated and A is simply destroyed

    return value:
      S        an array of length n containing the singular values of A sorted by
               decreasing magnitude. only the first min(m,n) elements are non-zero.

    This routine is a python translation of the fortran routine svd.f in the
    software library EISPACK (see netlib.org) which itself is based on the
    algol procedure svd described in:
      - num. math. 14, 403-420(1970) by golub and reinsch.
      - wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).

    """

    m, n = A.rows, A.cols

    S = ctx.zeros(n, 1)

    # work is a temporary array of size n
    work  = ctx.zeros(n, 1)
    lbeta = ctx.zeros(n, 1)
    rbeta = ctx.zeros(n, 1)
    dwork = ctx.zeros(n, 1)

    g = scale = anorm = 0
    maxits = 3 * ctx.dps

    for i in xrange(n):         # householder reduction to bidiagonal form
        dwork[i] = scale * g    # dwork are the side-diagonal elements
        g = s = scale = 0
        if i < m:
            for k in xrange(i, m):
                scale += ctx.fabs(ctx.re(A[k,i])) + ctx.fabs(ctx.im(A[k,i]))
            if scale != 0:
                for k in xrange(i, m):
                    A[k,i] /= scale
                    ar = ctx.re(A[k,i])
                    ai = ctx.im(A[k,i])
                    s += ar * ar + ai * ai
                f = A[i,i]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)
                beta /= ctx.conj(beta)
                beta += 1
                h = 2 * (ctx.re(f) * g - s)
                A[i,i] = f - g
                beta /= h
                lbeta[i] = (beta / scale) / scale
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i, m):
                        s += ctx.conj(A[k,i]) * A[k,j]
                    f = beta * s
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for k in xrange(i, m):
                    A[k,i] *= scale

        S[i] = scale * g     # S are the diagonal elements
        g = s = scale = 0

        if i < m and i != n - 1:
            for k in xrange(i+1, n):
                scale += ctx.fabs(ctx.re(A[i,k])) + ctx.fabs(ctx.im(A[i,k]))
            if scale:
                for k in xrange(i+1, n):
                    A[i,k] /= scale
                    ar = ctx.re(A[i,k])
                    ai = ctx.im(A[i,k])
                    s += ar * ar + ai * ai
                f = A[i,i+1]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)

                beta /= ctx.conj(beta)
                beta += 1

                h = 2 * (ctx.re(f) * g - s)
                A[i,i+1] = f - g

                beta /= h
                rbeta[i] = (beta / scale) / scale

                for k in xrange(i+1, n):
                    work[k] = A[i, k]

                for j in xrange(i+1, m):
                    s = 0
                    for k in xrange(i+1, n):
                        s += ctx.conj(A[i,k]) * A[j,k]
                    f = s * beta
                    for k in xrange(i+1,n):
                        A[j,k] += f * work[k]

                for k in xrange(i+1, n):
                    A[i,k] *= scale

        anorm = max(anorm,ctx.fabs(S[i]) + ctx.fabs(dwork[i]))

    if not isinstance(V, bool):
        for i in xrange(n-2, -1, -1):     # accumulation of right hand transformations
            V[i+1,i+1] = 1

            if dwork[i+1] != 0:
                f = ctx.conj(rbeta[i])
                for j in xrange(i+1, n):
                    V[i,j] = A[i,j] * f
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, n):
                        s += ctx.conj(A[i,k]) * V[j,k]
                    for k in xrange(i+1, n):
                        V[j,k] += s * V[i,k]

            for j in xrange(i+1,n):
                V[j,i] = V[i,j] = 0

        V[0,0] = 1

    if m < n : minnm = m
    else     : minnm = n

    if calc_u:
        for i in xrange(minnm-1, -1, -1): # accumulation of left hand transformations
            g = S[i]
            for j in xrange(i+1, n):
                A[i,j] = 0
            if g != 0:
                g = 1 / g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, m):
                        s += ctx.conj(A[k,i]) * A[k,j]
                    f = s * ctx.conj(lbeta[i])
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for j in xrange(i, m):
                    A[j,i] *= g
            else:
                for j in xrange(i, m):
                    A[j,i] = 0
            A[i,i] += 1

    for k in xrange(n-1, -1, -1):
        # diagonalization of the bidiagonal form:
        #   loop over singular values, and over allowed itations

        its = 0
        while 1:
            its += 1
            flag = True

            for l in xrange(k, -1, -1):
                nm = l - 1

                if ctx.fabs(dwork[l]) + anorm == anorm:
                    flag = False
                    break

                if ctx.fabs(S[nm]) + anorm == anorm:
                    break

            if flag:
                c = 0
                s = 1
                for i in xrange(l, k+1):
                    f = s * dwork[i]
                    dwork[i] *= c
                    if ctx.fabs(f) + anorm == anorm:
                        break
                    g = S[i]
                    h = ctx.hypot(f, g)
                    S[i] = h
                    h = 1 / h
                    c = g * h
                    s = -f * h

                    if calc_u:
                        for j in xrange(m):
                            y = A[j,nm]
                            z = A[j,i]
                            A[j,nm]= y * c + z * s
                            A[j,i] = z * c - y * s

            z = S[k]

            if l == k:         # convergence
                if z < 0:    # singular value is made nonnegative
                    S[k] = -z
                    if not isinstance(V, bool):
                        for j in xrange(n):
                            V[k,j] = -V[k,j]
                break

            if its >= maxits:
                raise RuntimeError("svd: no convergence to an eigenvalue after %d iterations" % its)

            x = S[l]         # shift from bottom 2 by 2 minor
            nm = k-1
            y = S[nm]
            g = dwork[nm]
            h = dwork[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
            g = ctx.hypot(f, 1)
            if f >=0: f = (( x - z) *( x + z) + h *((y / (f + g)) - h)) / x
            else:     f = (( x - z) *( x + z) + h *((y / (f - g)) - h)) / x

            c = s = 1         # next qt transformation

            for j in xrange(l, nm + 1):
                g = dwork[j+1]
                y = S[j+1]
                h = s * g
                g = c * g
                z = ctx.hypot(f, h)
                dwork[j] = z
                c = f / z
                s = h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                if not isinstance(V, bool):
                    for jj in xrange(n):
                        x = V[j  ,jj]
                        z = V[j+1,jj]
                        V[j    ,jj]= x * c + z * s
                        V[j+1,jj  ]= z * c - x * s
                z = ctx.hypot(f, h)
                S[j] = z
                if z != 0:            # rotation can be arbitray if z=0
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                x = c * y - s * g
                if calc_u:
                    for jj in xrange(m):
                        y = A[jj,j  ]
                        z = A[jj,j+1]
                        A[jj,j    ]= y * c + z * s
                        A[jj,j+1  ]= z * c - y * s

            dwork[l] = 0
            dwork[k] = f
            S[k] = x

    ##########################

    # Sort singular values into decreasing order (bubble-sort)

    for i in xrange(n):
        imax = i
        s = ctx.fabs(S[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = ctx.fabs(S[j])
            if c > s:
                s = c
                imax = j

        if imax != i:
            # swap singular values

            z = S[i]
            S[i] = S[imax]
            S[imax] = z

            if calc_u:
                for j in xrange(m):
                    z = A[j,i]
                    A[j,i] = A[j,imax]
                    A[j,imax] = z

            if not isinstance(V, bool):
                for j in xrange(n):
                    z = V[i,j]
                    V[i,j] = V[imax,j]
                    V[imax,j] = z

    return S

##################################################################################################

@defun
def svd_r(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal matrices U and V are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the transpose. The diagonal elements of S are the singular
    values of A, i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a real matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an orthogonal matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an orthogonal matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    examples:

       >>> from mpmath import mp
       >>> A = mp.matrix([[2, -2, -1], [3, 4, -2], [-2, -2, 0]])
       >>> S = mp.svd_r(A, compute_uv = False)
       >>> print(S)
       [6.0]
       [3.0]
       [1.0]

       >>> U, S, V = mp.svd_r(A)
       >>> print(mp.chop(A - U * mp.diag(S) * V))
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]


    see also: svd, svd_c
    """

    m, n = A.rows, A.cols

    if not compute_uv:
        if not overwrite_a:
            A = A.copy()
        S = svd_r_raw(ctx, A, V = False, calc_u = False)
        S = S[:min(m,n)]
        return S

    if full_matrices and n < m:
        V = ctx.zeros(m, m)
        A0 = ctx.zeros(m, m)
        A0[:,:n] = A
        S = svd_r_raw(ctx, A0, V, calc_u = True)

        S = S[:n]
        V = V[:n,:n]

        return (A0, S, V)
    else:
        if not overwrite_a:
            A = A.copy()
        V = ctx.zeros(n, n)
        S = svd_r_raw(ctx, A, V, calc_u = True)

        if n > m:
            if full_matrices == False:
                V = V[:m,:]

            S = S[:m]
            A = A[:,:m]

        return (A, S, V)

##############################

@defun
def svd_c(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the hermitian transpose (i.e. transposition and complex
    conjugation). The diagonal elements of S are the singular values of A,
    i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a complex matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an unitary matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an unitary matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[-2j, -1-3j, -2+2j], [2-2j, -1-3j, 1], [-3+1j,-2j,0]])
      >>> S = mp.svd_c(A, compute_uv = False)
      >>> print(mp.chop(S - mp.matrix([mp.sqrt(34), mp.sqrt(15), mp.sqrt(6)])))
      [0.0]
      [0.0]
      [0.0]

      >>> U, S, V = mp.svd_c(A)
      >>> print(mp.chop(A - U * mp.diag(S) * V))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    see also: svd, svd_r
    """

    m, n = A.rows, A.cols

    if not compute_uv:
        if not overwrite_a:
            A = A.copy()
        S = svd_c_raw(ctx, A, V = False, calc_u = False)
        S = S[:min(m,n)]
        return S

    if full_matrices and n < m:
        V = ctx.zeros(m, m)
        A0 = ctx.zeros(m, m)
        A0[:,:n] = A
        S = svd_c_raw(ctx, A0, V, calc_u = True)

        S = S[:n]
        V = V[:n,:n]

        return (A0, S, V)
    else:
        if not overwrite_a:
            A = A.copy()
        V = ctx.zeros(n, n)
        S = svd_c_raw(ctx, A, V, calc_u = True)

        if n > m:
            if full_matrices == False:
                V = V[:m,:]

            S = S[:m]
            A = A[:,:m]

        return (A, S, V)

@defun
def svd(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    "svd" is a unified interface for "svd_r" and "svd_c". Depending on
    whether A is real or complex the appropriate function is called.

    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal (A real) or unitary (A complex) matrices U and V
    are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the hermitian transpose (i.e. transposition and complex
    conjugation). The diagonal elements of S are the singular values of A,
    i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a real or complex matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an orthogonal or unitary matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an orthogonal or unitary matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    examples:

       >>> from mpmath import mp
       >>> A = mp.matrix([[2, -2, -1], [3, 4, -2], [-2, -2, 0]])
       >>> S = mp.svd(A, compute_uv = False)
       >>> print(S)
       [6.0]
       [3.0]
       [1.0]

       >>> U, S, V = mp.svd(A)
       >>> print(mp.chop(A - U * mp.diag(S) * V))
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]

    see also: svd_r, svd_c
    """

    iscomplex = any(type(x) is ctx.mpc for x in A)

    if iscomplex:
        return ctx.svd_c(A, full_matrices = full_matrices, compute_uv = compute_uv, overwrite_a = overwrite_a)
    else:
        return ctx.svd_r(A, full_matrices = full_matrices, compute_uv = compute_uv, overwrite_a = overwrite_a)
