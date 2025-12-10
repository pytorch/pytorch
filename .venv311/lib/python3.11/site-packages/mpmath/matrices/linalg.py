"""
Linear algebra
--------------

Linear equations
................

Basic linear algebra is implemented; you can for example solve the linear
equation system::

      x + 2*y = -10
    3*x + 4*y =  10

using ``lu_solve``::

    >>> from mpmath import *
    >>> mp.pretty = False
    >>> A = matrix([[1, 2], [3, 4]])
    >>> b = matrix([-10, 10])
    >>> x = lu_solve(A, b)
    >>> x
    matrix(
    [['30.0'],
     ['-20.0']])

If you don't trust the result, use ``residual`` to calculate the residual ||A*x-b||::

    >>> residual(A, x, b)
    matrix(
    [['3.46944695195361e-18'],
     ['3.46944695195361e-18']])
    >>> str(eps)
    '2.22044604925031e-16'

As you can see, the solution is quite accurate. The error is caused by the
inaccuracy of the internal floating point arithmetic. Though, it's even smaller
than the current machine epsilon, which basically means you can trust the
result.

If you need more speed, use NumPy, or ``fp.lu_solve`` for a floating-point computation.

    >>> fp.lu_solve(A, b)   # doctest: +ELLIPSIS
    matrix(...)

``lu_solve`` accepts overdetermined systems. It is usually not possible to solve
such systems, so the residual is minimized instead. Internally this is done
using Cholesky decomposition to compute a least squares approximation. This means
that that ``lu_solve`` will square the errors. If you can't afford this, use
``qr_solve`` instead. It is twice as slow but more accurate, and it calculates
the residual automatically.


Matrix factorization
....................

The function ``lu`` computes an explicit LU factorization of a matrix::

    >>> P, L, U = lu(matrix([[0,2,3],[4,5,6],[7,8,9]]))
    >>> print(P)
    [0.0  0.0  1.0]
    [1.0  0.0  0.0]
    [0.0  1.0  0.0]
    >>> print(L)
    [              1.0                0.0  0.0]
    [              0.0                1.0  0.0]
    [0.571428571428571  0.214285714285714  1.0]
    >>> print(U)
    [7.0  8.0                9.0]
    [0.0  2.0                3.0]
    [0.0  0.0  0.214285714285714]
    >>> print(P.T*L*U)
    [0.0  2.0  3.0]
    [4.0  5.0  6.0]
    [7.0  8.0  9.0]

Interval matrices
-----------------

Matrices may contain interval elements. This allows one to perform
basic linear algebra operations such as matrix multiplication
and equation solving with rigorous error bounds::

    >>> a = iv.matrix([['0.1','0.3','1.0'],
    ...             ['7.1','5.5','4.8'],
    ...             ['3.2','4.4','5.6']])
    >>>
    >>> b = iv.matrix(['4','0.6','0.5'])
    >>> c = iv.lu_solve(a, b)
    >>> print(c)
    [   [5.2582327113062568605927528666, 5.25823271130625686059275702219]]
    [[-13.1550493962678375411635581388, -13.1550493962678375411635540152]]
    [  [7.42069154774972557628979076189, 7.42069154774972557628979190734]]
    >>> print(a*c)
    [  [3.99999999999999999999999844904, 4.00000000000000000000000155096]]
    [[0.599999999999999999999968898009, 0.600000000000000000000031763736]]
    [[0.499999999999999999999979320485, 0.500000000000000000000020679515]]
"""

# TODO:
# *implement high-level qr()
# *test unitvector
# *iterative solving

from copy import copy

from ..libmp.backend import xrange

class LinearAlgebraMethods(object):

    def LU_decomp(ctx, A, overwrite=False, use_cache=True):
        """
        LU-factorization of a n*n matrix using the Gauss algorithm.
        Returns L and U in one matrix and the pivot indices.

        Use overwrite to specify whether A will be overwritten with L and U.
        """
        if not A.rows == A.cols:
            raise ValueError('need n*n matrix')
        # get from cache if possible
        if use_cache and isinstance(A, ctx.matrix) and A._LU:
            return A._LU
        if not overwrite:
            orig = A
            A = A.copy()
        tol = ctx.absmin(ctx.mnorm(A,1) * ctx.eps) # each pivot element has to be bigger
        n = A.rows
        p = [None]*(n - 1)
        for j in xrange(n - 1):
            # pivoting, choose max(abs(reciprocal row sum)*abs(pivot element))
            biggest = 0
            for k in xrange(j, n):
                s = ctx.fsum([ctx.absmin(A[k,l]) for l in xrange(j, n)])
                if ctx.absmin(s) <= tol:
                    raise ZeroDivisionError('matrix is numerically singular')
                current = 1/s * ctx.absmin(A[k,j])
                if current > biggest: # TODO: what if equal?
                    biggest = current
                    p[j] = k
            # swap rows according to p
            ctx.swap_row(A, j, p[j])
            if ctx.absmin(A[j,j]) <= tol:
                raise ZeroDivisionError('matrix is numerically singular')
            # calculate elimination factors and add rows
            for i in xrange(j + 1, n):
                A[i,j] /= A[j,j]
                for k in xrange(j + 1, n):
                    A[i,k] -= A[i,j]*A[j,k]
        if ctx.absmin(A[n - 1,n - 1]) <= tol:
            raise ZeroDivisionError('matrix is numerically singular')
        # cache decomposition
        if not overwrite and isinstance(orig, ctx.matrix):
            orig._LU = (A, p)
        return A, p

    def L_solve(ctx, L, b, p=None):
        """
        Solve the lower part of a LU factorized matrix for y.
        """
        if L.rows != L.cols:
            raise RuntimeError("need n*n matrix")
        n = L.rows
        if len(b) != n:
            raise ValueError("Value should be equal to n")
        b = copy(b)
        if p: # swap b according to p
            for k in xrange(0, len(p)):
                ctx.swap_row(b, k, p[k])
        # solve
        for i in xrange(1, n):
            for j in xrange(i):
                b[i] -= L[i,j] * b[j]
        return b

    def U_solve(ctx, U, y):
        """
        Solve the upper part of a LU factorized matrix for x.
        """
        if U.rows != U.cols:
            raise RuntimeError("need n*n matrix")
        n = U.rows
        if len(y) != n:
            raise ValueError("Value should be equal to n")
        x = copy(y)
        for i in xrange(n - 1, -1, -1):
            for j in xrange(i + 1, n):
                x[i] -= U[i,j] * x[j]
            x[i] /= U[i,i]
        return x

    def lu_solve(ctx, A, b, **kwargs):
        """
        Ax = b => x

        Solve a determined or overdetermined linear equations system.
        Fast LU decomposition is used, which is less accurate than QR decomposition
        (especially for overdetermined systems), but it's twice as efficient.
        Use qr_solve if you want more precision or have to solve a very ill-
        conditioned system.

        If you specify real=True, it does not check for overdeterminded complex
        systems.
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows < A.cols:
                raise ValueError('cannot solve underdetermined system')
            if A.rows > A.cols:
                # use least-squares method if overdetermined
                # (this increases errors)
                AH = A.H
                A = AH * A
                b = AH * b
                if (kwargs.get('real', False) or
                    not sum(type(i) is ctx.mpc for i in A)):
                    # TODO: necessary to check also b?
                    x = ctx.cholesky_solve(A, b)
                else:
                    x = ctx.lu_solve(A, b)
            else:
                # LU factorization
                A, p = ctx.LU_decomp(A)
                b = ctx.L_solve(A, b, p)
                x = ctx.U_solve(A, b)
        finally:
            ctx.prec = prec
        return x

    def improve_solution(ctx, A, x, b, maxsteps=1):
        """
        Improve a solution to a linear equation system iteratively.

        This re-uses the LU decomposition and is thus cheap.
        Usually 3 up to 4 iterations are giving the maximal improvement.
        """
        if A.rows != A.cols:
            raise RuntimeError("need n*n matrix") # TODO: really?
        for _ in xrange(maxsteps):
            r = ctx.residual(A, x, b)
            if ctx.norm(r, 2) < 10*ctx.eps:
                break
            # this uses cached LU decomposition and is thus cheap
            dx = ctx.lu_solve(A, -r)
            x += dx
        return x

    def lu(ctx, A):
        """
        A -> P, L, U

        LU factorisation of a square matrix A. L is the lower, U the upper part.
        P is the permutation matrix indicating the row swaps.

        P*A = L*U

        If you need efficiency, use the low-level method LU_decomp instead, it's
        much more memory efficient.
        """
        # get factorization
        A, p = ctx.LU_decomp(A)
        n = A.rows
        L = ctx.matrix(n)
        U = ctx.matrix(n)
        for i in xrange(n):
            for j in xrange(n):
                if i > j:
                    L[i,j] = A[i,j]
                elif i == j:
                    L[i,j] = 1
                    U[i,j] = A[i,j]
                else:
                    U[i,j] = A[i,j]
        # calculate permutation matrix
        P = ctx.eye(n)
        for k in xrange(len(p)):
            ctx.swap_row(P, k, p[k])
        return P, L, U

    def unitvector(ctx, n, i):
        """
        Return the i-th n-dimensional unit vector.
        """
        assert 0 < i <= n, 'this unit vector does not exist'
        return [ctx.zero]*(i-1) + [ctx.one] + [ctx.zero]*(n-i)

    def inverse(ctx, A, **kwargs):
        """
        Calculate the inverse of a matrix.

        If you want to solve an equation system Ax = b, it's recommended to use
        solve(A, b) instead, it's about 3 times more efficient.
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A
            A = ctx.matrix(A, **kwargs).copy()
            n = A.rows
            # get LU factorisation
            A, p = ctx.LU_decomp(A)
            cols = []
            # calculate unit vectors and solve corresponding system to get columns
            for i in xrange(1, n + 1):
                e = ctx.unitvector(n, i)
                y = ctx.L_solve(A, e, p)
                cols.append(ctx.U_solve(A, y))
            # convert columns to matrix
            inv = []
            for i in xrange(n):
                row = []
                for j in xrange(n):
                    row.append(cols[j][i])
                inv.append(row)
            result = ctx.matrix(inv, **kwargs)
        finally:
            ctx.prec = prec
        return result

    def householder(ctx, A):
        """
        (A|b) -> H, p, x, res

        (A|b) is the coefficient matrix with left hand side of an optionally
        overdetermined linear equation system.
        H and p contain all information about the transformation matrices.
        x is the solution, res the residual.
        """
        if not isinstance(A, ctx.matrix):
            raise TypeError("A should be a type of ctx.matrix")
        m = A.rows
        n = A.cols
        if m < n - 1:
            raise RuntimeError("Columns should not be less than rows")
        # calculate Householder matrix
        p = []
        for j in xrange(0, n - 1):
            s = ctx.fsum(abs(A[i,j])**2 for i in xrange(j, m))
            if not abs(s) > ctx.eps:
                raise ValueError('matrix is numerically singular')
            p.append(-ctx.sign(ctx.re(A[j,j])) * ctx.sqrt(s))
            kappa = ctx.one / (s - p[j] * A[j,j])
            A[j,j] -= p[j]
            for k in xrange(j+1, n):
                y = ctx.fsum(ctx.conj(A[i,j]) * A[i,k] for i in xrange(j, m)) * kappa
                for i in xrange(j, m):
                    A[i,k] -= A[i,j] * y
        # solve Rx = c1
        x = [A[i,n - 1] for i in xrange(n - 1)]
        for i in xrange(n - 2, -1, -1):
            x[i] -= ctx.fsum(A[i,j] * x[j] for j in xrange(i + 1, n - 1))
            x[i] /= p[i]
        # calculate residual
        if not m == n - 1:
            r = [A[m-1-i, n-1] for i in xrange(m - n + 1)]
        else:
            # determined system, residual should be 0
            r = [0]*m # maybe a bad idea, changing r[i] will change all elements
        return A, p, x, r

    #def qr(ctx, A):
    #    """
    #    A -> Q, R
    #
    #    QR factorisation of a square matrix A using Householder decomposition.
    #    Q is orthogonal, this leads to very few numerical errors.
    #
    #    A = Q*R
    #    """
    #    H, p, x, res = householder(A)
    # TODO: implement this

    def residual(ctx, A, x, b, **kwargs):
        """
        Calculate the residual of a solution to a linear equation system.

        r = A*x - b for A*x = b
        """
        oldprec = ctx.prec
        try:
            ctx.prec *= 2
            A, x, b = ctx.matrix(A, **kwargs), ctx.matrix(x, **kwargs), ctx.matrix(b, **kwargs)
            return A*x - b
        finally:
            ctx.prec = oldprec

    def qr_solve(ctx, A, b, norm=None, **kwargs):
        """
        Ax = b => x, ||Ax - b||

        Solve a determined or overdetermined linear equations system and
        calculate the norm of the residual (error).
        QR decomposition using Householder factorization is applied, which gives very
        accurate results even for ill-conditioned matrices. qr_solve is twice as
        efficient.
        """
        if norm is None:
            norm = ctx.norm
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows < A.cols:
                raise ValueError('cannot solve underdetermined system')
            H, p, x, r = ctx.householder(ctx.extend(A, b))
            res = ctx.norm(r)
            # calculate residual "manually" for determined systems
            if res == 0:
                res = ctx.norm(ctx.residual(A, x, b))
            return ctx.matrix(x, **kwargs), res
        finally:
            ctx.prec = prec

    def cholesky(ctx, A, tol=None):
        r"""
        Cholesky decomposition of a symmetric positive-definite matrix `A`.
        Returns a lower triangular matrix `L` such that `A = L \times L^T`.
        More generally, for a complex Hermitian positive-definite matrix,
        a Cholesky decomposition satisfying `A = L \times L^H` is returned.

        The Cholesky decomposition can be used to solve linear equation
        systems twice as efficiently as LU decomposition, or to
        test whether `A` is positive-definite.

        The optional parameter ``tol`` determines the tolerance for
        verifying positive-definiteness.

        **Examples**

        Cholesky decomposition of a positive-definite symmetric matrix::

            >>> from mpmath import *
            >>> mp.dps = 25; mp.pretty = True
            >>> A = eye(3) + hilbert(3)
            >>> nprint(A)
            [     2.0      0.5  0.333333]
            [     0.5  1.33333      0.25]
            [0.333333     0.25       1.2]
            >>> L = cholesky(A)
            >>> nprint(L)
            [ 1.41421      0.0      0.0]
            [0.353553  1.09924      0.0]
            [0.235702  0.15162  1.05899]
            >>> chop(A - L*L.T)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]

        Cholesky decomposition of a Hermitian matrix::

            >>> A = eye(3) + matrix([[0,0.25j,-0.5j],[-0.25j,0,0],[0.5j,0,0]])
            >>> L = cholesky(A)
            >>> nprint(L)
            [          1.0                0.0                0.0]
            [(0.0 - 0.25j)  (0.968246 + 0.0j)                0.0]
            [ (0.0 + 0.5j)  (0.129099 + 0.0j)  (0.856349 + 0.0j)]
            >>> chop(A - L*L.H)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]

        Attempted Cholesky decomposition of a matrix that is not positive
        definite::

            >>> A = -eye(3) + hilbert(3)
            >>> L = cholesky(A)
            Traceback (most recent call last):
              ...
            ValueError: matrix is not positive-definite

        **References**

        1. [Wikipedia]_ http://en.wikipedia.org/wiki/Cholesky_decomposition

        """
        if not isinstance(A, ctx.matrix):
            raise RuntimeError("A should be a type of ctx.matrix")
        if not A.rows == A.cols:
            raise ValueError('need n*n matrix')
        if tol is None:
            tol = +ctx.eps
        n = A.rows
        L = ctx.matrix(n)
        for j in xrange(n):
            c = ctx.re(A[j,j])
            if abs(c-A[j,j]) > tol:
                raise ValueError('matrix is not Hermitian')
            s = c - ctx.fsum((L[j,k] for k in xrange(j)),
                absolute=True, squared=True)
            if s < tol:
                raise ValueError('matrix is not positive-definite')
            L[j,j] = ctx.sqrt(s)
            for i in xrange(j, n):
                it1 = (L[i,k] for k in xrange(j))
                it2 = (L[j,k] for k in xrange(j))
                t = ctx.fdot(it1, it2, conjugate=True)
                L[i,j] = (A[i,j] - t) / L[j,j]
        return L

    def cholesky_solve(ctx, A, b, **kwargs):
        """
        Ax = b => x

        Solve a symmetric positive-definite linear equation system.
        This is twice as efficient as lu_solve.

        Typical use cases:
        * A.T*A
        * Hessian matrix
        * differential equations
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows !=  A.cols:
                raise ValueError('can only solve determined system')
            # Cholesky factorization
            L = ctx.cholesky(A)
            # solve
            n = L.rows
            if len(b) != n:
                raise ValueError("Value should be equal to n")
            for i in xrange(n):
                b[i] -= ctx.fsum(L[i,j] * b[j] for j in xrange(i))
                b[i] /= L[i,i]
            x = ctx.U_solve(L.T, b)
            return x
        finally:
            ctx.prec = prec

    def det(ctx, A):
        """
        Calculate the determinant of a matrix.
        """
        prec = ctx.prec
        try:
            # do not overwrite A
            A = ctx.matrix(A).copy()
            # use LU factorization to calculate determinant
            try:
                R, p = ctx.LU_decomp(A)
            except ZeroDivisionError:
                return 0
            z = 1
            for i, e in enumerate(p):
                if i != e:
                    z *= -1
            for i in xrange(A.rows):
                z *= R[i,i]
            return z
        finally:
            ctx.prec = prec

    def cond(ctx, A, norm=None):
        """
        Calculate the condition number of a matrix using a specified matrix norm.

        The condition number estimates the sensitivity of a matrix to errors.
        Example: small input errors for ill-conditioned coefficient matrices
        alter the solution of the system dramatically.

        For ill-conditioned matrices it's recommended to use qr_solve() instead
        of lu_solve(). This does not help with input errors however, it just avoids
        to add additional errors.

        Definition:    cond(A) = ||A|| * ||A**-1||
        """
        if norm is None:
            norm = lambda x: ctx.mnorm(x,1)
        return norm(A) * norm(ctx.inverse(A))

    def lu_solve_mat(ctx, a, b):
        """Solve a * x = b  where a and b are matrices."""
        r = ctx.matrix(a.rows, b.cols)
        for i in range(b.cols):
            c = ctx.lu_solve(a, b.column(i))
            for j in range(len(c)):
                r[j, i] = c[j]
        return r

    def qr(ctx, A, mode = 'full', edps = 10):
        """
        Compute a QR factorization $A = QR$ where
        A is an m x n matrix of real or complex numbers where m >= n

        mode has following meanings:
        (1) mode = 'raw' returns two matrixes (A, tau) in the
            internal format used by LAPACK
        (2) mode = 'skinny' returns the leading n columns of Q
            and n rows of R
        (3) Any other value returns the leading m columns of Q
            and m rows of R

        edps is the increase in mp precision used for calculations

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> mp.pretty = True
            >>> A = matrix([[1, 2], [3, 4], [1, 1]])
            >>> Q, R = qr(A)
            >>> Q
            [-0.301511344577764   0.861640436855329   0.408248290463863]
            [-0.904534033733291  -0.123091490979333  -0.408248290463863]
            [-0.301511344577764  -0.492365963917331   0.816496580927726]
            >>> R
            [-3.3166247903554  -4.52267016866645]
            [             0.0  0.738548945875996]
            [             0.0                0.0]
            >>> Q * R
            [1.0  2.0]
            [3.0  4.0]
            [1.0  1.0]
            >>> chop(Q.T * Q)
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]
            >>> B = matrix([[1+0j, 2-3j], [3+j, 4+5j]])
            >>> Q, R = qr(B)
            >>> nprint(Q)
            [     (-0.301511 + 0.0j)   (0.0695795 - 0.95092j)]
            [(-0.904534 - 0.301511j)  (-0.115966 + 0.278318j)]
            >>> nprint(R)
            [(-3.31662 + 0.0j)  (-5.72872 - 2.41209j)]
            [              0.0       (3.91965 + 0.0j)]
            >>> Q * R
            [(1.0 + 0.0j)  (2.0 - 3.0j)]
            [(3.0 + 1.0j)  (4.0 + 5.0j)]
            >>> chop(Q.T * Q.conjugate())
            [1.0  0.0]
            [0.0  1.0]

        """

        # check values before continuing
        assert isinstance(A, ctx.matrix)
        m = A.rows
        n = A.cols
        assert n >= 0
        assert m >= n
        assert edps >= 0

        # check for complex data type
        cmplx = any(type(x) is ctx.mpc for x in A)

        # temporarily increase the precision and initialize
        with ctx.extradps(edps):
            tau = ctx.matrix(n,1)
            A = A.copy()

            # ---------------
            # FACTOR MATRIX A
            # ---------------
            if cmplx:
                one = ctx.mpc('1.0', '0.0')
                zero = ctx.mpc('0.0', '0.0')
                rzero = ctx.mpf('0.0')

                # main loop to factor A (complex)
                for j in xrange(0, n):
                    alpha = A[j,j]
                    alphr = ctx.re(alpha)
                    alphi = ctx.im(alpha)

                    if (m-j) >= 2:
                        xnorm = ctx.fsum( A[i,j]*ctx.conj(A[i,j]) for i in xrange(j+1, m) )
                        xnorm = ctx.re( ctx.sqrt(xnorm) )
                    else:
                        xnorm = rzero

                    if (xnorm == rzero) and (alphi == rzero):
                        tau[j] = zero
                        continue

                    if alphr < rzero:
                        beta = ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)
                    else:
                        beta = -ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)

                    tau[j] = ctx.mpc( (beta - alphr) / beta, -alphi / beta )
                    t = -ctx.conj(tau[j])
                    za = one / (alpha - beta)

                    for i in xrange(j+1, m):
                        A[i,j] *= za

                    A[j,j] = one
                    for k in xrange(j+1, n):
                        y = ctx.fsum(A[i,j] * ctx.conj(A[i,k]) for i in xrange(j, m))
                        temp = t * ctx.conj(y)
                        for i in xrange(j, m):
                            A[i,k] += A[i,j] * temp

                    A[j,j] = ctx.mpc(beta, '0.0')
            else:
                one = ctx.mpf('1.0')
                zero = ctx.mpf('0.0')

                # main loop to factor A (real)
                for j in xrange(0, n):
                    alpha = A[j,j]

                    if (m-j) > 2:
                        xnorm = ctx.fsum( (A[i,j])**2 for i in xrange(j+1, m) )
                        xnorm = ctx.sqrt(xnorm)
                    elif (m-j) == 2:
                        xnorm = abs( A[m-1,j] )
                    else:
                        xnorm = zero

                    if xnorm == zero:
                        tau[j] = zero
                        continue

                    if alpha < zero:
                        beta = ctx.sqrt(alpha**2 + xnorm**2)
                    else:
                        beta = -ctx.sqrt(alpha**2 + xnorm**2)

                    tau[j] = (beta - alpha) / beta
                    t = -tau[j]
                    da = one / (alpha - beta)

                    for i in xrange(j+1, m):
                        A[i,j] *= da

                    A[j,j] = one
                    for k in xrange(j+1, n):
                        y = ctx.fsum( A[i,j] * A[i,k] for i in xrange(j, m) )
                        temp = t * y
                        for i in xrange(j,m):
                            A[i,k] += A[i,j] * temp

                    A[j,j] = beta

            # return factorization in same internal format as LAPACK
            if (mode == 'raw') or (mode == 'RAW'):
                return A, tau

            # ----------------------------------
            # FORM Q USING BACKWARD ACCUMULATION
            # ----------------------------------

            # form R before the values are overwritten
            R = A.copy()
            for j in xrange(0, n):
                for i in xrange(j+1, m):
                    R[i,j] = zero

            # set the value of p (number of columns of Q to return)
            p = m
            if (mode == 'skinny') or (mode == 'SKINNY'):
                p = n

            # add columns to A if needed and initialize
            A.cols += (p-n)
            for j in xrange(0, p):
                A[j,j] = one
                for i in xrange(0, j):
                    A[i,j] = zero

            # main loop to form Q
            for j in xrange(n-1, -1, -1):
                t = -tau[j]
                A[j,j] += t

                for k in xrange(j+1, p):
                    if cmplx:
                        y = ctx.fsum(A[i,j] * ctx.conj(A[i,k]) for i in xrange(j+1, m))
                        temp = t * ctx.conj(y)
                    else:
                        y = ctx.fsum(A[i,j] * A[i,k] for i in xrange(j+1, m))
                        temp = t * y
                    A[j,k] = temp
                    for i in xrange(j+1, m):
                        A[i,k] += A[i,j] * temp

                for i in xrange(j+1, m):
                    A[i, j] *= t

            return A, R[0:p,0:n]

        # ------------------
        # END OF FUNCTION QR
        # ------------------
