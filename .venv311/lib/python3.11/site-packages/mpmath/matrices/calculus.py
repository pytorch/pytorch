from ..libmp.backend import xrange

# TODO: should use diagonalization-based algorithms

class MatrixCalculusMethods(object):

    def _exp_pade(ctx, a):
        """
        Exponential of a matrix using Pade approximants.

        See G. H. Golub, C. F. van Loan 'Matrix Computations',
        third Ed., page 572

        TODO:
         - find a good estimate for q
         - reduce the number of matrix multiplications to improve
           performance
        """
        def eps_pade(p):
            return ctx.mpf(2)**(3-2*p) * \
                ctx.factorial(p)**2/(ctx.factorial(2*p)**2 * (2*p + 1))
        q = 4
        extraq = 8
        while 1:
            if eps_pade(q) < ctx.eps:
                break
            q += 1
        q += extraq
        j = int(max(1, ctx.mag(ctx.mnorm(a,'inf'))))
        extra = q
        prec = ctx.prec
        ctx.dps += extra + 3
        try:
            a = a/2**j
            na = a.rows
            den = ctx.eye(na)
            num = ctx.eye(na)
            x = ctx.eye(na)
            c = ctx.mpf(1)
            for k in range(1, q+1):
                c *= ctx.mpf(q - k + 1)/((2*q - k + 1) * k)
                x = a*x
                cx = c*x
                num += cx
                den += (-1)**k * cx
            f = ctx.lu_solve_mat(den, num)
            for k in range(j):
                f = f*f
        finally:
            ctx.prec = prec
        return f*1

    def expm(ctx, A, method='taylor'):
        r"""
        Computes the matrix exponential of a square matrix `A`, which is defined
        by the power series

        .. math ::

            \exp(A) = I + A + \frac{A^2}{2!} + \frac{A^3}{3!} + \ldots

        With method='taylor', the matrix exponential is computed
        using the Taylor series. With method='pade', Pade approximants
        are used instead.

        **Examples**

        Basic examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> expm(zeros(3))
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]
            >>> expm(eye(3))
            [2.71828182845905               0.0               0.0]
            [             0.0  2.71828182845905               0.0]
            [             0.0               0.0  2.71828182845905]
            >>> expm([[1,1,0],[1,0,1],[0,1,0]])
            [ 3.86814500615414  2.26812870852145  0.841130841230196]
            [ 2.26812870852145  2.44114713886289   1.42699786729125]
            [0.841130841230196  1.42699786729125    1.6000162976327]
            >>> expm([[1,1,0],[1,0,1],[0,1,0]], method='pade')
            [ 3.86814500615414  2.26812870852145  0.841130841230196]
            [ 2.26812870852145  2.44114713886289   1.42699786729125]
            [0.841130841230196  1.42699786729125    1.6000162976327]
            >>> expm([[1+j, 0], [1+j,1]])
            [(1.46869393991589 + 2.28735528717884j)                        0.0]
            [  (1.03776739863568 + 3.536943175722j)  (2.71828182845905 + 0.0j)]

        Matrices with large entries are allowed::

            >>> expm(matrix([[1,2],[2,3]])**25)
            [5.65024064048415e+2050488462815550  9.14228140091932e+2050488462815550]
            [9.14228140091932e+2050488462815550  1.47925220414035e+2050488462815551]

        The identity `\exp(A+B) = \exp(A) \exp(B)` does not hold for
        noncommuting matrices::

            >>> A = hilbert(3)
            >>> B = A + eye(3)
            >>> chop(mnorm(A*B - B*A))
            0.0
            >>> chop(mnorm(expm(A+B) - expm(A)*expm(B)))
            0.0
            >>> B = A + ones(3)
            >>> mnorm(A*B - B*A)
            1.8
            >>> mnorm(expm(A+B) - expm(A)*expm(B))
            42.0927851137247

        """
        if method == 'pade':
            prec = ctx.prec
            try:
                A = ctx.matrix(A)
                ctx.prec += 2*A.rows
                res = ctx._exp_pade(A)
            finally:
                ctx.prec = prec
            return res
        A = ctx.matrix(A)
        prec = ctx.prec
        j = int(max(1, ctx.mag(ctx.mnorm(A,'inf'))))
        j += int(0.5*prec**0.5)
        try:
            ctx.prec += 10 + 2*j
            tol = +ctx.eps
            A = A/2**j
            T = A
            Y = A**0 + A
            k = 2
            while 1:
                T *= A * (1/ctx.mpf(k))
                if ctx.mnorm(T, 'inf') < tol:
                    break
                Y += T
                k += 1
            for k in xrange(j):
                Y = Y*Y
        finally:
            ctx.prec = prec
        Y *= 1
        return Y

    def cosm(ctx, A):
        r"""
        Gives the cosine of a square matrix `A`, defined in analogy
        with the matrix exponential.

        Examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> cosm(X)
            [0.54030230586814               0.0               0.0]
            [             0.0  0.54030230586814               0.0]
            [             0.0               0.0  0.54030230586814]
            >>> X = hilbert(3)
            >>> cosm(X)
            [ 0.424403834569555  -0.316643413047167  -0.221474945949293]
            [-0.316643413047167   0.820646708837824  -0.127183694770039]
            [-0.221474945949293  -0.127183694770039   0.909236687217541]
            >>> X = matrix([[1+j,-2],[0,-j]])
            >>> cosm(X)
            [(0.833730025131149 - 0.988897705762865j)  (1.07485840848393 - 0.17192140544213j)]
            [                                     0.0               (1.54308063481524 + 0.0j)]
        """
        B = 0.5 * (ctx.expm(A*ctx.j) + ctx.expm(A*(-ctx.j)))
        if not sum(A.apply(ctx.im).apply(abs)):
            B = B.apply(ctx.re)
        return B

    def sinm(ctx, A):
        r"""
        Gives the sine of a square matrix `A`, defined in analogy
        with the matrix exponential.

        Examples::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> sinm(X)
            [0.841470984807897                0.0                0.0]
            [              0.0  0.841470984807897                0.0]
            [              0.0                0.0  0.841470984807897]
            >>> X = hilbert(3)
            >>> sinm(X)
            [0.711608512150994  0.339783913247439  0.220742837314741]
            [0.339783913247439  0.244113865695532  0.187231271174372]
            [0.220742837314741  0.187231271174372  0.155816730769635]
            >>> X = matrix([[1+j,-2],[0,-j]])
            >>> sinm(X)
            [(1.29845758141598 + 0.634963914784736j)  (-1.96751511930922 + 0.314700021761367j)]
            [                                    0.0                  (0.0 - 1.1752011936438j)]
        """
        B = (-0.5j) * (ctx.expm(A*ctx.j) - ctx.expm(A*(-ctx.j)))
        if not sum(A.apply(ctx.im).apply(abs)):
            B = B.apply(ctx.re)
        return B

    def _sqrtm_rot(ctx, A, _may_rotate):
        # If the iteration fails to converge, cheat by performing
        # a rotation by a complex number
        u = ctx.j**0.3
        return ctx.sqrtm(u*A, _may_rotate) / ctx.sqrt(u)

    def sqrtm(ctx, A, _may_rotate=2):
        r"""
        Computes a square root of the square matrix `A`, i.e. returns
        a matrix `B = A^{1/2}` such that `B^2 = A`. The square root
        of a matrix, if it exists, is not unique.

        **Examples**

        Square roots of some simple matrices::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> sqrtm([[1,0], [0,1]])
            [1.0  0.0]
            [0.0  1.0]
            >>> sqrtm([[0,0], [0,0]])
            [0.0  0.0]
            [0.0  0.0]
            >>> sqrtm([[2,0],[0,1]])
            [1.4142135623731  0.0]
            [            0.0  1.0]
            >>> sqrtm([[1,1],[1,0]])
            [ (0.920442065259926 - 0.21728689675164j)  (0.568864481005783 + 0.351577584254143j)]
            [(0.568864481005783 + 0.351577584254143j)  (0.351577584254143 - 0.568864481005783j)]
            >>> sqrtm([[1,0],[0,1]])
            [1.0  0.0]
            [0.0  1.0]
            >>> sqrtm([[-1,0],[0,1]])
            [(0.0 - 1.0j)           0.0]
            [         0.0  (1.0 + 0.0j)]
            >>> sqrtm([[j,0],[0,j]])
            [(0.707106781186547 + 0.707106781186547j)                                       0.0]
            [                                     0.0  (0.707106781186547 + 0.707106781186547j)]

        A square root of a rotation matrix, giving the corresponding
        half-angle rotation matrix::

            >>> t1 = 0.75
            >>> t2 = t1 * 0.5
            >>> A1 = matrix([[cos(t1), -sin(t1)], [sin(t1), cos(t1)]])
            >>> A2 = matrix([[cos(t2), -sin(t2)], [sin(t2), cos(t2)]])
            >>> sqrtm(A1)
            [0.930507621912314  -0.366272529086048]
            [0.366272529086048   0.930507621912314]
            >>> A2
            [0.930507621912314  -0.366272529086048]
            [0.366272529086048   0.930507621912314]

        The identity `(A^2)^{1/2} = A` does not necessarily hold::

            >>> A = matrix([[4,1,4],[7,8,9],[10,2,11]])
            >>> sqrtm(A**2)
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> sqrtm(A)**2
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> A = matrix([[-4,1,4],[7,-8,9],[10,2,11]])
            >>> sqrtm(A**2)
            [  7.43715112194995  -0.324127569985474   1.8481718827526]
            [-0.251549715716942    9.32699765900402  2.48221180985147]
            [  4.11609388833616   0.775751877098258   13.017955697342]
            >>> chop(sqrtm(A)**2)
            [-4.0   1.0   4.0]
            [ 7.0  -8.0   9.0]
            [10.0   2.0  11.0]

        For some matrices, a square root does not exist::

            >>> sqrtm([[0,1], [0,0]])
            Traceback (most recent call last):
              ...
            ZeroDivisionError: matrix is numerically singular

        Two examples from the documentation for Matlab's ``sqrtm``::

            >>> mp.dps = 15; mp.pretty = True
            >>> sqrtm([[7,10],[15,22]])
            [1.56669890360128  1.74077655955698]
            [2.61116483933547  4.17786374293675]
            >>>
            >>> X = matrix(\
            ...   [[5,-4,1,0,0],
            ...   [-4,6,-4,1,0],
            ...   [1,-4,6,-4,1],
            ...   [0,1,-4,6,-4],
            ...   [0,0,1,-4,5]])
            >>> Y = matrix(\
            ...   [[2,-1,-0,-0,-0],
            ...   [-1,2,-1,0,-0],
            ...   [0,-1,2,-1,0],
            ...   [-0,0,-1,2,-1],
            ...   [-0,-0,-0,-1,2]])
            >>> mnorm(sqrtm(X) - Y)
            4.53155328326114e-19

        """
        A = ctx.matrix(A)
        # Trivial
        if A*0 == A:
            return A
        prec = ctx.prec
        if _may_rotate:
            d = ctx.det(A)
            if abs(ctx.im(d)) < 16*ctx.eps and ctx.re(d) < 0:
                return ctx._sqrtm_rot(A, _may_rotate-1)
        try:
            ctx.prec += 10
            tol = ctx.eps * 128
            Y = A
            Z = I = A**0
            k = 0
            # Denman-Beavers iteration
            while 1:
                Yprev = Y
                try:
                    Y, Z = 0.5*(Y+ctx.inverse(Z)), 0.5*(Z+ctx.inverse(Y))
                except ZeroDivisionError:
                    if _may_rotate:
                        Y = ctx._sqrtm_rot(A, _may_rotate-1)
                        break
                    else:
                        raise
                mag1 = ctx.mnorm(Y-Yprev, 'inf')
                mag2 = ctx.mnorm(Y, 'inf')
                if mag1 <= mag2*tol:
                    break
                if _may_rotate and k > 6 and not mag1 < mag2 * 0.001:
                    return ctx._sqrtm_rot(A, _may_rotate-1)
                k += 1
                if k > ctx.prec:
                    raise ctx.NoConvergence
        finally:
            ctx.prec = prec
        Y *= 1
        return Y

    def logm(ctx, A):
        r"""
        Computes a logarithm of the square matrix `A`, i.e. returns
        a matrix `B = \log(A)` such that `\exp(B) = A`. The logarithm
        of a matrix, if it exists, is not unique.

        **Examples**

        Logarithms of some simple matrices::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> X = eye(3)
            >>> logm(X)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            >>> logm(2*X)
            [0.693147180559945                0.0                0.0]
            [              0.0  0.693147180559945                0.0]
            [              0.0                0.0  0.693147180559945]
            >>> logm(expm(X))
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]

        A logarithm of a complex matrix::

            >>> X = matrix([[2+j, 1, 3], [1-j, 1-2*j, 1], [-4, -5, j]])
            >>> B = logm(X)
            >>> nprint(B)
            [ (0.808757 + 0.107759j)    (2.20752 + 0.202762j)   (1.07376 - 0.773874j)]
            [ (0.905709 - 0.107795j)  (0.0287395 - 0.824993j)  (0.111619 + 0.514272j)]
            [(-0.930151 + 0.399512j)   (-2.06266 - 0.674397j)  (0.791552 + 0.519839j)]
            >>> chop(expm(B))
            [(2.0 + 1.0j)           1.0           3.0]
            [(1.0 - 1.0j)  (1.0 - 2.0j)           1.0]
            [        -4.0          -5.0  (0.0 + 1.0j)]

        A matrix `X` close to the identity matrix, for which
        `\log(\exp(X)) = \exp(\log(X)) = X` holds::

            >>> X = eye(3) + hilbert(3)/4
            >>> X
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]
            >>> logm(expm(X))
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]
            >>> expm(logm(X))
            [              1.25             0.125  0.0833333333333333]
            [             0.125  1.08333333333333              0.0625]
            [0.0833333333333333            0.0625                1.05]

        A logarithm of a rotation matrix, giving back the angle of
        the rotation::

            >>> t = 3.7
            >>> A = matrix([[cos(t),sin(t)],[-sin(t),cos(t)]])
            >>> chop(logm(A))
            [             0.0  -2.58318530717959]
            [2.58318530717959                0.0]
            >>> (2*pi-t)
            2.58318530717959

        For some matrices, a logarithm does not exist::

            >>> logm([[1,0], [0,0]])
            Traceback (most recent call last):
              ...
            ZeroDivisionError: matrix is numerically singular

        Logarithm of a matrix with large entries::

            >>> logm(hilbert(3) * 10**20).apply(re)
            [ 45.5597513593433  1.27721006042799  0.317662687717978]
            [ 1.27721006042799  42.5222778973542   2.24003708791604]
            [0.317662687717978  2.24003708791604    42.395212822267]

        """
        A = ctx.matrix(A)
        prec = ctx.prec
        try:
            ctx.prec += 10
            tol = ctx.eps * 128
            I = A**0
            B = A
            n = 0
            while 1:
                B = ctx.sqrtm(B)
                n += 1
                if ctx.mnorm(B-I, 'inf') < 0.125:
                    break
            T = X = B-I
            L = X*0
            k = 1
            while 1:
                if k & 1:
                    L += T / k
                else:
                    L -= T / k
                T *= X
                if ctx.mnorm(T, 'inf') < tol:
                    break
                k += 1
                if k > ctx.prec:
                    raise ctx.NoConvergence
        finally:
            ctx.prec = prec
        L *= 2**n
        return L

    def powm(ctx, A, r):
        r"""
        Computes `A^r = \exp(A \log r)` for a matrix `A` and complex
        number `r`.

        **Examples**

        Powers and inverse powers of a matrix::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> A = matrix([[4,1,4],[7,8,9],[10,2,11]])
            >>> powm(A, 2)
            [ 63.0  20.0   69.0]
            [174.0  89.0  199.0]
            [164.0  48.0  179.0]
            >>> chop(powm(powm(A, 4), 1/4.))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(20)(powm)(A, -4), -1/4.)
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> chop(powm(powm(A, 1+0.5j), 1/(1+0.5j)))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]
            >>> powm(extraprec(5)(powm)(A, -1.5), -1/(1.5))
            [ 4.0  1.0   4.0]
            [ 7.0  8.0   9.0]
            [10.0  2.0  11.0]

        A Fibonacci-generating matrix::

            >>> powm([[1,1],[1,0]], 10)
            [89.0  55.0]
            [55.0  34.0]
            >>> fib(10)
            55.0
            >>> powm([[1,1],[1,0]], 6.5)
            [(16.5166626964253 - 0.0121089837381789j)  (10.2078589271083 + 0.0195927472575932j)]
            [(10.2078589271083 + 0.0195927472575932j)  (6.30880376931698 - 0.0317017309957721j)]
            >>> (phi**6.5 - (1-phi)**6.5)/sqrt(5)
            (10.2078589271083 - 0.0195927472575932j)
            >>> powm([[1,1],[1,0]], 6.2)
            [ (14.3076953002666 - 0.008222855781077j)  (8.81733464837593 + 0.0133048601383712j)]
            [(8.81733464837593 + 0.0133048601383712j)  (5.49036065189071 - 0.0215277159194482j)]
            >>> (phi**6.2 - (1-phi)**6.2)/sqrt(5)
            (8.81733464837593 - 0.0133048601383712j)

        """
        A = ctx.matrix(A)
        r = ctx.convert(r)
        prec = ctx.prec
        try:
            ctx.prec += 10
            if ctx.isint(r):
                v = A ** int(r)
            elif ctx.isint(r*2):
                y = int(r*2)
                v = ctx.sqrtm(A) ** y
            else:
                v = ctx.expm(r*ctx.logm(A))
        finally:
            ctx.prec = prec
        v *= 1
        return v
