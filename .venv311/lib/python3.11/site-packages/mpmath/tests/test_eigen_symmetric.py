#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpmath import mp
from mpmath import libmp

xrange = libmp.backend.xrange

def run_eigsy(A, verbose = False):
    if verbose:
        print("original matrix:\n", str(A))

    D, Q = mp.eigsy(A)
    B = Q * mp.diag(D) * Q.transpose()
    C = A - B
    E = Q * Q.transpose() - mp.eye(A.rows)

    if verbose:
        print("eigenvalues:\n", D)
        print("eigenvectors:\n", Q)

    NC = mp.mnorm(C)
    NE = mp.mnorm(E)

    if verbose:
        print("difference:", NC, "\n", C, "\n")
        print("difference:", NE, "\n", E, "\n")

    eps = mp.exp( 0.8 * mp.log(mp.eps))

    assert NC < eps
    assert NE < eps

    return NC

def run_eighe(A, verbose = False):
    if verbose:
        print("original matrix:\n", str(A))

    D, Q = mp.eighe(A)
    B = Q * mp.diag(D) * Q.transpose_conj()
    C = A - B
    E = Q * Q.transpose_conj() - mp.eye(A.rows)

    if verbose:
        print("eigenvalues:\n", D)
        print("eigenvectors:\n", Q)

    NC = mp.mnorm(C)
    NE = mp.mnorm(E)

    if verbose:
        print("difference:", NC, "\n", C, "\n")
        print("difference:", NE, "\n", E, "\n")

    eps = mp.exp( 0.8 * mp.log(mp.eps))

    assert NC < eps
    assert NE < eps

    return NC

def run_svd_r(A, full_matrices = False, verbose = True):

    m, n = A.rows, A.cols

    eps = mp.exp(0.8 * mp.log(mp.eps))

    if verbose:
        print("original matrix:\n", str(A))
        print("full", full_matrices)

    U, S0, V = mp.svd_r(A, full_matrices = full_matrices)

    S = mp.zeros(U.cols, V.rows)
    for j in xrange(min(m, n)):
        S[j,j] = S0[j]

    if verbose:
        print("U:\n", str(U))
        print("S:\n", str(S0))
        print("V:\n", str(V))

    C = U * S * V - A
    err = mp.mnorm(C)
    if verbose:
        print("C\n", str(C), "\n", err)
    assert err < eps

    D = V * V.transpose() - mp.eye(V.rows)
    err = mp.mnorm(D)
    if verbose:
        print("D:\n", str(D), "\n", err)
    assert err < eps

    E = U.transpose() * U - mp.eye(U.cols)
    err = mp.mnorm(E)
    if verbose:
        print("E:\n", str(E), "\n", err)
    assert err < eps

def run_svd_c(A, full_matrices = False, verbose = True):

    m, n = A.rows, A.cols

    eps = mp.exp(0.8 * mp.log(mp.eps))

    if verbose:
        print("original matrix:\n", str(A))
        print("full", full_matrices)

    U, S0, V = mp.svd_c(A, full_matrices = full_matrices)

    S = mp.zeros(U.cols, V.rows)
    for j in xrange(min(m, n)):
        S[j,j] = S0[j]

    if verbose:
        print("U:\n", str(U))
        print("S:\n", str(S0))
        print("V:\n", str(V))

    C = U * S * V - A
    err = mp.mnorm(C)
    if verbose:
        print("C\n", str(C), "\n", err)
    assert err  < eps

    D = V * V.transpose_conj() - mp.eye(V.rows)
    err = mp.mnorm(D)
    if verbose:
        print("D:\n", str(D), "\n", err)
    assert err < eps

    E = U.transpose_conj() * U - mp.eye(U.cols)
    err = mp.mnorm(E)
    if verbose:
        print("E:\n", str(E), "\n", err)
    assert err < eps

def run_gauss(qtype, a, b):
    eps = 1e-5

    d, e = mp.gauss_quadrature(len(a), qtype)
    d -= mp.matrix(a)
    e -= mp.matrix(b)

    assert mp.mnorm(d) < eps
    assert mp.mnorm(e) < eps

def irandmatrix(n, range = 10):
    """
    random matrix with integer entries
    """
    A = mp.matrix(n, n)
    for i in xrange(n):
        for j in xrange(n):
            A[i,j]=int( (2 * mp.rand() - 1) * range)
    return A

#######################

def test_eighe_fixed_matrix():
    A = mp.matrix([[2, 3], [3, 5]])
    run_eigsy(A)
    run_eighe(A)

    A = mp.matrix([[7, -11], [-11, 13]])
    run_eigsy(A)
    run_eighe(A)

    A = mp.matrix([[2, 11, 7], [11, 3, 13], [7, 13, 5]])
    run_eigsy(A)
    run_eighe(A)

    A = mp.matrix([[2, 0, 7], [0, 3, 1], [7, 1, 5]])
    run_eigsy(A)
    run_eighe(A)

    #

    A = mp.matrix([[2, 3+7j], [3-7j, 5]])
    run_eighe(A)

    A = mp.matrix([[2, -11j, 0], [+11j, 3, 29j], [0, -29j, 5]])
    run_eighe(A)

    A = mp.matrix([[2, 11 + 17j, 7 + 19j], [11 - 17j, 3, -13 + 23j], [7 - 19j, -13 - 23j, 5]])
    run_eighe(A)

def test_eigsy_randmatrix():
    N = 5

    for a in xrange(10):
        A = 2 * mp.randmatrix(N, N) - 1

        for i in xrange(0, N):
            for j in xrange(i + 1, N):
                A[j,i] = A[i,j]

        run_eigsy(A)

def test_eighe_randmatrix():
    N = 5

    for a in xrange(10):
        A = (2 * mp.randmatrix(N, N) - 1) + 1j * (2 * mp.randmatrix(N, N) - 1)

        for i in xrange(0, N):
            A[i,i] = mp.re(A[i,i])
            for j in xrange(i + 1, N):
                A[j,i] = mp.conj(A[i,j])

        run_eighe(A)

def test_eigsy_irandmatrix():
    N = 4
    R = 4

    for a in xrange(10):
        A=irandmatrix(N, R)

        for i in xrange(0, N):
            for j in xrange(i + 1, N):
                A[j,i] = A[i,j]

        run_eigsy(A)

def test_eighe_irandmatrix():
    N = 4
    R = 4

    for a in xrange(10):
        A=irandmatrix(N, R) + 1j * irandmatrix(N, R)

        for i in xrange(0, N):
            A[i,i] = mp.re(A[i,i])
            for j in xrange(i + 1, N):
                A[j,i] = mp.conj(A[i,j])

        run_eighe(A)

def test_svd_r_rand():
    for i in xrange(5):
        full = mp.rand() > 0.5
        m = 1 + int(mp.rand() * 10)
        n = 1 + int(mp.rand() * 10)
        A = 2 * mp.randmatrix(m, n) - 1
        if mp.rand() > 0.5:
            A *= 10
            for x in xrange(m):
                for y in xrange(n):
                    A[x,y]=int(A[x,y])

        run_svd_r(A, full_matrices = full, verbose = False)

def test_svd_c_rand():
    for i in xrange(5):
        full = mp.rand() > 0.5
        m = 1 + int(mp.rand() * 10)
        n = 1 + int(mp.rand() * 10)
        A = (2 * mp.randmatrix(m, n) - 1) + 1j * (2 * mp.randmatrix(m, n) - 1)
        if mp.rand() > 0.5:
            A *= 10
            for x in xrange(m):
                for y in xrange(n):
                    A[x,y]=int(mp.re(A[x,y])) + 1j * int(mp.im(A[x,y]))

        run_svd_c(A, full_matrices=full, verbose=False)

def test_svd_test_case():
    # a test case from Golub and Reinsch
    #  (see wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).)

    eps = mp.exp(0.8 * mp.log(mp.eps))

    a = [[22, 10,  2,   3,  7],
         [14,  7, 10,   0,  8],
         [-1, 13, -1, -11,  3],
         [-3, -2, 13,  -2,  4],
         [ 9,  8,  1,  -2,  4],
         [ 9,  1, -7,   5, -1],
         [ 2, -6,  6,   5,  1],
         [ 4,  5,  0,  -2,  2]]

    a = mp.matrix(a)
    b = mp.matrix([mp.sqrt(1248), 20, mp.sqrt(384), 0, 0])

    S = mp.svd_r(a, compute_uv = False)
    S -= b
    assert mp.mnorm(S) < eps

    S = mp.svd_c(a, compute_uv = False)
    S -= b
    assert mp.mnorm(S) < eps


def test_gauss_quadrature_static():
    a = [-0.57735027,  0.57735027]
    b = [ 1,  1]
    run_gauss("legendre", a , b)

    a = [ -0.906179846,  -0.538469310,   0,           0.538469310,   0.906179846]
    b = [  0.23692689,    0.47862867,    0.56888889,  0.47862867,    0.23692689]
    run_gauss("legendre", a , b)

    a = [ 0.06943184,  0.33000948,  0.66999052,  0.93056816]
    b = [ 0.17392742,  0.32607258,  0.32607258,  0.17392742]
    run_gauss("legendre01", a , b)

    a = [-0.70710678,  0.70710678]
    b = [ 0.88622693,  0.88622693]
    run_gauss("hermite", a , b)

    a = [ -2.02018287,  -0.958572465,   0,           0.958572465,   2.02018287]
    b = [  0.01995324,   0.39361932,    0.94530872,  0.39361932,    0.01995324]
    run_gauss("hermite", a , b)

    a = [ 0.41577456,  2.29428036,  6.28994508]
    b = [ 0.71109301,  0.27851773,  0.01038926]
    run_gauss("laguerre", a , b)

def test_gauss_quadrature_dynamic(verbose = False):
    n = 5

    A = mp.randmatrix(2 * n, 1)

    def F(x):
        r = 0
        for i in xrange(len(A) - 1, -1, -1):
            r = r * x + A[i]
        return r

    def run(qtype, FW, R, alpha = 0, beta = 0):
        X, W = mp.gauss_quadrature(n, qtype, alpha = alpha, beta = beta)

        a = 0
        for i in xrange(len(X)):
            a += W[i] * F(X[i])

        b = mp.quad(lambda x: FW(x) * F(x), R)

        c = mp.fabs(a - b)

        if verbose:
            print(qtype, c, a, b)

        assert c < 1e-5

    run("legendre", lambda x: 1, [-1, 1])
    run("legendre01", lambda x: 1, [0, 1])
    run("hermite", lambda x: mp.exp(-x*x), [-mp.inf, mp.inf])
    run("laguerre", lambda x: mp.exp(-x), [0, mp.inf])
    run("glaguerre", lambda x: mp.sqrt(x)*mp.exp(-x), [0, mp.inf], alpha = 1 / mp.mpf(2))
    run("chebyshev1", lambda x: 1/mp.sqrt(1-x*x), [-1, 1])
    run("chebyshev2", lambda x: mp.sqrt(1-x*x), [-1, 1])
    run("jacobi", lambda x: (1-x)**(1/mp.mpf(3)) * (1+x)**(1/mp.mpf(5)), [-1, 1], alpha = 1 / mp.mpf(3), beta = 1 / mp.mpf(5) )
