#!/usr/bin/python
# -*- coding: utf-8 -*-

from mpmath import mp
from mpmath import libmp

xrange = libmp.backend.xrange

def run_hessenberg(A, verbose = 0):
    if verbose > 1:
        print("original matrix (hessenberg):\n", A)

    n = A.rows

    Q, H = mp.hessenberg(A)

    if verbose > 1:
        print("Q:\n",Q)
        print("H:\n",H)

    B = Q * H * Q.transpose_conj()

    eps = mp.exp(0.8 * mp.log(mp.eps))

    err0 = 0
    for x in xrange(n):
        for y in xrange(n):
            err0 += abs(A[y,x] - B[y,x])
    err0 /= n * n

    err1 = 0
    for x in xrange(n):
        for y in xrange(x + 2, n):
            err1 += abs(H[y,x])

    if verbose > 0:
        print("difference (H):", err0, err1)

    if verbose > 1:
        print("B:\n", B)

    assert err0 < eps
    assert err1 == 0


def run_schur(A, verbose = 0):
    if verbose > 1:
        print("original matrix (schur):\n", A)

    n = A.rows

    Q, R = mp.schur(A)

    if verbose > 1:
        print("Q:\n", Q)
        print("R:\n", R)

    B = Q * R * Q.transpose_conj()
    C = Q * Q.transpose_conj()

    eps = mp.exp(0.8 * mp.log(mp.eps))

    err0 = 0
    for x in xrange(n):
        for y in xrange(n):
            err0 += abs(A[y,x] - B[y,x])
    err0 /= n * n

    err1 = 0
    for x in xrange(n):
        for y in xrange(n):
            if x == y:
                C[y,x] -= 1
            err1 += abs(C[y,x])
    err1 /= n * n

    err2 = 0
    for x in xrange(n):
        for y in xrange(x + 1, n):
            err2 += abs(R[y,x])

    if verbose > 0:
        print("difference (S):", err0, err1, err2)

    if verbose > 1:
        print("B:\n", B)

    assert err0 < eps
    assert err1 < eps
    assert err2 == 0

def run_eig(A, verbose = 0):
    if verbose > 1:
        print("original matrix (eig):\n", A)

    n = A.rows

    E, EL, ER = mp.eig(A, left = True, right = True)

    if verbose > 1:
        print("E:\n", E)
        print("EL:\n", EL)
        print("ER:\n", ER)

    eps = mp.exp(0.8 * mp.log(mp.eps))

    err0 = 0
    for i in xrange(n):
        B = A * ER[:,i] - E[i] * ER[:,i]
        err0 = max(err0, mp.mnorm(B))

        B = EL[i,:] * A - EL[i,:] * E[i]
        err0 = max(err0, mp.mnorm(B))

    err0 /= n * n

    if verbose > 0:
        print("difference (E):", err0)

    assert err0 < eps

#####################

def test_eig_dyn():
    v = 0
    for i in xrange(5):
        n = 1 + int(mp.rand() * 5)
        if mp.rand() > 0.5:
            # real
            A = 2 * mp.randmatrix(n, n) - 1
            if mp.rand() > 0.5:
                A *= 10
                for x in xrange(n):
                    for y in xrange(n):
                        A[x,y] = int(A[x,y])
        else:
            A = (2 * mp.randmatrix(n, n) - 1) + 1j * (2 * mp.randmatrix(n, n) - 1)
            if mp.rand() > 0.5:
                A *= 10
                for x in xrange(n):
                    for y in xrange(n):
                        A[x,y] = int(mp.re(A[x,y])) + 1j * int(mp.im(A[x,y]))

        run_hessenberg(A, verbose = v)
        run_schur(A, verbose = v)
        run_eig(A, verbose = v)

def test_eig():
    v = 0
    AS = []

    A = mp.matrix([[2, 1, 0],  # jordan block of size 3
                   [0, 2, 1],
                   [0, 0, 2]])
    AS.append(A)
    AS.append(A.transpose())

    A = mp.matrix([[2, 0, 0],  # jordan block of size 2
                   [0, 2, 1],
                   [0, 0, 2]])
    AS.append(A)
    AS.append(A.transpose())

    A = mp.matrix([[2, 0, 1],  # jordan block of size 2
                   [0, 2, 0],
                   [0, 0, 2]])
    AS.append(A)
    AS.append(A.transpose())

    A=  mp.matrix([[0, 0, 1],  # cyclic
                   [1, 0, 0],
                   [0, 1, 0]])
    AS.append(A)
    AS.append(A.transpose())

    for A in AS:
        run_hessenberg(A, verbose = v)
        run_schur(A, verbose = v)
        run_eig(A, verbose = v)
