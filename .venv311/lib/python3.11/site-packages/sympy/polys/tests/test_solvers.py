"""Tests for low-level linear systems solver. """

from sympy.matrices import Matrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.fields import field
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys, eqs_to_matrix


def test_solve_lin_sys_2x2_one():
    domain, x1,x2 = ring("x1,x2", QQ)
    eqs = [x1 + x2 - 5,
           2*x1 - x2]
    sol = {x1: QQ(5, 3), x2: QQ(10, 3)}
    _sol = solve_lin_sys(eqs, domain)
    assert _sol == sol and all(s.ring == domain for s in _sol)

def test_solve_lin_sys_2x4_none():
    domain, x1,x2 = ring("x1,x2", QQ)
    eqs = [x1 - 1,
           x1 - x2,
           x1 - 2*x2,
           x2 - 1]
    assert solve_lin_sys(eqs, domain) is None


def test_solve_lin_sys_3x4_one():
    domain, x1,x2,x3 = ring("x1,x2,x3", QQ)
    eqs = [x1 + 2*x2 + 3*x3,
           2*x1 - x2 + x3,
           3*x1 + x2 + x3,
           5*x2 + 2*x3]
    sol = {x1: 0, x2: 0, x3: 0}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_3x3_inf():
    domain, x1,x2,x3 = ring("x1,x2,x3", QQ)
    eqs = [x1 - x2 + 2*x3 - 1,
           2*x1 + x2 + x3 - 8,
           x1 + x2 - 5]
    sol = {x1: -x3 + 3, x2: x3 + 2}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_3x4_none():
    domain, x1,x2,x3,x4 = ring("x1,x2,x3,x4", QQ)
    eqs = [2*x1 + x2 + 7*x3 - 7*x4 - 2,
           -3*x1 + 4*x2 - 5*x3 - 6*x4 - 3,
           x1 + x2 + 4*x3 - 5*x4 - 2]
    assert solve_lin_sys(eqs, domain) is None


def test_solve_lin_sys_4x7_inf():
    domain, x1,x2,x3,x4,x5,x6,x7 = ring("x1,x2,x3,x4,x5,x6,x7", QQ)
    eqs = [x1 + 4*x2 - x4 + 7*x6 - 9*x7 - 3,
           2*x1 + 8*x2 - x3 + 3*x4 + 9*x5 - 13*x6 + 7*x7 - 9,
           2*x3 - 3*x4 - 4*x5 + 12*x6 - 8*x7 - 1,
           -x1 - 4*x2 + 2*x3 + 4*x4 + 8*x5 - 31*x6 + 37*x7 - 4]
    sol = {x1: 4 - 4*x2 - 2*x5 - x6 + 3*x7,
           x3: 2 - x5 + 3*x6 - 5*x7,
           x4: 1 - 2*x5 + 6*x6 - 6*x7}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_5x5_inf():
    domain, x1,x2,x3,x4,x5 = ring("x1,x2,x3,x4,x5", QQ)
    eqs = [x1 - x2 - 2*x3 + x4 + 11*x5 - 13,
           x1 - x2 + x3 + x4 + 5*x5 - 16,
           2*x1 - 2*x2 + x4 + 10*x5 - 21,
           2*x1 - 2*x2 - x3 + 3*x4 + 20*x5 - 38,
           2*x1 - 2*x2 + x3 + x4 + 8*x5 - 22]
    sol = {x1: 6 + x2 - 3*x5,
           x3: 1 + 2*x5,
           x4: 9 - 4*x5}
    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_6x6_1():
    ground, d,r,e,g,i,j,l,o,m,p,q = field("d,r,e,g,i,j,l,o,m,p,q", ZZ)
    domain, c,f,h,k,n,b = ring("c,f,h,k,n,b", ground)

    eqs = [b + q/d - c/d, c*(1/d + 1/e + 1/g) - f/g - q/d, f*(1/g + 1/i + 1/j) - c/g - h/i, h*(1/i + 1/l + 1/m) - f/i - k/m, k*(1/m + 1/o + 1/p) - h/m - n/p, n/p - k/p]
    sol = {
         b: (e*i*l*q + e*i*m*q + e*i*o*q + e*j*l*q + e*j*m*q + e*j*o*q + e*l*m*q + e*l*o*q + g*i*l*q + g*i*m*q + g*i*o*q + g*j*l*q + g*j*m*q + g*j*o*q + g*l*m*q + g*l*o*q + i*j*l*q + i*j*m*q + i*j*o*q + j*l*m*q + j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         c: (-e*g*i*l*q - e*g*i*m*q - e*g*i*o*q - e*g*j*l*q - e*g*j*m*q - e*g*j*o*q - e*g*l*m*q - e*g*l*o*q - e*i*j*l*q - e*i*j*m*q - e*i*j*o*q - e*j*l*m*q - e*j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         f: (-e*i*j*l*q - e*i*j*m*q - e*i*j*o*q - e*j*l*m*q - e*j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         h: (-e*j*l*m*q - e*j*l*o*q)/(-d*e*i*l - d*e*i*m - d*e*i*o - d*e*j*l - d*e*j*m - d*e*j*o - d*e*l*m - d*e*l*o - d*g*i*l - d*g*i*m - d*g*i*o - d*g*j*l - d*g*j*m - d*g*j*o - d*g*l*m - d*g*l*o - d*i*j*l - d*i*j*m - d*i*j*o - d*j*l*m - d*j*l*o - e*g*i*l - e*g*i*m - e*g*i*o - e*g*j*l - e*g*j*m - e*g*j*o - e*g*l*m - e*g*l*o - e*i*j*l - e*i*j*m - e*i*j*o - e*j*l*m - e*j*l*o),
         k: e*j*l*o*q/(d*e*i*l + d*e*i*m + d*e*i*o + d*e*j*l + d*e*j*m + d*e*j*o + d*e*l*m + d*e*l*o + d*g*i*l + d*g*i*m + d*g*i*o + d*g*j*l + d*g*j*m + d*g*j*o + d*g*l*m + d*g*l*o + d*i*j*l + d*i*j*m + d*i*j*o + d*j*l*m + d*j*l*o + e*g*i*l + e*g*i*m + e*g*i*o + e*g*j*l + e*g*j*m + e*g*j*o + e*g*l*m + e*g*l*o + e*i*j*l + e*i*j*m + e*i*j*o + e*j*l*m + e*j*l*o),
         n: e*j*l*o*q/(d*e*i*l + d*e*i*m + d*e*i*o + d*e*j*l + d*e*j*m + d*e*j*o + d*e*l*m + d*e*l*o + d*g*i*l + d*g*i*m + d*g*i*o + d*g*j*l + d*g*j*m + d*g*j*o + d*g*l*m + d*g*l*o + d*i*j*l + d*i*j*m + d*i*j*o + d*j*l*m + d*j*l*o + e*g*i*l + e*g*i*m + e*g*i*o + e*g*j*l + e*g*j*m + e*g*j*o + e*g*l*m + e*g*l*o + e*i*j*l + e*i*j*m + e*i*j*o + e*j*l*m + e*j*l*o),
    }

    assert solve_lin_sys(eqs, domain) == sol

def test_solve_lin_sys_6x6_2():
    ground, d,r,e,g,i,j,l,o,m,p,q = field("d,r,e,g,i,j,l,o,m,p,q", ZZ)
    domain, c,f,h,k,n,b = ring("c,f,h,k,n,b", ground)

    eqs = [b + r/d - c/d, c*(1/d + 1/e + 1/g) - f/g - r/d, f*(1/g + 1/i + 1/j) - c/g - h/i, h*(1/i + 1/l + 1/m) - f/i - k/m, k*(1/m + 1/o + 1/p) - h/m - n/p, n*(1/p + 1/q) - k/p]
    sol = {
        b: -((l*q*e*o + l*q*g*o + i*m*q*e + i*l*q*e + i*l*p*e + i*j*o*q + j*e*o*q + g*j*o*q + i*e*o*q + g*i*o*q + e*l*o*p + e*l*m*p + e*l*m*o + e*i*o*p + e*i*m*p + e*i*m*o + e*i*l*o + j*e*o*p + j*e*m*q + j*e*m*p + j*e*m*o + j*l*m*q + j*l*m*p + j*l*m*o + i*j*m*p + i*j*m*o + i*j*l*q + i*j*l*o + i*j*m*q + j*l*o*p + j*e*l*o + g*j*o*p + g*j*m*q + g*j*m*p + i*j*l*p + i*j*o*p + j*e*l*q + j*e*l*p + j*l*o*q + g*j*m*o + g*j*l*q + g*j*l*p + g*j*l*o + g*l*o*p + g*l*m*p + g*l*m*o + g*i*m*o + g*i*o*p + g*i*m*q + g*i*m*p + g*i*l*q + g*i*l*p + g*i*l*o + l*m*q*e + l*m*q*g)*r)/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        c: (r*e*(l*q*g*o + i*j*o*q + g*j*o*q + g*i*o*q + j*l*m*q + j*l*m*p + j*l*m*o + i*j*m*p + i*j*m*o + i*j*l*q + i*j*l*o + i*j*m*q + j*l*o*p + g*j*o*p + g*j*m*q + g*j*m*p + i*j*l*p + i*j*o*p + j*l*o*q + g*j*m*o + g*j*l*q + g*j*l*p + g*j*l*o + g*l*o*p + g*l*m*p + g*l*m*o + g*i*m*o + g*i*o*p + g*i*m*q + g*i*m*p + g*i*l*q + g*i*l*p + g*i*l*o + l*m*q*g))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        f: (r*e*j*(l*q*o + l*o*p + l*m*q + l*m*p + l*m*o + i*o*q + i*o*p + i*m*q + i*m*p + i*m*o + i*l*q + i*l*p + i*l*o))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        h: (j*e*r*l*(o*q + o*p + m*q + m*p + m*o))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        k: (j*e*r*o*l*(q + p))/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
        n: (j*e*r*o*q*l)/(l*q*d*e*o + l*q*d*g*o + l*q*e*g*o + i*j*d*o*q + i*j*e*o*q + j*d*e*o*q + g*j*d*o*q + g*j*e*o*q + g*i*e*o*q + i*d*e*o*q + g*i*d*o*q + g*i*d*o*p + g*i*d*m*q + g*i*d*m*p + g*i*d*m*o + g*i*d*l*q + g*i*d*l*p + g*i*d*l*o + g*e*l*m*p + g*e*l*o*p + g*j*e*l*q + g*e*l*m*o + g*j*e*m*p + g*j*e*m*o + d*e*l*m*p + d*e*l*m*o + i*d*e*m*p + g*j*e*l*p + g*j*e*l*o + d*e*l*o*p + i*j*d*l*o + i*j*e*o*p + i*j*e*m*q + i*j*d*m*q + i*j*d*m*p + i*j*d*m*o + i*j*d*l*q + i*j*d*l*p + i*j*e*m*p + i*j*e*m*o + i*j*e*l*q + i*j*e*l*p + i*j*e*l*o + i*d*e*m*q + i*d*e*m*o + i*d*e*l*q + i*d*e*l*p + j*d*l*o*p + j*d*e*l*o + g*j*d*o*p + g*j*d*m*q + g*j*d*m*p + g*j*d*m*o + g*j*d*l*q + g*j*d*l*p + g*j*d*l*o + g*j*e*o*p + g*j*e*m*q + g*d*l*o*p + g*d*l*m*p + g*d*l*m*o + j*d*e*m*p + i*d*e*o*p + j*e*o*q*l + j*e*o*p*l + j*e*m*q*l + j*d*e*o*p + j*d*e*m*q + i*j*d*o*p + g*i*e*o*p + j*d*e*m*o + j*d*e*l*q + j*d*e*l*p + j*e*m*p*l + j*e*m*o*l + g*i*e*m*q + g*i*e*m*p + g*i*e*m*o + g*i*e*l*q + g*i*e*l*p + g*i*e*l*o + j*d*l*o*q + j*d*l*m*q + j*d*l*m*p + j*d*l*m*o + i*d*e*l*o + l*m*q*d*e + l*m*q*d*g + l*m*q*e*g),
    }

    assert solve_lin_sys(eqs, domain) == sol

def test_eqs_to_matrix():
    domain, x1,x2 = ring("x1,x2", QQ)
    eqs_coeff = [{x1: QQ(1), x2: QQ(1)}, {x1: QQ(2), x2: QQ(-1)}]
    eqs_rhs = [QQ(-5), QQ(0)]
    M = eqs_to_matrix(eqs_coeff, eqs_rhs, [x1, x2], QQ)
    assert M.to_Matrix() == Matrix([[1, 1, 5], [2, -1, 0]])
