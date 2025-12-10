r'''
unit test describing the hyperbolic half-plane with the Poincare metric. This
is a basic model of hyperbolic geometry on the (positive) half-space

{(x,y) \in R^2 | y > 0}

with the Riemannian metric

ds^2 = (dx^2 + dy^2)/y^2

It has constant negative scalar curvature = -2

https://en.wikipedia.org/wiki/Poincare_half-plane_model
'''
from sympy.matrices.dense import diag
from sympy.diffgeom import (twoform_to_matrix,
                            metric_to_Christoffel_1st, metric_to_Christoffel_2nd,
                            metric_to_Riemann_components, metric_to_Ricci_components)
import sympy.diffgeom.rn
from sympy.tensor.array import ImmutableDenseNDimArray


def test_H2():
    TP = sympy.diffgeom.TensorProduct
    R2 = sympy.diffgeom.rn.R2
    y = R2.y
    dy = R2.dy
    dx = R2.dx
    g = (TP(dx, dx) + TP(dy, dy))*y**(-2)
    automat = twoform_to_matrix(g)
    mat = diag(y**(-2), y**(-2))
    assert mat == automat

    gamma1 = metric_to_Christoffel_1st(g)
    assert gamma1[0, 0, 0] == 0
    assert gamma1[0, 0, 1] == -y**(-3)
    assert gamma1[0, 1, 0] == -y**(-3)
    assert gamma1[0, 1, 1] == 0

    assert gamma1[1, 1, 1] == -y**(-3)
    assert gamma1[1, 1, 0] == 0
    assert gamma1[1, 0, 1] == 0
    assert gamma1[1, 0, 0] == y**(-3)

    gamma2 = metric_to_Christoffel_2nd(g)
    assert gamma2[0, 0, 0] == 0
    assert gamma2[0, 0, 1] == -y**(-1)
    assert gamma2[0, 1, 0] == -y**(-1)
    assert gamma2[0, 1, 1] == 0

    assert gamma2[1, 1, 1] == -y**(-1)
    assert gamma2[1, 1, 0] == 0
    assert gamma2[1, 0, 1] == 0
    assert gamma2[1, 0, 0] == y**(-1)

    Rm = metric_to_Riemann_components(g)
    assert Rm[0, 0, 0, 0] == 0
    assert Rm[0, 0, 0, 1] == 0
    assert Rm[0, 0, 1, 0] == 0
    assert Rm[0, 0, 1, 1] == 0

    assert Rm[0, 1, 0, 0] == 0
    assert Rm[0, 1, 0, 1] == -y**(-2)
    assert Rm[0, 1, 1, 0] == y**(-2)
    assert Rm[0, 1, 1, 1] == 0

    assert Rm[1, 0, 0, 0] == 0
    assert Rm[1, 0, 0, 1] == y**(-2)
    assert Rm[1, 0, 1, 0] == -y**(-2)
    assert Rm[1, 0, 1, 1] == 0

    assert Rm[1, 1, 0, 0] == 0
    assert Rm[1, 1, 0, 1] == 0
    assert Rm[1, 1, 1, 0] == 0
    assert Rm[1, 1, 1, 1] == 0

    Ric = metric_to_Ricci_components(g)
    assert Ric[0, 0] == -y**(-2)
    assert Ric[0, 1] == 0
    assert Ric[1, 0] == 0
    assert Ric[0, 0] == -y**(-2)

    assert Ric == ImmutableDenseNDimArray([-y**(-2), 0, 0, -y**(-2)], (2, 2))

    ## scalar curvature is -2
    #TODO - it would be nice to have index contraction built-in
    R = (Ric[0, 0] + Ric[1, 1])*y**2
    assert R == -2

    ## Gauss curvature is -1
    assert R/2 == -1
