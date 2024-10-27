from __future__ import annotations

from math import floor as mfloor

from sympy.polys.domains import ZZ, QQ
from sympy.polys.matrices.exceptions import DMRankError, DMShapeError, DMValueError, DMDomainError


def _ddm_lll(x, delta=QQ(3, 4), return_transform=False):
    if QQ(1, 4) >= delta or delta >= QQ(1, 1):
        raise DMValueError("delta must lie in range (0.25, 1)")
    if x.shape[0] > x.shape[1]:
        raise DMShapeError("input matrix must have shape (m, n) with m <= n")
    if x.domain != ZZ:
        raise DMDomainError("input matrix domain must be ZZ")
    m = x.shape[0]
    n = x.shape[1]
    k = 1
    y = x.copy()
    y_star = x.zeros((m, n), QQ)
    mu = x.zeros((m, m), QQ)
    g_star = [QQ(0, 1) for _ in range(m)]
    half = QQ(1, 2)
    T = x.eye(m, ZZ) if return_transform else None
    linear_dependent_error = "input matrix contains linearly dependent rows"

    def closest_integer(x):
        return ZZ(mfloor(x + half))

    def lovasz_condition(k: int) -> bool:
        return g_star[k] >= ((delta - mu[k][k - 1] ** 2) * g_star[k - 1])

    def mu_small(k: int, j: int) -> bool:
        return abs(mu[k][j]) <= half

    def dot_rows(x, y, rows: tuple[int, int]):
        return sum(x[rows[0]][z] * y[rows[1]][z] for z in range(x.shape[1]))

    def reduce_row(T, mu, y, rows: tuple[int, int]):
        r = closest_integer(mu[rows[0]][rows[1]])
        y[rows[0]] = [y[rows[0]][z] - r * y[rows[1]][z] for z in range(n)]
        mu[rows[0]][:rows[1]] = [mu[rows[0]][z] - r * mu[rows[1]][z] for z in range(rows[1])]
        mu[rows[0]][rows[1]] -= r
        if return_transform:
            T[rows[0]] = [T[rows[0]][z] - r * T[rows[1]][z] for z in range(m)]

    for i in range(m):
        y_star[i] = [QQ.convert_from(z, ZZ) for z in y[i]]
        for j in range(i):
            row_dot = dot_rows(y, y_star, (i, j))
            try:
                mu[i][j] = row_dot / g_star[j]
            except ZeroDivisionError:
                raise DMRankError(linear_dependent_error)
            y_star[i] = [y_star[i][z] - mu[i][j] * y_star[j][z] for z in range(n)]
        g_star[i] = dot_rows(y_star, y_star, (i, i))
    while k < m:
        if not mu_small(k, k - 1):
            reduce_row(T, mu, y, (k, k - 1))
        if lovasz_condition(k):
            for l in range(k - 2, -1, -1):
                if not mu_small(k, l):
                    reduce_row(T, mu, y, (k, l))
            k += 1
        else:
            nu = mu[k][k - 1]
            alpha = g_star[k] + nu ** 2 * g_star[k - 1]
            try:
                beta = g_star[k - 1] / alpha
            except ZeroDivisionError:
                raise DMRankError(linear_dependent_error)
            mu[k][k - 1] = nu * beta
            g_star[k] = g_star[k] * beta
            g_star[k - 1] = alpha
            y[k], y[k - 1] = y[k - 1], y[k]
            mu[k][:k - 1], mu[k - 1][:k - 1] = mu[k - 1][:k - 1], mu[k][:k - 1]
            for i in range(k + 1, m):
                xi = mu[i][k]
                mu[i][k] = mu[i][k - 1] - nu * xi
                mu[i][k - 1] = mu[k][k - 1] * mu[i][k] + xi
            if return_transform:
                T[k], T[k - 1] = T[k - 1], T[k]
            k = max(k - 1, 1)
    assert all(lovasz_condition(i) for i in range(1, m))
    assert all(mu_small(i, j) for i in range(m) for j in range(i))
    return y, T


def ddm_lll(x, delta=QQ(3, 4)):
    return _ddm_lll(x, delta=delta, return_transform=False)[0]


def ddm_lll_transform(x, delta=QQ(3, 4)):
    return _ddm_lll(x, delta=delta, return_transform=True)
