"""Low-level linear systems solver. """


from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components

from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import ZZ, QQ

from sympy.polys.domains import EX
from sympy.polys.rings import sring
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix


class PolyNonlinearError(Exception):
    """Raised by solve_lin_sys for nonlinear equations"""
    pass


class RawMatrix(MutableDenseMatrix):
    """
    .. deprecated:: 1.9

       This class fundamentally is broken by design. Use ``DomainMatrix`` if
       you want a matrix over the polys domains or ``Matrix`` for a matrix
       with ``Expr`` elements. The ``RawMatrix`` class will be removed/broken
       in future in order to reestablish the invariant that the elements of a
       Matrix should be of type ``Expr``.

    """
    _sympify = staticmethod(lambda x, *args, **kwargs: x)

    def __init__(self, *args, **kwargs):
        sympy_deprecation_warning(
            """
            The RawMatrix class is deprecated. Use either DomainMatrix or
            Matrix instead.
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-rawmatrix",
        )

        domain = ZZ
        for i in range(self.rows):
            for j in range(self.cols):
                val = self[i,j]
                if getattr(val, 'is_Poly', False):
                    K = val.domain[val.gens]
                    val_sympy = val.as_expr()
                elif hasattr(val, 'parent'):
                    K = val.parent()
                    val_sympy = K.to_sympy(val)
                elif isinstance(val, (int, Integer)):
                    K = ZZ
                    val_sympy = sympify(val)
                elif isinstance(val, Rational):
                    K = QQ
                    val_sympy = val
                else:
                    for K in ZZ, QQ:
                        if K.of_type(val):
                            val_sympy = K.to_sympy(val)
                            break
                    else:
                        raise TypeError
                domain = domain.unify(K)
                self[i,j] = val_sympy
        self.ring = domain


def eqs_to_matrix(eqs_coeffs, eqs_rhs, gens, domain):
    """Get matrix from linear equations in dict format.

    Explanation
    ===========

    Get the matrix representation of a system of linear equations represented
    as dicts with low-level DomainElement coefficients. This is an
    *internal* function that is used by solve_lin_sys.

    Parameters
    ==========

    eqs_coeffs: list[dict[Symbol, DomainElement]]
        The left hand sides of the equations as dicts mapping from symbols to
        coefficients where the coefficients are instances of
        DomainElement.
    eqs_rhs: list[DomainElements]
        The right hand sides of the equations as instances of
        DomainElement.
    gens: list[Symbol]
        The unknowns in the system of equations.
    domain: Domain
        The domain for coefficients of both lhs and rhs.

    Returns
    =======

    The augmented matrix representation of the system as a DomainMatrix.

    Examples
    ========

    >>> from sympy import symbols, ZZ
    >>> from sympy.polys.solvers import eqs_to_matrix
    >>> x, y = symbols('x, y')
    >>> eqs_coeff = [{x:ZZ(1), y:ZZ(1)}, {x:ZZ(1), y:ZZ(-1)}]
    >>> eqs_rhs = [ZZ(0), ZZ(-1)]
    >>> eqs_to_matrix(eqs_coeff, eqs_rhs, [x, y], ZZ)
    DomainMatrix([[1, 1, 0], [1, -1, 1]], (2, 3), ZZ)

    See also
    ========

    solve_lin_sys: Uses :func:`~eqs_to_matrix` internally
    """
    sym2index = {x: n for n, x in enumerate(gens)}
    nrows = len(eqs_coeffs)
    ncols = len(gens) + 1
    rows = [[domain.zero] * ncols for _ in range(nrows)]
    for row, eq_coeff, eq_rhs in zip(rows, eqs_coeffs, eqs_rhs):
        for sym, coeff in eq_coeff.items():
            row[sym2index[sym]] = domain.convert(coeff)
        row[-1] = -domain.convert(eq_rhs)

    return DomainMatrix(rows, (nrows, ncols), domain)


def sympy_eqs_to_ring(eqs, symbols):
    """Convert a system of equations from Expr to a PolyRing

    Explanation
    ===========

    High-level functions like ``solve`` expect Expr as inputs but can use
    ``solve_lin_sys`` internally. This function converts equations from
    ``Expr`` to the low-level poly types used by the ``solve_lin_sys``
    function.

    Parameters
    ==========

    eqs: List of Expr
        A list of equations as Expr instances
    symbols: List of Symbol
        A list of the symbols that are the unknowns in the system of
        equations.

    Returns
    =======

    Tuple[List[PolyElement], Ring]: The equations as PolyElement instances
    and the ring of polynomials within which each equation is represented.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.polys.solvers import sympy_eqs_to_ring
    >>> a, x, y = symbols('a, x, y')
    >>> eqs = [x-y, x+a*y]
    >>> eqs_ring, ring = sympy_eqs_to_ring(eqs, [x, y])
    >>> eqs_ring
    [x - y, x + a*y]
    >>> type(eqs_ring[0])
    <class 'sympy.polys.rings.PolyElement'>
    >>> ring
    ZZ(a)[x,y]

    With the equations in this form they can be passed to ``solve_lin_sys``:

    >>> from sympy.polys.solvers import solve_lin_sys
    >>> solve_lin_sys(eqs_ring, ring)
    {y: 0, x: 0}
    """
    try:
        K, eqs_K = sring(eqs, symbols, field=True, extension=True)
    except NotInvertible:
        # https://github.com/sympy/sympy/issues/18874
        K, eqs_K = sring(eqs, symbols, domain=EX)
    return eqs_K, K.to_domain()


def solve_lin_sys(eqs, ring, _raw=True):
    """Solve a system of linear equations from a PolynomialRing

    Explanation
    ===========

    Solves a system of linear equations given as PolyElement instances of a
    PolynomialRing. The basic arithmetic is carried out using instance of
    DomainElement which is more efficient than :class:`~sympy.core.expr.Expr`
    for the most common inputs.

    While this is a public function it is intended primarily for internal use
    so its interface is not necessarily convenient. Users are suggested to use
    the :func:`sympy.solvers.solveset.linsolve` function (which uses this
    function internally) instead.

    Parameters
    ==========

    eqs: list[PolyElement]
        The linear equations to be solved as elements of a
        PolynomialRing (assumed equal to zero).
    ring: PolynomialRing
        The polynomial ring from which eqs are drawn. The generators of this
        ring are the unknowns to be solved for and the domain of the ring is
        the domain of the coefficients of the system of equations.
    _raw: bool
        If *_raw* is False, the keys and values in the returned dictionary
        will be of type Expr (and the unit of the field will be removed from
        the keys) otherwise the low-level polys types will be returned, e.g.
        PolyElement: PythonRational.

    Returns
    =======

    ``None`` if the system has no solution.

    dict[Symbol, Expr] if _raw=False

    dict[Symbol, DomainElement] if _raw=True.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.polys.solvers import solve_lin_sys, sympy_eqs_to_ring
    >>> x, y = symbols('x, y')
    >>> eqs = [x - y, x + y - 2]
    >>> eqs_ring, ring = sympy_eqs_to_ring(eqs, [x, y])
    >>> solve_lin_sys(eqs_ring, ring)
    {y: 1, x: 1}

    Passing ``_raw=False`` returns the same result except that the keys are
    ``Expr`` rather than low-level poly types.

    >>> solve_lin_sys(eqs_ring, ring, _raw=False)
    {x: 1, y: 1}

    See also
    ========

    sympy_eqs_to_ring: prepares the inputs to ``solve_lin_sys``.
    linsolve: ``linsolve`` uses ``solve_lin_sys`` internally.
    sympy.solvers.solvers.solve: ``solve`` uses ``solve_lin_sys`` internally.
    """
    as_expr = not _raw

    assert ring.domain.is_Field

    eqs_dict = [dict(eq) for eq in eqs]

    one_monom = ring.one.monoms()[0]
    zero = ring.domain.zero

    eqs_rhs = []
    eqs_coeffs = []
    for eq_dict in eqs_dict:
        eq_rhs = eq_dict.pop(one_monom, zero)
        eq_coeffs = {}
        for monom, coeff in eq_dict.items():
            if sum(monom) != 1:
                msg = "Nonlinear term encountered in solve_lin_sys"
                raise PolyNonlinearError(msg)
            eq_coeffs[ring.gens[monom.index(1)]] = coeff
        if not eq_coeffs:
            if not eq_rhs:
                continue
            else:
                return None
        eqs_rhs.append(eq_rhs)
        eqs_coeffs.append(eq_coeffs)

    result = _solve_lin_sys(eqs_coeffs, eqs_rhs, ring)

    if result is not None and as_expr:

        def to_sympy(x):
            as_expr = getattr(x, 'as_expr', None)
            if as_expr:
                return as_expr()
            else:
                return ring.domain.to_sympy(x)

        tresult = {to_sympy(sym): to_sympy(val) for sym, val in result.items()}

        # Remove 1.0x
        result = {}
        for k, v in tresult.items():
            if k.is_Mul:
                c, s = k.as_coeff_Mul()
                result[s] = v/c
            else:
                result[k] = v

    return result


def _solve_lin_sys(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. The role of this function is to split
    the system into connected components and pass those to
    :func:`_solve_lin_sys_component`.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """
    V = ring.gens
    E = []
    for eq_coeffs in eqs_coeffs:
        syms = list(eq_coeffs)
        E.extend(zip(syms[:-1], syms[1:]))
    G = V, E

    components = connected_components(G)

    sym2comp = {}
    for n, component in enumerate(components):
        for sym in component:
            sym2comp[sym] = n

    subsystems = [([], []) for _ in range(len(components))]
    for eq_coeff, eq_rhs in zip(eqs_coeffs, eqs_rhs):
        sym = next(iter(eq_coeff), None)
        sub_coeff, sub_rhs = subsystems[sym2comp[sym]]
        sub_coeff.append(eq_coeff)
        sub_rhs.append(eq_rhs)

    sol = {}
    for subsystem in subsystems:
        subsol = _solve_lin_sys_component(subsystem[0], subsystem[1], ring)
        if subsol is None:
            return None
        sol.update(subsol)

    return sol


def _solve_lin_sys_component(eqs_coeffs, eqs_rhs, ring):
    """Solve a linear system from dict of PolynomialRing coefficients

    Explanation
    ===========

    This is an **internal** function used by :func:`solve_lin_sys` after the
    equations have been preprocessed. After :func:`_solve_lin_sys` splits the
    system into connected components this function is called for each
    component. The system of equations is solved using Gauss-Jordan
    elimination with division followed by back-substitution.

    Examples
    ========

    Setup a system for $x-y=0$ and $x+y=2$ and solve:

    >>> from sympy import symbols, sring
    >>> from sympy.polys.solvers import _solve_lin_sys_component
    >>> x, y = symbols('x, y')
    >>> R, (xr, yr) = sring([x, y], [x, y])
    >>> eqs = [{xr:R.one, yr:-R.one}, {xr:R.one, yr:R.one}]
    >>> eqs_rhs = [R.zero, -2*R.one]
    >>> _solve_lin_sys_component(eqs, eqs_rhs, R)
    {y: 1, x: 1}

    See also
    ========

    solve_lin_sys: This function is used internally by :func:`solve_lin_sys`.
    """

    # transform from equations to matrix form
    matrix = eqs_to_matrix(eqs_coeffs, eqs_rhs, ring.gens, ring.domain)

    # convert to a field for rref
    if not matrix.domain.is_Field:
        matrix = matrix.to_field()

    # solve by row-reduction
    echelon, pivots = matrix.rref()

    # construct the returnable form of the solutions
    keys = ring.gens

    if pivots and pivots[-1] == len(keys):
        return None

    if len(pivots) == len(keys):
        sol = []
        for s in [row[-1] for row in echelon.rep.to_ddm()]:
            a = s
            sol.append(a)
        sols = dict(zip(keys, sol))
    else:
        sols = {}
        g = ring.gens
        # Extract ground domain coefficients and convert to the ring:
        if hasattr(ring, 'ring'):
            convert = ring.ring.ground_new
        else:
            convert = ring.ground_new
        echelon = echelon.rep.to_ddm()
        vals_set = {v for row in echelon for v in row}
        vals_map = {v: convert(v) for v in vals_set}
        echelon = [[vals_map[eij] for eij in ei] for ei in echelon]
        for i, p in enumerate(pivots):
            v = echelon[i][-1] - sum(echelon[i][j]*g[j] for j in range(p+1, len(g)) if echelon[i][j])
            sols[keys[p]] = v

    return sols
