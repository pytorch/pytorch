from sympy.core.expr import Expr
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import Poly, parallel_poly_from_expr
from sympy.polys.domains import QQ

from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.domainscalar import DomainScalar


class MutablePolyDenseMatrix:
    """
    A mutable matrix of objects from poly module or to operate with them.

    Examples
    ========

    >>> from sympy.polys.polymatrix import PolyMatrix
    >>> from sympy import Symbol, Poly
    >>> x = Symbol('x')
    >>> pm1 = PolyMatrix([[Poly(x**2, x), Poly(-x, x)], [Poly(x**3, x), Poly(-1 + x, x)]])
    >>> v1 = PolyMatrix([[1, 0], [-1, 0]], x)
    >>> pm1*v1
    PolyMatrix([
    [    x**2 + x, 0],
    [x**3 - x + 1, 0]], ring=QQ[x])

    >>> pm1.ring
    ZZ[x]

    >>> v1*pm1
    PolyMatrix([
    [ x**2, -x],
    [-x**2,  x]], ring=QQ[x])

    >>> pm2 = PolyMatrix([[Poly(x**2, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(1, x, domain='QQ'), \
            Poly(x**3, x, domain='QQ'), Poly(0, x, domain='QQ'), Poly(-x**3, x, domain='QQ')]])
    >>> v2 = PolyMatrix([1, 0, 0, 0, 0, 0], x)
    >>> v2.ring
    QQ[x]
    >>> pm2*v2
    PolyMatrix([[x**2]], ring=QQ[x])

    """

    def __new__(cls, *args, ring=None):

        if not args:
            # PolyMatrix(ring=QQ[x])
            if ring is None:
                raise TypeError("The ring needs to be specified for an empty PolyMatrix")
            rows, cols, items, gens = 0, 0, [], ()
        elif isinstance(args[0], list):
            elements, gens = args[0], args[1:]
            if not elements:
                # PolyMatrix([])
                rows, cols, items = 0, 0, []
            elif isinstance(elements[0], (list, tuple)):
                # PolyMatrix([[1, 2]], x)
                rows, cols = len(elements), len(elements[0])
                items = [e for row in elements for e in row]
            else:
                # PolyMatrix([1, 2], x)
                rows, cols = len(elements), 1
                items = elements
        elif [type(a) for a in args[:3]] == [int, int, list]:
            # PolyMatrix(2, 2, [1, 2, 3, 4], x)
            rows, cols, items, gens = args[0], args[1], args[2], args[3:]
        elif [type(a) for a in args[:3]] == [int, int, type(lambda: 0)]:
            # PolyMatrix(2, 2, lambda i, j: i+j, x)
            rows, cols, func, gens = args[0], args[1], args[2], args[3:]
            items = [func(i, j) for i in range(rows) for j in range(cols)]
        else:
            raise TypeError("Invalid arguments")

        # PolyMatrix([[1]], x, y) vs PolyMatrix([[1]], (x, y))
        if len(gens) == 1 and isinstance(gens[0], tuple):
            gens = gens[0]
            # gens is now a tuple (x, y)

        return cls.from_list(rows, cols, items, gens, ring)

    @classmethod
    def from_list(cls, rows, cols, items, gens, ring):

        # items can be Expr, Poly, or a mix of Expr and Poly
        items = [_sympify(item) for item in items]
        if items and all(isinstance(item, Poly) for item in items):
            polys = True
        else:
            polys = False

        # Identify the ring for the polys
        if ring is not None:
            # Parse a domain string like 'QQ[x]'
            if isinstance(ring, str):
                ring = Poly(0, Dummy(), domain=ring).domain
        elif polys:
            p = items[0]
            for p2 in items[1:]:
                p, _ = p.unify(p2)
            ring = p.domain[p.gens]
        else:
            items, info = parallel_poly_from_expr(items, gens, field=True)
            ring = info['domain'][info['gens']]
            polys = True

        # Efficiently convert when all elements are Poly
        if polys:
            p_ring = Poly(0, ring.symbols, domain=ring.domain)
            to_ring = ring.ring.from_list
            convert_poly = lambda p: to_ring(p.unify(p_ring)[0].rep.to_list())
            elements = [convert_poly(p) for p in items]
        else:
            convert_expr = ring.from_sympy
            elements = [convert_expr(e.as_expr()) for e in items]

        # Convert to domain elements and construct DomainMatrix
        elements_lol = [[elements[i*cols + j] for j in range(cols)] for i in range(rows)]
        dm = DomainMatrix(elements_lol, (rows, cols), ring)
        return cls.from_dm(dm)

    @classmethod
    def from_dm(cls, dm):
        obj = super().__new__(cls)
        dm = dm.to_sparse()
        R = dm.domain
        obj._dm = dm
        obj.ring = R
        obj.domain = R.domain
        obj.gens = R.symbols
        return obj

    def to_Matrix(self):
        return self._dm.to_Matrix()

    @classmethod
    def from_Matrix(cls, other, *gens, ring=None):
        return cls(*other.shape, other.flat(), *gens, ring=ring)

    def set_gens(self, gens):
        return self.from_Matrix(self.to_Matrix(), gens)

    def __repr__(self):
        if self.rows * self.cols:
            return 'Poly' + repr(self.to_Matrix())[:-1] + f', ring={self.ring})'
        else:
            return f'PolyMatrix({self.rows}, {self.cols}, [], ring={self.ring})'

    @property
    def shape(self):
        return self._dm.shape

    @property
    def rows(self):
        return self.shape[0]

    @property
    def cols(self):
        return self.shape[1]

    def __len__(self):
        return self.rows * self.cols

    def __getitem__(self, key):

        def to_poly(v):
            ground = self._dm.domain.domain
            gens = self._dm.domain.symbols
            return Poly(v.to_dict(), gens, domain=ground)

        dm = self._dm

        if isinstance(key, slice):
            items = dm.flat()[key]
            return [to_poly(item) for item in items]
        elif isinstance(key, int):
            i, j = divmod(key, self.cols)
            e = dm[i,j]
            return to_poly(e.element)

        i, j = key
        if isinstance(i, int) and isinstance(j, int):
            return to_poly(dm[i, j].element)
        else:
            return self.from_dm(dm[i, j])

    def __eq__(self, other):
        if not isinstance(self, type(other)):
            return NotImplemented
        return self._dm == other._dm

    def __add__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm + other._dm)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm - other._dm)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return self.from_dm(self._dm * other._dm)
        elif isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            Kx = self.ring
            try:
                other_ds = DomainScalar(Kx.from_sympy(other), Kx)
            except (CoercionFailed, ValueError):
                other_ds = DomainScalar.from_sympy(other)
            return self.from_dm(self._dm * other_ds)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, int):
            other = _sympify(other)
        if isinstance(other, Expr):
            other_ds = DomainScalar.from_sympy(other)
            return self.from_dm(other_ds * self._dm)
        return NotImplemented

    def __truediv__(self, other):

        if isinstance(other, Poly):
            other = other.as_expr()
        elif isinstance(other, int):
            other = _sympify(other)
        if not isinstance(other, Expr):
            return NotImplemented

        other = self.domain.from_sympy(other)
        inverse = self.ring.convert_from(1/other, self.domain)
        inverse = DomainScalar(inverse, self.ring)
        dm = self._dm * inverse
        return self.from_dm(dm)

    def __neg__(self):
        return self.from_dm(-self._dm)

    def transpose(self):
        return self.from_dm(self._dm.transpose())

    def row_join(self, other):
        dm = DomainMatrix.hstack(self._dm, other._dm)
        return self.from_dm(dm)

    def col_join(self, other):
        dm = DomainMatrix.vstack(self._dm, other._dm)
        return self.from_dm(dm)

    def applyfunc(self, func):
        M = self.to_Matrix().applyfunc(func)
        return self.from_Matrix(M, self.gens)

    @classmethod
    def eye(cls, n, gens):
        return cls.from_dm(DomainMatrix.eye(n, QQ[gens]))

    @classmethod
    def zeros(cls, m, n, gens):
        return cls.from_dm(DomainMatrix.zeros((m, n), QQ[gens]))

    def rref(self, simplify='ignore', normalize_last='ignore'):
        # If this is K[x] then computes RREF in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix rref is only for ground field elements")
        dm = self._dm
        dm_ground = dm.convert_to(dm.domain.domain)
        dm_rref, pivots = dm_ground.rref()
        dm_rref = dm_rref.convert_to(dm.domain)
        return self.from_dm(dm_rref), pivots

    def nullspace(self):
        # If this is K[x] then computes nullspace in ground field K.
        if not (self.domain.is_Field and all(p.is_ground for p in self)):
            raise ValueError("PolyMatrix nullspace is only for ground field elements")
        dm = self._dm
        K, Kx = self.domain, self.ring
        dm_null_rows = dm.convert_to(K).nullspace(divide_last=True).convert_to(Kx)
        dm_null = dm_null_rows.transpose()
        dm_basis = [dm_null[:,i] for i in range(dm_null.shape[1])]
        return [self.from_dm(dmvec) for dmvec in dm_basis]

    def rank(self):
        return self.cols - len(self.nullspace())

MutablePolyMatrix = PolyMatrix = MutablePolyDenseMatrix
