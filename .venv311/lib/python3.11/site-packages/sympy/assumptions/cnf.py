"""
The classes used here are for the internal use of assumptions system
only and should not be used anywhere else as these do not possess the
signatures common to SymPy objects. For general use of logic constructs
please refer to sympy.logic classes And, Or, Not, etc.
"""
from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)


class Literal:
    """
    The smallest element of a CNF object.

    Parameters
    ==========

    lit : Boolean expression

    is_Not : bool

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import Literal
    >>> from sympy.abc import x
    >>> Literal(Q.even(x))
    Literal(Q.even(x), False)
    >>> Literal(~Q.even(x))
    Literal(Q.even(x), True)
    """

    def __new__(cls, lit, is_Not=False):
        if isinstance(lit, Not):
            lit = lit.args[0]
            is_Not = True
        elif isinstance(lit, (AND, OR, Literal)):
            return ~lit if is_Not else lit
        obj = super().__new__(cls)
        obj.lit = lit
        obj.is_Not = is_Not
        return obj

    @property
    def arg(self):
        return self.lit

    def rcall(self, expr):
        if callable(self.lit):
            lit = self.lit(expr)
        else:
            lit = self.lit.apply(expr)
        return type(self)(lit, self.is_Not)

    def __invert__(self):
        is_Not = not self.is_Not
        return Literal(self.lit, is_Not)

    def __str__(self):
        return '{}({}, {})'.format(type(self).__name__, self.lit, self.is_Not)

    __repr__ = __str__

    def __eq__(self, other):
        return self.arg == other.arg and self.is_Not == other.is_Not

    def __hash__(self):
        h = hash((type(self).__name__, self.arg, self.is_Not))
        return h


class OR:
    """
    A low-level implementation for Or
    """
    def __init__(self, *args):
        self._args = args

    @property
    def args(self):
        return sorted(self._args, key=str)

    def rcall(self, expr):
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    def __invert__(self):
        return AND(*[~arg for arg in self._args])

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(self, other):
        return self.args == other.args

    def __str__(self):
        s = '(' + ' | '.join([str(arg) for arg in self.args]) + ')'
        return s

    __repr__ = __str__


class AND:
    """
    A low-level implementation for And
    """
    def __init__(self, *args):
        self._args = args

    def __invert__(self):
        return OR(*[~arg for arg in self._args])

    @property
    def args(self):
        return sorted(self._args, key=str)

    def rcall(self, expr):
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(self, other):
        return self.args == other.args

    def __str__(self):
        s = '('+' & '.join([str(arg) for arg in self.args])+')'
        return s

    __repr__ = __str__


def to_NNF(expr, composite_map=None):
    """
    Generates the Negation Normal Form of any boolean expression in terms
    of AND, OR, and Literal objects.

    Examples
    ========

    >>> from sympy import Q, Eq
    >>> from sympy.assumptions.cnf import to_NNF
    >>> from sympy.abc import x, y
    >>> expr = Q.even(x) & ~Q.positive(x)
    >>> to_NNF(expr)
    (Literal(Q.even(x), False) & Literal(Q.positive(x), True))

    Supported boolean objects are converted to corresponding predicates.

    >>> to_NNF(Eq(x, y))
    Literal(Q.eq(x, y), False)

    If ``composite_map`` argument is given, ``to_NNF`` decomposes the
    specified predicate into a combination of primitive predicates.

    >>> cmap = {Q.nonpositive: Q.negative | Q.zero}
    >>> to_NNF(Q.nonpositive, cmap)
    (Literal(Q.negative, False) | Literal(Q.zero, False))
    >>> to_NNF(Q.nonpositive(x), cmap)
    (Literal(Q.negative(x), False) | Literal(Q.zero(x), False))
    """
    from sympy.assumptions.ask import Q

    if composite_map is None:
        composite_map = {}


    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
    if type(expr) in binrelpreds:
        pred = binrelpreds[type(expr)]
        expr = pred(*expr.args)

    if isinstance(expr, Not):
        arg = expr.args[0]
        tmp = to_NNF(arg, composite_map)  # Strategy: negate the NNF of expr
        return ~tmp

    if isinstance(expr, Or):
        return OR(*[to_NNF(x, composite_map) for x in Or.make_args(expr)])

    if isinstance(expr, And):
        return AND(*[to_NNF(x, composite_map) for x in And.make_args(expr)])

    if isinstance(expr, Nand):
        tmp = AND(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    if isinstance(expr, Nor):
        tmp = OR(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    if isinstance(expr, Xor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return AND(*cnfs)

    if isinstance(expr, Xnor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return ~AND(*cnfs)

    if isinstance(expr, Implies):
        L, R = to_NNF(expr.args[0], composite_map), to_NNF(expr.args[1], composite_map)
        return OR(~L, R)

    if isinstance(expr, Equivalent):
        cnfs = []
        for a, b in zip_longest(expr.args, expr.args[1:], fillvalue=expr.args[0]):
            a = to_NNF(a, composite_map)
            b = to_NNF(b, composite_map)
            cnfs.append(OR(~a, b))
        return AND(*cnfs)

    if isinstance(expr, ITE):
        L = to_NNF(expr.args[0], composite_map)
        M = to_NNF(expr.args[1], composite_map)
        R = to_NNF(expr.args[2], composite_map)
        return AND(OR(~L, M), OR(L, R))

    if isinstance(expr, AppliedPredicate):
        pred, args = expr.function, expr.arguments
        newpred = composite_map.get(pred, None)
        if newpred is not None:
            return to_NNF(newpred.rcall(*args), composite_map)

    if isinstance(expr, Predicate):
        newpred = composite_map.get(expr, None)
        if newpred is not None:
            return to_NNF(newpred, composite_map)

    return Literal(expr)


def distribute_AND_over_OR(expr):
    """
    Distributes AND over OR in the NNF expression.
    Returns the result( Conjunctive Normal Form of expression)
    as a CNF object.
    """
    if not isinstance(expr, (AND, OR)):
        tmp = set()
        tmp.add(frozenset((expr,)))
        return CNF(tmp)

    if isinstance(expr, OR):
        return CNF.all_or(*[distribute_AND_over_OR(arg)
                            for arg in expr._args])

    if isinstance(expr, AND):
        return CNF.all_and(*[distribute_AND_over_OR(arg)
                             for arg in expr._args])


class CNF:
    """
    Class to represent CNF of a Boolean expression.
    Consists of set of clauses, which themselves are stored as
    frozenset of Literal objects.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.abc import x
    >>> cnf = CNF.from_prop(Q.real(x) & ~Q.zero(x))
    >>> cnf.clauses
    {frozenset({Literal(Q.zero(x), True)}),
    frozenset({Literal(Q.negative(x), False),
    Literal(Q.positive(x), False), Literal(Q.zero(x), False)})}
    """
    def __init__(self, clauses=None):
        if not clauses:
            clauses = set()
        self.clauses = clauses

    def add(self, prop):
        clauses = CNF.to_CNF(prop).clauses
        self.add_clauses(clauses)

    def __str__(self):
        s = ' & '.join(
            ['(' + ' | '.join([str(lit) for lit in clause]) +')'
            for clause in self.clauses]
        )
        return s

    def extend(self, props):
        for p in props:
            self.add(p)
        return self

    def copy(self):
        return CNF(set(self.clauses))

    def add_clauses(self, clauses):
        self.clauses |= clauses

    @classmethod
    def from_prop(cls, prop):
        res = cls()
        res.add(prop)
        return res

    def __iand__(self, other):
        self.add_clauses(other.clauses)
        return self

    def all_predicates(self):
        predicates = set()
        for c in self.clauses:
            predicates |= {arg.lit for arg in c}
        return predicates

    def _or(self, cnf):
        clauses = set()
        for a, b in product(self.clauses, cnf.clauses):
            tmp = set(a)
            tmp.update(b)
            clauses.add(frozenset(tmp))
        return CNF(clauses)

    def _and(self, cnf):
        clauses = self.clauses.union(cnf.clauses)
        return CNF(clauses)

    def _not(self):
        clss = list(self.clauses)
        ll = {frozenset((~x,)) for x in clss[-1]}
        ll = CNF(ll)

        for rest in clss[:-1]:
            p = {frozenset((~x,)) for x in rest}
            ll = ll._or(CNF(p))
        return ll

    def rcall(self, expr):
        clause_list = []
        for clause in self.clauses:
            lits = [arg.rcall(expr) for arg in clause]
            clause_list.append(OR(*lits))
        expr = AND(*clause_list)
        return distribute_AND_over_OR(expr)

    @classmethod
    def all_or(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._or(rest)
        return b

    @classmethod
    def all_and(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._and(rest)
        return b

    @classmethod
    def to_CNF(cls, expr):
        from sympy.assumptions.facts import get_composite_predicates
        expr = to_NNF(expr, get_composite_predicates())
        expr = distribute_AND_over_OR(expr)
        return expr

    @classmethod
    def CNF_to_cnf(cls, cnf):
        """
        Converts CNF object to SymPy's boolean expression
        retaining the form of expression.
        """
        def remove_literal(arg):
            return Not(arg.lit) if arg.is_Not else arg.lit

        return And(*(Or(*(remove_literal(arg) for arg in clause)) for clause in cnf.clauses))


class EncodedCNF:
    """
    Class for encoding the CNF expression.
    """
    def __init__(self, data=None, encoding=None):
        if not data and not encoding:
            data = []
            encoding = {}
        self.data = data
        self.encoding = encoding
        self._symbols = list(encoding.keys())

    def from_cnf(self, cnf):
        self._symbols = list(cnf.all_predicates())
        n = len(self._symbols)
        self.encoding = dict(zip(self._symbols, range(1, n + 1)))
        self.data = [self.encode(clause) for clause in cnf.clauses]

    @property
    def symbols(self):
        return self._symbols

    @property
    def variables(self):
        return range(1, len(self._symbols) + 1)

    def copy(self):
        new_data = [set(clause) for clause in self.data]
        return EncodedCNF(new_data, dict(self.encoding))

    def add_prop(self, prop):
        cnf = CNF.from_prop(prop)
        self.add_from_cnf(cnf)

    def add_from_cnf(self, cnf):
        clauses = [self.encode(clause) for clause in cnf.clauses]
        self.data += clauses

    def encode_arg(self, arg):
        literal = arg.lit
        value = self.encoding.get(literal, None)
        if value is None:
            n = len(self._symbols)
            self._symbols.append(literal)
            value = self.encoding[literal] = n + 1
        if arg.is_Not:
            return -value
        else:
            return value

    def encode(self, clause):
        return {self.encode_arg(arg) if not arg.lit == S.false else 0 for arg in clause}
