#TODO:
# -Implement Clebsch-Gordan symmetries
# -Improve simplification method
# -Implement new simplifications
"""Clebsch-Gordon Coefficients."""

from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict

from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE

__all__ = [
    'CG',
    'Wigner3j',
    'Wigner6j',
    'Wigner9j',
    'cg_simp'
]

#-----------------------------------------------------------------------------
# CG Coefficients
#-----------------------------------------------------------------------------


class Wigner3j(Expr):
    """Class for the Wigner-3j symbols.

    Explanation
    ===========

    Wigner 3j-symbols are coefficients determined by the coupling of
    two angular momenta. When created, they are expressed as symbolic
    quantities that, for numerical parameters, can be evaluated using the
    ``.doit()`` method [1]_.

    Parameters
    ==========

    j1, m1, j2, m2, j3, m3 : Number, Symbol
        Terms determining the angular momentum of coupled angular momentum
        systems.

    Examples
    ========

    Declare a Wigner-3j coefficient and calculate its value

        >>> from sympy.physics.quantum.cg import Wigner3j
        >>> w3j = Wigner3j(6,0,4,0,2,0)
        >>> w3j
        Wigner3j(6, 0, 4, 0, 2, 0)
        >>> w3j.doit()
        sqrt(715)/143

    See Also
    ========

    CG: Clebsch-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """

    is_commutative = True

    def __new__(cls, j1, m1, j2, m2, j3, m3):
        args = map(sympify, (j1, m1, j2, m2, j3, m3))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def m1(self):
        return self.args[1]

    @property
    def j2(self):
        return self.args[2]

    @property
    def m2(self):
        return self.args[3]

    @property
    def j3(self):
        return self.args[4]

    @property
    def m3(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.m1)),
            (printer._print(self.j2), printer._print(self.m2)),
            (printer._print(self.j3), printer._print(self.m3)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(2))
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens())
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j3,
                    self.m1, self.m2, self.m3))
        return r'\left(\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right)' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_3j(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)


class CG(Wigner3j):
    r"""Class for Clebsch-Gordan coefficient.

    Explanation
    ===========

    Clebsch-Gordan coefficients describe the angular momentum coupling between
    two systems. The coefficients give the expansion of a coupled total angular
    momentum state and an uncoupled tensor product state. The Clebsch-Gordan
    coefficients are defined as [1]_:

    .. math ::
        C^{j_3,m_3}_{j_1,m_1,j_2,m_2} = \left\langle j_1,m_1;j_2,m_2 | j_3,m_3\right\rangle

    Parameters
    ==========

    j1, m1, j2, m2 : Number, Symbol
        Angular momenta of states 1 and 2.

    j3, m3: Number, Symbol
        Total angular momentum of the coupled system.

    Examples
    ========

    Define a Clebsch-Gordan coefficient and evaluate its value

        >>> from sympy.physics.quantum.cg import CG
        >>> from sympy import S
        >>> cg = CG(S(3)/2, S(3)/2, S(1)/2, -S(1)/2, 1, 1)
        >>> cg
        CG(3/2, 3/2, 1/2, -1/2, 1, 1)
        >>> cg.doit()
        sqrt(3)/2
        >>> CG(j1=S(1)/2, m1=-S(1)/2, j2=S(1)/2, m2=+S(1)/2, j3=1, m3=0).doit()
        sqrt(2)/2


    Compare [2]_.

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    .. [2] `Clebsch-Gordan Coefficients, Spherical Harmonics, and d Functions
        <https://pdg.lbl.gov/2020/reviews/rpp2020-rev-clebsch-gordan-coefs.pdf>`_
        in P.A. Zyla *et al.* (Particle Data Group), Prog. Theor. Exp. Phys.
        2020, 083C01 (2020).
    """
    precedence = PRECEDENCE["Pow"] - 1

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return clebsch_gordan(self.j1, self.j2, self.j3, self.m1, self.m2, self.m3)

    def _pretty(self, printer, *args):
        bot = printer._print_seq(
            (self.j1, self.m1, self.j2, self.m2), delimiter=',')
        top = printer._print_seq((self.j3, self.m3), delimiter=',')

        pad = max(top.width(), bot.width())
        bot = prettyForm(*bot.left(' '))
        top = prettyForm(*top.left(' '))

        if not pad == bot.width():
            bot = prettyForm(*bot.right(' '*(pad - bot.width())))
        if not pad == top.width():
            top = prettyForm(*top.right(' '*(pad - top.width())))
        s = stringPict('C' + ' '*pad)
        s = prettyForm(*s.below(bot))
        s = prettyForm(*s.above(top))
        return s

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j3, self.m3, self.j1,
                    self.m1, self.j2, self.m2))
        return r'C^{%s,%s}_{%s,%s,%s,%s}' % tuple(label)


class Wigner6j(Expr):
    """Class for the Wigner-6j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j, j23):
        args = map(sympify, (j1, j2, j12, j3, j, j23))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j(self):
        return self.args[4]

    @property
    def j23(self):
        return self.args[5]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = ((printer._print(self.j1), printer._print(self.j3)),
            (printer._print(self.j2), printer._print(self.j)),
            (printer._print(self.j12), printer._print(self.j23)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(2))
        D = None
        for i in range(2):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12,
                    self.j3, self.j, self.j23))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_6j(self.j1, self.j2, self.j12, self.j3, self.j, self.j23)


class Wigner9j(Expr):
    """Class for the Wigner-9j symbols

    See Also
    ========

    Wigner3j: Wigner-3j symbols

    """
    def __new__(cls, j1, j2, j12, j3, j4, j34, j13, j24, j):
        args = map(sympify, (j1, j2, j12, j3, j4, j34, j13, j24, j))
        return Expr.__new__(cls, *args)

    @property
    def j1(self):
        return self.args[0]

    @property
    def j2(self):
        return self.args[1]

    @property
    def j12(self):
        return self.args[2]

    @property
    def j3(self):
        return self.args[3]

    @property
    def j4(self):
        return self.args[4]

    @property
    def j34(self):
        return self.args[5]

    @property
    def j13(self):
        return self.args[6]

    @property
    def j24(self):
        return self.args[7]

    @property
    def j(self):
        return self.args[8]

    @property
    def is_symbolic(self):
        return not all(arg.is_number for arg in self.args)

    # This is modified from the _print_Matrix method
    def _pretty(self, printer, *args):
        m = (
            (printer._print(
                self.j1), printer._print(self.j3), printer._print(self.j13)),
            (printer._print(
                self.j2), printer._print(self.j4), printer._print(self.j24)),
            (printer._print(self.j12), printer._print(self.j34), printer._print(self.j)))
        hsep = 2
        vsep = 1
        maxw = [-1]*3
        for j in range(3):
            maxw[j] = max(m[j][i].width() for i in range(3))
        D = None
        for i in range(3):
            D_row = None
            for j in range(3):
                s = m[j][i]
                wdelta = maxw[j] - s.width()
                wleft = wdelta //2
                wright = wdelta - wleft

                s = prettyForm(*s.right(' '*wright))
                s = prettyForm(*s.left(' '*wleft))

                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(' '*hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(' '))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens(left='{', right='}'))
        return D

    def _latex(self, printer, *args):
        label = map(printer._print, (self.j1, self.j2, self.j12, self.j3,
                self.j4, self.j34, self.j13, self.j24, self.j))
        return r'\left\{\begin{array}{ccc} %s & %s & %s \\ %s & %s & %s \\ %s & %s & %s \end{array}\right\}' % \
            tuple(label)

    def doit(self, **hints):
        if self.is_symbolic:
            raise ValueError("Coefficients must be numerical")
        return wigner_9j(self.j1, self.j2, self.j12, self.j3, self.j4, self.j34, self.j13, self.j24, self.j)


def cg_simp(e):
    """Simplify and combine CG coefficients.

    Explanation
    ===========

    This function uses various symmetry and properties of sums and
    products of Clebsch-Gordan coefficients to simplify statements
    involving these terms [1]_.

    Examples
    ========

    Simplify the sum over CG(a,alpha,0,0,a,alpha) for all alpha to
    2*a+1

        >>> from sympy.physics.quantum.cg import CG, cg_simp
        >>> a = CG(1,1,0,0,1,1)
        >>> b = CG(1,0,0,0,1,0)
        >>> c = CG(1,-1,0,0,1,-1)
        >>> cg_simp(a+b+c)
        3

    See Also
    ========

    CG: Clebsh-Gordan coefficients

    References
    ==========

    .. [1] Varshalovich, D A, Quantum Theory of Angular Momentum. 1988.
    """
    if isinstance(e, Add):
        return _cg_simp_add(e)
    elif isinstance(e, Sum):
        return _cg_simp_sum(e)
    elif isinstance(e, Mul):
        return Mul(*[cg_simp(arg) for arg in e.args])
    elif isinstance(e, Pow):
        return Pow(cg_simp(e.base), e.exp)
    else:
        return e


def _cg_simp_add(e):
    #TODO: Improve simplification method
    """Takes a sum of terms involving Clebsch-Gordan coefficients and
    simplifies the terms.

    Explanation
    ===========

    First, we create two lists, cg_part, which is all the terms involving CG
    coefficients, and other_part, which is all other terms. The cg_part list
    is then passed to the simplification methods, which return the new cg_part
    and any additional terms that are added to other_part
    """
    cg_part = []
    other_part = []

    e = expand(e)
    for arg in e.args:
        if arg.has(CG):
            if isinstance(arg, Sum):
                other_part.append(_cg_simp_sum(arg))
            elif isinstance(arg, Mul):
                terms = 1
                for term in arg.args:
                    if isinstance(term, Sum):
                        terms *= _cg_simp_sum(term)
                    else:
                        terms *= term
                if terms.has(CG):
                    cg_part.append(terms)
                else:
                    other_part.append(terms)
            else:
                cg_part.append(arg)
        else:
            other_part.append(arg)

    cg_part, other = _check_varsh_871_1(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_871_2(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_872_9(cg_part)
    other_part.append(other)
    return Add(*cg_part) + Add(*other_part)


def _check_varsh_871_1(term_list):
    # Sum( CG(a,alpha,b,0,a,alpha), (alpha, -a, a)) == KroneckerDelta(b,0)
    a, alpha, b, lt = map(Wild, ('a', 'alpha', 'b', 'lt'))
    expr = lt*CG(a, alpha, b, 0, a, alpha)
    simp = (2*a + 1)*KroneckerDelta(b, 0)
    sign = lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, lt), (a, b), build_expr, index_expr)


def _check_varsh_871_2(term_list):
    # Sum((-1)**(a-alpha)*CG(a,alpha,a,-alpha,c,0),(alpha,-a,a))
    a, alpha, c, lt = map(Wild, ('a', 'alpha', 'c', 'lt'))
    expr = lt*CG(a, alpha, a, -alpha, c, 0)
    simp = sqrt(2*a + 1)*KroneckerDelta(c, 0)
    sign = (-1)**(a - alpha)*lt/abs(lt)
    build_expr = 2*a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, c, lt), (a, c), build_expr, index_expr)


def _check_varsh_872_9(term_list):
    # Sum( CG(a,alpha,b,beta,c,gamma)*CG(a,alpha',b,beta',c,gamma), (gamma, -c, c), (c, abs(a-b), a+b))
    a, alpha, alphap, b, beta, betap, c, gamma, lt = map(Wild, (
        'a', 'alpha', 'alphap', 'b', 'beta', 'betap', 'c', 'gamma', 'lt'))
    # Case alpha==alphap, beta==betap

    # For numerical alpha,beta
    expr = lt*CG(a, alpha, b, beta, c, gamma)**2
    simp = S.One
    sign = lt/abs(lt)
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other1 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # For symbolic alpha,beta
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    term_list, other2 = _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, beta, c, gamma, lt), (a, alpha, b, beta), build_expr, index_expr)

    # Case alpha!=alphap or beta!=betap
    # Note: this only works with leading term of 1, pattern matching is unable to match when there is a Wild leading term
    # For numerical alpha,alphap,beta,betap
    expr = CG(a, alpha, b, beta, c, gamma)*CG(a, alphap, b, betap, c, gamma)
    simp = KroneckerDelta(alpha, alphap)*KroneckerDelta(beta, betap)
    sign = S.One
    x = abs(a - b)
    y = abs(alpha + beta)
    build_expr = a + b + 1 - Piecewise((x, x > y), (0, Eq(x, y)), (y, y > x))
    index_expr = a + b - c
    term_list, other3 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    # For symbolic alpha,alphap,beta,betap
    x = abs(a - b)
    y = a + b
    build_expr = (y + 1 - x)*(x + y + 1)
    index_expr = (c - x)*(x + c) + c + gamma
    term_list, other4 = _check_cg_simp(expr, simp, sign, S.One, term_list, (a, alpha, alphap, b, beta, betap, c, gamma), (a, alpha, alphap, b, beta, betap), build_expr, index_expr)

    return term_list, other1 + other2 + other4


def _check_cg_simp(expr, simp, sign, lt, term_list, variables, dep_variables, build_index_expr, index_expr):
    """ Checks for simplifications that can be made, returning a tuple of the
    simplified list of terms and any terms generated by simplification.

    Parameters
    ==========

    expr: expression
        The expression with Wild terms that will be matched to the terms in
        the sum

    simp: expression
        The expression with Wild terms that is substituted in place of the CG
        terms in the case of simplification

    sign: expression
        The expression with Wild terms denoting the sign that is on expr that
        must match

    lt: expression
        The expression with Wild terms that gives the leading term of the
        matched expr

    term_list: list
        A list of all of the terms is the sum to be simplified

    variables: list
        A list of all the variables that appears in expr

    dep_variables: list
        A list of the variables that must match for all the terms in the sum,
        i.e. the dependent variables

    build_index_expr: expression
        Expression with Wild terms giving the number of elements in cg_index

    index_expr: expression
        Expression with Wild terms giving the index terms have when storing
        them to cg_index

    """
    other_part = 0
    i = 0
    while i < len(term_list):
        sub_1 = _check_cg(term_list[i], expr, len(variables))
        if sub_1 is None:
            i += 1
            continue
        if not build_index_expr.subs(sub_1).is_number:
            i += 1
            continue
        sub_dep = [(x, sub_1[x]) for x in dep_variables]
        cg_index = [None]*build_index_expr.subs(sub_1)
        for j in range(i, len(term_list)):
            sub_2 = _check_cg(term_list[j], expr.subs(sub_dep), len(variables) - len(dep_variables), sign=(sign.subs(sub_1), sign.subs(sub_dep)))
            if sub_2 is None:
                continue
            if not index_expr.subs(sub_dep).subs(sub_2).is_number:
                continue
            cg_index[index_expr.subs(sub_dep).subs(sub_2)] = j, expr.subs(lt, 1).subs(sub_dep).subs(sub_2), lt.subs(sub_2), sign.subs(sub_dep).subs(sub_2)
        if not any(i is None for i in cg_index):
            min_lt = min(*[ abs(term[2]) for term in cg_index ])
            indices = [ term[0] for term in cg_index]
            indices.sort()
            indices.reverse()
            [ term_list.pop(j) for j in indices ]
            for term in cg_index:
                if abs(term[2]) > min_lt:
                    term_list.append( (term[2] - min_lt*term[3])*term[1] )
            other_part += min_lt*(sign*simp).subs(sub_1)
        else:
            i += 1
    return term_list, other_part


def _check_cg(cg_term, expr, length, sign=None):
    """Checks whether a term matches the given expression"""
    # TODO: Check for symmetries
    matches = cg_term.match(expr)
    if matches is None:
        return
    if sign is not None:
        if not isinstance(sign, tuple):
            raise TypeError('sign must be a tuple')
        if not sign[0] == (sign[1]).subs(matches):
            return
    if len(matches) == length:
        return matches


def _cg_simp_sum(e):
    e = _check_varsh_sum_871_1(e)
    e = _check_varsh_sum_871_2(e)
    e = _check_varsh_sum_872_4(e)
    return e


def _check_varsh_sum_871_1(e):
    a = Wild('a')
    alpha = symbols('alpha')
    b = Wild('b')
    match = e.match(Sum(CG(a, alpha, b, 0, a, alpha), (alpha, -a, a)))
    if match is not None and len(match) == 2:
        return ((2*a + 1)*KroneckerDelta(b, 0)).subs(match)
    return e


def _check_varsh_sum_871_2(e):
    a = Wild('a')
    alpha = symbols('alpha')
    c = Wild('c')
    match = e.match(
        Sum((-1)**(a - alpha)*CG(a, alpha, a, -alpha, c, 0), (alpha, -a, a)))
    if match is not None and len(match) == 2:
        return (sqrt(2*a + 1)*KroneckerDelta(c, 0)).subs(match)
    return e


def _check_varsh_sum_872_4(e):
    alpha = symbols('alpha')
    beta = symbols('beta')
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    cp = Wild('cp')
    gamma = Wild('gamma')
    gammap = Wild('gammap')
    cg1 = CG(a, alpha, b, beta, c, gamma)
    cg2 = CG(a, alpha, b, beta, cp, gammap)
    match1 = e.match(Sum(cg1*cg2, (alpha, -a, a), (beta, -b, b)))
    if match1 is not None and len(match1) == 6:
        return (KroneckerDelta(c, cp)*KroneckerDelta(gamma, gammap)).subs(match1)
    match2 = e.match(Sum(cg1**2, (alpha, -a, a), (beta, -b, b)))
    if match2 is not None and len(match2) == 4:
        return S.One
    return e


def _cg_list(term):
    if isinstance(term, CG):
        return (term,), 1, 1
    cg = []
    coeff = 1
    if not isinstance(term, (Mul, Pow)):
        raise NotImplementedError('term must be CG, Add, Mul or Pow')
    if isinstance(term, Pow) and term.exp.is_number:
        if term.exp.is_number:
            [ cg.append(term.base) for _ in range(term.exp) ]
        else:
            return (term,), 1, 1
    if isinstance(term, Mul):
        for arg in term.args:
            if isinstance(arg, CG):
                cg.append(arg)
            else:
                coeff *= arg
        return cg, coeff, coeff/abs(coeff)
