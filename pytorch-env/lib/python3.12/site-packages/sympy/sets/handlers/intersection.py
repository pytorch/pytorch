from sympy.core.basic import _aresame
from sympy.core.function import Lambda, expand_complex
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, Float
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.core.sorting import ordered
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor, ceiling
from sympy.sets.fancysets import ComplexRegion
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.multipledispatch import Dispatcher
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import (Integers, Naturals, Reals, Range,
    ImageSet, Rationals)
from sympy.sets.sets import EmptySet, UniversalSet, imageset, ProductSet
from sympy.simplify.radsimp import numer


intersection_sets = Dispatcher('intersection_sets')


@intersection_sets.register(ConditionSet, ConditionSet)
def _(a, b):
    return None

@intersection_sets.register(ConditionSet, Set)
def _(a, b):
    return ConditionSet(a.sym, a.condition, Intersection(a.base_set, b))

@intersection_sets.register(Naturals, Integers)
def _(a, b):
    return a

@intersection_sets.register(Naturals, Naturals)
def _(a, b):
    return a if a is S.Naturals else b

@intersection_sets.register(Interval, Naturals)
def _(a, b):
    return intersection_sets(b, a)

@intersection_sets.register(ComplexRegion, Set)
def _(self, other):
    if other.is_ComplexRegion:
        # self in rectangular form
        if (not self.polar) and (not other.polar):
            return ComplexRegion(Intersection(self.sets, other.sets))

        # self in polar form
        elif self.polar and other.polar:
            r1, theta1 = self.a_interval, self.b_interval
            r2, theta2 = other.a_interval, other.b_interval
            new_r_interval = Intersection(r1, r2)
            new_theta_interval = Intersection(theta1, theta2)

            # 0 and 2*Pi means the same
            if ((2*S.Pi in theta1 and S.Zero in theta2) or
               (2*S.Pi in theta2 and S.Zero in theta1)):
                new_theta_interval = Union(new_theta_interval,
                                           FiniteSet(0))
            return ComplexRegion(new_r_interval*new_theta_interval,
                                polar=True)


    if other.is_subset(S.Reals):
        new_interval = []
        x = symbols("x", cls=Dummy, real=True)

        # self in rectangular form
        if not self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)

        # self in polar form
        elif self.polar:
            for element in self.psets:
                if S.Zero in element.args[1]:
                    new_interval.append(element.args[0])
                if S.Pi in element.args[1]:
                    new_interval.append(ImageSet(Lambda(x, -x), element.args[0]))
                if S.Zero in element.args[0]:
                    new_interval.append(FiniteSet(0))
            new_interval = Union(*new_interval)
            return Intersection(new_interval, other)

@intersection_sets.register(Integers, Reals)
def _(a, b):
    return a

@intersection_sets.register(Range, Interval)
def _(a, b):
    # Check that there are no symbolic arguments
    if not all(i.is_number for i in a.args + b.args[:2]):
        return

    # In case of null Range, return an EmptySet.
    if a.size == 0:
        return S.EmptySet

    # trim down to self's size, and represent
    # as a Range with step 1.
    start = ceiling(max(b.inf, a.inf))
    if start not in b:
        start += 1
    end = floor(min(b.sup, a.sup))
    if end not in b:
        end -= 1
    return intersection_sets(a, Range(start, end + 1))

@intersection_sets.register(Range, Naturals)
def _(a, b):
    return intersection_sets(a, Interval(b.inf, S.Infinity))

@intersection_sets.register(Range, Range)
def _(a, b):
    # Check that there are no symbolic range arguments
    if not all(all(v.is_number for v in r.args) for r in [a, b]):
        return None

    # non-overlap quick exits
    if not b:
        return S.EmptySet
    if not a:
        return S.EmptySet
    if b.sup < a.inf:
        return S.EmptySet
    if b.inf > a.sup:
        return S.EmptySet

    # work with finite end at the start
    r1 = a
    if r1.start.is_infinite:
        r1 = r1.reversed
    r2 = b
    if r2.start.is_infinite:
        r2 = r2.reversed

    # If both ends are infinite then it means that one Range is just the set
    # of all integers (the step must be 1).
    if r1.start.is_infinite:
        return b
    if r2.start.is_infinite:
        return a

    from sympy.solvers.diophantine.diophantine import diop_linear

    # this equation represents the values of the Range;
    # it's a linear equation
    eq = lambda r, i: r.start + i*r.step

    # we want to know when the two equations might
    # have integer solutions so we use the diophantine
    # solver
    va, vb = diop_linear(eq(r1, Dummy('a')) - eq(r2, Dummy('b')))

    # check for no solution
    no_solution = va is None and vb is None
    if no_solution:
        return S.EmptySet

    # there is a solution
    # -------------------

    # find the coincident point, c
    a0 = va.as_coeff_Add()[0]
    c = eq(r1, a0)

    # find the first point, if possible, in each range
    # since c may not be that point
    def _first_finite_point(r1, c):
        if c == r1.start:
            return c
        # st is the signed step we need to take to
        # get from c to r1.start
        st = sign(r1.start - c)*step
        # use Range to calculate the first point:
        # we want to get as close as possible to
        # r1.start; the Range will not be null since
        # it will at least contain c
        s1 = Range(c, r1.start + st, st)[-1]
        if s1 == r1.start:
            pass
        else:
            # if we didn't hit r1.start then, if the
            # sign of st didn't match the sign of r1.step
            # we are off by one and s1 is not in r1
            if sign(r1.step) != sign(st):
                s1 -= st
        if s1 not in r1:
            return
        return s1

    # calculate the step size of the new Range
    step = abs(ilcm(r1.step, r2.step))
    s1 = _first_finite_point(r1, c)
    if s1 is None:
        return S.EmptySet
    s2 = _first_finite_point(r2, c)
    if s2 is None:
        return S.EmptySet

    # replace the corresponding start or stop in
    # the original Ranges with these points; the
    # result must have at least one point since
    # we know that s1 and s2 are in the Ranges
    def _updated_range(r, first):
        st = sign(r.step)*step
        if r.start.is_finite:
            rv = Range(first, r.stop, st)
        else:
            rv = Range(r.start, first + st, st)
        return rv
    r1 = _updated_range(a, s1)
    r2 = _updated_range(b, s2)

    # work with them both in the increasing direction
    if sign(r1.step) < 0:
        r1 = r1.reversed
    if sign(r2.step) < 0:
        r2 = r2.reversed

    # return clipped Range with positive step; it
    # can't be empty at this point
    start = max(r1.start, r2.start)
    stop = min(r1.stop, r2.stop)
    return Range(start, stop, step)


@intersection_sets.register(Range, Integers)
def _(a, b):
    return a


@intersection_sets.register(Range, Rationals)
def _(a, b):
    return a


@intersection_sets.register(ImageSet, Set)
def _(self, other):
    from sympy.solvers.diophantine import diophantine

    # Only handle the straight-forward univariate case
    if (len(self.lamda.variables) > 1
            or self.lamda.signature != self.lamda.variables):
        return None
    base_set = self.base_sets[0]

    # Intersection between ImageSets with Integers as base set
    # For {f(n) : n in Integers} & {g(m) : m in Integers} we solve the
    # diophantine equations f(n)=g(m).
    # If the solutions for n are {h(t) : t in Integers} then we return
    # {f(h(t)) : t in integers}.
    # If the solutions for n are {n_1, n_2, ..., n_k} then we return
    # {f(n_i) : 1 <= i <= k}.
    if base_set is S.Integers:
        gm = None
        if isinstance(other, ImageSet) and other.base_sets == (S.Integers,):
            gm = other.lamda.expr
            var = other.lamda.variables[0]
            # Symbol of second ImageSet lambda must be distinct from first
            m = Dummy('m')
            gm = gm.subs(var, m)
        elif other is S.Integers:
            m = gm = Dummy('m')
        if gm is not None:
            fn = self.lamda.expr
            n = self.lamda.variables[0]
            try:
                solns = list(diophantine(fn - gm, syms=(n, m), permute=True))
            except (TypeError, NotImplementedError):
                # TypeError if equation not polynomial with rational coeff.
                # NotImplementedError if correct format but no solver.
                return
            # 3 cases are possible for solns:
            # - empty set,
            # - one or more parametric (infinite) solutions,
            # - a finite number of (non-parametric) solution couples.
            # Among those, there is one type of solution set that is
            # not helpful here: multiple parametric solutions.
            if len(solns) == 0:
                return S.EmptySet
            elif any(s.free_symbols for tupl in solns for s in tupl):
                if len(solns) == 1:
                    soln, solm = solns[0]
                    (t,) = soln.free_symbols
                    expr = fn.subs(n, soln.subs(t, n)).expand()
                    return imageset(Lambda(n, expr), S.Integers)
                else:
                    return
            else:
                return FiniteSet(*(fn.subs(n, s[0]) for s in solns))

    if other == S.Reals:
        from sympy.solvers.solvers import denoms, solve_linear

        def _solution_union(exprs, sym):
            # return a union of linear solutions to i in expr;
            # if i cannot be solved, use a ConditionSet for solution
            sols = []
            for i in exprs:
                x, xis = solve_linear(i, 0, [sym])
                if x == sym:
                    sols.append(FiniteSet(xis))
                else:
                    sols.append(ConditionSet(sym, Eq(i, 0)))
            return Union(*sols)

        f = self.lamda.expr
        n = self.lamda.variables[0]

        n_ = Dummy(n.name, real=True)
        f_ = f.subs(n, n_)

        re, im = f_.as_real_imag()
        im = expand_complex(im)

        re = re.subs(n_, n)
        im = im.subs(n_, n)
        ifree = im.free_symbols
        lam = Lambda(n, re)
        if im.is_zero:
            # allow re-evaluation
            # of self in this case to make
            # the result canonical
            pass
        elif im.is_zero is False:
            return S.EmptySet
        elif ifree != {n}:
            return None
        else:
            # univarite imaginary part in same variable;
            # use numer instead of as_numer_denom to keep
            # this as fast as possible while still handling
            # simple cases
            base_set &= _solution_union(
                Mul.make_args(numer(im)), n)
        # exclude values that make denominators 0
        base_set -= _solution_union(denoms(f), n)
        return imageset(lam, base_set)

    elif isinstance(other, Interval):
        from sympy.solvers.solveset import (invert_real, invert_complex,
                                            solveset)

        f = self.lamda.expr
        n = self.lamda.variables[0]
        new_inf, new_sup = None, None
        new_lopen, new_ropen = other.left_open, other.right_open

        if f.is_real:
            inverter = invert_real
        else:
            inverter = invert_complex

        g1, h1 = inverter(f, other.inf, n)
        g2, h2 = inverter(f, other.sup, n)

        if all(isinstance(i, FiniteSet) for i in (h1, h2)):
            if g1 == n:
                if len(h1) == 1:
                    new_inf = h1.args[0]
            if g2 == n:
                if len(h2) == 1:
                    new_sup = h2.args[0]
            # TODO: Design a technique to handle multiple-inverse
            # functions

            # Any of the new boundary values cannot be determined
            if any(i is None for i in (new_sup, new_inf)):
                return


            range_set = S.EmptySet

            if all(i.is_real for i in (new_sup, new_inf)):
                # this assumes continuity of underlying function
                # however fixes the case when it is decreasing
                if new_inf > new_sup:
                    new_inf, new_sup = new_sup, new_inf
                new_interval = Interval(new_inf, new_sup, new_lopen, new_ropen)
                range_set = base_set.intersect(new_interval)
            else:
                if other.is_subset(S.Reals):
                    solutions = solveset(f, n, S.Reals)
                    if not isinstance(range_set, (ImageSet, ConditionSet)):
                        range_set = solutions.intersect(other)
                    else:
                        return

            if range_set is S.EmptySet:
                return S.EmptySet
            elif isinstance(range_set, Range) and range_set.size is not S.Infinity:
                range_set = FiniteSet(*list(range_set))

            if range_set is not None:
                return imageset(Lambda(n, f), range_set)
            return
        else:
            return


@intersection_sets.register(ProductSet, ProductSet)
def _(a, b):
    if len(b.args) != len(a.args):
        return S.EmptySet
    return ProductSet(*(i.intersect(j) for i, j in zip(a.sets, b.sets)))


@intersection_sets.register(Interval, Interval)
def _(a, b):
    # handle (-oo, oo)
    infty = S.NegativeInfinity, S.Infinity
    if a == Interval(*infty):
        l, r = a.left, a.right
        if l.is_real or l in infty or r.is_real or r in infty:
            return b

    # We can't intersect [0,3] with [x,6] -- we don't know if x>0 or x<0
    if not a._is_comparable(b):
        return None

    empty = False

    if a.start <= b.end and b.start <= a.end:
        # Get topology right.
        if a.start < b.start:
            start = b.start
            left_open = b.left_open
        elif a.start > b.start:
            start = a.start
            left_open = a.left_open
        else:
            start = a.start
            if not _aresame(a.start, b.start):
                # For example Integer(2) != Float(2)
                # Prefer the Float boundary because Floats should be
                # contagious in calculations.
                if b.start.has(Float) and not a.start.has(Float):
                    start = b.start
                elif a.start.has(Float) and not b.start.has(Float):
                    start = a.start
                else:
                    #this is to ensure that if Eq(a.start, b.start) but
                    #type(a.start) != type(b.start) the order of a and b
                    #does not matter for the result
                    start = list(ordered([a,b]))[0].start
            left_open = a.left_open or b.left_open

        if a.end < b.end:
            end = a.end
            right_open = a.right_open
        elif a.end > b.end:
            end = b.end
            right_open = b.right_open
        else:
            # see above for logic with start
            end = a.end
            if not _aresame(a.end, b.end):
                if b.end.has(Float) and not a.end.has(Float):
                    end = b.end
                elif a.end.has(Float) and not b.end.has(Float):
                    end = a.end
                else:
                    end = list(ordered([a,b]))[0].end
            right_open = a.right_open or b.right_open

        if end - start == 0 and (left_open or right_open):
            empty = True
    else:
        empty = True

    if empty:
        return S.EmptySet

    return Interval(start, end, left_open, right_open)

@intersection_sets.register(EmptySet, Set)
def _(a, b):
    return S.EmptySet

@intersection_sets.register(UniversalSet, Set)
def _(a, b):
    return b

@intersection_sets.register(FiniteSet, FiniteSet)
def _(a, b):
    return FiniteSet(*(a._elements & b._elements))

@intersection_sets.register(FiniteSet, Set)
def _(a, b):
    try:
        return FiniteSet(*[el for el in a if el in b])
    except TypeError:
        return None  # could not evaluate `el in b` due to symbolic ranges.

@intersection_sets.register(Set, Set)
def _(a, b):
    return None

@intersection_sets.register(Integers, Rationals)
def _(a, b):
    return a

@intersection_sets.register(Naturals, Rationals)
def _(a, b):
    return a

@intersection_sets.register(Rationals, Reals)
def _(a, b):
    return a

def _intlike_interval(a, b):
    try:
        if b._inf is S.NegativeInfinity and b._sup is S.Infinity:
            return a
        s = Range(max(a.inf, ceiling(b.left)), floor(b.right) + 1)
        return intersection_sets(s, b)  # take out endpoints if open interval
    except ValueError:
        return None

@intersection_sets.register(Integers, Interval)
def _(a, b):
    return _intlike_interval(a, b)

@intersection_sets.register(Naturals, Interval)
def _(a, b):
    return _intlike_interval(a, b)
