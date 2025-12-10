"""
A Printer for generating executable code.

The most important function here is srepr that returns a string so that the
relation eval(srepr(expr))=expr holds in an appropriate environment.
"""

from __future__ import annotations
from typing import Any

from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from mpmath.libmp import repr_dps, to_str as mlib_to_str

from .printer import Printer, print_function


class ReprPrinter(Printer):
    printmethod = "_sympyrepr"

    _default_settings: dict[str, Any] = {
        "order": None,
        "perm_cyclic" : True,
    }

    def reprify(self, args, sep):
        """
        Prints each item in `args` and joins them with `sep`.
        """
        return sep.join([self.doprint(item) for item in args])

    def emptyPrinter(self, expr):
        """
        The fallback printer.
        """
        if isinstance(expr, str):
            return expr
        elif hasattr(expr, "__srepr__"):
            return expr.__srepr__()
        elif hasattr(expr, "args") and hasattr(expr.args, "__iter__"):
            l = []
            for o in expr.args:
                l.append(self._print(o))
            return expr.__class__.__name__ + '(%s)' % ', '.join(l)
        elif hasattr(expr, "__module__") and hasattr(expr, "__name__"):
            return "<'%s.%s'>" % (expr.__module__, expr.__name__)
        else:
            return str(expr)

    def _print_Add(self, expr, order=None):
        args = self._as_ordered_terms(expr, order=order)
        args = map(self._print, args)
        clsname = type(expr).__name__
        return clsname + "(%s)" % ", ".join(args)

    def _print_Cycle(self, expr):
        return expr.__repr__()

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
        from sympy.utilities.exceptions import sympy_deprecation_warning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=7,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            if not expr.size:
                return 'Permutation()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            return 'Permutation%s' %s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % str(expr.array_form)
                return 'Permutation([], size=%s)' % expr.size
            trim = str(expr.array_form[:s[-1] + 1]) + ', size=%s' % expr.size
            use = full = str(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use

    def _print_Function(self, expr):
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.args])
        return r

    def _print_Heaviside(self, expr):
        # Same as _print_Function but uses pargs to suppress default value for
        # 2nd arg.
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.pargs])
        return r

    def _print_FunctionClass(self, expr):
        if issubclass(expr, AppliedUndef):
            return 'Function(%r)' % (expr.__name__)
        else:
            return expr.__name__

    def _print_Half(self, expr):
        return 'Rational(1, 2)'

    def _print_RationalConstant(self, expr):
        return str(expr)

    def _print_AtomicExpr(self, expr):
        return str(expr)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Integer(self, expr):
        return 'Integer(%i)' % expr.p

    def _print_Complexes(self, expr):
        return 'Complexes'

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Rationals(self, expr):
        return 'Rationals'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_UniversalSet(self, expr):
        return 'UniversalSet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_list(self, expr):
        return "[%s]" % self.reprify(expr, ", ")

    def _print_dict(self, expr):
        sep = ", "
        dict_kvs = ["%s: %s" % (self.doprint(key), self.doprint(value)) for key, value in expr.items()]
        return "{%s}" % sep.join(dict_kvs)

    def _print_set(self, expr):
        if not expr:
            return "set()"
        return "{%s}" % self.reprify(expr, ", ")

    def _print_MatrixBase(self, expr):
        # special case for some empty matrices
        if (expr.rows == 0) ^ (expr.cols == 0):
            return '%s(%s, %s, %s)' % (expr.__class__.__name__,
                                       self._print(expr.rows),
                                       self._print(expr.cols),
                                       self._print([]))
        l = []
        for i in range(expr.rows):
            l.append([])
            for j in range(expr.cols):
                l[-1].append(expr[i, j])
        return '%s(%s)' % (expr.__class__.__name__, self._print(l))

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_NaN(self, expr):
        return "nan"

    def _print_Mul(self, expr, order=None):
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        args = map(self._print, args)
        clsname = type(expr).__name__
        return clsname + "(%s)" % ", ".join(args)

    def _print_Rational(self, expr):
        return 'Rational(%s, %s)' % (self._print(expr.p), self._print(expr.q))

    def _print_PythonRational(self, expr):
        return "%s(%d, %d)" % (expr.__class__.__name__, expr.p, expr.q)

    def _print_Fraction(self, expr):
        return 'Fraction(%s, %s)' % (self._print(expr.numerator), self._print(expr.denominator))

    def _print_Float(self, expr):
        r = mlib_to_str(expr._mpf_, repr_dps(expr._prec))
        return "%s('%s', precision=%i)" % (expr.__class__.__name__, r, expr._prec)

    def _print_Sum2(self, expr):
        return "Sum2(%s, (%s, %s, %s))" % (self._print(expr.f), self._print(expr.i),
                                           self._print(expr.a), self._print(expr.b))

    def _print_Str(self, s):
        return "%s(%s)" % (s.__class__.__name__, self._print(s.name))

    def _print_Symbol(self, expr):
        d = expr._assumptions_orig
        # print the dummy_index like it was an assumption
        if expr.is_Dummy:
            d = d.copy()
            d['dummy_index'] = expr.dummy_index

        if d == {}:
            return "%s(%s)" % (expr.__class__.__name__, self._print(expr.name))
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s)" % (expr.__class__.__name__,
                                   self._print(expr.name), ', '.join(attr))

    def _print_CoordinateSymbol(self, expr):
        d = expr._assumptions.generator

        if d == {}:
            return "%s(%s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index)
            )
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index),
                ', '.join(attr)
            )

    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    def _print_AppliedPredicate(self, expr):
        # will be changed to just expr.args when args overriding is removed
        args = expr._args
        return "%s(%s)" % (expr.__class__.__name__, self.reprify(args, ", "))

    def _print_str(self, expr):
        return repr(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.reprify(expr, ", ")

    def _print_WildFunction(self, expr):
        return "%s('%s')" % (expr.__class__.__name__, expr.name)

    def _print_AlgebraicNumber(self, expr):
        return "%s(%s, %s)" % (expr.__class__.__name__,
            self._print(expr.root), self._print(expr.coeffs()))

    def _print_PolyRing(self, ring):
        return "%s(%s, %s, %s)" % (ring.__class__.__name__,
            self._print(ring.symbols), self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "%s(%s, %s, %s)" % (field.__class__.__name__,
            self._print(field.symbols), self._print(field.domain), self._print(field.order))

    def _print_PolyElement(self, poly):
        terms = list(poly.terms())
        terms.sort(key=poly.ring.order, reverse=True)
        return "%s(%s, %s)" % (poly.__class__.__name__, self._print(poly.ring), self._print(terms))

    def _print_FracElement(self, frac):
        numer_terms = list(frac.numer.terms())
        numer_terms.sort(key=frac.field.order, reverse=True)
        denom_terms = list(frac.denom.terms())
        denom_terms.sort(key=frac.field.order, reverse=True)
        numer = self._print(numer_terms)
        denom = self._print(denom_terms)
        return "%s(%s, %s, %s)" % (frac.__class__.__name__, self._print(frac.field), numer, denom)

    def _print_FractionField(self, domain):
        cls = domain.__class__.__name__
        field = self._print(domain.field)
        return "%s(%s)" % (cls, field)

    def _print_PolynomialRingBase(self, ring):
        cls = ring.__class__.__name__
        dom = self._print(ring.domain)
        gens = ', '.join(map(self._print, ring.gens))
        order = str(ring.order)
        if order != ring.default_order:
            orderstr = ", order=" + order
        else:
            orderstr = ""
        return "%s(%s, %s%s)" % (cls, dom, gens, orderstr)

    def _print_DMP(self, p):
        cls = p.__class__.__name__
        rep = self._print(p.to_list())
        dom = self._print(p.dom)
        return "%s(%s, %s)" % (cls, rep, dom)

    def _print_MonogenicFiniteExtension(self, ext):
        # The expanded tree shown by srepr(ext.modulus)
        # is not practical.
        return "FiniteExtension(%s)" % str(ext.modulus)

    def _print_ExtensionElement(self, f):
        rep = self._print(f.rep)
        ext = self._print(f.ext)
        return "ExtElem(%s, %s)" % (rep, ext)

@print_function(ReprPrinter)
def srepr(expr, **settings):
    """return expr in repr form"""
    return ReprPrinter(settings).doprint(expr)
