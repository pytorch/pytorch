"""Definitions of common exceptions for `polys` module. """


from sympy.utilities import public

@public
class BasePolynomialError(Exception):
    """Base class for polynomial related exceptions. """

    def new(self, *args):
        raise NotImplementedError("abstract base class")

@public
class ExactQuotientFailed(BasePolynomialError):

    def __init__(self, f, g, dom=None):
        self.f, self.g, self.dom = f, g, dom

    def __str__(self):  # pragma: no cover
        from sympy.printing.str import sstr

        if self.dom is None:
            return "%s does not divide %s" % (sstr(self.g), sstr(self.f))
        else:
            return "%s does not divide %s in %s" % (sstr(self.g), sstr(self.f), sstr(self.dom))

    def new(self, f, g):
        return self.__class__(f, g, self.dom)

@public
class PolynomialDivisionFailed(BasePolynomialError):

    def __init__(self, f, g, domain):
        self.f = f
        self.g = g
        self.domain = domain

    def __str__(self):
        if self.domain.is_EX:
            msg = "You may want to use a different simplification algorithm. Note " \
                  "that in general it's not possible to guarantee to detect zero "  \
                  "in this domain."
        elif not self.domain.is_Exact:
            msg = "Your working precision or tolerance of computations may be set " \
                  "improperly. Adjust those parameters of the coefficient domain "  \
                  "and try again."
        else:
            msg = "Zero detection is guaranteed in this coefficient domain. This "  \
                  "may indicate a bug in SymPy or the domain is user defined and "  \
                  "doesn't implement zero detection properly."

        return "couldn't reduce degree in a polynomial division algorithm when "    \
               "dividing %s by %s. This can happen when it's not possible to "      \
               "detect zero in the coefficient domain. The domain of computation "  \
               "is %s. %s" % (self.f, self.g, self.domain, msg)

@public
class OperationNotSupported(BasePolynomialError):

    def __init__(self, poly, func):
        self.poly = poly
        self.func = func

    def __str__(self):  # pragma: no cover
        return "`%s` operation not supported by %s representation" % (self.func, self.poly.rep.__class__.__name__)

@public
class HeuristicGCDFailed(BasePolynomialError):
    pass

class ModularGCDFailed(BasePolynomialError):
    pass

@public
class HomomorphismFailed(BasePolynomialError):
    pass

@public
class IsomorphismFailed(BasePolynomialError):
    pass

@public
class ExtraneousFactors(BasePolynomialError):
    pass

@public
class EvaluationFailed(BasePolynomialError):
    pass

@public
class RefinementFailed(BasePolynomialError):
    pass

@public
class CoercionFailed(BasePolynomialError):
    pass

@public
class NotInvertible(BasePolynomialError):
    pass

@public
class NotReversible(BasePolynomialError):
    pass

@public
class NotAlgebraic(BasePolynomialError):
    pass

@public
class DomainError(BasePolynomialError):
    pass

@public
class PolynomialError(BasePolynomialError):
    pass

@public
class UnificationFailed(BasePolynomialError):
    pass

@public
class UnsolvableFactorError(BasePolynomialError):
    """Raised if ``roots`` is called with strict=True and a polynomial
     having a factor whose solutions are not expressible in radicals
     is encountered."""

@public
class GeneratorsError(BasePolynomialError):
    pass

@public
class GeneratorsNeeded(GeneratorsError):
    pass

@public
class ComputationFailed(BasePolynomialError):

    def __init__(self, func, nargs, exc):
        self.func = func
        self.nargs = nargs
        self.exc = exc

    def __str__(self):
        return "%s(%s) failed without generators" % (self.func, ', '.join(map(str, self.exc.exprs[:self.nargs])))

@public
class UnivariatePolynomialError(PolynomialError):
    pass

@public
class MultivariatePolynomialError(PolynomialError):
    pass

@public
class PolificationFailed(PolynomialError):

    def __init__(self, opt, origs, exprs, seq=False):
        if not seq:
            self.orig = origs
            self.expr = exprs
            self.origs = [origs]
            self.exprs = [exprs]
        else:
            self.origs = origs
            self.exprs = exprs

        self.opt = opt
        self.seq = seq

    def __str__(self):  # pragma: no cover
        if not self.seq:
            return "Cannot construct a polynomial from %s" % str(self.orig)
        else:
            return "Cannot construct polynomials from %s" % ', '.join(map(str, self.origs))

@public
class OptionError(BasePolynomialError):
    pass

@public
class FlagError(OptionError):
    pass
