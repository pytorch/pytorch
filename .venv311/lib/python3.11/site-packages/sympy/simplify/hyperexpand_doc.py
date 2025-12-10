""" This module cooks up a docstring when imported. Its only purpose is to
    be displayed in the sphinx documentation. """

from sympy.core.relational import Eq
from sympy.functions.special.hyper import hyper
from sympy.printing.latex import latex
from sympy.simplify.hyperexpand import FormulaCollection

c = FormulaCollection()

doc = ""

for f in c.formulae:
    obj = Eq(hyper(f.func.ap, f.func.bq, f.z),
             f.closed_form.rewrite('nonrepsmall'))
    doc += ".. math::\n  %s\n" % latex(obj)

__doc__ = doc
