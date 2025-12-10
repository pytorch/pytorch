import re
from sympy.concrete.products import product
from sympy.concrete.summations import Sum
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)


class MaximaHelpers:
    def maxima_expand(expr):
        return expr.expand()

    def maxima_float(expr):
        return expr.evalf()

    def maxima_trigexpand(expr):
        return expr.expand(trig=True)

    def maxima_sum(a1, a2, a3, a4):
        return Sum(a1, (a2, a3, a4)).doit()

    def maxima_product(a1, a2, a3, a4):
        return product(a1, (a2, a3, a4))

    def maxima_csc(expr):
        return 1/sin(expr)

    def maxima_sec(expr):
        return 1/cos(expr)

sub_dict = {
    'pi': re.compile(r'%pi'),
    'E': re.compile(r'%e'),
    'I': re.compile(r'%i'),
    '**': re.compile(r'\^'),
    'oo': re.compile(r'\binf\b'),
    '-oo': re.compile(r'\bminf\b'),
    "'-'": re.compile(r'\bminus\b'),
    'maxima_expand': re.compile(r'\bexpand\b'),
    'maxima_float': re.compile(r'\bfloat\b'),
    'maxima_trigexpand': re.compile(r'\btrigexpand'),
    'maxima_sum': re.compile(r'\bsum\b'),
    'maxima_product': re.compile(r'\bproduct\b'),
    'cancel': re.compile(r'\bratsimp\b'),
    'maxima_csc': re.compile(r'\bcsc\b'),
    'maxima_sec': re.compile(r'\bsec\b')
}

var_name = re.compile(r'^\s*(\w+)\s*:')


def parse_maxima(str, globals=None, name_dict={}):
    str = str.strip()
    str = str.rstrip('; ')

    for k, v in sub_dict.items():
        str = v.sub(k, str)

    assign_var = None
    var_match = var_name.search(str)
    if var_match:
        assign_var = var_match.group(1)
        str = str[var_match.end():].strip()

    dct = MaximaHelpers.__dict__.copy()
    dct.update(name_dict)
    obj = sympify(str, locals=dct)

    if assign_var and globals:
        globals[assign_var] = obj

    return obj
