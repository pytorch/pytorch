from sympy.core.symbol import symbols
from sympy.functions.special.spherical_harmonics import Ynm

x, y = symbols('x,y')


def timeit_Ynm_xy():
    Ynm(1, 1, x, y)
