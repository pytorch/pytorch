from sympy.core import Atom, Basic


class CartanType_generator():
    """
    Constructor for actually creating things
    """
    def __call__(self, *args):
        c = args[0]
        if isinstance(c, list):
            letter, n = c[0], int(c[1])
        elif isinstance(c, str):
            letter, n = c[0], int(c[1:])
        else:
            raise TypeError("Argument must be a string (e.g. 'A3') or a list (e.g. ['A', 3])")

        if n < 0:
            raise ValueError("Lie algebra rank cannot be negative")
        if letter == "A":
            from . import type_a
            return type_a.TypeA(n)
        if letter == "B":
            from . import type_b
            return type_b.TypeB(n)

        if letter == "C":
            from . import type_c
            return type_c.TypeC(n)

        if letter == "D":
            from . import type_d
            return type_d.TypeD(n)

        if letter == "E":
            if n >= 6 and n <= 8:
                from . import type_e
                return type_e.TypeE(n)

        if letter == "F":
            if n == 4:
                from . import type_f
                return type_f.TypeF(n)

        if letter == "G":
            if n == 2:
                from . import type_g
                return type_g.TypeG(n)

CartanType = CartanType_generator()


class Standard_Cartan(Atom):
    """
    Concrete base class for Cartan types such as A4, etc
    """

    def __new__(cls, series, n):
        obj = Basic.__new__(cls)
        obj.n = n
        obj.series = series
        return obj

    def rank(self):
        """
        Returns the rank of the Lie algebra
        """
        return self.n

    def series(self):
        """
        Returns the type of the Lie algebra
        """
        return self.series
