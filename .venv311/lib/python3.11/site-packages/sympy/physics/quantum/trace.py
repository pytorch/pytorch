from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import sympify
from sympy.matrices import Matrix


def _is_scalar(e):
    """ Helper method used in Tr"""

    # sympify to set proper attributes
    e = sympify(e)
    if isinstance(e, Expr):
        if (e.is_Integer or e.is_Float or
            e.is_Rational or e.is_Number or
            (e.is_Symbol and e.is_commutative)
                ):
            return True

    return False


def _cycle_permute(l):
    """ Cyclic permutations based on canonical ordering

    Explanation
    ===========

    This method does the sort based ascii values while
    a better approach would be to used lexicographic sort.

    TODO: Handle condition such as symbols have subscripts/superscripts
    in case of lexicographic sort

    """

    if len(l) == 1:
        return l

    min_item = min(l, key=default_sort_key)
    indices = [i for i, x in enumerate(l) if x == min_item]

    le = list(l)
    le.extend(l)  # duplicate and extend string for easy processing

    # adding the first min_item index back for easier looping
    indices.append(len(l) + indices[0])

    # create sublist of items with first item as min_item and last_item
    # in each of the sublist is item just before the next occurrence of
    # minitem in the cycle formed.
    sublist = [[le[indices[i]:indices[i + 1]]] for i in
               range(len(indices) - 1)]

    # we do comparison of strings by comparing elements
    # in each sublist
    idx = sublist.index(min(sublist))
    ordered_l = le[indices[idx]:indices[idx] + len(l)]

    return ordered_l


def _rearrange_args(l):
    """ this just moves the last arg to first position
     to enable expansion of args
     A,B,A ==> A**2,B
    """
    if len(l) == 1:
        return l

    x = list(l[-1:])
    x.extend(l[0:-1])
    return Mul(*x).args


class Tr(Expr):
    """ Generic Trace operation than can trace over:

    a) SymPy matrix
    b) operators
    c) outer products

    Parameters
    ==========
    o : operator, matrix, expr
    i : tuple/list indices (optional)

    Examples
    ========

    # TODO: Need to handle printing

    a) Trace(A+B) = Tr(A) + Tr(B)
    b) Trace(scalar*Operator) = scalar*Trace(Operator)

    >>> from sympy.physics.quantum.trace import Tr
    >>> from sympy import symbols, Matrix
    >>> a, b = symbols('a b', commutative=True)
    >>> A, B = symbols('A B', commutative=False)
    >>> Tr(a*A,[2])
    a*Tr(A)
    >>> m = Matrix([[1,2],[1,1]])
    >>> Tr(m)
    2

    """
    def __new__(cls, *args):
        """ Construct a Trace object.

        Parameters
        ==========
        args = SymPy expression
        indices = tuple/list if indices, optional

        """

        # expect no indices,int or a tuple/list/Tuple
        if (len(args) == 2):
            if not isinstance(args[1], (list, Tuple, tuple)):
                indices = Tuple(args[1])
            else:
                indices = Tuple(*args[1])

            expr = args[0]
        elif (len(args) == 1):
            indices = Tuple()
            expr = args[0]
        else:
            raise ValueError("Arguments to Tr should be of form "
                             "(expr[, [indices]])")

        if isinstance(expr, Matrix):
            return expr.trace()
        elif hasattr(expr, 'trace') and callable(expr.trace):
            #for any objects that have trace() defined e.g numpy
            return expr.trace()
        elif isinstance(expr, Add):
            return Add(*[Tr(arg, indices) for arg in expr.args])
        elif isinstance(expr, Mul):
            c_part, nc_part = expr.args_cnc()
            if len(nc_part) == 0:
                return Mul(*c_part)
            else:
                obj = Expr.__new__(cls, Mul(*nc_part), indices )
                #this check is needed to prevent cached instances
                #being returned even if len(c_part)==0
                return Mul(*c_part)*obj if len(c_part) > 0 else obj
        elif isinstance(expr, Pow):
            if (_is_scalar(expr.args[0]) and
                    _is_scalar(expr.args[1])):
                return expr
            else:
                return Expr.__new__(cls, expr, indices)
        else:
            if (_is_scalar(expr)):
                return expr

            return Expr.__new__(cls, expr, indices)

    @property
    def kind(self):
        expr = self.args[0]
        expr_kind = expr.kind
        return expr_kind.element_kind

    def doit(self, **hints):
        """ Perform the trace operation.

        #TODO: Current version ignores the indices set for partial trace.

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy.physics.quantum.operator import OuterProduct
        >>> from sympy.physics.quantum.spin import JzKet, JzBra
        >>> t = Tr(OuterProduct(JzKet(1,1), JzBra(1,1)))
        >>> t.doit()
        1

        """
        if hasattr(self.args[0], '_eval_trace'):
            return self.args[0]._eval_trace(indices=self.args[1])

        return self

    @property
    def is_number(self):
        # TODO : improve this implementation
        return True

    #TODO: Review if the permute method is needed
    # and if it needs to return a new instance
    def permute(self, pos):
        """ Permute the arguments cyclically.

        Parameters
        ==========

        pos : integer, if positive, shift-right, else shift-left

        Examples
        ========

        >>> from sympy.physics.quantum.trace import Tr
        >>> from sympy import symbols
        >>> A, B, C, D = symbols('A B C D', commutative=False)
        >>> t = Tr(A*B*C*D)
        >>> t.permute(2)
        Tr(C*D*A*B)
        >>> t.permute(-2)
        Tr(C*D*A*B)

        """
        if pos > 0:
            pos = pos % len(self.args[0].args)
        else:
            pos = -(abs(pos) % len(self.args[0].args))

        args = list(self.args[0].args[-pos:] + self.args[0].args[0:-pos])

        return Tr(Mul(*(args)))

    def _hashable_content(self):
        if isinstance(self.args[0], Mul):
            args = _cycle_permute(_rearrange_args(self.args[0].args))
        else:
            args = [self.args[0]]

        return tuple(args) + (self.args[1], )
