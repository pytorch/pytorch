from sympy import permutedims
from sympy.core.numbers import Number
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.tensor.tensor import Tensor, TensExpr, TensAdd, TensMul


class PartialDerivative(TensExpr):
    """
    Partial derivative for tensor expressions.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead
    >>> from sympy.tensor.toperators import PartialDerivative
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> A = TensorHead("A", [L])
    >>> B = TensorHead("B", [L])
    >>> i, j, k = symbols("i j k")

    >>> expr = PartialDerivative(A(i), A(j))
    >>> expr
    PartialDerivative(A(i), A(j))

    The ``PartialDerivative`` object behaves like a tensorial expression:

    >>> expr.get_indices()
    [i, -j]

    Notice that the deriving variables have opposite valence than the
    printed one: ``A(j)`` is printed as covariant, but the index of the
    derivative is actually contravariant, i.e. ``-j``.

    Indices can be contracted:

    >>> expr = PartialDerivative(A(i), A(i))
    >>> expr
    PartialDerivative(A(L_0), A(L_0))
    >>> expr.get_indices()
    [L_0, -L_0]

    The method ``.get_indices()`` always returns all indices (even the
    contracted ones). If only uncontracted indices are needed, call
    ``.get_free_indices()``:

    >>> expr.get_free_indices()
    []

    Nested partial derivatives are flattened:

    >>> expr = PartialDerivative(PartialDerivative(A(i), A(j)), A(k))
    >>> expr
    PartialDerivative(A(i), A(j), A(k))
    >>> expr.get_indices()
    [i, -j, -k]

    Replace a derivative with array values:

    >>> from sympy.abc import x, y
    >>> from sympy import sin, log
    >>> compA = [sin(x), log(x)*y**3]
    >>> compB = [x, y]
    >>> expr = PartialDerivative(A(i), B(j))
    >>> expr.replace_with_arrays({A(i): compA, B(i): compB})
    [[cos(x), 0], [y**3/x, 3*y**2*log(x)]]

    The returned array is indexed by `(i, -j)`.

    Be careful that other SymPy modules put the indices of the deriving
    variables before the indices of the derivand in the derivative result.
    For example:

    >>> expr.get_free_indices()
    [i, -j]

    >>> from sympy import Matrix, Array
    >>> Matrix(compA).diff(Matrix(compB)).reshape(2, 2)
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    >>> Array(compA).diff(Array(compB))
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]

    These are the transpose of the result of ``PartialDerivative``,
    as the matrix and the array modules put the index `-j` before `i` in the
    derivative result. An array read with index order `(-j, i)` is indeed the
    transpose of the same array read with index order `(i, -j)`. By specifying
    the index order to ``.replace_with_arrays`` one can get a compatible
    expression:

    >>> expr.replace_with_arrays({A(i): compA, B(i): compB}, [-j, i])
    [[cos(x), y**3/x], [0, 3*y**2*log(x)]]
    """

    def __new__(cls, expr, *variables):

        # Flatten:
        if isinstance(expr, PartialDerivative):
            variables = expr.variables + variables
            expr = expr.expr

        args, indices, free, dum = cls._contract_indices_for_derivative(
            S(expr), variables)

        obj = TensExpr.__new__(cls, *args)

        obj._indices = indices
        obj._free = free
        obj._dum = dum
        return obj

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    @classmethod
    def _contract_indices_for_derivative(cls, expr, variables):
        variables_opposite_valence = []

        for i in variables:
            if isinstance(i, Tensor):
                i_free_indices = i.get_free_indices()
                variables_opposite_valence.append(
                        i.xreplace({k: -k for k in i_free_indices}))
            elif isinstance(i, Symbol):
                variables_opposite_valence.append(i)

        args, indices, free, dum = TensMul._tensMul_contract_indices(
            [expr] + variables_opposite_valence, replace_indices=True)

        for i in range(1, len(args)):
            args_i = args[i]
            if isinstance(args_i, Tensor):
                i_indices = args[i].get_free_indices()
                args[i] = args[i].xreplace({k: -k for k in i_indices})

        return args, indices, free, dum

    def doit(self, **hints):
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        return obj

    def _expand_partial_derivative(self):
        args, indices, free, dum = self._contract_indices_for_derivative(self.expr, self.variables)

        obj = self.func(*args)
        obj._indices = indices
        obj._free = free
        obj._dum = dum

        result = obj

        if not args[0].free_symbols:
            return S.Zero
        elif isinstance(obj.expr, TensAdd):
            # take care of sums of multi PDs
            result = obj.expr.func(*[
                    self.func(a, *obj.variables)._expand_partial_derivative()
                    for a in result.expr.args])
        elif isinstance(obj.expr, TensMul):
            # take care of products of multi PDs
            if len(obj.variables) == 1:
                # derivative with respect to single variable
                terms = []
                mulargs = list(obj.expr.args)
                for ind in range(len(mulargs)):
                    if not isinstance(sympify(mulargs[ind]), Number):
                        # a number coefficient is not considered for
                        # expansion of PartialDerivative
                        d = self.func(mulargs[ind], *obj.variables)._expand_partial_derivative()
                        terms.append(TensMul(*(mulargs[:ind]
                                               + [d]
                                               + mulargs[(ind + 1):])))
                result = TensAdd.fromiter(terms)
            else:
                # derivative with respect to multiple variables
                # decompose:
                # partial(expr, (u, v))
                # = partial(partial(expr, u).doit(), v).doit()
                result = obj.expr  # init with expr
                for v in obj.variables:
                    result = self.func(result, v)._expand_partial_derivative()
                    # then throw PD on it

        return result

    def _perform_derivative(self):
        result = self.expr
        for v in self.variables:
            if isinstance(result, TensExpr):
                result = result._eval_partial_derivative(v)
            else:
                if v._diff_wrt:
                    result = result._eval_derivative(v)
                else:
                    result = S.Zero
        return result

    def get_indices(self):
        return self._indices

    def get_free_indices(self):
        free = sorted(self._free, key=lambda x: x[1])
        return [i[0] for i in free]

    def _replace_indices(self, repl):
        expr = self.expr.xreplace(repl)
        mirrored = {-k: -v for k, v in repl.items()}
        variables = [i.xreplace(mirrored) for i in self.variables]
        return self.func(expr, *variables)

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        return self.args[1:]

    def _extract_data(self, replacement_dict):
        from .array import derive_by_array, tensorcontraction
        indices, array = self.expr._extract_data(replacement_dict)
        for variable in self.variables:
            var_indices, var_array = variable._extract_data(replacement_dict)
            var_indices = [-i for i in var_indices]
            coeff_array, var_array = zip(*[i.as_coeff_Mul() for i in var_array])
            dim_before = len(array.shape)
            array = derive_by_array(array, var_array)
            dim_after = len(array.shape)
            dim_increase = dim_after - dim_before
            array = permutedims(array, [i + dim_increase for i in range(dim_before)] + list(range(dim_increase)))
            array = array.as_mutable()
            varindex = var_indices[0]
            # Remove coefficients of base vector:
            coeff_index = [0] + [slice(None) for i in range(len(indices))]
            for i, coeff in enumerate(coeff_array):
                coeff_index[0] = i
                array[tuple(coeff_index)] /= coeff
            if -varindex in indices:
                pos = indices.index(-varindex)
                array = tensorcontraction(array, (0, pos+1))
                indices.pop(pos)
            else:
                indices.append(varindex)
        return indices, array
