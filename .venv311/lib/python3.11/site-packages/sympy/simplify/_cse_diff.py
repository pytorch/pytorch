"""Module for differentiation using CSE."""

from sympy import cse, Matrix, Derivative, MatrixBase
from sympy.utilities.iterables import iterable


def _remove_cse_from_derivative(replacements, reduced_expressions):
    """
    This function is designed to postprocess the output of a common subexpression
    elimination (CSE) operation. Specifically, it removes any CSE replacement
    symbols from the arguments of ``Derivative`` terms in the expression. This
    is necessary to ensure that the forward Jacobian function correctly handles
    derivative terms.

    Parameters
    ==========

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation.

    reduced_expressions : list of SymPy expressions
        The reduced expressions with all the replacements from the
        replacements list above.

    Returns
    =======

    processed_replacements : list of (Symbol, expression) pairs
        Processed replacement list, in the same format of the
        ``replacements`` input list.

    processed_reduced : list of SymPy expressions
        Processed reduced list, in the same format of the
        ``reduced_expressions`` input list.
    """

    def traverse(node, repl_dict):
        if isinstance(node, Derivative):
            return replace_all(node, repl_dict)
        if not node.args:
            return node
        new_args = [traverse(arg, repl_dict) for arg in node.args]
        return node.func(*new_args)

    def replace_all(node, repl_dict):
        result = node
        while True:
            free_symbols = result.free_symbols
            symbols_dict = {k: repl_dict[k] for k in free_symbols if k in repl_dict}
            if not symbols_dict:
                break
            result = result.xreplace(symbols_dict)
        return result

    repl_dict = dict(replacements)
    processed_replacements = [
        (rep_sym, traverse(sub_exp, repl_dict))
        for rep_sym, sub_exp in replacements
    ]
    processed_reduced = [
        red_exp.__class__([traverse(exp, repl_dict) for exp in red_exp])
        for red_exp in reduced_expressions
    ]

    return processed_replacements, processed_reduced


def _forward_jacobian_cse(replacements, reduced_expr, wrt):
    """
    Core function to compute the Jacobian of an input Matrix of expressions
    through forward accumulation. Takes directly the output of a CSE operation
    (replacements and reduced_expr), and an iterable of variables (wrt) with
    respect to which to differentiate the reduced expression and returns the
    reduced Jacobian matrix and the ``replacements`` list.

    The function also returns a list of precomputed free symbols for each
    subexpression, which are useful in the substitution process.

    Parameters
    ==========

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation.

    reduced_expr : list of SymPy expressions
        The reduced expressions with all the replacements from the
        replacements list above.

    wrt : iterable
        Iterable of expressions with respect to which to compute the
        Jacobian matrix.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation. Compared to the input replacement list,
        the output one doesn't contain replacement symbols inside
        ``Derivative``'s arguments.

    jacobian : list of SymPy expressions
        The list only contains one element, which is the Jacobian matrix with
        elements in reduced form (replacement symbols are present).

    precomputed_fs: list
        List of sets, which store the free symbols present in each sub-expression.
        Useful in the substitution process.
    """

    if not isinstance(reduced_expr[0], MatrixBase):
        raise TypeError("``expr`` must be of matrix type")

    if not (reduced_expr[0].shape[0] == 1 or reduced_expr[0].shape[1] == 1):
        raise TypeError("``expr`` must be a row or a column matrix")

    if not iterable(wrt):
        raise TypeError("``wrt`` must be an iterable of variables")

    elif not isinstance(wrt, MatrixBase):
        wrt = Matrix(wrt)

    if not (wrt.shape[0] == 1 or wrt.shape[1] == 1):
        raise TypeError("``wrt`` must be a row or a column matrix")

    replacements, reduced_expr = _remove_cse_from_derivative(replacements, reduced_expr)

    if replacements:
        rep_sym, sub_expr = map(Matrix, zip(*replacements))
    else:
        rep_sym, sub_expr = Matrix([]), Matrix([])

    l_sub, l_wrt, l_red = len(sub_expr), len(wrt), len(reduced_expr[0])

    f1 = reduced_expr[0].__class__.from_dok(l_red, l_wrt,
        {
            (i, j): diff_value
            for i, r in enumerate(reduced_expr[0])
            for j, w in enumerate(wrt)
            if (diff_value := r.diff(w)) != 0
        },
    )

    if not replacements:
        return [], [f1], []

    f2 = Matrix.from_dok(l_red, l_sub,
        {
            (i, j): diff_value
            for i, (r, fs) in enumerate([(r, r.free_symbols) for r in reduced_expr[0]])
            for j, s in enumerate(rep_sym)
            if s in fs and (diff_value := r.diff(s)) != 0
        },
    )

    rep_sym_set = set(rep_sym)
    precomputed_fs = [s.free_symbols & rep_sym_set for s in sub_expr ]

    c_matrix = Matrix.from_dok(1, l_wrt,
                               {(0, j): diff_value for j, w in enumerate(wrt)
                                if (diff_value := sub_expr[0].diff(w)) != 0})

    for i in range(1, l_sub):

        bi_matrix = Matrix.from_dok(1, i,
                                    {(0, j): diff_value for j in range(i + 1)
                                     if rep_sym[j] in precomputed_fs[i]
                                     and (diff_value := sub_expr[i].diff(rep_sym[j])) != 0})

        ai_matrix = Matrix.from_dok(1, l_wrt,
                                    {(0, j): diff_value for j, w in enumerate(wrt)
                                     if (diff_value := sub_expr[i].diff(w)) != 0})

        if bi_matrix._rep.nnz():
            ci_matrix = bi_matrix.multiply(c_matrix).add(ai_matrix)
            c_matrix = Matrix.vstack(c_matrix, ci_matrix)
        else:
            c_matrix = Matrix.vstack(c_matrix, ai_matrix)

    jacobian = f2.multiply(c_matrix).add(f1)
    jacobian = [reduced_expr[0].__class__(jacobian)]

    return replacements, jacobian, precomputed_fs


def _forward_jacobian_norm_in_cse_out(expr, wrt):
    """
    Function to compute the Jacobian of an input Matrix of expressions through
    forward accumulation. Takes a sympy Matrix of expressions (expr) as input
    and an iterable of variables (wrt) with respect to which to compute the
    Jacobian matrix. The matrix is returned in reduced form (containing
    replacement symbols) along with the ``replacements`` list.

    The function also returns a list of precomputed free symbols for each
    subexpression, which are useful in the substitution process.

    Parameters
    ==========

    expr : Matrix
        The vector to be differentiated.

    wrt : iterable
        The vector with respect to which to perform the differentiation.
        Can be a matrix or an iterable of variables.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        Replacement symbols and relative common subexpressions that have been
        replaced during a CSE operation. The output replacement list doesn't
        contain replacement symbols inside ``Derivative``'s arguments.

    jacobian : list of SymPy expressions
        The list only contains one element, which is the Jacobian matrix with
        elements in reduced form (replacement symbols are present).

    precomputed_fs: list
        List of sets, which store the free symbols present in each
        sub-expression. Useful in the substitution process.
    """

    replacements, reduced_expr = cse(expr)
    replacements, jacobian, precomputed_fs = _forward_jacobian_cse(replacements, reduced_expr, wrt)

    return replacements, jacobian, precomputed_fs


def _forward_jacobian(expr, wrt):
    """
    Function to compute the Jacobian of an input Matrix of expressions through
    forward accumulation. Takes a sympy Matrix of expressions (expr) as input
    and an iterable of variables (wrt) with respect to which to compute the
    Jacobian matrix.

    Explanation
    ===========

    Expressions often contain repeated subexpressions. Using a tree structure,
    these subexpressions are duplicated and differentiated multiple times,
    leading to inefficiency.

    Instead, if a data structure called a directed acyclic graph (DAG) is used
    then each of these repeated subexpressions will only exist a single time.
    This function uses a combination of representing the expression as a DAG and
    a forward accumulation algorithm (repeated application of the chain rule
    symbolically) to more efficiently calculate the Jacobian matrix of a target
    expression ``expr`` with respect to an expression or set of expressions
    ``wrt``.

    Note that this function is intended to improve performance when
    differentiating large expressions that contain many common subexpressions.
    For small and simple expressions it is likely less performant than using
    SymPy's standard differentiation functions and methods.

    Parameters
    ==========

    expr : Matrix
        The vector to be differentiated.

    wrt : iterable
        The vector with respect to which to do the differentiation.
        Can be a matrix or an iterable of variables.

    See Also
    ========

    Direct Acyclic Graph : https://en.wikipedia.org/wiki/Directed_acyclic_graph
    """

    replacements, reduced_expr = cse(expr)

    if replacements:
        rep_sym, _ = map(Matrix, zip(*replacements))
    else:
        rep_sym = Matrix([])

    replacements, jacobian, precomputed_fs = _forward_jacobian_cse(replacements, reduced_expr, wrt)

    if not replacements: return jacobian[0]

    sub_rep = dict(replacements)
    for i, ik in enumerate(precomputed_fs):
        sub_dict = {j: sub_rep[j] for j in ik}
        sub_rep[rep_sym[i]] = sub_rep[rep_sym[i]].xreplace(sub_dict)

    return jacobian[0].xreplace(sub_rep)
