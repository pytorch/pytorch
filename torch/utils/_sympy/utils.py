import sympy


def is_integer_scalar_expr(expr: sympy.Expr) -> bool:
    """
    Returns True if the expression is a scalar integer expression
    that can be used as a scalar index. This is meant to differentiate
    between value like 3 and Add(Identity(3), Identity(0)), which would
    both be "scalar" from a variable like xindex.
    """
    stack = [expr]
    while stack:
        current = stack.pop()
        if current.args:
            stack.extend(current.args)
        elif not isinstance(current, sympy.Integer):
            return False
    return True
