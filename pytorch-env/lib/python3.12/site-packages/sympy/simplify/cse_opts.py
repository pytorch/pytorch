""" Optimizations of the expression tree representation for better CSE
opportunities.
"""
from sympy.core import Add, Basic, Mul
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.traversal import preorder_traversal


def sub_pre(e):
    """ Replace y - x with -(x - y) if -1 can be extracted from y - x.
    """
    # replacing Add, A, from which -1 can be extracted with -1*-A
    adds = [a for a in e.atoms(Add) if a.could_extract_minus_sign()]
    reps = {}
    ignore = set()
    for a in adds:
        na = -a
        if na.is_Mul:  # e.g. MatExpr
            ignore.add(a)
            continue
        reps[a] = Mul._from_args([S.NegativeOne, na])

    e = e.xreplace(reps)

    # repeat again for persisting Adds but mark these with a leading 1, -1
    # e.g. y - x -> 1*-1*(x - y)
    if isinstance(e, Basic):
        negs = {}
        for a in sorted(e.atoms(Add), key=default_sort_key):
            if a in ignore:
                continue
            if a in reps:
                negs[a] = reps[a]
            elif a.could_extract_minus_sign():
                negs[a] = Mul._from_args([S.One, S.NegativeOne, -a])
        e = e.xreplace(negs)
    return e


def sub_post(e):
    """ Replace 1*-1*x with -x.
    """
    replacements = []
    for node in preorder_traversal(e):
        if isinstance(node, Mul) and \
            node.args[0] is S.One and node.args[1] is S.NegativeOne:
            replacements.append((node, -Mul._from_args(node.args[2:])))
    for node, replacement in replacements:
        e = e.xreplace({node: replacement})

    return e
