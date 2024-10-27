"""Strategies to Traverse a Tree."""
from sympy.strategies.util import basic_fns
from sympy.strategies.core import chain, do_one


def top_down(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the top nodes first."""
    return chain(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))


def bottom_up(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the bottom nodes first."""
    return chain(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)


def top_down_once(rule, fns=basic_fns):
    """Apply a rule down a tree - stop on success."""
    return do_one(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))


def bottom_up_once(rule, fns=basic_fns):
    """Apply a rule up a tree - stop on success."""
    return do_one(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)


def sall(rule, fns=basic_fns):
    """Strategic all - apply rule to args."""
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    def all_rl(expr):
        if leaf(expr):
            return expr
        else:
            args = map(rule, children(expr))
            return new(op(expr), *args)

    return all_rl
