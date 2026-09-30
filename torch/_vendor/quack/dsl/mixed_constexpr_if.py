# Copyright (c) 2026, Tri Dao.
"""Teach CuTe's AST preprocessor mixed static/dynamic ``if`` tests.

The stock DSL only treats a test that is *entirely* ``const_expr(...)`` as a
compile-time branch; anything else — including ``const_expr(s) and d`` —
lowers to a dynamic if-region.  Kernels that want to statically gate a
dynamic branch *without touching the codegen of the ungated path* are forced
to duplicate the body::

    if const_expr(static_cond):
        if dynamic_cond:
            body
    else:
        body

This module rewrites, before normal CuTe lowering::

    if const_expr(S) and D:        ->    if const_expr(S):
        body                                 if D:
    else:                                        body
        orelse                               else:
                                                 orelse
                                         else:
                                             orelse

    if const_expr(S) or D:         ->    if const_expr(S):
        body                                 body
    else:                                else:
        orelse                               if D:
                                                 body
                                             else:
                                                 orelse

so the static term folds at trace time and the dynamic remainder only
materializes an if-region in the branch where it matters.  The body/orelse is
duplicated in the rewritten AST exactly as in the manual pattern, but only the
branch selected by the constexpr test is ever traced, so codegen is identical
to the hand-written version.

Rules:

- Only a *leading* ``const_expr(...)`` prefix is folded — write the static
  term(s) first.  A trailing ``const_expr`` operand is left in the dynamic
  remainder, where it is a runtime identity (valid, just not folded).
- Multiple leading ``const_expr`` calls fold into one static test combined
  with the same ``and``/``or``.  If *every* operand is a ``const_expr`` call,
  the whole test becomes a single constexpr if (the stock DSL would treat
  that BoolOp as dynamic — a known trap).
- ``elif`` positions are supported: the patch normalizes the whole elif chain
  when it sees the head of the chain, because the DSL recurses into elifs via
  ``create_if_function`` which bypasses ``visit_If``.
- ``not const_expr(S)`` is not recognized; write ``const_expr(not S)``.

Like flash-attention's block-sparse loop patch, this intentionally
monkey-patches the installed CuTe DSL at import time rather than editing
site-packages; ``quack.dsl.__init__`` installs it before any quack kernel is
preprocessed.
"""

import ast
from copy import deepcopy

import cutlass.base_dsl.ast_preprocessor as ast_preprocessor

__all__ = ["rewrite_mixed_constexpr_if"]


def _is_constexpr_call(node: ast.expr) -> bool:
    """True for a ``const_expr(x)`` / ``<mod>.const_expr(x)`` call with one arg."""
    if not isinstance(node, ast.Call) or node.keywords or len(node.args) != 1:
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr == "const_expr"
    return isinstance(func, ast.Name) and func.id == "const_expr"


def rewrite_mixed_constexpr_if(node: ast.If) -> ast.If | None:
    """Rewrite an ``if`` whose BoolOp test starts with ``const_expr(...)``.

    Returns the rewritten node (a constexpr ``if`` whose branches hold the
    dynamic remainder as documented in the module docstring), or None when the
    test does not match — callers then fall through to stock DSL handling.
    """
    test = node.test
    if not isinstance(test, ast.BoolOp):
        return None
    static_prefix = []
    for value in test.values:
        if not _is_constexpr_call(value):
            break
        static_prefix.append(value)
    if not static_prefix:
        return None

    if len(static_prefix) == 1:
        static_test = static_prefix[0]
    else:
        static_test = ast.Call(
            func=deepcopy(static_prefix[0].func),
            args=[ast.BoolOp(op=test.op, values=[call.args[0] for call in static_prefix])],
            keywords=[],
        )

    dynamic_rest = test.values[len(static_prefix) :]
    if not dynamic_rest:
        # Every operand was const_expr: a single fully-static if.
        rewritten = ast.If(test=static_test, body=node.body, orelse=node.orelse)
    else:
        dynamic_test = (
            dynamic_rest[0]
            if len(dynamic_rest) == 1
            else ast.BoolOp(op=test.op, values=dynamic_rest)
        )
        if isinstance(test.op, ast.And):
            # S and D: S picks between `if D: body else: orelse` and `orelse`.
            inner = ast.If(test=dynamic_test, body=node.body, orelse=deepcopy(node.orelse))
            rewritten = ast.If(test=static_test, body=[inner], orelse=node.orelse)
        else:
            # S or D: S picks between `body` and `if D: body else: orelse`.
            inner = ast.If(test=dynamic_test, body=deepcopy(node.body), orelse=node.orelse)
            rewritten = ast.If(test=static_test, body=node.body, orelse=[inner])
        ast.copy_location(inner, node)
    ast.copy_location(rewritten, node)
    return ast.fix_missing_locations(rewritten)


def _rewrite_elif_chain(node: ast.If) -> ast.If:
    """Normalize an if/elif chain tail-first so copied orelse trees are normalized."""
    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node.orelse[0] = _rewrite_elif_chain(node.orelse[0])
    return rewrite_mixed_constexpr_if(node) or node


def _install_mixed_constexpr_if_ast_patch() -> None:
    preprocessor_cls = ast_preprocessor.DSLPreprocessor
    if getattr(preprocessor_cls, "_quack_mixed_constexpr_if_patch", False):
        return

    original_visit_if = preprocessor_cls.visit_If

    def visit_if(self, node: ast.If):
        # The DSL recurses into elifs via create_if_function, which bypasses
        # visit_If, so normalize the whole chain while we still hold its head.
        # Work tail-first: rewriting an earlier mixed elif duplicates its
        # orelse, and both copies must already contain the normalized tail.
        return original_visit_if(self, _rewrite_elif_chain(node))

    preprocessor_cls._quack_original_visit_If = original_visit_if
    preprocessor_cls.visit_If = visit_if
    preprocessor_cls._quack_mixed_constexpr_if_patch = True


_install_mixed_constexpr_if_ast_patch()
