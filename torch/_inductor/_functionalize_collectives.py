"""Rewrite raw ``c10d.{op}_`` inplace ops produced by ``make_fx`` into
``_c10d_functional.{op}`` + ``wait_tensor`` + ``aten.copy_`` form so
``standalone_compile`` (and Inductor's collective machinery underneath) can
recognize and lower them.

The structure separates concerns:
  * Arg resolution (PG, ReduceOp) lives in ``_resolve_*``.
  * The functional + wait + copy_ chain construction lives in
    ``_emit_collective_chain`` (used by every per-op rewrite).
  * Use redirection lives in ``_redirect_*_work_uses`` keyed by output schema.
  * Per-op rewrites (``_rewrite_<op>_``) glue these together.
  * The driver ``_functionalize_inplace_collectives`` walks the graph,
    dispatches by target, and runs final cleanup.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import torch
from torch.fx.graph_module import _del_attr, _get_attr


def _resolve_reduce_op_str(
    gm: torch.fx.GraphModule, arg: torch.fx.node.Argument
) -> str:
    """Get the lower-case op string (``"sum"``/``"avg"``/...) for a c10d
    ReduceOp ``get_attr`` arg, converting from the torchbind ScriptObject
    form if needed.

    Only the ``get_attr`` (constant-baked) shape is supported today. Any
    other producer (placeholder, call_function, ...) would require
    ``_c10d_functional.*`` to accept ReduceOp directly (today it only
    takes the string form); deferred to a follow-up PR.
    """
    import torch.distributed as dist
    from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

    if not isinstance(arg, torch.fx.Node) or arg.op != "get_attr":
        raise NotImplementedError(
            f"ReduceOp arg must be a constant ``get_attr``; got {arg!r}."
        )
    reduce_op = _get_attr(gm, arg.target)  # type: ignore[arg-type]
    if isinstance(reduce_op, torch.ScriptObject):
        reduce_op = dist.ReduceOp.RedOpType(reduce_op.op())  # type: ignore[attr-defined]
    return REDUCE_OP_TO_STR[reduce_op]


def _emit_collective_chain(
    gm: torch.fx.GraphModule,
    before: torch.fx.Node,
    input_t: torch.fx.Node,
    output_t: torch.fx.Node,
    functional_target: Any,
    extra_args: tuple[Any, ...],
    pg_arg: torch.fx.Node,
    custom: Any,
) -> torch.fx.Node:
    """Insert ``_c10d_functional.<op>(input, *extra_args, pg_arg)`` ->
    ``wait_tensor`` -> ``aten.copy_(output, wait)`` immediately before
    ``before`` and return the ``wait_tensor`` node.

    Propagates ``output_t.meta["val"]`` onto every new node — every functional
    collective we rewrite to produces a tensor with the same shape/dtype as
    ``output_t``, and ``aten.copy_`` returns ``output_t``. Also carries the
    rewritten node's ``custom`` annotation forward so region scooping continues
    to work.
    """
    val = output_t.meta.get("val")
    with gm.graph.inserting_before(before):
        ar = gm.graph.call_function(functional_target, (input_t, *extra_args, pg_arg))
        wait = gm.graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default, (ar,)
        )
        copy_ = gm.graph.call_function(torch.ops.aten.copy_.default, (output_t, wait))
        if val is not None:
            ar.meta["val"] = wait.meta["val"] = copy_.meta["val"] = val
        if custom is not None:
            ar.meta["custom"] = wait.meta["custom"] = copy_.meta["custom"] = custom
    return wait


def _redirect_inplace_collective_uses(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    wait_nodes: list[torch.fx.Node],
) -> None:
    """Re-route uses for an inplace c10d op with output schema
    ``(Tensor[], Work)`` — e.g. ``allreduce_``, ``broadcast_``,
    ``reduce_scatter_``, ``alltoall_``.

    ``make_fx`` lowers tensor consumers into ``getitem(node, 0)[i]``; replace
    each with ``wait_nodes[i]``. The work-handle ``getitem(node, 1)`` is erased
    if unused (live users raise — functional collectives have no Work object
    to forward to, so we can't preserve that synchronization).

    TODO: extend to ``(Tensor, Work)`` schemas (``_allgather_base_``,
    ``_reduce_scatter_base_``) when those rewrites are added. There the
    ``getitem(node, 0)`` is itself the tensor consumer (no inner
    ``getitem`` chain), so the inner loop below would need to fall back to
    ``use.replace_all_uses_with(wait_nodes[0])`` when ``use`` has no inner
    ``getitem`` users.
    """
    for use in list(node.users):
        if use.op != "call_function" or use.target is not operator.getitem:
            raise AssertionError(
                f"Cannot functionalize {node.target}: unexpected use {use} "
                f"(op={use.op!r}, target={use.target!r}). The c10d inplace "
                f"output is a tuple consumed via ``getitem`` in ``make_fx`` "
                f"graphs; non-getitem users are not supported by this pass."
            )
        if use.args[1] == 0:
            for sub_use in list(use.users):
                if (
                    sub_use.op != "call_function"
                    or sub_use.target is not operator.getitem
                ):
                    raise AssertionError(
                        f"Cannot functionalize {node.target}: tensor list "
                        f"consumed by non-getitem user {sub_use}. Extend "
                        f"_redirect_inplace_collective_uses to support this "
                        f"schema if needed."
                    )
                sub_use.replace_all_uses_with(wait_nodes[sub_use.args[1]])  # type: ignore[index]
                gm.graph.erase_node(sub_use)
            gm.graph.erase_node(use)
        elif use.args[1] == 1:
            if use.users:
                raise AssertionError(
                    f"Cannot functionalize {node.target}: Work handle has "
                    f"{len(use.users)} live use(s) ({list(use.users)}). "
                    f"Functional collectives express synchronization via "
                    f"``wait_tensor`` on the data tensor, so the Work object "
                    f"has no equivalent to forward to."
                )
            gm.graph.erase_node(use)


def _rewrite_allreduce_(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Reference rewrite — use this as a template for new collectives.

    Schema: ``c10d::allreduce_(Tensor[] tensors, ProcessGroup pg, ReduceOp op,
    Tensor? sparse_indices, bool async_op, int timeout=-1) -> (Tensor[], Work)``.

    Before::

        allreduce_ = c10d.allreduce_([t0, t1], pg, reduce_op, None, False)
        getitem    = allreduce_[0]
        getitem_1  = getitem[0]      # consumer of t0 post-allreduce
        getitem_2  = getitem[1]      # consumer of t1 post-allreduce
        ...        = allreduce_[1]   # work handle (unused)

    After (per input tensor; ``pg`` is the original ``get_attr`` Node — a
    later pass will bake it into a ``group_name`` string)::

        ar_i = _c10d_functional.all_reduce(t_i, op_str, pg)
        wait_i = _c10d_functional.wait_tensor(ar_i)
        _ = aten.copy_(t_i, wait_i)  # preserves inplace semantics
        # downstream uses of ``getitem_<i>`` are redirected to ``wait_i``
    """
    tensors = node.args[0]
    pg_arg = node.args[1]
    if not isinstance(tensors, (list, tuple)):
        raise AssertionError(f"expected ``Tensor[]`` arg list, got {type(tensors)}")
    if not isinstance(pg_arg, torch.fx.Node):
        raise AssertionError(f"expected ProcessGroup as fx.Node, got {type(pg_arg)}")
    op_str = _resolve_reduce_op_str(gm, node.args[2])
    custom = node.meta.get("custom")
    waits = []
    for t in tensors:
        if not isinstance(t, torch.fx.Node):
            raise AssertionError(f"expected tensor entry as fx.Node, got {type(t)}")
        waits.append(
            _emit_collective_chain(
                gm,
                node,
                t,
                t,
                torch.ops._c10d_functional.all_reduce.default,
                (op_str,),
                pg_arg,
                custom,
            )
        )
    _redirect_inplace_collective_uses(gm, node, waits)


_InplaceCollectiveRewrite = Callable[[torch.fx.GraphModule, torch.fx.Node], None]


def _inplace_c10d_rewrites() -> dict[Any, _InplaceCollectiveRewrite] | None:
    """Map inplace ``c10d.{op}_`` op targets to per-op rewrite functions.

    Returns ``None`` if ``torch.distributed`` is not available.

    To support a new collective: write ``_rewrite_<op>_`` (typically a few
    lines using ``_resolve_reduce_op_str`` / ``_emit_collective_chain`` /
    ``_redirect_inplace_collective_uses``), then register it here.
    """
    if not torch.distributed.is_available():
        return None
    return {
        torch.ops.c10d.allreduce_.default: _rewrite_allreduce_,
    }


def _functionalize_inplace_collectives(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Rewrite raw ``torch.ops.c10d.{op}_`` inplace calls in ``gm`` into
    ``_c10d_functional.{op}`` + ``wait_tensor`` + ``aten.copy_`` form so
    Inductor's collective machinery can recognize and lower them.
    """
    rewrites = _inplace_c10d_rewrites()
    if rewrites is None:
        return gm

    found = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        rewrite = rewrites.get(node.target, None)
        if rewrite is None:
            continue
        rewrite(gm, node)
        # Snapshot ``get_attr`` args before erasing the rewritten node.
        # Inplace c10d ops are impure; ``eliminate_dead_code`` keeps them
        # alive even with no users, so erase explicitly. Then drop any
        # ``get_attr`` arg that became orphan (e.g. the ReduceOp whose
        # value got baked into ``op_str``) along with its backing module
        # attribute. Live ones (e.g. the ProcessGroup, still referenced
        # by the new ``_c10d_functional`` call) are kept.
        get_attr_args = [
            a for a in node.args if isinstance(a, torch.fx.Node) and a.op == "get_attr"
        ]
        gm.graph.erase_node(node)
        for a in get_attr_args:
            if not a.users:
                _del_attr(gm, a.target)  # type: ignore[arg-type]
                gm.graph.erase_node(a)
        found = True

    if found:
        gm.recompile()
    return gm
