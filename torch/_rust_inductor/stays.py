# mypy: allow-untyped-defs
"""What a lowering may take for granted of the memory of a graph's tensors:
whether the graph writes to a tensor it did not make, and whether every
tensor it makes stays inside the tensor it is made of. Both go by the
operators' schemas, and say "it may" for whatever has none.

``writes(gm)``: whether a node of ``gm`` may write to one of its arguments,
or change its sizes, strides or offset: an operator whose schema marks an
argument as written, one of UNDECLARED, or a call of anything without a
schema (but the taking of one result of a node that has several). An
operator with a schema that marks no argument, and is not listed, is taken
to write to none: what follows goes by the schemas too. AOT Autograd puts
overlapping inputs of which one is written under one base and builds each
from it at the offset the traced tensor had in its storage: what is made of
a graph that writes is for those offsets (see exec/overlap.rs).

What a lowering makes of a graph takes two of the graph's tensors that are in
one storage without overlapping for independent (AOT Autograd gives aliased
inputs that do not overlap no common base). They are, while no node makes a
tensor over memory its argument is not over: ``as_strided`` past the last
element of its argument, ``set_``, ``resize_`` of a view.

``leaves(gm)`` names the first node of ``gm`` that may do so; None if none
can. It goes by the operator's schema:

- no result aliases an argument: the result is new memory;
- every result that aliases is an argument the operator writes in place
  (``add_``), the operator is not tagged ``inplace_view`` (it changes no
  sizes, strides or offset) and the argument is not an ``out=`` (which is
  resized to the result): the result is the argument where it was;
- else (a view, an in-place change of sizes, strides or offset): only an
  operator of STAYS, whose result is over elements of its argument only.

Every other node may leave: an operator that is not listed, an ``out=``, an
operator outside ``aten``, a call of anything without a schema (but the
taking of one result of a node that has several, and a call whose recorded
value holds no tensor: arithmetic on numbers). Imports nothing of this
package.
"""

import operator

import torch
import torch.utils._pytree as pytree


# The aten operators with a result that is a view of an argument, or that
# change an argument's sizes, strides or offset in place, whose result has
# no element the argument has not.
STAYS = frozenset(
    {
        "_autocast_to_full_precision",
        "_autocast_to_reduced_precision",
        "_conj",
        "_neg_view",
        "_reshape_alias",
        "adjoint",
        "alias",
        "broadcast_to",
        "chunk",
        "conj",
        "contiguous",
        "detach",
        "detach_",
        "diagonal",
        "dsplit",
        "expand",
        "expand_as",
        "flatten",
        "hsplit",
        "imag",
        "lift_fresh",
        "linalg_diagonal",
        "mH",
        "mT",
        "matrix_H",
        "moveaxis",
        "movedim",
        "narrow",
        "numpy_T",
        "permute",
        "positive",
        "ravel",
        "real",
        "reshape",
        "reshape_as",
        "resolve_conj",
        "resolve_neg",
        "select",
        "slice",
        "split",
        "split_with_sizes",
        "squeeze",
        "squeeze_",
        "swapaxes",
        "swapaxes_",
        "swapdims",
        "swapdims_",
        "t",
        "t_",
        "tensor_split",
        "to",
        "transpose",
        "transpose_",
        "unbind",
        "unflatten",
        "unfold",
        "unsqueeze",
        "unsqueeze_",
        "view",
        "view_as",
        "view_as_complex",
        "view_as_real",
        "vsplit",
    }
)


_NUMBERS = (int, float, bool, torch.SymInt, torch.SymFloat, torch.SymBool)


def _schemas(gm):
    """The nodes of ``gm`` that call something, each with the schema of the
    operator it calls (None: it calls no operator)."""
    for n in gm.graph.nodes:
        if n.op in ("placeholder", "output", "get_attr"):
            continue
        if n.op == "call_function" and n.target is operator.getitem:
            continue
        yield n, getattr(n.target, "_schema", None) if n.op == "call_function" else None


# The aten operators that write to an argument their schema does not mark:
# the running statistics of a normalization in training mode.
UNDECLARED = frozenset(
    {
        "_batch_norm_impl_index",
        "batch_norm",
        "cudnn_batch_norm",
        "instance_norm",
        "miopen_batch_norm",
        "native_batch_norm",
    }
)


def writes(gm):
    return any(
        schema is None
        or schema.name.partition("::")[2] in UNDECLARED
        or any(
            a.alias_info is not None and a.alias_info.is_write for a in schema.arguments
        )
        for _, schema in _schemas(gm)
    )


def leaves(gm):
    for n, schema in _schemas(gm):
        if schema is None:
            made = pytree.tree_leaves(n.meta["val"]) if "val" in n.meta else []
            if made and all(isinstance(v, _NUMBERS) for v in made):
                continue
            return f"its graph calls {n.target!r}, which is no operator"
        aliasing = [r.alias_info for r in schema.returns if r.alias_info is not None]
        if not aliasing:
            continue
        space, _, name = schema.name.partition("::")
        in_place = all(a.is_write for a in aliasing) and not any(
            a.is_out for a in schema.arguments
        )
        if in_place and torch.Tag.inplace_view not in n.target.tags:
            continue
        if space != "aten" or name not in STAYS:
            return (
                f"its graph has {schema.name}, whose result may be over memory "
                "its argument is not over"
            )
    return None
