# mypy: allow-untyped-defs
"""What a lowering may take for granted of the memory of a graph's tensors:
whether the graph writes to a tensor it did not make, and whether it names
a place in a storage. Both go by the operators' schemas, and say "it may"
for whatever has none.

``writes(gm)``: whether a node of ``gm`` may write to one of its arguments,
or change its sizes, strides or offset: an operator whose schema marks an
argument as written, one of UNDECLARED, or a call of anything without a
schema (but the taking of one result of a node that has several). An
operator with a schema that marks no argument, and is not listed, is taken
to write to none: what follows goes by the schemas too. AOT Autograd puts
overlapping inputs of which one is written under one base and builds each
from it at the offset the traced tensor had in its storage: what is made of
a graph that writes is for those offsets (see exec/overlap.rs).

What a lowering makes of a graph is handed the graph's tensors, each by its
own elements: where a tensor starts in its storage, and what else is in
that storage, is not the lowering's to know. Inductor turns the offset of
``as_strided(a, (8,), (1,), 0)``, which is from the start of the storage,
into a distance from its input's first element; its CUDA graphs run on a
copy of an input's elements, where nothing is behind the last; AOT Autograd
takes two inputs of one storage that do not overlap for independent. A
graph can be lowered while no node of it is handed, or takes, a place in a
storage outside the elements of its arguments.

``written(func, args, kwargs)``: the tensors a call of an operator may
write, by the same two sources (the schema's marks, UNDECLARED): what a
trace keeps to take back (undo.py).

``names(gm)`` names the first node of ``gm`` that may; None if none can. A
node cannot only if all of:

- it calls an operator (it has a schema), or it is the taking of one result
  of a node that has several, or its recorded value holds numbers only
  (arithmetic on numbers);
- no argument of the operator is called ``storage_offset`` or is a
  ``Storage`` (``as_strided``, ``as_strided_``, ``as_strided_copy``,
  ``as_strided_scatter``, ``set_``), and the operator's name has neither
  ``storage`` nor ``data_ptr`` in it (what returns where a tensor is:
  ``sym_storage_offset``);
- of its results, by its schema: none aliases an argument (new memory); or
  every one that does is an argument the operator writes in place
  (``add_``), the operator is not tagged ``inplace_view`` (it changes no
  sizes, strides or offset) and the argument is not an ``out=`` (which is
  resized to the result); or the operator is one of STAYS, the views whose
  result is over elements of its argument only.

Every other node may: ``as_strided`` whatever its offset, ``set_``,
``resize_``, ``_reshape_alias``, an aliasing operator that is not listed,
an ``out=``, an operator outside ``aten`` with an aliasing result, a call
of anything without a schema that made a tensor. Imports nothing of this
package.
"""

import operator

import torch
import torch.utils._pytree as pytree


# The aten operators with a result that is a view of an argument, or that
# change an argument's sizes, strides or offset in place, whose result has
# no element the argument has not. (Not `_reshape_alias`: it takes the sizes
# and strides it is given, like `as_strided`.)
STAYS = frozenset(
    {
        "_autocast_to_full_precision",
        "_autocast_to_reduced_precision",
        "_conj",
        "_neg_view",
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
# MEASURED: those with a line in the table of measured calls whose column W
# has a write no schema declares (exec/rules/torch_members.tsv has the lines
# measured on the CPU and on CUDA; the two others were measured on the CPU).
MEASURED = ("batch_norm", "instance_norm", "native_batch_norm")
# UNMEASURED: no line there. `batch_norm_update_stats` changes the statistics
# it is handed (a call shows it); `_batch_norm_impl_index`, `cudnn_batch_norm`
# and `miopen_batch_norm` are what `batch_norm` hands them on to; the two
# `batch_norm_gather_stats` are SyncBatchNorm's, whose CUDA kernel stores into
# them.
UNMEASURED = (
    "_batch_norm_impl_index",
    "batch_norm_gather_stats",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_update_stats",
    "cudnn_batch_norm",
    "miopen_batch_norm",
)
UNDECLARED = frozenset(MEASURED + UNMEASURED)


def unmarked(n):
    """Whether the node calls one of UNDECLARED: an operator that writes to
    an argument its schema does not mark. Such a call is no dead code when
    nothing reads its results, and the order of reads and writes that the
    schemas of a graph tell is not the program's where it is."""
    schema = getattr(n.target, "_schema", None) if n.op == "call_function" else None
    return schema is not None and schema.name.partition("::")[2] in UNDECLARED


# The arguments that an operator of UNDECLARED writes and does not mark.
STATISTICS = ("running_mean", "running_var")
# By operator: the places of the arguments it may write, each (position,
# name, whether the schema marks it, whether it is an `out=`), and the place
# of the `training` of one of UNDECLARED.
_places: dict = {}


def written(func, args, kwargs):
    """The tensors that a call of the operator ``func`` with these arguments
    may write, of those it is handed: (tensor, declared, out) each.
    ``declared``: the schema marks the argument as written. Not declared:
    a statistic of one of UNDECLARED, unless the call hands a ``training``
    that is False (calls of ``batch_norm`` that are not in training were
    measured to write nothing). ``out``: the argument is an ``out=``, which
    the operator resizes to its result before it writes it."""
    found = _places.get(func)
    if found is None:
        schema = func._schema
        listed = schema.name.partition("::")[2] in UNDECLARED
        places, training = [], None
        for k, a in enumerate(schema.arguments):
            declared = a.alias_info is not None and a.alias_info.is_write
            if declared or (listed and a.name in STATISTICS):
                places.append((k, a.name, declared, a.is_out))
            elif listed and a.name == "training":
                training = (k, a.name)
        found = _places[func] = (places, training)
    places, training = found

    def handed(k, name):
        return args[k] if k < len(args) else kwargs.get(name)

    idle = training is not None and handed(*training) is False
    tensors = []
    for k, name, declared, out in places:
        if idle and not declared:
            continue
        x = handed(k, name)
        for t in x if isinstance(x, (list, tuple)) else (x,):
            if isinstance(t, torch.Tensor):
                tensors.append((t, declared, out))
    return tensors


def writes(gm):
    return any(
        schema is None
        or schema.name.partition("::")[2] in UNDECLARED
        or any(
            a.alias_info is not None and a.alias_info.is_write for a in schema.arguments
        )
        for _, schema in _schemas(gm)
    )


def names(gm):
    for n, schema in _schemas(gm):
        if schema is None:
            made = pytree.tree_leaves(n.meta["val"]) if "val" in n.meta else []
            if made and all(isinstance(v, _NUMBERS) for v in made):
                continue
            return f"its graph calls {n.target!r}, which is no operator"
        space, _, name = schema.name.partition("::")
        if (
            "storage" in name
            or "data_ptr" in name
            or any(
                a.name == "storage_offset" or str(a.type) == "Storage"
                for a in schema.arguments
            )
        ):
            return (
                f"its graph has {schema.name}, which is handed or returns a "
                "place in a storage"
            )
        aliasing = [r.alias_info for r in schema.returns if r.alias_info is not None]
        if not aliasing:
            continue
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
