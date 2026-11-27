# mypy: allow-untyped-defs
"""Values read out of tensors, for the lowering of a call list (lower.py).

A list may read a number out of a tensor (``Tensor.item``) and do Python
arithmetic on it before handing the result to an operator: an optimizer's
step count and bias corrections. The data-free run has no data, so such a
number is a symbolic scalar there (``torch.SymFloat``, ``SymInt``,
``SymBool``), and the graph gets a node for every read and one for each step
of the arithmetic: the graph computes the number of every later call, as the
list does. Nothing of the traced value is in the graph.

Given: the entries of a spec. ``symbols`` says which results are such
scalars, from the entries alone: a read (``READS``) and whatever an entry
makes of a scalar and asserts to be a float, an int or a bool. The fact
recorded at the trace is that class, never the value. ``in_run`` gives an
entry the form it has in the data-free run; ``settled`` holds the run
against the one thing a graph cannot do with such a number, need its value:

- a branch on it inside a callee (``bool``, ``int``, ``float``, ``range``,
  ``max``, a ``math`` function): the callee raises in the data-free run,
  at its entry;
- an index: the key of a subscript entry, whatever the run does with it;
  inside another callee (``torch.select``), a range the run assumes of an
  int;
- a size made of it (``torch.zeros(n)``): a tensor of the run whose sizes
  are no ints;
- a branch the run took on it (a guard of the shape environment);
- the number as an output of the list (lower.py compares no number).

Each is NotLowerable with the entry where there is one. What the run only
assumed of a float, to keep its class (`(1 - b ** step) ** 0.5` is a float
while the base is not negative), becomes an assert node of the graph, and
the list is NotLowerable if it cannot (a bound of the read number itself).
Imports nothing of this package.
"""

import operator

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


# The callees that hand a tensor's value to Python as one number.
READS = (torch.Tensor.item,)
# The class a scalar has at the trace, and in the data-free run.
SYMBOL = {float: torch.SymFloat, int: torch.SymInt, bool: torch.SymBool}
NEEDS_VALUE = (GuardOnDataDependentSymNode,)
_ASSERT = torch.ops.aten._assert_scalar.default
_READ_OPS = (torch.ops.aten._local_scalar_dense.default, torch.ops.aten.item.default)
# What a slot callee runs, as the C function of the operator module: the
# interpreter resolves a slot from classes, and takes no class that defines
# the slot's method in Python, as the symbolic scalars do.
_BINARY = {
    "+": operator.add, "&": operator.and_, "//": operator.floordiv,
    "<<": operator.lshift, "@": operator.matmul, "*": operator.mul,
    "%": operator.mod, "|": operator.or_, "**": operator.pow,
    ">>": operator.rshift, "-": operator.sub, "/": operator.truediv,
    "^": operator.xor, "+=": operator.iadd, "&=": operator.iand,
    "//=": operator.ifloordiv, "<<=": operator.ilshift, "@=": operator.imatmul,
    "*=": operator.imul, "%=": operator.imod, "|=": operator.ior,
    "**=": operator.ipow, ">>=": operator.irshift, "-=": operator.isub,
    "/=": operator.itruediv, "^=": operator.ixor,
}  # fmt: skip
_COMPARE = {
    "<": operator.lt, "<=": operator.le, "==": operator.eq,
    "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
}  # fmt: skip
_UNARY = {"-": operator.neg, "~": operator.invert, "+": operator.pos}
_SLOTS = {"binary": _BINARY, "compare": _COMPARE, "unary": _UNARY}
# The subscripts, and where their key is among the operands.
_KEYED = {"getitem": 1, "setitem": 2, "delitem": 1}


class Reads(FakeTensorMode):
    """The fake mode of a data-free run that reads values: every read is a
    number of its own, as it is a call of its own in the list. The mode it
    is made of hands two reads of one tensor one number while nothing wrote
    that tensor, which it cannot know of a tensor that shares memory with
    another tensor of the call: here the keys of every memo change with
    each read."""

    def dispatch(self, func, types, args=(), kwargs=None):
        try:
            return super().dispatch(func, types, args, kwargs or {})
        finally:
            if func in _READ_OPS:
                self.epoch += 1


def _leaves(t):
    if t[0] not in ("seq", "map", "slice"):
        return [t]
    parts = t[2] if t[0] != "slice" else t[1:]
    return [leaf for part in parts for leaf in _leaves(part)]


def _tensors(x):
    """The tensors in ``x``: itself, or what tuples and lists in it hold."""
    if isinstance(x, torch.Tensor):
        return [x]
    held = x if isinstance(x, (tuple, list)) else ()
    return [t for item in held for t in _tensors(item)]


def _uses(args, symbols):
    """Whether one of the operands ``args`` is, or holds, a symbolic scalar."""
    named = [t for a in args for t in _leaves(a)]
    return any(t[0] in ("res", "move") and t[1] in symbols for t in named)


def symbols(entries):
    """By entry whose result is a scalar of the data-free run: the class the
    entry asserts of it (float, int or bool). Such an entry is a read, or
    has such a scalar among its operands. Also (why, entry) of the first
    entry that is not lowered whatever the run does, or None: a read that
    asserts no such class (the class is the fact a read is lowered on), a
    scalar as the key of a subscript."""
    out, refused = {}, None
    for k, e in enumerate(entries):
        if e[0] == "drop":
            continue
        asserted = [a[1] for a in e[6] if a[0] == "class" and a[1] in SYMBOL]
        reads = e[0][0] == "vector" and any(e[0][1] is f for f in READS)
        key = _KEYED.get(e[0][0])
        if key is not None and _uses(e[1][key:], out):
            refused = refused or ("a value read from a tensor is an index", k)
        if asserted and (reads or _uses(e[1], out)):
            out[k] = asserted[0]
        elif reads:
            why = "a value is read from a tensor, of no asserted class"
            refused = refused or (why, k)
    return out, refused


def in_run(e, k, symbols):
    """The callee and the asserts of entry ``k`` (``e``) for the data-free
    run: a slot callee with a symbolic operand is the operator module's
    function of the same instruction, and a result that is a scalar is
    asserted to be of the symbolic class."""
    callee, asserts = e[0], e[6]
    table = _SLOTS.get(callee[0])
    if table is not None and _uses(e[1], symbols):
        f = table[callee[1]]
        callee = ("vector", f, f"operator.{f.__name__}")
    if k in symbols:
        asserts = tuple(
            ("class", SYMBOL[a[1]]) if a[0] == "class" and a[1] in SYMBOL else a
            for a in asserts
        )
    return callee, asserts


def settled(gm, results, shape_env):
    """Holds a finished data-free run against what a graph cannot do with a
    scalar: (why, entry) of the first thing that needs its value, or None.
    What the run assumed of a scalar to go on (the base of a root is not
    negative, so that the result is the float the entry asserts) is put
    into ``gm`` as an assert on the number of every call: where it does not
    hold the graph raises, as the list stops at its entry's assert."""
    for k, x in results.items():
        for t in _tensors(x):
            if any(type(n) is not int for n in (*t.shape, *t.stride())):
                return "a size of its result is a value read from a tensor", k
    if shape_env.guards:
        return "the data-free run took a branch on a value read from a tensor", None
    assumed = [a.expr for of in shape_env.deferred_runtime_asserts.values() for a in of]
    if any(symbol.is_integer for expr in assumed for symbol in expr.free_symbols):
        why = "the data-free run assumed a range of an int read from a tensor"
        return f"{why} (an index, a size)", None
    if assumed:
        insert_deferred_runtime_asserts(gm, shape_env, "lowering")
        gm.recompile()
        # The pass leaves some out (a bound of a float symbol itself).
        nodes = gm.graph.find_nodes(op="call_function", target=_ASSERT)
        said = [str(node.args[1]) for node in nodes]
        for expr in assumed:
            if not any(f"expression {expr} on node" in text for text in said):
                why = "what the data-free run assumed of a value read from a tensor"
                return f"{why} ({expr}) could not be made an assert of the graph", None
    return None
