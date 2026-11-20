# mypy: allow-untyped-defs
"""Compiles pure blocks of the torch._rust CFG executor with Inductor.

The executor calls ``trace_region`` on a run of a pure block's version (see
torch/_rust/src/exec/region.rs) with the block's params and inputs. It runs
the body once under ``make_fx`` on the real inputs, so the run is also the
trace, and compiles the aten graph with ``compile_fx``. Tensors the body read
that aren't inputs (module parameters and buffers) become extra graph inputs,
held by the region, so autograd reaches them. The executor adds a version
running the compiled graph, guarded on what the trace specialized on.
"""

import logging
import operator
import os
from typing import Any

import torch
import torch.fx
from torch.fx.experimental.proxy_tensor import make_fx


log = logging.getLogger(__name__)


class _Null:
    def __repr__(self) -> str:
        return "NULL"


# Stands for CPython's NULL (an empty LoadMethod self slot) in args and
# outputs crossing into Python.
NULL = _Null()

MIN_OPS = int(os.environ.get("TORCH_RUST_REGION_MIN_OPS", "4"))
DEBUG = bool(os.environ.get("TORCH_RUST_REGION_DEBUG"))

_PLAIN = (torch.Tensor, torch.nn.Parameter)
_SCALARS = (bool, int, float, type(None))
_IMMUTABLE = (bool, int, float, type(None), str, torch.dtype, torch.device, torch.layout, torch.memory_format)
_SKIP_OPS = {torch.ops.aten._local_scalar_dense.default, torch.ops.aten.item.default}

stats = {"traced": 0, "compiled": 0, "small": 0, "refused": 0}


class _Refused(Exception):
    pass


def _checks(args, unused):
    checks: list[tuple[Any, ...]] = [("grad_enabled", torch.is_grad_enabled())]
    tensor_pos = []
    for i, a in enumerate(args):
        if i in unused:
            continue
        if a is NULL:
            checks.append(("null", i))
        elif isinstance(a, torch.Tensor):
            if type(a) not in _PLAIN:
                raise _Refused(f"arg {i} is a {type(a).__name__}")
            tensor_pos.append(i)
            checks.append(("tensor", i, a))
            checks.append(("requires_grad", i, a.requires_grad))
        elif type(a) in _SCALARS:
            checks.append(("value", i, a))
        elif type(a) is tuple and all(type(x) is int for x in a):
            checks.append(("ints", i, True, list(a)))
        elif isinstance(a, (tuple, list, dict, set)):
            raise _Refused(f"arg {i} is a {type(a).__name__}")
        else:
            checks.append(("is", i, a))
    return checks, tensor_pos


def _template(o, args, tensor_pos, tensor_outs, objects):
    if o is NULL:
        return ("null",)
    for i in tensor_pos:
        if o is args[i]:
            return ("arg", i)
    if isinstance(o, torch.Tensor):
        tensor_outs.append(o)
        return ("out", len(tensor_outs) - 1)
    for i, a in enumerate(args):
        if o is a:
            return ("arg", i)
    if type(o) in _IMMUTABLE:
        objects.append(o)
        return ("obj", len(objects) - 1)
    if type(o) in (tuple, list):
        kind = "tuple" if type(o) is tuple else "list"
        return (kind, [_template(x, args, tensor_pos, tensor_outs, objects) for x in o])
    if isinstance(o, (list, dict, set)):
        raise _Refused(f"an output is a {type(o).__name__}")
    # Read from the objects the guard pins (a submodule, a bound method), which
    # are assumed not to be reassigned, like the parameters a region holds.
    objects.append(o)
    return ("obj", len(objects) - 1)


def _lift_constants(gm: torch.fx.GraphModule) -> list[torch.Tensor]:
    """Turns the graph's tensor constants into trailing placeholders."""
    consts = []
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    last = placeholders[-1] if placeholders else None
    for node in list(gm.graph.nodes):
        if node.op != "get_attr":
            continue
        value = getattr(gm, node.target)
        if not isinstance(value, torch.Tensor):
            continue
        if last is None:
            with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
                ph = gm.graph.placeholder(f"const_{len(consts)}")
        else:
            with gm.graph.inserting_after(last):
                ph = gm.graph.placeholder(f"const_{len(consts)}")
        ph.meta = dict(node.meta)
        last = ph
        node.replace_all_uses_with(ph)
        gm.graph.erase_node(node)
        delattr(gm, node.target)
        consts.append(value)
    gm.graph.lint()
    gm.recompile()
    return consts


def _compile(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    from torch._inductor.compile_fx import compile_fx

    # aot_module_simplified's forward takes the inputs unboxed (whatever its
    # copied _boxed_call says).
    compiled = compile_fx(gm, example_inputs)
    return lambda inputs: compiled(*inputs)


def trace_region(name: str, args: tuple[Any, ...], unused: list[int]):
    """Runs the block being traced (via torch._rust._region_body) and returns
    (func, tensor_args, outputs, objects, checks) for its region, or None. A
    None before the body ran leaves running it to the executor."""
    from torch._rust import _region_body

    try:
        checks, tensor_pos = _checks(args, set(unused))
    except _Refused as e:
        stats["refused"] += 1
        if DEBUG:
            print(f"region {name}: refused: {e}")
        return None

    result: dict[str, Any] = {}

    def body(*tensors):
        full = list(args)
        for i, t in zip(tensor_pos, tensors):
            full[i] = t
        outs = _region_body(*full)
        tensor_outs: list[torch.Tensor] = []
        objects: list[Any] = []
        try:
            result["outputs"] = [_template(o, full, tensor_pos, tensor_outs, objects) for o in outs]
        except _Refused as e:
            result["refused"] = str(e)
            return []
        result["objects"] = objects
        return tensor_outs

    gm = make_fx(body, tracing_mode="real")(*[args[i] for i in tensor_pos])
    stats["traced"] += 1
    if "refused" in result:
        stats["refused"] += 1
        if DEBUG:
            print(f"region {name}: refused: {result['refused']}")
        return None
    ops = [n for n in gm.graph.nodes if n.op == "call_function"]
    if any(n.target in _SKIP_OPS for n in ops):
        stats["refused"] += 1
        if DEBUG:
            print(f"region {name}: refused: reads a tensor's value")
        return None
    n_ops = sum(1 for n in ops if n.target is not operator.getitem)
    if n_ops < MIN_OPS:
        stats["small"] += 1
        if DEBUG:
            print(f"region {name}: {n_ops} ops, too small")
        return None
    consts = _lift_constants(gm)
    example = [args[i] for i in tensor_pos] + consts
    compiled = _compile(gm, example)
    stats["compiled"] += 1
    if DEBUG:
        print(f"region {name}: compiled {n_ops} ops, {len(tensor_pos)} inputs + {len(consts)} constants")

    n_inputs = len(tensor_pos) + len(consts)

    def func(tensors):
        inputs = [*tensors, *consts]
        if len(inputs) != n_inputs or not all(isinstance(t, torch.Tensor) for t in tensors):
            raise RuntimeError(
                f"region {name}: called with {[type(t).__name__ for t in tensors]} + {len(consts)} constants, "
                f"compiled for {n_inputs} inputs at {tensor_pos} of {[type(a).__name__ for a in args]}"
            )
        return compiled(inputs)

    return func, tensor_pos, result["outputs"], result["objects"], checks
