"""
Contains various utils for AOTAutograd, including those for handling collections.
"""

import dataclasses
import warnings
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Optional, Tuple

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import py_sym_types

KNOWN_TYPES = tuple(
    [torch.Tensor, int, str, float, bool, type(None)] + list(py_sym_types)
)

original_zip = zip


def strict_zip(*iterables, strict=True, **kwargs):
    if not strict:
        return original_zip(*iterables, **kwargs)

    shortest_length = min(len(it) for it in iterables)
    for iterable in iterables:
        if len(iterable) != shortest_length:
            raise ValueError(
                "The iterables have different lengths and strict mode is enabled."
            )

    return original_zip(*iterables, **kwargs)


def _get_symint_hints(exprs):
    """
    Get the hints of a list/tuple of int/SymInt.
    """
    if isinstance(exprs, (list, tuple)):
        return type(exprs)(_get_symint_hints(e) for e in exprs)
    elif isinstance(exprs, torch.SymInt):
        return exprs.node.shape_env.size_hint(exprs.node.expr)
    else:
        return exprs


def partial_flatten_asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {
            field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)
        }
    elif isinstance(obj, (list, tuple)):
        return obj.__class__([partial_flatten_asdict(item) for item in obj])
    elif isinstance(obj, dict):
        return {k: partial_flatten_asdict(v) for k, v in obj.items()}
    else:
        return obj


def normalize_as_list(x):
    if isinstance(x, tuple):
        return list(x)
    elif isinstance(x, list):
        return x
    return [x]


def _get_autocast_states():
    return [
        torch.is_autocast_enabled(),
        torch.is_autocast_cpu_enabled(),
        torch.get_autocast_gpu_dtype(),
        torch.get_autocast_cpu_dtype(),
        torch.is_autocast_cache_enabled(),
    ]


def make_boxed_func(f):
    def g(args):
        return f(*args)

    g._boxed_call = True  # type: ignore[attr-defined]
    return g


def make_boxed_compiler(compiler):
    @wraps(compiler)
    def f(fx_g, inps):
        out_f = compiler(fx_g, inps)
        fx_g = make_boxed_func(out_f)
        return fx_g

    return f


def call_func_at_runtime_with_args(f, args, steal_args=False, disable_amp=False):
    if not steal_args:
        args = list(args)
    assert isinstance(args, list)

    context = torch._C._DisableAutocast if disable_amp else nullcontext
    with context():
        if hasattr(f, "_boxed_call"):
            out = normalize_as_list(f(args))
        else:
            # TODO: Please remove soon
            # https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670
            warnings.warn(
                "Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. "
                "Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. "
                "See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale."
            )
            out = normalize_as_list(f(*args))
    return out


# Inspired by autodidax (thanks!)
class PytreeThunk:
    spec: Optional[pytree.TreeSpec] = None
    # These are some kinda dumb microoptimizations that save about 3-4 us of overhead.
    is_simple = (
        None  # if the output spec is a tuple/list, we won't bother unflattening it.
    )
    is_really_simple = None  # if the output spec is a LeafSpec

    def set(self, spec):
        assert self.spec is None or self.spec == spec
        self.spec = spec
        if type(self.spec) in [tuple, list] and all(
            isinstance(i, pytree.LeafSpec) for i in spec.children_specs
        ):
            self.is_simple = True
        if isinstance(self.spec, pytree.LeafSpec):
            self.is_really_simple = True

    def unflatten(self, x):
        if self.is_really_simple:
            return x[0]
        if self.is_simple:
            return x
        assert self.spec is not None
        return pytree.tree_unflatten(x, self.spec)


# Creates a function that returns flattened inputs and outputs
# Also returns the output tree spec, which is needed to recover the "unflattened"
# output tree structure later.
def create_tree_flattened_fn(fn, args, kwargs=None) -> Tuple[Callable, PytreeThunk]:
    if kwargs is None:
        kwargs = {}
    # Save the args_spec for flat_tensor_args to unflatten while tracing
    _, tensor_args_spec = pytree.tree_flatten((args, kwargs))
    out_spec = PytreeThunk()

    def flat_fn(*flat_args):
        # The input are flattened tensor args. Prepare the args in the
        # order that original function expects. Add static args as well.
        # They will appear as tensor constants in the traced graph.
        nonlocal out_spec
        args, kwargs = pytree.tree_unflatten(flat_args, tensor_args_spec)
        tree_out = fn(*args, **kwargs)
        flat_out, spec = pytree.tree_flatten(tree_out)
        for i in flat_out:
            is_known_type = False
            for j in KNOWN_TYPES:
                if isinstance(i, j):
                    is_known_type = True
                    break
            if not is_known_type:
                raise RuntimeError(
                    f"Found {type(i)} in output, which is not a known type. "
                    "If this type holds tensors, you need to register a pytree for it. "
                    "See https://github.com/pytorch/functorch/issues/475 for a brief "
                    "explanation why. If you don't need to register a pytree, please "
                    "leave a comment explaining your use case and we'll make this more "
                    "ergonomic to deal with"
                )
        out_spec.set(spec)
        return flat_out

    return flat_fn, out_spec


# This function takes in a tensor t, and returns one of t, t.view(), or t.clone().
# When tracing the joint forward + backward, for any inputs in the graph that are mutated,
# we need to clone them first (and similarly for metadata-only mutations, we need to view them first).
# The idea is that when we trace the backward, we need to pass in the *original* primals
# to autograd.grad(), before they were mutated.
# Note: when we have synthetic base inputs, we need to clone them *before* creating views off of them.
# This means that "idx" here represents the index of the (potentially) synthetic base.
# What we need to do is:
# (1) map the current (post-synthetic-base calling convention) input argument index
#     to int index pre-synthetic-base-calling-convention.
# (2) There could be multiple, if this index corresponds to a synthetic base
#     that has multiple input aliases.
# (3) If any of those corresponding inputs get metadata mutations, then we clone the base.
def maybe_to_fresh_input(idx, t, meta):
    if not isinstance(t, torch.Tensor):
        return t
    if idx in meta.mutated_inp_runtime_indices:
        # We only need to bother cloning mutated inputs that participate in autograd.
        mutated_inp_idx = meta.mutated_inp_runtime_indices.index(idx)
        if meta.input_info[idx].requires_grad and meta.input_info[idx].mutates_data:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the mutation
            return t.clone()
        if meta.input_info[idx] and meta.input_info[idx].mutates_metadata:
            # Make sure the primal we pass to autograd.grad()
            # sees the tensor before the metadata mutation
            return t.view(t.shape)
    return t
