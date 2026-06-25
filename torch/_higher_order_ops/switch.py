import contextlib
import functools
import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    _maybe_run_with_interpreter,
    reenter_make_fx,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


log = logging.getLogger(__name__)


class SwitchOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("switch")

    def __call__(self, index, branches, operands):
        validate_subgraph_args_types(operands)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(index, branches, operands)


switch_op = SwitchOp()


def wrap_branch_fn_flat(*args, branch_fn, spec_operands):
    operands = pytree.tree_unflatten(args, spec_operands)
    return branch_fn(*operands)


@exposed_in("torch")
def switch(
    index: int | torch.Tensor,
    branches: tuple[Callable, ...] | list[Callable],
    operands: tuple | list = (),
) -> Any:
    r"""
    Selects and runs one of N branch functions by index.

    .. warning::

        `torch.switch` is a prototype feature in PyTorch. It has limited support for input and
        output types. Please look forward to a more stable implementation in a future version of
        PyTorch. Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Equivalent to: ``branches[index](*operands)`` with index in ``[0, len(branches))``.

    Args:
        index (Union[int, torch.Tensor]): An int or single-element integer tensor
          indicating which branch to run. Out-of-range values are clamped into
          ``[0, len(branches))``.

        branches (Union[tuple[Callable, ...], list[Callable]]): Non-empty sequence of
          callables. Each must accept operands and return the same structure of outputs.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): Inputs to
          the branch functions. Defaults to ().

    Restrictions:
        - Each branch must have the same signature as operands and return the same
          output structure (shape, dtype, etc.). Constant ``int`` and ``None`` leaves
          are also permitted in branch outputs and are merged across branches (an
          unbacked SymInt is introduced when ``int`` leaves differ between branches).
        - Branches cannot have in-place mutations on inputs or global variables.
        - Autograd is not supported in this prototype: the autograd dispatch
          key is a no-op that redispatches below autograd, so gradients will
          not flow through ``torch.switch``. Full autograd support is planned
          for a future release.
    """

    # Flatten operands so the HOP only sees a flat list of tensors.
    # Each branch is wrapped to unflatten the leaves
    # back to the user-facing pytree before invocation.
    leaves_operands, spec_operands = pytree.tree_flatten(operands)

    def _validate_input(index, branches, operands):
        if not isinstance(index, (int, torch.SymInt, torch.Tensor)):
            raise RuntimeError(
                f"Expected index to be int or single-element tensor, but got {index}."
            )

        if isinstance(index, torch.Tensor) and index.numel() != 1:
            raise RuntimeError(
                f"Expected index to be int or single-element tensor, but got {index}."
            )

        if not isinstance(branches, (tuple, list)) or len(branches) == 0:
            raise RuntimeError(
                "Expected branches to be a non-empty tuple or list of callables."
            )

        for i, branch in enumerate(branches):
            if not callable(branch):
                raise RuntimeError(
                    f"Expected all branches to be callable. branch{i} is not callable."
                )

        for x in operands:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All operand leaves must be a Tensor but got {x}")

    _validate_input(index, branches, leaves_operands)

    wrapped_branches = tuple(
        functools.partial(
            wrap_branch_fn_flat,
            branch_fn=branch_fn,
            spec_operands=spec_operands,
        )
        for branch_fn in branches
    )

    # Early shortcut: single-branch switch degenerates to a plain call
    if len(wrapped_branches) == 1:
        return wrapped_branches[0](*leaves_operands)

    # Constant index shortcut for eager mode.
    if not torch.compiler.is_dynamo_compiling() and isinstance(index, int):
        # This is the non-strict export case. Strict export and torch.compile are
        # handled below via _maybe_compile_and_run_fn.
        if torch.compiler.is_compiling():
            warnings.warn(
                "Index is a Python constant. When used with torch.switch, it specializes on one of the branches."
                " If you want torch.switch to preserve the branches, please make the index an int tensor or a SymInt.",
                UserWarning,
                stacklevel=2,
            )

        # Clamp out-of-range indices rather than raising for consistency with compiled behavior.
        clamped_index = min(max(0, index), len(wrapped_branches) - 1)
        return wrapped_branches[clamped_index](*leaves_operands)

    # Use _maybe_compile_and_run_fn pattern from scan/associative_scan
    def run_switch(index, wrapped_branches, leaves_operands):
        return switch_op(index, wrapped_branches, leaves_operands)

    return _maybe_compile_and_run_fn(
        run_switch, index, wrapped_branches, leaves_operands
    )


def trace_switch(proxy_mode, func_overload, index, branches, operands):
    if not isinstance(operands, (list, tuple)):
        raise AssertionError(
            f"Switch operands must be a list or tuple of tensors and SymInts {operands}"
        )
    if not isinstance(branches, (list, tuple)) or len(branches) == 0:
        raise AssertionError(
            "Switch branches must be a non-empty list or tuple of callables"
        )

    branch_graphs = [reenter_make_fx(branch)(*operands) for branch in branches]

    branch_outs = [list() for _ in branches]
    for i, branch_graph in enumerate(branch_graphs):
        for node in branch_graph.graph.nodes:
            if node.op == "output":
                branch_outs[i].extend(node.args)

    branch_out_spec = [pytree.tree_flatten(outs)[1] for outs in branch_outs]
    for i, spec in enumerate(branch_out_spec):
        if branch_out_spec[0] != spec:
            raise RuntimeError(
                "Unmatched output spec from torch.switch branches: "
                f"branch0 tree_spec {branch_out_spec[0]} vs branch{i} tree_spec {spec}"
            )

    uid, _ = unique_graph_id(proxy_mode, prefix="branch0_graph")
    for i, branch_graph in enumerate(branch_graphs):
        proxy_mode.tracer.root.register_module(f"branch{i}_graph_{uid}", branch_graph)

    args = (index, branch_graphs, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}
    )

    out = func_overload(index, branch_graphs, operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@switch_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def switch_op_dense(index, branches, operands):
    if not all(isinstance(o, (torch.Tensor, int)) for o in operands):
        raise RuntimeError(
            f"Dense implementation operands must be a list of tensors and ints {operands}"
        )
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    idx: int = int(index.item()) if isinstance(index, torch.Tensor) else int(index)
    # Clamp out-of-range indices rather than raising for consistency with compiled behavior.
    clamped_idx = min(max(0, idx), len(branches) - 1)
    return branches[clamped_idx](*operands)


@switch_op.py_autograd_impl
def switch_autograd(index, branches, operands):
    with torch._C._AutoDispatchBelowAutograd():
        return switch_op(index, branches, operands)


@switch_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, index, branches, operands):
    return trace_switch(mode, switch_op, index, branches, operands)


@switch_op.py_impl(FakeTensorMode)
def switch_fake_tensor_mode(mode, index, branches, operands):
    # Ignore here, because if you've gotten here but you're not manually
    # tracing the inner graphs, that means that you intend to reuse the graph
    # directly.  Which means the old unbacked symbol bindings are appropriate.
    # This strategy will not work if unbacked symbols can escape.
    ignore_fresh_unbacked = contextlib.nullcontext()
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    with mode, ignore_fresh_unbacked:
        flat_branch_outs, branch_out_spec = zip(
            *[pytree.tree_flatten(branch(*operands)) for branch in branches]
        )
        for i, spec in enumerate(branch_out_spec):
            if branch_out_spec[0] != spec:
                raise RuntimeError(
                    "Unmatched output spec from torch.switch branches: "
                    f"branch0 tree_spec {branch_out_spec[0]} vs branch{i} tree_spec {spec}"
                )

    merged_outs = []
    for branches_out in zip(*flat_branch_outs):
        merged_outs.append(_merge_output(branches_out, mode))
    return pytree.tree_unflatten(merged_outs, branch_out_spec[0])


def _merge_output(xs: tuple[torch.Tensor | int | None, ...], mode: FakeTensorMode):
    from torch._higher_order_ops.cond import _merge_output as cond_merge_output

    # Shortcut if a branch produces None outputs; then all branches need to produce None
    if any(x is None for x in xs):
        if not all(x is None for x in xs):
            raise AssertionError(f"expected all leaves to be None, got {xs}")
        return None

    # In case all branches return an int, use an unbacked symbol as the merge result
    if all(type(x) is int for x in xs):
        if all(x == xs[0] for x in xs):
            return xs[0]
        if mode.shape_env is None:
            raise AssertionError("mode.shape_env is None")
        merged_out = mode.shape_env.create_unbacked_symint()
        mode.shape_env.constrain_symbol_range(
            merged_out.node.expr,
            min(xs),  # type: ignore[type-var]
            max(xs),  # type: ignore[type-var]
        )
        return merged_out

    return functools.reduce(lambda a, b: cond_merge_output(a, b, mode), xs)


@switch_op.py_functionalize_impl
def switch_func(ctx, index, branches, inputs):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_index = ctx.unwrap_tensors(index)
    with ctx.redispatch_to_next():
        functional_branches = [
            ctx.functionalize(_maybe_run_with_interpreter(fn)) for fn in branches
        ]
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for i, branch in enumerate(branches):
            _check_alias_and_mutation(
                branch, unwrapped_inputs, f"switch_branch{i}", pre_dispatch
            )
        switch_return = switch_op(
            unwrapped_index, functional_branches, unwrapped_inputs
        )
        return ctx.wrap_tensors(switch_return)
