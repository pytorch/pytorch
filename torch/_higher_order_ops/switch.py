# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import contextlib
import functools
import logging
import warnings
from collections.abc import Callable
from typing import Any, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import (
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


@exposed_in("torch")
def switch(
    index: Union[int, torch.Tensor],
    branches: Union[tuple[Callable, ...], list[Callable]],
    operands: Union[tuple, list] = (),
) -> Any:
    r"""
    Selects and runs one of N branch functions by index.

    .. warning::

        `torch.switch` is a prototype feature in PyTorch. It has limited support for input and
        output types. Please look forward to a more stable implementation in a future version of
        PyTorch. Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Equivalent to: branches[index](*operands) with index in [0, len(branches)).

    Args:
        index (Union[int, torch.Tensor]): An int or 0-dim tensor in [0, len(branches)),
          indicating which branch to run.

        branches (Union[tuple[Callable, ...], list[Callable]]): Non-empty sequence of
          callables. Each must accept operands and return the same structure of outputs.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): Inputs to
          the branch functions. Defaults to ().

    Restrictions:
        - index must be an int or a torch.Tensor with a single element (in [0, len(branches))).
        - Each branch must have the same signature as operands and return the same
          output structure (shape, dtype, etc.).
        - Branches cannot have in-place mutations on inputs or global variables.
    """
    # Early checks mirroring jax.lax.switch: an empty branch sequence is an
    # error regardless of index type, and a single-branch switch degenerates
    # to a plain call. Keeping these before the integer/tensor fast paths
    # means torch.switch(idx, [], ops) always raises RuntimeError (not
    # IndexError) and the 1-branch case never enters the HOP.
    if not isinstance(branches, (tuple, list)) or len(branches) == 0:
        raise RuntimeError(
            "Expected branches to be a non-empty tuple or list of callables."
        )
    if len(branches) == 1:
        return branches[0](*operands)

    if torch.compiler.is_dynamo_compiling():
        return switch_op(index, branches, operands)

    if isinstance(index, int):
        # This is the non-strict export case. Strict export and torch.compile are
        # handled above in dynamo.
        if torch.compiler.is_compiling():
            warnings.warn(
                "Index is a Python constant. When used with torch.switch, it specializes on one of the branches."
                " If you want torch.switch to preserve the branches, please make the predicate an int tensor or a SymInt.",
                UserWarning,
                stacklevel=2,
            )
        # Clamp out-of-range indices rather than raising for consistency with compiled behavior.
        clamped_index = min(max(0, index), len(branches) - 1)
        return branches[clamped_index](*operands)

    def _validate_input(index, branches, operands):
        if not isinstance(index, (int, torch.Tensor, torch.SymInt)):
            raise RuntimeError(
                f"Expected index to be an int or tensor, but got {index}."
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

        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only "
                f"consists of tensor leaves, but got {operands}."
            )

    _validate_input(index, branches, operands)

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.switch requires dynamo support.")

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass switch_op to it. So we wrap it in a dummy function.
    def _switch_op_wrapper(*args, **kwargs):
        return switch_op(*args, **kwargs)

    from torch._higher_order_ops.utils import setup_compilation_env

    with setup_compilation_env() as backend:
        return torch.compile(_switch_op_wrapper, backend=backend, fullgraph=True)(
            index, branches, operands
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

    flat_branch_outs = [pytree.arg_tree_leaves(*outs) for outs in branch_outs]
    for i, outs in enumerate(flat_branch_outs):
        if len(flat_branch_outs[0]) != len(outs):
            raise torch._dynamo.exc.SwitchOpArgsMismatchError(
                f"Expected to return same number of outputs from all branches but got:"
                f"\n  branch0 returns {len(flat_branch_outs[0])} item(s)"
                f"\n  branch{i} returns {len(outs)} item(s)"
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
        raise AssertionError(
            f"Dense implementation operands must be a list of tensors and ints {operands}"
        )
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    idx: int = int(index.item() if isinstance(index, torch.Tensor) else int(index))
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


def _merge_output(
    xs: tuple[Optional[Union[torch.Tensor, int]], ...], mode: FakeTensorMode
):
    from torch._higher_order_ops.cond import _merge_output as cond_merge_output

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
