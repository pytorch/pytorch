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
    check_input_alias_and_mutation_return_outputs,
    create_bw_fn,
    create_fn_remove_none,
    fill_none_with_masks,
    materialize_as_graph,
    reenter_make_fx,
    save_values_for_backward,
    saved_values,
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

    # pyrefly: ignore [bad-override]
    def gen_schema(self, index, branches, operands):
        from torch._guards import detect_fake_mode
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        branch_gms: list[torch.fx.GraphModule] = []
        all_branch_outputs: list[tuple[Any, ...] | list[Any]] = []
        mutated_inputs: set[int] = set()
        for branch in branches:
            gm = (
                branch
                if isinstance(branch, torch.fx.GraphModule)
                else materialize_as_graph(branch, operands)
            )
            (
                _,
                _,
                _,
                branch_mutated_inputs,
                branch_outputs,
            ) = check_input_alias_and_mutation_return_outputs(gm)
            branch_gms.append(gm)
            all_branch_outputs.append(branch_outputs)
            mutated_inputs |= set(branch_mutated_inputs)

        # Merge outputs to detect int -> SymInt change
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        fake_mode = detect_fake_mode(operands)
        if fake_mode is None or fake_mode.shape_env is None:
            fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        # pyrefly: ignore [missing-attribute]
        with fake_mode, fake_mode.shape_env.ignore_fresh_unbacked_symbols():
            merged_outputs = [
                _merge_output(branch_outs, fake_mode)
                for branch_outs in zip(*all_branch_outputs)
            ]

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("index", index)
        for i, gm in enumerate(branch_gms):
            schema_gen.add_arg(f"branch{i}_fn", gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"operand{idx}", arg, is_mutated=idx in mutated_inputs)

        for out in merged_outputs:
            schema_gen.add_output(out)
        schema_gen.add_schema_tree_spec(index, branches, operands)
        return schema_gen.gen_schema()


switch_op = SwitchOp()


def wrap_branch_fn_flat(*args, branch_fn, spec_operands):
    operands = pytree.tree_unflatten(args, spec_operands)
    return branch_fn(*operands)


def _get_branch(branches, idx):
    if not 0 <= idx < len(branches):
        raise AssertionError(
            f"switch index {idx} out of range for {len(branches)} branches"
        )
    return branches[idx]


@exposed_in("torch")
def switch(
    index: int | torch.SymInt | torch.Tensor,
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
    num_branches = len(wrapped_branches)
    if num_branches == 1:
        return wrapped_branches[0](*leaves_operands)

    # Clamp out-of-range indices to [0, len(branches) - 1]
    if isinstance(index, torch.Tensor):
        index = index.clamp(0, num_branches - 1)
    elif isinstance(index, torch.SymInt):
        index = torch.sym_max(0, torch.sym_min(index, num_branches - 1))
    elif isinstance(index, int):
        index = max(0, min(index, num_branches - 1))

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

        return _get_branch(wrapped_branches, index)(*leaves_operands)

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
    return _get_branch(branches, idx)(*operands)


class SwitchAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        index,
        branches,
        *operands,
    ):
        ctx._index = index
        # Build one bw fn per branch.
        ctx._branch_bw_fns = [
            create_bw_fn(create_fn_remove_none(branch)[0], operands)
            for branch in branches
        ]

        # We snapshot the dispatch keys in forward for materializing the
        # bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_values_for_backward(ctx, operands)

        with torch._C._AutoDispatchBelowAutograd():
            outs = switch_op(index, branches, operands)

        # Record which output slots are Tensors. Non-Tensor slots (None,
        # int/SymInt) carry no tangent in backward and the joint is built
        # to omit them.
        ctx._fw_output_is_tensor = [
            isinstance(o, torch.Tensor) for o in pytree.tree_leaves(outs)
        ]
        return outs

    @staticmethod
    def backward(ctx, *flat_grads):
        operands = saved_values(ctx)
        tensor_grads = tuple(
            g for g, keep in zip(flat_grads, ctx._fw_output_is_tensor) if keep
        )
        args = operands + tensor_grads
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.

        branches_bw_gm: list[torch.fx.GraphModule] = []
        grads_tensor_masks: list[bool] = []
        # All branches share the same input signature (see _validate_input)
        for bw_fn in ctx._branch_bw_fns:
            wrapped_bw, mask = create_fn_remove_none(bw_fn)
            bw_gm = materialize_as_graph(
                wrapped_bw,
                args,
                ctx._fw_include_key_set,
                ctx._fw_exclude_key_set,
                force_enable_grad=True,
            )
            branches_bw_gm.append(bw_gm)
            if not grads_tensor_masks:
                grads_tensor_masks = mask

        grads = switch_op(
            ctx._index,
            branches_bw_gm,
            args,
        )
        return None, None, *fill_none_with_masks(grads, grads_tensor_masks)


@switch_op.py_autograd_impl
def switch_autograd(index, branches, operands):
    return SwitchAutogradOp.apply(
        index,
        branches,
        *operands,
    )


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
