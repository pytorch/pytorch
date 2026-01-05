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
from torch._C._functorch import (
    _add_batch_dim,
    get_unwrapped,
    is_batchedtensor,
    maybe_get_bdim,
)
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    check_input_alias_and_mutation_return_outputs,
    create_bw_fn,
    fill_none_with_masks,
    filter_with_masks,
    materialize_as_graph,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


log = logging.getLogger(__name__)

"""
We're going to define a `cond_op` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""


class CondOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("cond")

    def __call__(self, pred, true_fn, false_fn, operands):
        validate_subgraph_args_types(operands)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(pred, true_fn, false_fn, operands)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, pred, true_fn, false_fn, operands):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        then_gm: torch.fx.GraphModule = materialize_as_graph(true_fn, operands)
        else_gm: torch.fx.GraphModule = materialize_as_graph(false_fn, operands)
        (
            _,
            _,
            _,
            then_mutated_inputs,
            then_outputs,
        ) = check_input_alias_and_mutation_return_outputs(then_gm)
        (
            _,
            _,
            _,
            else_mutated_inputs,
            else_outputs,
        ) = check_input_alias_and_mutation_return_outputs(else_gm)
        mutated_inputs = set(then_mutated_inputs) | set(else_mutated_inputs)

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("pred", pred)
        schema_gen.add_arg("true_fn", then_gm)
        schema_gen.add_arg("false_fn", else_gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"operand{idx}", arg, is_mutated=idx in mutated_inputs)

        for out in then_outputs:
            schema_gen.add_output(out)
        schema_gen.add_schema_tree_spec(pred, true_fn, false_fn, operands)
        return schema_gen.gen_schema()


cond_op = CondOp()


@exposed_in("torch")
def cond(
    pred: Union[bool, int, float, torch.Tensor],
    true_fn: Callable,
    false_fn: Callable,
    operands: Union[tuple, list] = (),
) -> Any:
    r"""
    Conditionally applies `true_fn` or `false_fn`.

    .. warning::
        `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types.
        Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `cond` is structured control flow operator. That is, it is like a Python if-statement,
    but has restrictions on `true_fn`, `false_fn`, and `operands` that enable it to be
    capturable using torch.compile and torch.export.

    Assuming the constraints on `cond`'s arguments are met, `cond` is equivalent to the following::

        def cond(pred, true_branch, false_branch, operands):
            if pred:
                return true_branch(*operands)
            else:
                return false_branch(*operands)

    Args:
        pred (Union[bool, torch.Tensor]): A boolean expression or a tensor with one element,
          indicating which branch function to apply.

        true_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced.

        false_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced. The true branch and false branch must
          have consistent input and outputs, meaning the inputs have to be
          the same, and the outputs have to be the same type and shape. Int
          output is also allowed. We'll make the output dynamic by turning it
          into a symint.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): A tuple of inputs to the
          true/false functions. It can be empty if true_fn/false_fn doesn't require input. Defaults to ().

    Example::

        def true_fn(x: torch.Tensor):
            return x.cos()


        def false_fn(x: torch.Tensor):
            return x.sin()


        return cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    Restrictions:
        - The conditional statement (aka `pred`) must meet one of the following constraints:

          - It's a `torch.Tensor` with only one element, and torch.bool dtype

          - It's a boolean expression, e.g. `x.shape[0] > 10` or `x.dim() > 1 and x.shape[1] > 10`

        - The branch function (aka `true_fn`/`false_fn`) must meet all of the following constraints:

          - The function signature must match with operands.

          - The function must return a tensor with the same metadata, e.g. shape,
            dtype, etc.

          - The function cannot have in-place mutations on inputs or global variables.
            (Note: in-place tensor operations such as `add_` for intermediate results
            are allowed in a branch)

    """
    if torch.compiler.is_dynamo_compiling():
        return cond_op(pred, true_fn, false_fn, operands)

    if isinstance(pred, (bool, int, float)):
        # This is the non-strict export case. Strict export and torch.compile are
        # handled above in dynamo.
        if torch.compiler.is_compiling():
            warnings.warn(
                "Pred is a Python constant. When used with torch.cond, it specializes on one of the branches."
                " If you want torch.cond to preserve two branches, please make the predicate a boolean tensor or a SymBool.",
                UserWarning,
                stacklevel=2,
            )
        # This is the eager case. We can just run the true or false branch.
        if pred:
            return true_fn(*operands)
        else:
            return false_fn(*operands)

    def _validate_input(pred, true_fn, false_fn, operands):
        if not isinstance(pred, (bool, torch.Tensor, torch.SymBool)):
            raise RuntimeError(f"Expected pred to be bool or tensor, but got {pred}.")

        if isinstance(pred, torch.Tensor) and pred.numel() != 1:
            raise RuntimeError(
                f"Expected pred to be bool or single-element tensor, but got {pred}."
            )

        if not callable(true_fn) or not callable(false_fn):
            raise RuntimeError("Expect both branches to be callable.")

        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only "
                f"consists of tensor leaves, but got {operands}."
            )

    _validate_input(pred, true_fn, false_fn, operands)

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.cond requires dynamo support.")

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _cond_op_wrapper(*args, **kwargs):
        return cond_op(*args, **kwargs)

    from torch._higher_order_ops.utils import setup_compilation_env

    with setup_compilation_env() as backend:
        return torch.compile(_cond_op_wrapper, backend=backend, fullgraph=True)(
            pred, true_fn, false_fn, operands
        )


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    assert isinstance(operands, (list, tuple)), (
        f"Cond operands must be a list or tuple of tensors and SymInts {operands}"
    )

    true_graph = reenter_make_fx(true_fn)(*operands)
    false_graph = reenter_make_fx(false_fn)(*operands)

    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == "output":
            true_outs.extend(node.args)

    for node in false_graph.graph.nodes:
        if node.op == "output":
            false_outs.extend(node.args)

    flat_true_outs = pytree.arg_tree_leaves(*true_outs)
    flat_false_outs = pytree.arg_tree_leaves(*false_outs)
    if len(flat_true_outs) != len(flat_false_outs):
        raise torch._dynamo.exc.CondOpArgsMismatchError(
            f"Expected to return same number of outputs but got:"
            f"\n  true branch returns {len(flat_true_outs)} item(s)"
            f"\n  false branch returns {len(flat_false_outs)} item(s)"
        )

    i, true_name = unique_graph_id(proxy_mode, prefix="true_graph")

    false_name = f"false_graph_{i}"
    assert not hasattr(proxy_mode.tracer.root, false_name)

    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    args = (pred, true_graph, false_graph, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}
    )

    out = func_overload(pred, true_graph, false_graph, operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@cond_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def cond_op_dense(pred, true_fn, false_fn, operands):
    assert all(isinstance(o, (torch.Tensor, int)) for o in operands), (
        f"Dense implementation operands must be a list of tensors and ints {operands}"
    )
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)


class CondAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        pred,
        true_fn,
        false_fn,
        *operands,
    ):
        ctx._pred = pred
        ctx._true_bw_fn = create_bw_fn(
            true_fn,
            operands,
        )
        ctx._false_bw_fn = create_bw_fn(
            false_fn,
            operands,
        )
        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_tensors_and_symints_for_backward(ctx, operands)

        with torch._C._AutoDispatchBelowAutograd():
            return cond_op(pred, true_fn, false_fn, operands)

    @staticmethod
    def backward(ctx, *flat_grads):
        operands = saved_tensors_and_symints(ctx)
        args = operands + flat_grads
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.

        grads_tensor_masks = []

        def create_fn_remove_none(fn):
            @functools.wraps(fn)
            def wrapped(*args):
                nonlocal grads_tensor_masks

                true_outputs = fn(*args)
                grads_tensor_masks = [
                    bool(isinstance(out, torch.Tensor)) for out in true_outputs
                ]
                return filter_with_masks(true_outputs, grads_tensor_masks)

            return wrapped

        true_bw_gm = materialize_as_graph(
            create_fn_remove_none(ctx._true_bw_fn),
            args,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )
        false_bw_gm = materialize_as_graph(
            create_fn_remove_none(ctx._false_bw_fn),
            args,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )
        grads = cond_op(
            ctx._pred,
            true_bw_gm,
            false_bw_gm,
            args,
        )
        return None, None, None, *fill_none_with_masks(grads, grads_tensor_masks)


# Note:
# As long as one of the tensors in pred or operands requires grad,
# all the output would require grad with backward fn set to be the CondAutogradOp.
# This is consistent with autograd.Function's semantic.
@cond_op.py_autograd_impl
def cond_autograd(pred, true_fn, false_fn, operands):
    return CondAutogradOp.apply(
        pred,
        true_fn,
        false_fn,
        *operands,
    )


@cond_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, pred, true_fn, false_fn, operands):
    return trace_cond(mode, cond_op, pred, true_fn, false_fn, operands)


@cond_op.py_impl(FakeTensorMode)
def cond_fake_tensor_mode(mode, pred, true_fn, false_fn, operands):
    # Ignore here, because if you've gotten here but you're not manually
    # tracing the inner graphs, that means that you intend to reuse the graph
    # directly.  Which means the old unbacked symbol bindings are appropriate.
    # This strategy will not work if unbacked symbols can escape.
    ignore_fresh_unbacked = contextlib.nullcontext()
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    with mode, ignore_fresh_unbacked:
        flat_true_outs, true_out_spec = pytree.tree_flatten(true_fn(*operands))
        flat_false_outs, false_out_spec = pytree.tree_flatten(false_fn(*operands))
        if true_out_spec != false_out_spec:
            raise RuntimeError(
                "Unmatched output spec from torch.cond branches: "
                f"true branch tree_spec {true_out_spec} vs false branch tree_spec {false_out_spec}."
            )

    merged_outs = []
    for true_out, false_out in zip(flat_true_outs, flat_false_outs):
        merged_outs.append(_merge_output(true_out, false_out, mode))
    return pytree.tree_unflatten(merged_outs, true_out_spec)


def check_tensor_meta_match(
    t1: torch.Tensor, t2: torch.Tensor, attr_names: tuple[str, ...], msg_prefix: str
) -> None:
    def _get_attr_maybe_call(t: torch.Tensor, attr_name: str) -> Any:
        attr = getattr(t, attr_name)
        if callable(attr):
            return attr()
        return attr

    for attr_name in attr_names:
        lattr = _get_attr_maybe_call(t1, attr_name)
        rattr = _get_attr_maybe_call(t2, attr_name)
        torch._check(
            lattr == rattr,
            lambda: f"{msg_prefix} expected same {attr_name} but got {lattr} and {rattr}.",
        )


def _merge_output(
    a: Optional[Union[torch.Tensor, int]],
    b: Optional[Union[torch.Tensor, int]],
    mode: FakeTensorMode,
):
    from torch.fx.experimental.symbolic_shapes import (
        has_free_unbacked_symbols,
        SymIntEqByExpr,
    )

    if a is None or b is None:
        assert a is None and b is None, (a, b)
        return None

    def min_max(s0, s1):
        def _bound(s0, lower_bound: bool):
            if isinstance(s0, int):
                return s0
            r = mode.shape_env.var_to_range.get(  # type: ignore[union-attr]
                s0.node.expr,
                torch.utils._sympy.value_ranges.ValueRanges.unknown(),
            )
            return r.lower if lower_bound else r.upper

        return min(_bound(s0, True), _bound(s1, True)), max(
            _bound(s0, False), _bound(s1, False)
        )

    if type(a) is int and type(b) is int:
        if a == b:
            return a
        assert mode.shape_env is not None
        merged_out = mode.shape_env.create_unbacked_symint()
        mode.shape_env.constrain_symbol_range(merged_out.node.expr, *min_max(a, b))
        return merged_out

    assert type(a) is FakeTensor and type(b) is FakeTensor, (a, type(a), b, type(b))

    # Note: we don't check size, stride because
    # they'll be merged with unbacked symints if they differ.
    _meta_to_check = {
        "dtype",
        "device",
        "layout",
        "dim",
        "is_quantized",
        "is_conj",
        "is_sparse",
        "storage_offset",
    }
    check_tensor_meta_match(
        a,
        b,
        tuple(_meta_to_check),
        msg_prefix="When merging two branches' output in torch.cond, ",
    )
    # NYI
    assert not a.is_quantized and not b.is_quantized
    assert not a.is_sparse and not b.is_sparse
    assert not a.is_conj() and not b.is_conj()

    """
    Step 1: create unbacked symints for sizes that are different
    along the same axis. For example:
        a.size is [s0, 4, s0, 5, 4, 5]
        b.size is [s1, 4, s2, 8, 4, 7]
        merged_size will be [u0, 4, u1, u2, 4, u3], where
        u0 has range [min(s0, s1), max(s0, s1)]
        u1 has range [min(s0, s2), max(s0, s2)]
        u2 has range [5, 8]
        u3 has range [5, 7]
    """
    merged_size: list[Union[int, torch.SymInt]] = []

    def _has_unbacked_symbols(s: Union[int, torch.SymInt]) -> bool:
        if isinstance(s, int):
            return False
        else:
            return has_free_unbacked_symbols(s.node.expr)

    for s0, s1 in zip(a.size(), b.size()):
        # If there are unbacked symbols leaked out of true_branch or false_branch
        # we need to merge them with a new unbacked symbol and track in parent graph.
        if (
            not _has_unbacked_symbols(s0)
            and not _has_unbacked_symbols(s1)
            and SymIntEqByExpr(s0) == SymIntEqByExpr(s1)
        ):
            merged_size.append(s0)
        else:
            assert mode.shape_env is not None
            new_size = mode.shape_env.create_unbacked_symint()
            mode.shape_env.constrain_symbol_range(new_size.node.expr, *min_max(s0, s1))
            merged_size.append(new_size)

    """
    This follows the logic in symbolic_shapes._compute_symbolic_stride
    Step 2: Since tensor stride is an accumulative multiplication of the sizes, which is a permutated
        (due to view ops) non-descending sequence.

        Case 1: No size is 1. In this case, strides have unique values.
            For example, suppose we have a tensor with:
            size [3, 4, 3, 5, 4, 5],
            stride (1200, 300, 1, 12, 3, 60),
            merged_size [u0, u1, u2, u3, u4, u5].

            We visit the strides in ascending order: 1, 3, 12, 60, 300, 1200. In each step, we check whether
            the current stride is bounded or not and bound next stride by setting.
                stride_expr[next_stride] = current_stride_expr * current_size_expr
            1st round:
                current_stride is 1, current_size is 3, so next_stride is 1 * 3 = 3,
                current_stride_expr is set to 1, current_size_expr is u2, so stride_expr[3] is therefore 1 * u2 = u2
            2nd round:
                current_stride is 3, current_size is 4, so next_stride is 3 * 4 = 12,
                current_stride_expr is stride_expr[3] i.e. u2, current_size_expr is u4, so stride_expr[12] = u2 * u4
                ...

        Case 2: At least one dimension has size 1, which can produce duplicates in strides.
            In this case, theoretically, we cannot uniquely determine the expr of strides because
            the accessing stride_expr with same key in different order causes the final stride expression
            to be different.

            Suppose we have:
                size: (3, 1)
                stride: (1, 1)
                merged_size: (u0, u1)

            The stride expr could either be (u1, 1) or (1, u0) depending on whether we start with u1 or u0.
            For this reason, we try to break tie by sorting via descending index so we always get (u1, 1).

            Note that backend might optimize the strides anyway so this is usually not a problem as long
            as two branches matches. See relevant discussions in https://github.com/pytorch/pytorch/issues/142024.

        Case 3: Dim has 0 stride. 0 stride doesn't participate in the accumulative multiplication of
            sizes. So they're always treated as constant even if their corresponding size is turned into unbacked symint.

            Suppose we have:
                size: (3, 3)
                stride: (0, 1)
                merged_size: (u0, u1)

            The merged stride would be (0, 1)
    """

    def _bound_stride(
        a_ex_size: torch.Size,
        b_ex_size: torch.Size,
        a_ex_stride: tuple[int, ...],
        b_ex_stride: tuple[int, ...],
        merged_size: list[Union[int, torch.SymInt]],
    ) -> list[Union[int, torch.SymInt]]:
        from torch._inductor.ir import get_stride_order

        a_sorted_stride_idx = get_stride_order(a_ex_stride, mode.shape_env)
        b_sorted_stride_idx = get_stride_order(b_ex_stride, mode.shape_env)

        a_stride_li: list[Optional[tuple[Union[int, torch.SymInt], int]]] = [
            None
        ] * len(a_ex_stride)
        b_stride_li: list[Optional[tuple[Union[int, torch.SymInt], int]]] = [
            None
        ] * len(b_ex_stride)
        for i, idx in enumerate(a_sorted_stride_idx):
            a_stride_li[idx] = (a_ex_stride[i], -i)
        for i, idx in enumerate(b_sorted_stride_idx):
            b_stride_li[idx] = (b_ex_stride[i], -i)

        for a_pair, b_pair in zip(a_stride_li, b_stride_li):
            assert a_pair is not None and b_pair is not None
            _, a_idx = a_pair
            _, b_idx = b_pair

            if a_idx != b_idx:
                raise RuntimeError(
                    f"The sorted order of strides of the two branches' output doesn't match."
                    f"this indicates the contiguousness of the two branches are different. "
                    f"True branch has stride {a_ex_stride} but false branch has stride {b_ex_stride}."
                    f"Consider using contiguous() to make the two branches have the same contiguousness."
                )

        def _maybe_expr(s: Union[int, torch.SymInt]):
            if isinstance(s, int):
                return s
            return s.node.expr

        a_stride_expr: dict[Any, Union[int, torch.SymInt]] = {}
        b_stride_expr: dict[Any, Union[int, torch.SymInt]] = {}
        merged_strides: list[Union[int, torch.SymInt]] = [None] * len(a_ex_stride)  # type: ignore[list-item]
        for a_pair, b_pair in zip(a_stride_li, b_stride_li):
            assert a_pair is not None and b_pair is not None
            a_val, neg_i = a_pair
            b_val, _ = b_pair

            i = -neg_i
            if a_val == 0:
                assert b_val == 0, (a_val, b_val)
                merged_strides[i] = 0
                continue

            if _maybe_expr(a_val) in a_stride_expr:
                a_expr = a_stride_expr[_maybe_expr(a_val)]
                assert b_stride_expr[_maybe_expr(b_val)] == a_expr, (
                    f"a_stride_expr:{a_stride_expr}, b_stride_expr:{b_stride_expr}"
                )
                merged_strides[i] = a_expr
            else:
                if a_val == 1:
                    assert b_val == 1
                    a_stride_expr[_maybe_expr(a_val)] = 1
                    b_stride_expr[_maybe_expr(b_val)] = 1
                    merged_strides[i] = 1
                else:
                    # If we cannot find the expr of a_val in a_stride_expr, it means
                    # the strides is not a simple accumulative multiplication of sizes.
                    # In this case, we cannot determine the expr of strides from the new
                    # shapes so we error out and hint users to call contiguous().
                    raise RuntimeError(
                        f"It seems one of cond's output stride is not a simple accumulative multiplication of sizes. "
                        f"This could be because cond returns a slice of a tensor, which is not dense in memory. "
                        f"True branch has size {a_ex_size}, stride {a_ex_stride} and false branch has size {b_ex_size} "
                        f"stride {b_ex_stride}. Hint: can call t.contiguous(). "
                    )
            nxt_merged_stride_expr = merged_strides[i] * merged_size[i]
            a_stride_expr[_maybe_expr(a_val * a_ex_size[i])] = nxt_merged_stride_expr
            b_stride_expr[_maybe_expr(b_val * b_ex_size[i])] = nxt_merged_stride_expr
        return merged_strides

    merged_stride: list[Union[int, torch.SymInt]] = _bound_stride(
        a.size(), b.size(), a.stride(), b.stride(), merged_size
    )

    with mode:
        return torch.empty_strided(
            merged_size, merged_stride, dtype=a.dtype, device=a.device
        )


@cond_op.py_functionalize_impl
def cond_func(ctx, pred, true_fn, false_fn, inputs):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_pred = ctx.unwrap_tensors(pred)
    with ctx.redispatch_to_next():
        functional_true = ctx.functionalize(_maybe_run_with_interpreter(true_fn))
        functional_false = ctx.functionalize(_maybe_run_with_interpreter(false_fn))
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for branch, branch_name in [(true_fn, "cond_true"), (false_fn, "cond_false")]:
            _check_alias_and_mutation(
                branch, unwrapped_inputs, branch_name, pre_dispatch
            )

        cond_return = cond_op(
            unwrapped_pred, functional_true, functional_false, unwrapped_inputs
        )
        return ctx.wrap_tensors(cond_return)


@cond_op.py_impl(torch._C._functorch.TransformType.Vmap)
def cond_batch_rule(interpreter, pred, true_fn, false_fn, inputs):
    assert isinstance(inputs, (list, tuple)), (
        "Cond inputs must be a list or tuple of tensors"
    )
    assert all(isinstance(i, torch.Tensor) for i in inputs), (
        "Cond inputs must be a list of tensors"
    )

    pred_is_batched = isinstance(pred, torch.Tensor) and is_batchedtensor(pred)
    pred_ = get_unwrapped(pred) if pred_is_batched else pred

    # unbatched tensors are not vmapped
    tensors, in_dims = zip(
        *[
            (get_unwrapped(t), maybe_get_bdim(t)) if is_batchedtensor(t) else (t, None)
            for t in inputs
        ]
    )

    if pred_is_batched:
        # prepend "pred" and vmap everything
        tensors = (pred_,) + tensors
        in_dims = (0,) + in_dims

        def fn(p, *args):
            t = true_fn(*args)
            f = false_fn(*args)
            return torch.where(p, t[0], f[0])

        with interpreter.lower():
            result = torch.vmap(fn, in_dims=in_dims)(*tensors)

    else:
        # predicate is known at this stage and it is a boolean expression or a
        # tensor with one element.
        true_fn = torch.vmap(true_fn, in_dims=in_dims)
        false_fn = torch.vmap(false_fn, in_dims=in_dims)

        with interpreter.lower():
            result = cond_op(pred, true_fn, false_fn, tensors)

    if not isinstance(result, tuple):
        result = (result,)
    lvl = interpreter.level()
    return tuple(_add_batch_dim(r, 0, lvl) for r in result)
