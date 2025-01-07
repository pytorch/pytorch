# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable, List, Tuple

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    _temp_remove_metadata_torch_function_mode,
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


aten = torch._ops.ops.aten


def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
):
    assert len(args) == (num_init_leaves + num_inp_leaves)
    carry = pytree.tree_unflatten(args[:num_init_leaves], spec_init)
    xs = pytree.tree_unflatten(args[num_init_leaves:], spec_xs)
    carry, combined = combine_fn(carry, xs)
    carry_flat = pytree.tree_leaves(carry)
    combined_flat = pytree.tree_leaves(combined)
    assert num_init_leaves == len(carry_flat)
    return [*carry_flat, *combined_flat]


def _extract_carry_and_out(flat_out: List[Any], num_carry: int):
    return flat_out[:num_carry], flat_out[num_carry:]


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], Tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> Tuple[pytree.PyTree, pytree.PyTree]:
    r"""
    Performs an inclusive scan with a combine function.

    .. warning::
        `torch.scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> (Tensor, Tensor)``,
            or if xs is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
            The first input to ``combine_fn`` is the previous or initial scan carry
            and the second input element to ``combine_fn`` is a slice of the input along dim.
            The first output element of ``combine_fn`` is the next scan carry
            and the second output  of ``combine_fn`` represents a slice of the output.
            This function must be pure, i.e., no lifted arguments are supported at the moment
            and may not have any side effects.
        init (torch.Tensor or pytree with tensor leaves): The inital scan carry, a tensor, or nested pytree of tensors.
            The ``init`` is expected to have the same pytree structure as the first output element (i.e. carry)
            of ``combine_fn``.
        xs (torch.Tensor or pytree with tensor leaves): The input tensor, or nested pytree of tensors.

    Kwargs:
        dim (int): the dimension to scan over, default 0.
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.

    Returns:
        final_carry (torch.Tensor or pytree with tensor leaves),
            the final carry of the scan operation with same pytree structure as init.
        out (torch.Tensor or pytree with tensor leaves),
            each tensor leaf is a stacked output along first dim, where each slice is the output of a scan iteration.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            return next_carry, y

        i0 = torch.zeros(1)
        xs = torch.arange(5)
        # returns torch.tensor([10.]), torch.tensor([[0], [1.], [3.], [6.], [10.]])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    if not callable(combine_fn):
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))
    if not isinstance(reverse, bool):
        raise RuntimeError("Reverse must be a bool, but got " + str(type(reverse)))

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs, spec_xs = pytree.tree_flatten(xs)

    if len(leaves_init) == 0:
        raise RuntimeError("Init tensors must be provided")
    for x in leaves_init:
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"All init leaves must be a Tensor but got {x}")
    for x in leaves_xs:
        if not isinstance(x, torch.Tensor):
            raise RuntimeError(f"All xs leaves must be a Tensor but got {x}")
        if x.shape[dim] == 0:
            raise RuntimeError(
                f"All xs leaves must have a scan dimension > 0 but got {x}"
            )

    if len(leaves_xs) == 0:
        return pytree.tree_unflatten(leaves_init, spec_init), xs

    shape = leaves_xs[0].shape
    ndim = len(shape)
    dim = utils.canonicalize_dim(ndim, dim)

    out = combine_fn(
        pytree.tree_unflatten(leaves_init, spec_init),
        pytree.tree_unflatten([elem.select(dim, 0) for elem in leaves_xs], spec_xs),
    )

    # The first output needs to have the same pytree as init
    carry_leaves = pytree.tree_leaves(out[0])
    if len(carry_leaves) != len(leaves_init):
        raise RuntimeError(
            f"The number of leaves of the pytree of the new carry produced by the operator is {len(carry_leaves)}\
doesn't match the length of the pytree of the init {len(leaves_init)}"
        )

    def _check_new_carry_match_init(leaves_init, carry_leaves):
        for i, (init, new_carry) in enumerate(zip(leaves_init, carry_leaves)):
            if init.shape != new_carry.shape:
                raise RuntimeError(
                    f"The shape of the new_carry[{i}] {new_carry.shape} doesn't match that of the init[{i}] {init.shape}."
                )
            if init.stride() != new_carry.stride():
                raise RuntimeError(
                    f"The stride of the new_carry[{i}] {new_carry.stride()} doesn't match that of the init[{i}] {init.stride()}."
                )
            if init.dtype != new_carry.dtype:
                raise RuntimeError(
                    f"The dtype of the new_carry[{i}] {new_carry.dtype} doesn't match that of the init[{i}] {init.dtype}."
                )
            if init.requires_grad != new_carry.requires_grad:
                raise RuntimeError(
                    f"The requires_grad of the new_carry[{i}] {new_carry.requires_grad} doesn't match that of the init[{i}] {init.requires_grad}."  # noqa: B950
                )

    _check_new_carry_match_init(leaves_init, carry_leaves)

    # There are no pytree restrictions on the second output of the operator
    out_leaves, tree_out = pytree.tree_flatten(out[1])

    # TODO: Support closures/nn_modules in order to be able represent RNNs with scan
    # TODO: Support _inductor lowering
    # TODO: Support Autograd
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.
    # TODO: Unify the list inputs of control flow ops to tuple.

    combine_fn = functools.partial(
        wrap_combine_fn_flat,
        combine_fn=combine_fn,
        spec_init=spec_init,
        spec_xs=spec_xs,
        num_init_leaves=len(leaves_init),
        num_inp_leaves=len(leaves_xs),
    )

    def run_flattened_scan(combine_fn, leaves_init, leaves_xs, dim, reverse):
        return scan_op(
            combine_fn, leaves_init, leaves_xs, dim, reverse, additional_inputs=[]
        )

    if not torch._dynamo.is_compiling():
        from torch._dynamo.backends.debugging import (
            make_eager_backend_with_torch_function_mode,
        )

        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                if metadata_mode:
                    backend = make_eager_backend_with_torch_function_mode(metadata_mode)
                else:
                    backend = "eager"
                result = torch.compile(
                    run_flattened_scan, backend=backend, fullgraph=True
                )(
                    combine_fn,
                    leaves_init,
                    leaves_xs,
                    dim=dim,
                    reverse=reverse,
                )
    else:
        result = run_flattened_scan(combine_fn, leaves_init, leaves_xs, dim, reverse)

    result_carry, result_flat = _extract_carry_and_out(
        result,
        len(leaves_init),
    )

    return pytree.tree_unflatten(result_carry, spec_init), pytree.tree_unflatten(
        result_flat, tree_out
    )


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, init, xs, dim, reverse, additional_inputs):
        assert isinstance(additional_inputs, list), "additional_inputs must be a list."
        validate_subgraph_args_types(additional_inputs)
        return super().__call__(combine_fn, init, xs, dim, reverse, additional_inputs)


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, reverse=False, additional_inputs=None):
    additional_inputs = additional_inputs if additional_inputs is not None else []

    def _scan(init, xs):
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(xs) == 0:
            return carry, []

        num_elems = xs[0].shape[dim]
        if reverse:
            ind = num_elems - 1
        else:
            ind = 0

        # Compute dummy shapes for the pre-allocation
        num_init_leaves = len(init)
        dummy_carry, dummy_out = _extract_carry_and_out(
            operator(
                *carry,
                *[first_slice_copy(elem, dim) for elem in xs],
                *additional_inputs,
            ),
            num_init_leaves,
        )

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        # out: (num_elems, M, N, ...)
        # idx: (1, M, N)
        outs, idxs = zip(
            *[
                [
                    torch.zeros(
                        [num_elems] + list(e.size()),
                        dtype=e.dtype,
                        device=e.device,
                    ),
                    torch.ones_like(e, dtype=torch.int64).unsqueeze(0),
                ]
                for i, e in enumerate(dummy_out)
            ]
        )

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                # o: (num_elems, M, N ...)
                # x: (M, N, ...) -> (1, M, N)
                # ind * idx: (1, M, N,) with values to be ind
                # essentially: o[ind][n][k] = x[0][n][k]
                o.scatter_(0, ind * idx, x.unsqueeze(0))

        for i in range(num_elems):
            ind = i if not reverse else num_elems - i - 1
            carry, out = _extract_carry_and_out(
                operator(
                    *carry,
                    *[elem.select(dim, ind) for elem in xs],
                    *additional_inputs,
                ),
                num_init_leaves,
            )

            # Store the inits in the outs matrix.
            store_out_in_outs(out, ind)

        return [*carry, *list(outs)]

    scans = _scan(init, xs)
    return scans


def first_slice_copy(t: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.select_copy(t, dim, 0)


# We also do a clone with contiguous_format. This is to be consistent with
# eager semantic of scan, which stacks the outputs. The result is contiguous
# as a result of the stack operation.
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor:
    return (
        y.unsqueeze(0)
        .repeat(*([scan_length] + [1] * y.ndim))
        .clone(memory_format=torch.contiguous_format)
    )


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: List[torch.Tensor],
    xs: List[torch.Tensor],
    dim: int,
    reverse: bool,
    additional_inputs: List[torch.Tensor],
):
    from torch._dynamo.utils import clone_input

    with disable_proxy_modes_tracing():
        sample_inits = [clone_input(x_init) for x_init in init]
        sample_inputs = [first_slice_copy(x, dim) for x in xs]
        sample_additional_inputs = [
            clone_input(x) if isinstance(x, torch.Tensor) else x
            for x in additional_inputs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_inits, *sample_inputs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None

    carry, output = _extract_carry_and_out(outputs, len(init))

    for ini, ca in zip(init, carry):
        ini_meta = ini
        carry_meta = ca.meta["tensor_meta"]
        carry_val = ca.meta["val"]
        if (
            carry_val.device != ini_meta.device
            or carry_meta.dtype != ini_meta.dtype
            or carry_meta.shape != ini_meta.shape
        ):
            raise RuntimeError(
                f"Expected metadata of the combine_fn result {carry_meta} to be the same as "
                + f"the metadata of init with {ini_meta}"
            )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, init, xs, dim, reverse, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="scan"
    )

    with disable_proxy_modes_tracing():
        scan_length = xs[0].shape[dim]
        fake_carry, fake_outputs = _extract_carry_and_out(
            [o.meta["val"] for o in outputs], len(init)
        )
        out = (
            *fake_carry,
            *(stack_y(t, scan_length) for t in fake_outputs),
        )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, dim, reverse, additional_inputs):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, init, xs, dim, reverse, additional_inputs)


scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(scan_op, deferred_error=True)
)


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, init, xs, dim, reverse, additional_inputs):
    return trace_scan(
        mode, scan_op, combine_fn, init, xs, dim, reverse, additional_inputs
    )


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(mode, combine_fn, init, xs, dim, reverse, additional_inputs):
    with mode:
        scan_length = xs[0].shape[dim]
        carry, outputs = _extract_carry_and_out(
            combine_fn(
                *init,
                *[first_slice_copy(inp, dim) for inp in xs],
                *additional_inputs,
            ),
            len(init),
        )
        out = (
            *carry,
            *(stack_y(t, scan_length) for t in outputs),
        )
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, dim, reverse, additional_inputs):
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [
            first_slice_copy(inp, dim) for inp in unwrapped_xs
        ]
        sample_inputs = list(
            itertools.chain(
                unwrapped_init,
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        if _has_potential_branch_input_mutation(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be modifying the input!"
            )
        if _has_potential_branch_input_alias(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be aliasing the input!"
            )
        ret = scan_op(
            functional_combine_fn,
            unwrapped_init,
            unwrapped_xs,
            dim,
            reverse,
            unwrapped_additional_inputs,
        )
    return ctx.wrap_tensors(ret)


# dense implementation for scan. Used for testing only.
def _fake_scan(combine_fn, init, xs=None, dim=0, reverse=False):
    carry_leaves, carry_spec = pytree.tree_flatten(init)
    inp_leaves, inp_spec = pytree.tree_flatten(xs)
    if xs is None or len(inp_leaves) == 0:
        return init, []
    result_flat = []
    carry = carry_leaves
    op = reversed if reverse else lambda x: x

    dummy_carry, dummy_out = combine_fn(
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(
            [first_slice_copy(elem, dim) for elem in inp_leaves],
            inp_spec,
        ),
    )
    dummy_out_leaves, dummy_out_spec = pytree.tree_flatten(dummy_out)
    num_leaves = len(dummy_out_leaves)

    for ind in op(range(inp_leaves[0].size(dim))):
        xs = [elem.select(dim, ind) for elem in inp_leaves]

        carry, y = combine_fn(
            pytree.tree_unflatten(carry, carry_spec),
            pytree.tree_unflatten(xs, inp_spec),
        )
        carry, _ = pytree.tree_flatten(carry)
        y, _ = pytree.tree_flatten(y)
        result_flat.append(y)

    results = [
        torch.stack([e[leave_ind] for e in op(result_flat)])
        for leave_ind in range(num_leaves)
    ]
    return (
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(results, dummy_out_spec),
    )
