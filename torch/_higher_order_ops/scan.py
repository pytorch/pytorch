# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List, Tuple

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
    return (carry_flat, combined_flat)


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
            each tensor leaf is a stacked output along dim, where each slice is the output of a scan iteration.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            return next_carry, y

        i0 = torch.zeros(1)
        xs = torch.arange(1, 5)
        # returns torch.tensor([10]), torch.tensor([1., 3., 6., 10.])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    if not callable(combine_fn):
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))
    if not isinstance(reverse, bool):
        raise RuntimeError("Reverse must be a bool, but got " + str(type(reverse)))

    # TODO: Support closures/nn_modules in order to be able represent RNNs with scan
    # TODO: Support _inductor lowering
    # TODO: Support Autograd
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

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
                return torch.compile(scan, backend=backend, fullgraph=True)(
                    combine_fn, init, xs, dim=dim, reverse=reverse
                )

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs, spec_xs = pytree.tree_flatten(xs)

    if len(leaves_init) == 0:
        raise RuntimeError("Init tensors must be provided")
    if any(not isinstance(x, torch.Tensor) for x in leaves_init):
        raise RuntimeError("All init leaves must be a Tensor")
    if any(not isinstance(x, torch.Tensor) for x in leaves_xs):
        raise RuntimeError("All xs leaves must be a Tensor")
    if any(x.shape[dim] == 0 for x in leaves_xs):
        raise RuntimeError("All xs leaves must have a scan dimension > 0")

    if len(leaves_xs) > 0:
        shape = leaves_xs[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)

        out = combine_fn(
            pytree.tree_unflatten(leaves_init, spec_init),
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_xs], spec_xs
            ),
        )

        # The first output needs to have the same pytree as init
        carry_leaves = pytree.tree_leaves(out[0])
        if len(carry_leaves) != len(leaves_init):
            raise RuntimeError(
                "The number of leaves of the pytree of the new carry produced by the operator\
 needs to match the length of the pytree of the init"
            )
        if any(
            in_l.shape != out_l.shape for in_l, out_l in zip(leaves_init, carry_leaves)
        ):
            raise RuntimeError(
                "The pytree of the new carry produced by the operator needs to match the pytree of the init"
            )

        # There are no pytree restrictions on the second output of the operator
        out_leaves, tree_out = pytree.tree_flatten(out[1])

        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec_init=spec_init,
            spec_xs=spec_xs,
            num_init_leaves=len(leaves_init),
            num_inp_leaves=len(leaves_xs),
        )

        result_carry, result_flat = scan_op(
            combine_fn, leaves_init, leaves_xs, dim, reverse
        )

        return pytree.tree_unflatten(result_carry, spec_init), pytree.tree_unflatten(
            result_flat, tree_out
        )

    else:
        return pytree.tree_unflatten(leaves_init, spec_init), xs


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, init, xs, dim, reverse):
        return super().__call__(combine_fn, init, xs, dim, reverse)


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, reverse=False):
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
        dummy_carry, dummy_out = operator(
            *carry, *[aten.slice(elem, dim, 0, 1, 1) for elem in xs]
        )
        output_scanned_dim = dummy_out[0].shape[dim]

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        outs, outs_idxs = zip(
            *[
                [
                    torch.zeros(
                        list(e.size())[:dim]
                        + [list(e.size())[dim] * num_elems]
                        + list(e.size())[dim + 1 :],
                        dtype=e.dtype,
                        device=e.device,
                    ),
                    torch.cat(
                        [
                            id * t
                            for id, t in zip(
                                range(output_scanned_dim),
                                torch.tensor_split(
                                    torch.ones_like(e, dtype=torch.int64),
                                    output_scanned_dim,
                                    dim=dim,
                                ),
                            )
                        ],
                        dim,
                    ),
                ]
                for i, e in enumerate(dummy_out)
            ]
        )

        def store_in_mat(mat, out, d, index, index_modifier):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(mat, out, index):
                o.scatter_(d, idx + index_modifier, x)

        def cond(i, n, r):
            if (r and i < 0) or (not r and i > (n - 1)):
                return False
            else:
                return True

        def op(i):
            if reverse:
                return i - 1
            else:
                return i + 1

        while cond(ind, num_elems, reverse):
            carry, out = operator(
                *carry,
                *[aten.slice(elem, dim, ind, ind + 1, 1) for elem in xs],
            )

            # Store the inits in the outs matrix.
            store_in_mat(outs, out, dim, outs_idxs, ind * output_scanned_dim)

            ind = op(ind)

        return (carry, list(outs))

    scans = _scan(init, xs)
    return scans


def make_expanded_output_shape(dim, scan_length, shapes, use_sh=False):
    expanded_shapes = [
        tuple(
            (s if use_sh else -1) if i != dim else scan_length for i, s in enumerate(sh)
        )
        for sh in shapes
    ]
    return expanded_shapes


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: List[torch.Tensor],
    xs: List[torch.Tensor],
    dim: int,
    reverse: bool,
):
    with disable_proxy_modes_tracing():
        sample_inits = [
            torch.empty_like(
                x_init,
                dtype=x_init.dtype,
                device=x_init.device,
                requires_grad=x_init.requires_grad,
            )
            for x_init in init
        ]
        sample_xs = [
            torch.empty_like(
                aten.slice(x, dim, 0, 1, 1),
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x in xs
        ]
        combine_graph = reenter_make_fx(combine_fn)(*sample_inits, *sample_xs)

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None
    if len(outputs) != 2:
        raise RuntimeError(
            f"Expected to return 2 outputs: carry, out_matrix, but got:"
            f"\n  {len(outputs)} elements"
        )

    for ini, carry in zip(init, outputs[0]):
        ini_meta = ini
        carry_meta = carry.meta["tensor_meta"]
        carry_val = carry.meta["val"]
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

    args = (combine_graph, init, xs, dim, reverse)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="scan"
    )

    with disable_proxy_modes_tracing():
        scan_length = xs[0].shape[dim]
        fake_out_shapes = make_expanded_output_shape(
            dim, scan_length, [o.meta["val"].size() for o in outputs[1]]
        )

        def expand_tensor(t, sh):
            if isinstance(t, torch.Tensor):
                return t.expand(*sh)
            return t

        expanded_outs = [
            pytree.tree_map(expand_tensor, t.meta["val"], sh)
            for t, sh in zip(outputs[1], fake_out_shapes)
        ]
        out = (init, expanded_outs)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, dim, reverse):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, init, xs, dim, reverse)


scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(scan_op, deferred_error=True)
)


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, init, xs, dim, reverse):
    return trace_scan(mode, scan_op, combine_fn, init, xs, dim, reverse)


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(mode, combine_fn, init, xs, dim, reverse):
    with mode:
        dim_len = xs[0].shape[dim]
        carry, outputs = combine_fn(
            *init, *[aten.slice(inp, dim, 0, 1, 1) for inp in xs]
        )
        fake_out_shapes = [
            tuple(-1 if i != dim else dim_len for i, sh in enumerate(o.size()))
            for o in outputs
        ]
        out = (
            carry,
            tuple(t.expand(*sh).clone() for t, sh in zip(outputs, fake_out_shapes)),
        )
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, dim, reverse):
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_xs = list(itertools.chain(unwrapped_init, unwrapped_init))
        if _has_potential_branch_input_mutation(
            functional_combine_fn, sample_xs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be modifying the input!"
            )
        if _has_potential_branch_input_alias(
            functional_combine_fn, sample_xs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be aliasing the input!"
            )
        ret = scan_op(functional_combine_fn, unwrapped_init, unwrapped_xs, dim, reverse)
    return ctx.wrap_tensors(ret)
