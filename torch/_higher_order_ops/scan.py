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
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


aten = torch._ops.ops.aten


def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_input, num_init_leaves, num_inp_leaves
):
    assert len(args) == (num_init_leaves + num_inp_leaves)
    carry = pytree.tree_unflatten(args[:num_init_leaves], spec_init)
    input = pytree.tree_unflatten(args[num_init_leaves:], spec_input)
    carry, combined = combine_fn(carry, input)
    carry_flat = pytree.tree_leaves(carry)
    combined_flat = pytree.tree_leaves(combined)
    assert num_init_leaves == len(carry_flat)
    return (carry_flat, combined_flat)


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], Tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    input: pytree.PyTree,
    dim: int,
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
            or if input is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
            The first input to ``combine_fn`` is the previous or initial scan carry
            and the second input element to ``combine_fn`` is a slice of the input along dim.
            The first output element of ``combine_fn`` is the next scan carry
            and the second output  of ``combine_fn`` represents a slice of the output.
            This function must be pure, i.e., no lifted arguments are supported at the moment
            and may not have any side effects.
        init (torch.Tensor): The inital scan carry, a tensor, or nested pytree of tensors.
            The ``init`` is expected to have the same pytree structure as the first output element
            of ``combine_fn``.
        input (torch.Tensor): The input tensor, or nested pytree of tensors.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.

    Returns:
        final_carry (torch.Tensor): The final carry of the scan operation
        out (torch.Tensor): The output matrix for which each scan iteration produced a slice along dim

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y, x + y

        i0 = 0.
        x = [1., 2., 3., 4.]
        cumsum = scan(add, init=i0, input=x, dim)
        # This produces the output
        carry, cumsum = 10., torch.stack([1., 3., 7., 10.], dim)


    """
    if not callable(combine_fn):
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))

    # TODO: Support closures/nn_modules in order to be able represent RNNs with scan
    # TODO: Support _inductor lowering
    # TODO: Support Autograd
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _scan_op_wrapper(*args, **kwargs):
        return scan(*args, **kwargs)

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(_scan_op_wrapper, backend="eager", fullgraph=True)(
                combine_fn, init=init, input=input, dim=dim, reverse=reverse
            )

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_input, spec_input = pytree.tree_flatten(input)

    def check_arg(arg):
        if any(not isinstance(x, torch.Tensor) for x in arg):
            raise RuntimeError("All leaves must be a Tensor")
        if any(x.shape[dim] == 0 for x in arg):
            raise RuntimeError("All leaves must have a scan dimension > 0")

    check_arg(leaves_init)
    if len(leaves_init) == 0:
        raise RuntimeError("Init tensors must be provided")
    check_arg(leaves_input)

    if len(leaves_input) > 0:
        if reverse:
            leaves_input = [torch.flip(elem, [dim]) for elem in leaves_input]

        shape = leaves_input[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)

        out = combine_fn(
            pytree.tree_unflatten(leaves_init, spec_init),
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input], spec_input
            ),
        )

        # The first output needs to have the same pytree as init
        carry_leaves, tree_carry = pytree.tree_flatten(out[0])
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
            spec_input=spec_input,
            num_init_leaves=len(leaves_init),
            num_inp_leaves=len(leaves_input),
        )

        result_carry, result_flat = scan_op(
            combine_fn, leaves_init, leaves_input, dim, reverse
        )

        if reverse:
            result_flat = [torch.flip(elem, [dim]) for elem in result_flat]

        return pytree.tree_unflatten(result_carry, spec_init), pytree.tree_unflatten(
            result_flat, tree_out
        )

    else:
        return pytree.tree_unflatten(leaves_init, spec_init), input


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, init, input, dim, reverse):
        return super().__call__(combine_fn, init, input, dim, reverse)


scan_op = ScanOp()


def generic_scan(operator, init, input, dim=0, reverse=False):
    def _scan(init, input):
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(input) == 0:
            return carry, []

        num_elems = input[0].shape[dim]
        ind = 0

        # Compute dummy shapes for the pre-allocation
        dummy_carry, dummy_out = operator(
            *carry, *[aten.slice(elem, dim, 0, 1, 1) for elem in input]
        )

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        outs, idxs = zip(
            *[
                [
                    torch.zeros(
                        list(e.size())[:dim]
                        + [list(e.size())[dim] * num_elems]
                        + list(e.size())[dim + 1 :],
                        dtype=e.dtype,
                        device=e.device,
                    ),
                    torch.ones_like(e, dtype=torch.int64),
                ]
                for i, e in enumerate(dummy_out)
            ]
        )
        op = reversed if reverse else lambda x: x
        output_scanned_dim = dummy_out[0].shape[dim]
        real_idx = []
        for idx in idxs:
            if output_scanned_dim > 1:
                real_idx.append(
                    torch.cat(
                        [
                            id * t
                            for id, t in zip(
                                op(range(output_scanned_dim)),
                                torch.tensor_split(idx, output_scanned_dim, dim=dim),
                            )
                        ],
                        dim,
                    )
                )
            else:
                real_idx.append(torch.zeros_like(idx))

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, real_idx):
                o.scatter_(dim, idx + (ind * output_scanned_dim), x)

        while ind < num_elems:
            carry, out = operator(
                *carry,
                *[aten.slice(elem, dim, ind, ind + 1, 1) for elem in input],
            )

            # Store the inits in the outs matrix.
            store_out_in_outs(out, ind)

            ind += 1

        return (carry, list(outs))

    scans = _scan(init, input)
    return scans


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: List[torch.Tensor],
    input: List[torch.Tensor],
    dim: int,
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
        sample_inputs = [
            torch.empty_like(
                aten.slice(x, 0, 1, 1),
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x in input
        ]
        combine_graph = reenter_make_fx(combine_fn)(*sample_inits, *sample_inputs)

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None
    assert len(outputs) == len(
        input
    ), f"expected combine_fn to return {len(input)} results but got {len(outputs)}"

    for i, si, o in zip(input, sample_inputs, outputs):
        o_meta = o.meta["tensor_meta"]
        assert o_meta.dtype == i.dtype, (
            f"combine_fn output type mismatch, expected {i.dtype} "
            + f"but got {o_meta.dtype}"
        )
        assert (
            si.shape == o_meta.shape
        ), "The pytree of the out of the operator needs to match the input pytree"

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, input, dim)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="scan"
    )

    with disable_proxy_modes_tracing():
        dim_len = input[0].shape[dim]
        fake_out_shapes = [
            tuple(-1 if i != dim else dim_len for i, sh in enumerate(o.size()))
            for o in outputs
        ]
        out = (
            init,
            tuple(t.expand(*sh).clone() for t, sh in zip(outputs, fake_out_shapes)),
        )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, input, dim, reverse):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, init, input, dim, reverse)


scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(scan_op, deferred_error=True)
)


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, init, input, dim, reverse):
    return trace_scan(mode, scan_op, combine_fn, init, input, dim)


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(mode, combine_fn, init, input, dim, reverse):
    with mode:
        dim_len = input[0].shape[dim]
        carry, outputs = combine_fn(
            *init, *[aten.slice(inp, dim, 0, 1, 1) for inp in input]
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
def scan_functionalize(ctx, combine_fn, init, input, dim, reverse):
    unwrapped_input = ctx.unwrap_tensors(input)
    unwrapped_init = ctx.unwrap_tensors(init)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_inputs = list(itertools.chain(unwrapped_init, unwrapped_init))
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
            functional_combine_fn, unwrapped_init, unwrapped_input, dim, reverse
        )
    return ctx.wrap_tensors(ret)
