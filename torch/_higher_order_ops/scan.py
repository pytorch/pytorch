# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List

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


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    assert len(args) == 2 * num_leaves
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    combined = combine_fn(lhs, rhs)
    combined_leaves = pytree.tree_leaves(combined)
    assert num_leaves == len(combined_leaves)
    return combined_leaves


def scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    input: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    init: pytree.PyTree = None,
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with a combine function.

    .. warning::
        `torch.scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment.
        input (torch.Tensor): The input tensor, or nested pytree of tensors.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        init (torch.Tensor): The inital scan carry, a tensor, or nested pytree of tensors that
            represents the first output of the scan, default ``None``. The ``init`` is expected to have the
            same pytree structure and shape as the output tensors of ``combine_fn``.
            In case the ``init`` is ``None``, the first element of input is used as ``init``.


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        # Usage without the usage of ``init``
        cumsum = scan(add, x, dim)
        # This produces the output
        cumsum = torch.stack([x0, add(x0, x1), add(add(x0, x1), x2)], dim)

        # Usage with the usage of ``init``
        cumsum = scan(add, x, dim, init=i0)
        # This produces the output
        cumsum = torch.stack([i0, add(i0, x0), add(add(i0, x0), x1)], dim)


    """
    if not callable(combine_fn):
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))

    # TODO: Support closures/nn_modules in order to be able represent RNNs with scan
    # TODO: Support _inductor lowering
    # TODO: Support Autograd
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.
    # TODO: Implement new "user interface" that allows the init and the output of scan to have
    # different shapes. E.g. We compute cumsum, [a, a+b, a+b+c], but output [a, (a+b).sin(), (a+b+c).sin()]

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _scan_op_wrapper(*args, **kwargs):
        return scan(*args, **kwargs)

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(_scan_op_wrapper, backend="eager", fullgraph=True)(
                combine_fn, input, dim=dim, reverse=reverse, init=init
            )

    init = [] if init is None else init

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_input, spec_input = pytree.tree_flatten(input)

    if reverse:
        leaves_input = [torch.flip(elem, [dim]) for elem in leaves_input]

    if len(leaves_init) == 0 and len(leaves_input) == 0:
        raise RuntimeError("Expected at least 1 init or input leaf")
    if len(leaves_init) > 0:
        if any(not isinstance(x, torch.Tensor) for x in leaves_init):
            raise RuntimeError("Init leaves must be a Tensor")
        if any(x.shape[dim] == 0 for x in leaves_init):
            raise RuntimeError("Init leaves must have a scan dimension > 0")
    if len(leaves_input) > 0:
        if any(not isinstance(x, torch.Tensor) for x in leaves_input):
            raise RuntimeError("Input leaves must be a Tensor")
        if any(x.shape[dim] == 0 for x in leaves_input):
            raise RuntimeError("Input leaves must have a scan dimension > 0")

    if len(leaves_init) > 0:
        shape = leaves_init[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)
        num_el = shape[dim]
        output_spec = spec_init
    else:
        shape = leaves_input[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)
        num_el = shape[dim]
        output_spec = spec_input

        # If no init is provided, take the first time step of input as the init
        # and crop it off the original input
        leaves_init = [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input]
        if num_el > 1:
            leaves_input = [aten.slice(elem, dim, 1, None, 1) for elem in leaves_input]
        else:
            leaves_input = []

    if len(leaves_input) > 0:
        out = combine_fn(
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input], output_spec
            ),
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input], output_spec
            ),
        )
        out_leaves, tree_out = pytree.tree_flatten(out)
        if len(leaves_input) != len(out_leaves):
            raise RuntimeError(
                "The number of leaves of the pytree of the output of the operator\
 needs to match the length of the pytree of the input"
            )
        if any(
            in_l.shape != out_l.shape for in_l, out_l in zip(leaves_init, out_leaves)
        ):
            raise RuntimeError(
                "The pytree of the output of the operator needs to match the pytree of the init"
            )

        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=output_spec,
            num_leaves=len(leaves_input),
        )

        result_flat = scan_op(combine_fn, leaves_input, leaves_init, dim)

        if reverse:
            result_flat = [torch.flip(elem, [dim]) for elem in result_flat]

        return pytree.tree_unflatten(result_flat, output_spec)

    else:
        return pytree.tree_unflatten(leaves_init, output_spec)


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, input, init, dim):
        return super().__call__(combine_fn, input, init, dim)


scan_op = ScanOp()


def generic_scan(operator, input, init, dim=0):
    def _scan(input, init):
        """Perform scan on `elems` using `elems_init."""
        num_elems = input[0].shape[dim]
        ind = 0
        out = init

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        outs, idxs = zip(
            *[
                (
                    torch.zeros(
                        list(e.size())[:dim]
                        + [num_elems + 1]
                        + list(e.size())[dim + 1 :],
                        dtype=e.dtype,
                        device=e.device,
                    ),
                    torch.ones_like(e, dtype=torch.int64),
                )
                for i, e in enumerate(init)
            ]
        )

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                o.scatter_(dim, idx * ind, x)

        # Store the inits in the outs matrix.
        # These are the first elements of the scan outputs
        store_out_in_outs(out, ind)

        while ind < num_elems:
            out = operator(
                *out,
                *[aten.slice(elem, dim, ind, ind + 1, 1) for elem in input],
            )

            # Store the inits in the outs matrix.
            store_out_in_outs(out, ind + 1)

            ind += 1

        return outs

    if len(input) == 0:
        raise ValueError(
            "Input needs to be a list of tensors with a non-zero scan dimension"
        )

    scans = _scan(input, init)
    return scans


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    input: List[torch.Tensor],
    init: List[torch.Tensor],
    dim: int,
):
    with disable_proxy_modes_tracing():
        sample_inputs = [
            torch.empty_like(
                x_init,
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x, x_init in itertools.chain(zip(input, init), zip(input, init))
        ]
        combine_graph = reenter_make_fx(combine_fn)(*sample_inputs)

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
            tuple(-1 if i != dim else (dim_len + 1) for i, sh in enumerate(inp.size()))
            for inp in input
        ]
        out = tuple(t.expand(*sh).clone() for t, sh in zip(init, fake_out_shapes))

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, input, init, dim):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, input, init, dim)


scan_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(scan_op, deferred_error=True)
)


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, input, init, dim):
    return trace_scan(mode, scan_op, combine_fn, input, init, dim)


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(mode, combine_fn, input, init, dim):
    with mode:
        dim_len = input[0].shape[dim]
        fake_out_shapes = [
            tuple(-1 if i != dim else (dim_len + 1) for i, sh in enumerate(inp.size()))
            for inp in input
        ]
        out = tuple(t.expand(*sh).clone() for t, sh in zip(init, fake_out_shapes))
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, input, init, dim):
    unwrapped_input = ctx.unwrap_tensors(input)
    unwrapped_init = ctx.unwrap_tensors(init)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_inputs = [
            x for x in itertools.chain(unwrapped_init, unwrapped_init)
        ]  # noqa: C416
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
        ret = scan_op(functional_combine_fn, unwrapped_input, unwrapped_init, dim)
    return ctx.wrap_tensors(ret)
