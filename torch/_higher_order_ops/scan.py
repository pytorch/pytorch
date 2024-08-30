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
    _set_compilation_env,
    autograd_not_implemented,
    reenter_make_fx,
    unique_graph_id,
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


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        if len(arg) != n:
            raise ValueError("length mismatch: {list(map(len, args))}")

    def nf(a):
        return f(*a)

    return list(map(nf, zip(*args)))


def scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    input: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    init: pytree.PyTree = [],  # noqa: B006
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative pointwise combine function.

    .. warning::
        `torch.scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, pointwise, and satisfy the associative property.
        input (torch.Tensor): The input tensor, or nested pytree of tensors.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to the dimension.
        init (torch.Tensor): The inital scan carry, a tensor, or nested pytree of tensors.
            The tensors are expected to have the same shape as the output tensors of ``combine_fn``

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = scan(add, x, dim)

    """
    assert callable(combine_fn), "combine_fn must be a callable, but got {combine_fn}"
    assert isinstance(dim, int), "dim must be an int, but got {type(dim)}"

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _scan_op_wrapper(*args, **kwargs):
        return scan(*args, **kwargs)

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(_scan_op_wrapper, backend="eager", fullgraph=True)(
                combine_fn, input, dim=dim, reverse=reverse, init=init
            )

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_input, spec_input = pytree.tree_flatten(input)

    if reverse:
        leaves_input = [torch.flip(elem, [dim]) for elem in leaves_input]

    assert (
        len(leaves_init) > 0 or len(leaves_input) > 0
    ), "expected at least 1 init or input leaf"
    if len(leaves_init) > 0:
        shape = leaves_init[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)
        num_el = shape[dim]
        output_spec = spec_init

        assert all(
            isinstance(x, torch.Tensor) for x in leaves_init
        ), "If init leaves are provided, they must be a Tensor"
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
        assert all(
            isinstance(x, torch.Tensor) for x in leaves_input
        ), "If input leaves are provided, they must be a Tensor"

        assert all(
            x.shape[dim] > 0 for x in leaves_input
        ), "If input leaves are provided, the scan dimension must be > 0"

        out = combine_fn(
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input], output_spec
            ),
            pytree.tree_unflatten(
                [aten.slice(elem, dim, 0, 1, 1) for elem in leaves_input], output_spec
            ),
        )
        out_leaves, tree_out = pytree.tree_flatten(out)
        assert len(leaves_input) == len(
            out_leaves
        ), "The number of leaves of the pytree of the output of the operator needs to match the lenght of the pytree of the input"
        for in_l, out_l in zip(leaves_init, out_leaves):
            assert (
                in_l.shape == out_l.shape
            ), "The pytree of the output of the operator needs to match the pytree of the init"

    # Add the init back to the result_flat as the first element
    if len(leaves_input) > 0:
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


def generic_scan(operator, elems_flat, elems_init_flat, dim=0):
    def combine(a, b, dim):
        return torch.concatenate([a, b], dim=dim)

    cmb = functools.partial(combine, dim=dim)

    def _scan(elems, elems_init):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[dim]

        ind = 0
        xs = elems_init

        # Approach with concatenate
        # outs = xs

        # Approach without concatenation
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
                for i, e in enumerate(elems_init)
            ]
        )

        # Approach with concatenate
        # outs = list(safe_map(cmb, outs, xs))

        # Approach without concatenation
        for o, x, idx in zip(outs, xs, idxs):
            o.scatter_(dim, idx * ind, x)

        while ind < num_elems:
            xs = operator(
                *xs,
                *[aten.slice(elem, dim, ind, ind + 1, 1) for elem in elems],
            )

            # Approach with concatenate
            # outs = list(safe_map(cmb, outs, xs))

            # Approach without concatenation
            for o, x, idx in zip(outs, xs, idxs):
                o.scatter_(dim, idx * (ind + 1), x)

            ind += 1

        return outs

    if len(elems_flat) == 0:
        return []

    scans = _scan(elems_flat, elems_init_flat)
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
        out = [aten.clone(x) for x in input]

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
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, input, init, dim):
    with mode:
        if len(init) > 0:
            return [x.clone() for x in init]
        else:
            return combine_fn(*input, *input)


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, input, init, dim):
    unwrapped_input = ctx.unwrap_tensors(input)
    unwrapped_init = ctx.unwrap_tensors(init)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        ret = scan_op(functional_combine_fn, unwrapped_input, unwrapped_init, dim)
    return ctx.wrap_tensors(ret)
