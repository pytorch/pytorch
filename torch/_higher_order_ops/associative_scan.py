# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable, List

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
    first_slice_copy,
    reenter_make_fx,
    unique_graph_id,
)
from torch._inductor.utils import is_pointwise_use
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)

from .utils import _from_fun, create_fw_bw_graph


aten = torch._ops.ops.aten


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    assert len(args) == 2 * num_leaves
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    combined = combine_fn(lhs, rhs)
    combined_leaves = pytree.tree_leaves(combined)
    assert num_leaves == len(combined_leaves)
    return combined_leaves


def _interleave(a, b, dim=0):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[dim] == b.shape[dim] + 1):
        pad = (
            [0] * ((b.ndim - dim - 1) * 2 + 1)
            + [1]
            + [0] * (b.ndim * 2 - ((b.ndim - dim - 1) * 2 + 2))
        )
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=dim + 1)
    interleaved = torch.flatten(stacked, start_dim=dim, end_dim=dim + 1)
    if b_trunc:
        # TODO: find torch alternative for slice_along dim for torch.jit.script to work
        interleaved = aten.slice(interleaved, dim, 0, b.shape[dim] + a.shape[dim] - 1)
    return interleaved


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        if len(arg) != n:
            raise ValueError("length mismatch: {list(map(len, args))}")

    def nf(a):
        return f(*a)

    return list(map(nf, zip(*args)))


def create_fw_bw_graph_combinefn(combine_fn, xs):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    # Helper wrapper for the autograd forward.
    # This wrapper ensures that the forward returns all carries
    # instead of only the last one
    # The gradients of the carries forwarded to the output are
    # detached in order not to raise problems with the function aliasing outputs

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            fw_xs_1 = [first_slice_copy(pytree.tree_map(_from_fun, x)) for x in xs]
            fw_xs_2 = [first_slice_copy(pytree.tree_map(_from_fun, x)) for x in xs]
            outs = combine_fn(*fw_xs_1, *fw_xs_2)

            # TODO: Support partial Autograd for associative_scan later
            if pytree.tree_any(
                lambda t: not t.requires_grad,  # type: ignore[union-attr]
                (outs),
            ):
                raise RuntimeError(
                    "gradient flags of the combine_fn output differ from the xs. Consider checking for `torch.no_grad()` statements?"
                )

            fw_outputs = [pytree.tree_map(_from_fun, o) for o in outs]
            if any(not isinstance(out, torch.Tensor) for out in fw_outputs):
                raise RuntimeError(
                    "Expect outputs produced by combine_fn to only contains tensors. "
                    f"Got types {[type(out) for out in fw_outputs]}."
                )

            fw_graph, joint_graph = create_fw_bw_graph(
                combine_fn,
                False,
                (*fw_xs_1, *fw_xs_2),
                (*fw_outputs,),
            )

        return fw_graph, joint_graph


class AssociativeScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("associative_scan")

    def __call__(self, combine_fn, xs):
        return super().__call__(combine_fn, xs)


associative_scan_op = AssociativeScanOp()


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative combine function.

    .. warning::
        `torch.associative_scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment,
            satisfy the associative property and have no side-effects.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure, may only contain pointwise operations
            and ``xs`` must be CUDA tensors.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
    if not callable(combine_fn):
        raise ValueError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise ValueError("Dim must be an int, but got " + str(type(dim)))
    if combine_mode not in ["pointwise", "generic"]:
        raise ValueError(
            "Combine_mode must either 'pointwise' or 'generic', but got {combine_mode}"
        )

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(associative_scan, fullgraph=True)(
                combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
            )

    leaves, spec = pytree.tree_flatten(xs)

    if combine_mode == "pointwise" and not all(l.device.type == "cuda" for l in leaves):
        raise ValueError(
            "For combine_mode='pointwise', all input tensors need to be on CUDA"
        )

    if len(leaves) == 0:
        raise ValueError("Expected at least 1 xs leaf")
    if any(not isinstance(x, torch.Tensor) for x in leaves):
        raise ValueError("xs leaves must be a Tensor")
    if any(x.is_sparse for x in leaves):
        raise ValueError("xs leaves must dense Tensors, consider using `to_dense()`")
    if any(x.ndim < dim for x in leaves):
        raise ValueError(
            "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
        )
    if any(x.shape[dim] == 0 for x in leaves):
        raise ValueError(
            "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
        )

    if reverse:
        leaves = [torch.flip(elem, [dim]) for elem in leaves]

    ndim = leaves[0].ndim
    orig_scan_dim = utils.canonicalize_dim(ndim, dim)
    leaves = [torch.movedim(elem, dim, 0) for elem in leaves]
    shape = leaves[0].shape

    for x in leaves[1:]:
        assert x.shape == shape, "All xs tensors must have the same shape"

    # Call the combine_fn with only a slice along the scan dim
    # and check whether the output leaves have the same slice dimensions
    sliced_leaves = [first_slice_copy(leaf, dim) for leaf in leaves]
    sliced_shape = sliced_leaves[0].shape

    out = combine_fn(
        pytree.tree_unflatten(sliced_leaves, spec),
        pytree.tree_unflatten(sliced_leaves, spec),
    )
    out_leaves = pytree.tree_leaves(out)
    if len(leaves) != len(out_leaves):
        raise RuntimeError(
            "The number of leaves of the pytree of the output of the operator needs to match the length of the pytree of the input"
        )
    if any(
        x.shape != sliced_shape
        or x.dtype != x_sliced.dtype
        or x.device != x_sliced.device
        or x.stride() != x_sliced.stride()
        for x, x_sliced in zip(out_leaves, sliced_leaves)
    ):
        raise RuntimeError(
            f"The metadata of the output of the operator needs to match the meta data of the xs pytree"
            f"\n  xs metadata             : {[(x.shape, x.dtype, x.device, x.stride()) for x in sliced_leaves]}"
            f"\n  operator output metadata: {[(x.shape, x.dtype, x.device, x.stride()) for x in out_leaves]}"
        )

    if combine_mode == "generic":
        # The generic_associative_scan implementation calls the combine_fn with a `batch` along the scan dimension
        # For example, consider:
        # def add(x: torch.Tensor, y: torch.Tensor):
        #     return x + y
        # leaves = torch.tensor([[0.0, 1.0, 2.0, 3.0]
        #                        [0.0, 1.0, 2.0, 3.0]])
        # which has shape 2 x 4;
        # dim = 1;
        # In the first iteration of `_scan` the combine_fn gets invoked with
        # combine_fn([torch.tensor([[0.0, 2.0],
        #                           [0.0, 2.0]])],
        #            [torch.tensor([[1.0, 3.0],
        #                           [1.0, 3.0]])])
        # The arguments are of shape 2 x 2, but can be evaluated in parallel along the scan dimension.
        # TODO: In case of the additional inputs, we the in_dims should be set to None
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=torch.vmap(
                combine_fn,
                in_dims=(
                    pytree.tree_unflatten([0] * len(leaves), spec),
                    pytree.tree_unflatten([0] * len(leaves), spec),
                ),
                out_dims=0,
            ),
            spec=spec,
            num_leaves=len(leaves),
        )
        result_flat = generic_associative_scan(combine_fn, leaves)
    else:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=spec,
            num_leaves=len(leaves),
        )
        result_flat = associative_scan_op(combine_fn, leaves)

    if reverse:
        result_flat = [torch.flip(elem, [0]) for elem in result_flat]

    result_flat = [torch.movedim(elem, 0, orig_scan_dim) for elem in result_flat]

    return pytree.tree_unflatten(result_flat, spec)


def generic_associative_scan(operator, leaves, dim=0):
    r"""
    This function performs the associative_scan operation.
    The algorithm works by recursively collecting neighbours of ``leaves`` and subsequently
    applying the ``operator`` on all pairs in parallel along ``dim``.
    The results of the recursive calls are later combined.

    Args:
        operator (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, pointwise, and satisfy the associative property.
        leaves (torch.Tensor): A list of torch.Tensors converted from the pytree of
            ``xs`` provided to ``associative_scan``.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        leaves = torch.tensor([0.0, 1.0, 2.0, 3.0])

        First iteration of _scan ->
            # odd_elems -> apply operator on all neighbours
            # odd_elems = operator([torch.tensor([0.0, 2.0])],
            #                      [torch.tensor([1.0, 3.0])])
            odd_elems = torch.tensor([1.0, 5.0])
            Second iteration of _scan ->
                # odd_elems = operator([torch.tensor([1.0])],
                #                      [torch.tensor([5.0])])
                odd_elems = torch.tensor([6.0])
                # even_elems -> apply operator on all odd_elems and
                # every second element of ``elems``, starting from the second element.
                # even_elems is expanded with the first element of ``elems``
                even_elems = [1.0]
                # Merges odd_elems and even_elems
                res = torch.tensor([1.0, 6.0])
            # even_elems -> apply operator on all odd_elems and
            # every second element of ``elems``, starting from the second element.
            # even_elems is expanded with the first element of ``elems``
            even_elems = [0.0, 3.0]
            # Merges odd_elems and even_elems
            res = torch.tensor([0.0, 1.0, 3.0, 6.0])

    """

    def _scan(elems):
        """Perform the actual recursive scan on ``elems``."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = operator(
            *[aten.slice(elem, dim, 0, -1, 2) for elem in elems],
            *[aten.slice(elem, dim, 1, None, 2) for elem in elems],
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = operator(
                *[aten.slice(e, dim, 0, -1) for e in odd_elems],
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
            )
        else:
            even_elems = operator(
                *odd_elems,
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
            )

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
            torch.cat([aten.slice(elem, dim, 0, 1), result], dim=dim)
            if result.shape.numel() > 0 and elem.shape[dim] > 0
            else result
            if result.shape.numel() > 0
            else aten.slice(
                elem, dim, 0, 1
            )  # Jax allows/ignores concat with 0-dim, Pytorch does not
            for (elem, result) in zip(elems, even_elems)
        ]

        return list(
            safe_map(functools.partial(_interleave, dim=dim), even_elems, odd_elems)
        )

    scans = _scan(leaves)

    return scans


def trace_associative_scan(
    proxy_mode, func_overload, combine_fn: Callable, xs: List[torch.Tensor]
):
    with disable_proxy_modes_tracing():
        sample_xs = [first_slice_copy(x) for x in itertools.chain(xs, xs)]
        combine_graph = reenter_make_fx(combine_fn)(*sample_xs)

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

        if not all(is_pointwise_use(use) or use.op == "output" for use in node.users):
            raise ValueError(
                "For combine_mode='pointwise', the combine_fn needs to be pointwise"
            )

    assert outputs is not None
    assert len(outputs) == len(
        xs
    ), f"expected combine_fn to return {len(xs)} results but got {len(outputs)}"

    for i, o in zip(xs, outputs):
        o_meta = o.meta["tensor_meta"]
        assert o_meta.dtype == i.dtype, (
            f"combine_fn output type mismatch, expected {i.dtype} "
            + f"but got {o_meta.dtype}"
        )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, xs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = [aten.clone(x) for x in xs]

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, xs):
    return generic_associative_scan(combine_fn, xs)


class ScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph,
        *xs,
    ):
        ctx._joint_graph = joint_graph
        num_xs = len(xs)
        ctx._num_xs = num_xs

        scan_length = xs[0].shape[0]
        ctx._scan_length = scan_length
        ctx._mapped_joint_graph = torch.vmap(joint_graph, 0, 0)

        with torch._C._AutoDispatchBelowAutograd():
            outs = associative_scan_op(fw_graph, xs)
            ctx.save_for_backward(*(*xs, *outs))

        return (*outs,)

    @staticmethod
    def backward(ctx, *flat_grads):
        r"""
        This function computes the gradients of the scan operation.
        It does so by factorizing the components of the chainrule into
        a elementwise multiplcation of a matrix and a vector.
        The rows of the matrix can be efficiently computed using ``cumprod``.

        Args:
            flat_grads (torch.Tensor): The tensor of upstream gradients, or anested pytree of tensors.

        Example::

            The ``fw_graph`` f(.,.), used in the forward function, is the operator used during the scan. For example
            def f(x: torch.Tensor, y: torch.Tensor):
                return x * y

            The ``joint_graph`` g(.,.), used in the backward function, is the gradient of the function f(.,.).
            It computes the gradients for x and y of f. For example for the function f above
            def g(x: torch.Tensor, y: torch.Tensor):
                return y, x
            In other words, the first output of g represents df(x,y)/dx, while the second one represents df(x,y)/dy.
            This will be exploited in the algorithm below.

            The inputs to ``associative_scan`` in the forward path are x_1, x_2, ..., x_T
            The outputs of ``associative_scan`` in the forward path are y_1, y_2, ..., y_T, where
            y_1 = x_1
            y_2 = f(y_1, x_2)
            ...
            y_T = f(y_{T-1}, x_T)

            The gradients of y_T with respect to the vector x are computed as:
            dy_T / dx = dy_T/dx_1 + dy_T/dx_2 + ... + dy_T/dx_T

            A few examples:
            dy_T/dx_T = df(y_{T-1}, x_T)/dx_T -> second output of g(y_{T-1}, x_T)

            dy_T/dx_{T-1} = df(y_{T-1}, x_T)/dy_{T-1} . df(y_{T-2}, x_{T-1})/dx_{T-1}
                          -> first output of g(y_{T-1}, x_T) . second output of g(y_{T-2}, x_{T-1})

            dy_T/dx_{T-2} = df(y_{T-1}, x_T)/dy_{T-1}
                            . df(y_{T-2}, x_{T-1})/dy_{T-2}
                            . df(y_{T-3}, x_{T-2})/dx_{T-2}
                          ->  first output of g(y_{T-1}, x_T)
                            . first output of g(y_{T-2}, x_{T-1})
                            . second output of g(y_{T-3}, x_{T-2})

            A conceptually similar pattern can be observerd for dy_{T-1} / dx
            dy_{T-1}/dx_T = 0

            dy_{T-1}/dx_{T-1} = df(y_{T-2}, x_{T-1})/dx_{T-1} -> second output of g(y_{T-2}, x_{T-1})

            dy_{T-1}/dx_{T-2} = df(y_{T-2}, x_{T-1})/dy_{T-2} . df(y_{T-3}, x_{T-2})/dx_{T-2}
                              -> first output of g(y_{T-2}, x_{T-1})
                              . second output of g(y_{T-3}, x_{T-2})

            If one inspects the pattern carefully, it becomes aparant that there is a product of
            'first outputs', followed by the last term which is a 'second output'.
            This can be represented with a matrix-vector multiplication, where the rows of the matrix contain
            the products of the 'first ouputs' and the vector contains the 'second outputs'.
            Furthermore, the product of 'first outputs' is continuously expanded leftwards with
            additional time steps. Therefore, the products can also be computed utilizing cumprod.
            The final gradients can be computed using an elementwise matrix-vector multiplication.

            This implementation is inspired by the "grid form" outlined on
            https://justintchiu.com/blog/pscan_diff/

            As one example, consider:
            x = torch.arange(1, 4)
            o = torch.cumprod(x)

            The gradients of `o` with respect to `x` can be computed as follows:
            Step 1.: Compute the gradients at every scan element with respect to h and x
            grads_h = [1, 2, 3, 4]
            grads_x = [1, 1, 2, 6]
            Step 2.: Compute the gradient matrix
            h_mat = [[do_0/dh_0, do_1/dh_0, do_2/dh_0, do_3/dh_0],
                     [do_1/dh_1, do_1/dh_1, do_2/dh_1, do_3/dh_1],
                     [do_0/dh_2, do_1/dh_2, do_2/dh_2, do_3/dh_2],
                     [do_0/dh_3, do_1/dh_3, do_2/dh_3, do_3/dh_3]]
            which results in
            h_mat = [[1, 2, 6, 24],
                     [0, 1, 3, 12],
                     [0, 0, 1,  4],
                     [0, 0, 0,  1]]
            Note: This involves the cumprod
            Step 3.: scale the matrix with the upstream gradients, e.g., dL_i/do_i
            h_mat * dL_i/do_i
            Step 4.: Reduce the matrix with sum along the columns to get the total contributions for x_i
            sum_mat = [33, 16, 5, 1]
            Step 5.: Scale with the grads_x, e.g., do_i/dx_i, to obtain the final gradients
            grads = [33, 16, 5, 1] * [1, 1, 2, 6]
            grads = [33, 16, 10, 6]
        """

        dim = 0
        scan_length = ctx._scan_length
        num_xs = ctx._num_xs

        # Extract the inputs to the forward path and outputs from the forward path
        flat_args = ctx.saved_tensors
        xs, outs = flat_args[:num_xs], flat_args[num_xs:]

        # Helper variables for grads_h and grads_x computation
        ones = torch.unsqueeze(torch.ones_like(first_slice_copy(xs[0])), 0)

        # vmap joint graph over scan dimension
        mapped_joint_graph = ctx._mapped_joint_graph

        with torch._C._AutoDispatchBelowAutograd():
            """Step 1"""
            shifted_outs = [
                torch.concat([ones, aten.slice(o, 0, 0, -1, 1)], 0) for o in outs
            ]

            # Function to compute the gradients with respect
            # *) to the inputs (xs) -> grads_xs
            # *) to the previosus outputs -> grads_hs
            def compute_grad_hs_xs():
                # Compute the partial grads_x and grads_h by only setting part of the gradients for the joint_graph to 1
                # This is requried in some cases, especially where tuples are used as inputs to the combine_fn
                def compute_part_grads(flat_grad_ind):
                    flat_grads_init = [
                        torch.ones_like(x)
                        if flat_grad_ind == ind
                        else torch.zeros_like(x)
                        for ind, x in enumerate(xs)
                    ]
                    grads = mapped_joint_graph(*flat_grads_init, *shifted_outs, *xs)
                    return (*grads,)

                # Compute all the partial gradients
                grad_parts = [torch.unsqueeze(g, 0) for g in compute_part_grads(0)]
                for part_ind in range(1, num_xs):
                    grad_parts = [
                        torch.concat([gp, torch.unsqueeze(g, 0)], 0)
                        for gp, g in zip(grad_parts, compute_part_grads(part_ind))
                    ]

                return grad_parts

            # Compute the grads_xs and grads_hs by collecting all the partial gradients
            grads_intermediate = compute_grad_hs_xs()
            grads_h_parts, grads_x_parts = (
                grads_intermediate[:num_xs],
                grads_intermediate[num_xs:],
            )

            # Helper variables to generate the gradient matrix
            zeros_p = torch.zeros_like(aten.slice(grads_x_parts[0], dim + 1, 0, 1, 1))
            ones_d = torch.ones((scan_length, scan_length), device=outs[0].device)
            len_shape = len(grads_x_parts[0].shape) - 1

            def expand_to_equal_dims(el, target_ndim):
                while len(el.shape) < target_ndim:
                    el = torch.unsqueeze(el, -1)
                return el

            triu = torch.triu(ones_d, diagonal=1)
            triu = torch.unsqueeze(triu, 0)
            triu = expand_to_equal_dims(triu, len_shape + 2)
            tril = torch.tril(ones_d, diagonal=0)
            tril = torch.unsqueeze(tril, 0)
            tril = expand_to_equal_dims(tril, len_shape + 2)
            tril2 = torch.tril(ones_d, diagonal=-1)
            tril2 = torch.unsqueeze(tril2, 0)
            tril2 = expand_to_equal_dims(tril2, len_shape + 2)
            eye = torch.eye(scan_length, device=outs[0].device)
            eye = expand_to_equal_dims(eye, len_shape + 1)
            a_eye = triu + tril2

            def compute_gradient_for_leaf(grads_h_parts, grads_x_parts, flat_grads):
                # The first output of the associative_scan operation is always the first element of xs.
                # Therefore, the first grads_h is always zero and the first grads_x is always 1
                grads_h_parts = torch.concat(
                    [
                        torch.zeros_like(aten.slice(grads_h_parts, dim + 1, 0, 1, 1)),
                        aten.slice(grads_h_parts, dim + 1, 1, None, 1),
                    ],
                    dim + 1,
                )
                grads_x_parts = torch.concat(
                    [
                        torch.stack(
                            [
                                torch.ones_like(gp)
                                if ind == 0
                                else torch.zeros_like(gp)
                                for ind, gp in enumerate(
                                    aten.slice(grads_x_parts, dim + 1, 0, 1, 1)
                                )
                            ],
                            0,
                        ),
                        aten.slice(grads_x_parts, dim + 1, 1, None, 1),
                    ],
                    dim + 1,
                )

                # Prepare the components for the gradient computation
                grads_h_parts = aten.slice(
                    torch.concat(
                        [zeros_p, torch.flip(grads_h_parts, [dim + 1])], dim + 1
                    ),
                    dim + 1,
                    0,
                    -1,
                    1,
                )
                grads_x = torch.flip(torch.sum(grads_x_parts, 0), [dim])
                flat_grads = torch.unsqueeze(
                    aten.slice(
                        torch.concat([torch.flip(flat_grads, [dim]), ones], dim),
                        dim,
                        0,
                        -1,
                        1,
                    ),
                    dim + 1,
                )

                def create_grads_h_matrix_part(gh_p):
                    # Create the gradient matrix from the grads_h by duplicating grads_h and masking the appropriate regions
                    # Dim 1 is the dimention of the individual parts
                    # Dim 2 will be the newly expanded dim that is of scan_length
                    # Dim 3 is the scan dim with scan_length
                    h_mat = torch.tile(
                        torch.unsqueeze(gh_p, 1), [1, scan_length] + [1] * len_shape
                    )
                    h_mat = h_mat * triu + tril
                    return h_mat

                def compute_gradient_factors_part(h_mat):
                    # Comput the gradient by applying the cumprod on the scan dim and masking the irrelevant values
                    return torch.cumprod(h_mat, dim + 2) - tril2

                def stack_and_reduce_parts(mat_fact):
                    # Summing the different parts (dim 0) without the diagonal and adding the diagonal back
                    return torch.sum(mat_fact * a_eye, 0) + eye

                """Step 2 + 3"""
                grads_h_prod_mat = (
                    stack_and_reduce_parts(
                        compute_gradient_factors_part(
                            create_grads_h_matrix_part(grads_h_parts)
                        )
                    )
                    * flat_grads
                )

                """Step 4 + 5"""
                grad = torch.flip(torch.sum(grads_h_prod_mat * grads_x, 0), [dim])
                return grad

            compute_gradient_for_leaf_mapped = torch.vmap(
                compute_gradient_for_leaf, 0, 0
            )

            # Compute the gradients for all the leaves in parallel
            grads = torch.split(
                compute_gradient_for_leaf_mapped(
                    torch.stack(grads_h_parts),
                    torch.stack(grads_x_parts),
                    torch.stack(flat_grads),
                ),
                1,
                0,
            )

            return *[None] * 2, *grads


@associative_scan_op.py_impl(DispatchKey.Autograd)
def associative_scan_autograd(combine_fn, xs):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    # TODO: Figure out how to do this in dispatcher so that we don't have to do this check here
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (xs),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return associative_scan_op(combine_fn, xs)

    # TODO: Support partial Autograd for associative_scan later
    if pytree.tree_any(
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (xs),
    ):
        raise RuntimeError(
            "associative_scan currently only supports Autograd if all xs require gradients."
        )

    # TODO: The create_fw_bw is always invoked twice:
    # Once in the forward path and
    # once in the backward path, where it should only be invoked for the grad grad case.
    # We don't support this currently
    if not torch.is_grad_enabled():
        # This clause is hit in the case of double backward.
        # Currently scan does not support this and thus we just dummy call another scan
        with torch._C._AutoDispatchBelowAutograd():
            return associative_scan_op(combine_fn, xs)

    (
        fw_graph,
        joint_graph,
    ) = create_fw_bw_graph_combinefn(combine_fn, xs)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph,
        *xs,
    )
    return (*flat_out,)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, xs):
    return trace_associative_scan(mode, associative_scan_op, combine_fn, xs)


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, xs):
    with mode:
        return [x.clone() for x in xs]


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs):
    unwrapped_xs = ctx.unwrap_tensors(xs)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        ret = associative_scan_op(functional_combine_fn, unwrapped_xs)
    return ctx.wrap_tensors(ret)


def _fake_associative_scan(combine_fn, xs, dim, reverse=False):
    inp_leaves, spec = pytree.tree_flatten(xs)
    result_flat: List[Any] = []
    num_leaves = len(inp_leaves)
    op = reversed if reverse else lambda x: x

    for ind in op(range(inp_leaves[0].size(dim))):
        r = [
            inp_leaves[leave_ind][(slice(None),) * dim + (ind,)]
            for leave_ind in range(num_leaves)
        ]
        if (ind > 0 and not reverse) or (
            ind < (inp_leaves[0].size(dim) - 1) and reverse
        ):
            r = combine_fn(
                pytree.tree_unflatten(result_flat[-1], spec),
                pytree.tree_unflatten(r, spec),
            )
        r_flat, _ = pytree.tree_flatten(r)
        result_flat.append(r_flat)

    results = [
        torch.stack([e[leave_ind] for e in op(result_flat)], dim)
        for leave_ind in range(num_leaves)
    ]
    return pytree.tree_unflatten(results, spec)
