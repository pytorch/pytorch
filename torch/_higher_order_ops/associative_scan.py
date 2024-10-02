# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
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

from .utils import _from_fun, _maybe_reenter_make_fx, create_fw_bw_graph


aten = torch._ops.ops.aten

# TODO: These functions can be merged with the corresponding functions from scan
# once it is merged
def first_slice_copy(t: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.select_copy(t, dim, 0)

def get_gradient_mask(tensor_list):
    return [True if v is not None and v.requires_grad else False for v in tensor_list]


def mask_gradient(grads, mask):
    return [g for g, m in zip(grads, mask) if m]


def wrap_combine_fn_flat(*args, combine_fn, spec, num_leaves):
    assert len(args) == 2 * num_leaves
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    combined = combine_fn(lhs, rhs)
    combined_leaves = pytree.tree_leaves(combined)
    assert num_leaves == len(combined_leaves)
    return combined_leaves


def _interleave(a, b, dim):
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


def create_fw_bw_graph_combinefn(combine_fn, xs, dim):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    # Helper wrapper for the autograd forward.
    # This wrapper ensures that the forward returns all carries
    # instead of only the last one
    # The gradients of the carries forwarded to the output are
    # detached in order not to raise problems with the function aliasing outputs

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            fw_xs_1 = [first_slice_copy(pytree.tree_map(_from_fun, x), dim) for x in xs]
            fw_xs_2 = [first_slice_copy(pytree.tree_map(_from_fun, x), dim) for x in xs]
            outs = combine_fn(*fw_xs_1, *fw_xs_2)

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
            
            gradient_mask_post = get_gradient_mask(outs)
            def wrapper(*args):
                grads = joint_graph(*args)
                grads = mask_gradient(grads, gradient_mask_post * 2)
                return *grads,
            joint_graph = _maybe_reenter_make_fx(wrapper)(*fw_outputs, *fw_xs_1, *fw_xs_2)

        return fw_graph, joint_graph, gradient_mask_post


class AssociativeScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("associative_scan")

    def __call__(self, combine_fn, xs, dim):
        return super().__call__(combine_fn, xs, dim)


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
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))
    if combine_mode not in ["pointwise", "generic"]:
        raise RuntimeError(
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
        raise RuntimeError("Expected at least 1 xs leaf")
    if any(not isinstance(x, torch.Tensor) for x in leaves):
        raise RuntimeError("xs leaves must be a Tensor")

    if reverse:
        leaves = [torch.flip(elem, [dim]) for elem in leaves]

    shape = leaves[0].shape
    ndim = len(shape)
    dim = utils.canonicalize_dim(ndim, dim)

    for x in leaves[1:]:
        assert x.shape == shape, "All xs tensors must have the same shape"

    out = combine_fn(
        pytree.tree_unflatten(leaves, spec),
        pytree.tree_unflatten(leaves, spec),
    )
    out_leaves = pytree.tree_leaves(out)
    if len(leaves) != len(out_leaves):
        raise RuntimeError(
            "The number of leaves of the pytree of the output of the operator needs to match the length of the pytree of the input"
        )
    if any(x.shape != shape for x in out_leaves):
        raise RuntimeError(
            "The pytree of the output of the operator needs to match the xs pytree"
        )

    combine_fn = functools.partial(
        wrap_combine_fn_flat, combine_fn=combine_fn, spec=spec, num_leaves=len(leaves)
    )

    if combine_mode == "generic":
        result_flat = generic_associative_scan(combine_fn, leaves, dim)
    else:
        result_flat = associative_scan_op(combine_fn, leaves, dim)

    if reverse:
        result_flat = [torch.flip(elem, [dim]) for elem in result_flat]

    return pytree.tree_unflatten(result_flat, spec)


def generic_associative_scan(operator, elems_flat, dim=0):
    r"""
    This function performs the associative_scan operation.
    The algorithm works by recursively collecting neighbours of ``elems_flat`` and subsequently
    applying the ``operator`` on all pairs in parallel along ``dim``.
    The results of the recursive calls are later combined.

    Args:
        operator (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, pointwise, and satisfy the associative property.
        elems_flat (torch.Tensor): A list of torch.Tensors converted from the pytree of
            ``xs`` provided to ``associative_scan``.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        elems_flat = torch.tensor([0.0, 1.0, 2.0, 3.0])

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

    scans = _scan(elems_flat)

    return scans


def trace_associative_scan(
    proxy_mode, func_overload, combine_fn: Callable, xs: List[torch.Tensor], dim: int
):
    with disable_proxy_modes_tracing():
        sample_xs = [
            torch.empty_like(
                x,
                dtype=x.dtype,
                device=x.device,
                requires_grad=x.requires_grad,
            )
            for x in itertools.chain(xs, xs)
        ]
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

    args = (combine_graph, xs, dim)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = [aten.clone(x) for x in xs]

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, xs, dim):
    raise NotImplementedError("associative_scan is not implemented for eager")


class ScanAutogradOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph,
        dim,
        gradient_mask,
        *flat_args,
    ):
        xs = flat_args
        
        ctx._joint_graph = joint_graph
        ctx._dim = dim
        
        num_xs = len(xs)
        ctx._num_xs = num_xs
        
        scan_length = xs[0].shape[dim]
        ctx._scan_length = scan_length
        ctx._gradient_mask = gradient_mask
        ctx._mapped_joint_graph = torch.vmap(joint_graph, int(dim), int(dim))
        
        with torch._C._AutoDispatchBelowAutograd():
            outs = associative_scan_op(fw_graph, xs, dim)
            ctx.save_for_backward(*(*xs, *outs))
            
        return *outs,

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

        """
        dim = int(ctx._dim)
        scan_length = ctx._scan_length
        num_xs = ctx._num_xs
        gradient_mask = ctx._gradient_mask
        num_xs_masked = sum(gradient_mask)
        
        # Extract the inputs to the forward path and outputs from the forward path
        flat_args = ctx.saved_tensors
        xs, outs = flat_args[:num_xs], flat_args[num_xs:]
        
        # Helper variables
        ones = torch.unsqueeze(torch.ones_like(first_slice_copy(xs[0], dim)), dim)
        zeros = torch.unsqueeze(torch.zeros_like(first_slice_copy(xs[0], dim)), dim)
        shifted_outs = [torch.concat([ones, aten.slice(o, dim, 0, -1, 1)], dim) for o in outs]

        # vmap joint graph over scan dimension
        mapped_joint_graph = ctx._mapped_joint_graph
        
        def expand_grads_with_None(real_grads, mask):
            g_list = []
            true_cnt = 0
            for m in mask:
                if m:
                    g_list.append(real_grads[true_cnt])
                    true_cnt += 1
                else:
                    g_list.append(None)
            return g_list
        
        with torch._C._AutoDispatchBelowAutograd():
            # Mask the gradients for the variables that do not require gradients for partial gradient support
            flat_grads = [fg for fg, m in zip(flat_grads, gradient_mask) if m]
            
            # Function to compute the gradients with respect 
            # *) to the inputs (xs) -> grads_xs
            # *) to the previous outputs -> grads_hs
            def compute_grad_hs_xs():
                
                # Compute the partial grads_x and grads_h by only setting part of the gradients for the joint_graph to 1
                def compute_part_grads(flat_grad_ind):
                    flat_grads_init = [torch.ones_like(x) if flat_grad_ind == ind else torch.zeros_like(x) for ind, x in enumerate(xs)]
                    grads = mapped_joint_graph(*flat_grads_init, *shifted_outs, *xs)
                    return *grads,
            
                # Compute all the partial gradients
                grad_parts = [torch.unsqueeze(g, 0) for g in compute_part_grads(0)]
                for part_ind in range(1, num_xs):
                    grad_parts = [torch.concat([gp, torch.unsqueeze(g, 0)], 0) for gp, g in zip(grad_parts, compute_part_grads(part_ind))]

                return grad_parts
            
            # Compute the grads_xs and grads_hs by collecting all the partial gradients
            grads_intermediate = compute_grad_hs_xs()
            grads_hs, grads_xs = grads_intermediate[:num_xs_masked], grads_intermediate[num_xs_masked:]
            
            # In case of the associative_scan, the first output mirrors the first scan element of xs
            # Therefore, the first grads_hs are all zeros and the first grads_xs are only ones
            grads_hs = [torch.concat([torch.zeros_like(aten.slice(g, dim + 1, 0, 1, 1)), aten.slice(g, dim + 1, 1, None, 1)], dim + 1) for g in grads_hs]
            grads_xs = [torch.concat([torch.stack([torch.ones_like(gp) if ind == 0 else torch.zeros_like(gp) for ind, gp in enumerate(aten.slice(g, dim + 1, 0, 1, 1))], 0), aten.slice(g, dim + 1, 1, None, 1)], dim + 1) for g in grads_xs]
            
            # Compute the cumprod of the rows of the gradient matrix and fill the remainder with zeros
            def cumprod_and_prepad(fg, val, size):
                return torch.concat([zeros] * max(size - 1 - val.shape[dim + 1], 0) + [ones * fg] + [fg * torch.sum(torch.cumprod(val, dim + 1), 0)], dim)

            # Compute the gradients for a single element of xs
            # The computations are done on dim + 1, because g_x and g_h have all partial gradients on dim 0
            # The partial gradients are combined in the process of this function
            def compute_grad_xs(fg, g_x, g_h):
                g_x = torch.flip(g_x, [dim + 1])
                g_h = torch.flip(g_h, [dim + 1])
                fg = torch.concat([torch.flip(fg, [dim]), ones], dim)
                
                # Create the matrix consisting of 
                gradient_mat = [cumprod_and_prepad(aten.slice(fg, dim, n, n + 1, 1), aten.slice(g_h, dim + 1, n, -1, 1), scan_length) for n in range(0, scan_length, 1)]
                grads = torch.flip(torch.sum(torch.stack(gradient_mat, 0) * torch.sum(g_x, 0), 0), [dim])
                return grads
            
            # Compute the gradients in parallel for all elements of xs
            compute_grad_xs_mapped = torch.vmap(compute_grad_xs, 0, 0)
            grads = [torch.squeeze(el, 0) for el in torch.split(compute_grad_xs_mapped(torch.stack(flat_grads, 0), torch.stack(grads_xs, 0), torch.stack(grads_hs, 0)), 1, 0)]

            # Expand the gradients with Nones for partial gradient support
            grads = expand_grads_with_None(grads, gradient_mask)
            
            return *[None] * 4, *grads


@associative_scan_op.py_impl(DispatchKey.Autograd)
def associative_scan_autograd(combine_fn, xs, dim):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    # TODO: Figure out how to do this in dispatcher so that we don't have to do this check here
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (xs),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return associative_scan_op(combine_fn, xs, dim)

    # TODO: The create_fw_bw is always invoked twice:
    # Once in the forward path and
    # once in the backward path, where it should only be invoked for the grad grad case.
    # We don't support this currently
    if not torch.is_grad_enabled():
        # This clause is hit in the case of double backward.
        # Currently scan does not support this and thus we just dummy call another scan
        with torch._C._AutoDispatchBelowAutograd():
            return associative_scan_op(combine_fn, xs, dim)

    (
        fw_graph,
        joint_graph,
        gradient_mask,
    ) = create_fw_bw_graph_combinefn(combine_fn, xs, dim)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph,
        dim,
        gradient_mask,
        *xs,
    )
    return *flat_out,


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, xs, dim):
    return trace_associative_scan(mode, associative_scan_op, combine_fn, xs, dim)


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, xs, dim):
    with mode:
        return [x.clone() for x in xs]


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs, dim):
    unwrapped_xs = ctx.unwrap_tensors(xs)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        ret = associative_scan_op(functional_combine_fn, unwrapped_xs, dim)
    return ctx.wrap_tensors(ret)
