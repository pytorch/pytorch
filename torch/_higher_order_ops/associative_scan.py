# mypy: allow-untyped-defs
import functools
import itertools
from typing import Callable, List

import torch
import torch._higher_order_ops
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
    autograd_not_implemented,
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

def first_slice_copy(t: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.select_copy(t, dim, 0)


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
            num_xs = len(xs)
            fw_xs_1 = [pytree.tree_map(_from_fun, x).select(dim, 0) for x in xs]
            fw_xs_2 = [pytree.tree_map(_from_fun, x).select(dim, 0) for x in xs]
            outs = combine_fn(*fw_xs_1, *fw_xs_2)

            fw_outputs = [pytree.tree_map(_from_fun, o) for o in outs]
            fw_outputs_2 = [pytree.tree_map(_from_fun, o) for o in outs]
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

        def joint_wrapper_x(*args):
            # ones = [torch.ones_like(el) for el in args[:num_xs]]
            # g_out = joint_graph(*(*ones, *args))
            g_out = joint_graph(*args)
            return [*g_out[:num_xs],]
        
        def joint_wrapper_h(*args):
            # ones = [torch.ones_like(el) for el in args[:num_xs]]
            # g_out = joint_graph(*(*ones, *args))
            g_out = joint_graph(*args)
            return [*g_out[num_xs:],]
        
        # def joint_wrapper_bwd_asssociative_scan(x: torch.Tensor, y: torch.Tensor):
        #     # grad_L_h, grad_h_h = args[0][:num_xs], args[0][num_xs:2*num_xs]
        #     # grad_L_x, g = args[1][:num_xs], args[1][num_xs:2*num_xs]
            
        #     # def mul(x: torch.Tensor, y: torch.Tensor):
        #     #     return x * y
    
        #     # g_out = mul(*grad_h_h, *g)
        #     # g_out = [g_L_h + g_h * go for g_L_h, g_h, go in zip(x[:num_xs], x[num_xs:], y[num_xs:])]
        #     g_out = [g_L_h + g_h * go for g_L_h, g_h, go in zip(y[:num_xs], y[num_xs:], x[num_xs:])]
            
        #     return (*x[:num_xs], *g_out)
        
        # def joint_wrapper_bwd_scan(x: torch.Tensor, y: torch.Tensor):
        #     g_out = [g_L_h + g_h * go for g_L_h, g_h, go in zip(y[:num_xs], y[num_xs:], x[num_xs:])]
        #     return ((*g_out, *g_out), *g_out)

        joint_graph_x = _maybe_reenter_make_fx(joint_wrapper_x)(*fw_outputs, *fw_xs_1, *fw_xs_2)
        joint_graph_h = _maybe_reenter_make_fx(joint_wrapper_h)(*fw_outputs, *fw_xs_1, *fw_xs_2)
        # # joint_graph_bwd_associative_scan = _maybe_reenter_make_fx(joint_wrapper_bwd_asssociative_scan)((*fw_outputs, *fw_xs_1), (*fw_outputs_2, *fw_xs_2))
        
        # num_scan = xs[0].shape[dim]
        # fw_xs_1_s = [pytree.tree_map(_from_fun, x) for x in xs]
        # xs_joint = [torch.split(x, [1, num_scan - 1], dim=dim) for x in fw_xs_1_s]
        # xs_init, xs_grads = zip(*[[xs_j[0], xs_j[1]] for xs_j in xs_joint])
        # fw_xs_init = [pytree.tree_map(_from_fun, x) for x in xs_init]
        # fw_xs_grads = [pytree.tree_map(_from_fun, x) for x in xs_grads]
        
        # fw_outputs_s = [pytree.tree_map(_from_fun, x) for x in xs]
        # outs_joint = [torch.split(o, [1, num_scan - 1], dim=dim) for o in fw_outputs_s]
        # outs_init, outs_grads = zip(*[[xs_j[0], xs_j[1]] for xs_j in outs_joint])
        # fw_outs_init = [pytree.tree_map(_from_fun, x) for x in outs_init]
        # fw_outs_grads = [pytree.tree_map(_from_fun, x) for x in outs_grads]
        # joint_graph_bwd_associative_scan = _maybe_reenter_make_fx(joint_wrapper_bwd_scan)((*fw_xs_init, *fw_outs_init), (*fw_outs_grads, *fw_xs_grads))

        # return fw_graph, joint_graph_x, joint_graph_h, joint_graph_bwd_associative_scan
        return fw_graph, joint_graph_x, joint_graph_h


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

    # if not torch._dynamo.is_compiling():
    #     with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
    #         return torch.compile(associative_scan, fullgraph=True)(
    #             combine_fn, xs, dim, reverse=reverse, combine_mode=combine_mode
    #         )

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
    out_leaves, tree_out = pytree.tree_flatten(out)
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
    # raise NotImplementedError("associative_scan is not implemented for eager")
    return generic_associative_scan(combine_fn, xs, dim)


class ScanAutogradOp(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph_x, 
        joint_graph_h,
        # joint_graph_bwd_associative_scan,
        dim,
        num_xs,
        *flat_args,
    ):
        ctx._joint_graph_x = joint_graph_x
        ctx._joint_graph_h = joint_graph_h
        # ctx._joint_graph_bwd_associative_scan = joint_graph_bwd_associative_scan
        ctx._dim = dim
        ctx._num_xs = num_xs
        xs = flat_args
        
        scan_length = xs[0].shape[dim]
        ctx._scan_length = scan_length

        with torch._C._AutoDispatchBelowAutograd():
            outs = associative_scan_op(fw_graph, xs, dim)
            ctx.save_for_backward(*(*xs, *outs))
            
            # # #BW in FWD
            # mapped_joint_graph_x = torch.vmap(joint_graph_x, dim, dim)
            # mapped_joint_graph_h = torch.vmap(joint_graph_h, dim, dim)
            
            # # Compute the derivatives of the parts of the combine_fn
            # ones = torch.unsqueeze(torch.ones_like(first_slice_copy(xs[0], dim)), dim)
            # zeros = torch.unsqueeze(torch.zeros_like(first_slice_copy(xs[0], dim)), dim)
            
            # grads_x = mapped_joint_graph_x(*xs, *outs)
            # grads_x = [torch.concat([aten.slice(torch.flip(g, [dim]), dim, 1, None, 1), ones], dim) for g in grads_x]
            
            # grads_h = mapped_joint_graph_h(*xs, *outs)
            # grads_h = [torch.flip(g, [dim]) for g in grads_h]
        
            # # Form the gradient matrix
            # def cumprod_and_prepad(val, size):
            #     return torch.concat([zeros] * max(size - 1 - val.shape[dim], 0.) + [ones] + [torch.cumprod(val, dim)], dim)
            
            # # ones_inp = [torch.unsqueeze(torch.ones_like(first_slice_copy(g, dim)), dim) for g in grads_h]
            # # grads_h_ext = [torch.concat([o, aten.slice(h, dim, 0, -1, 1)], dim) for h, o in zip(grads_h, ones_inp)]
            # # grads_h_ext = [torch.concat([o, h], dim) for h, o in zip(grads_x, ones_inp)]
            
            # # helper_mat = torch.triu(torch.ones(num_xs + 1, num_xs + 1))
            # # helper_mat *= grads_h_ext
            # # helper_mat = [[torch.cumprod(aten.slice(el, dim, 0, n, 1), dim) for el in grads_h_ext] for n in range(scan_length)]
            # gradient_mat = [[cumprod_and_prepad(aten.slice(el, dim, n, -1, 1), scan_length)for n in range(0, scan_length, 1)]  for el in grads_h]
            # # gradient_mat = [ for n in range(num_xs)]
            
            # grads = [torch.flip(torch.sum(torch.stack(g_mat, 0) * g_x, 0), [dim]) for g_mat, g_x in zip(gradient_mat, grads_x)]
            
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

            #TODO: Use a more complicated example such as mul
            The ``fw_graph`` f(.,.), used in the forward function, is the operator used during the scan. For example
            def f(x: torch.Tensor, y: torch.Tensor):
                return x + y

            The ``joint_graph`` g(.,.), used in the backward function, is the gradient of the function f(.,.).
            It computes the gradients for x and y of f. For example for the function f above
            def g(x: torch.Tensor, y: torch.Tensor):
                return 1., 1.
            In other words, the first output of g represents df(x,y)/dx, while the second one represents df(x,y)/dy.
            This will be exploited in the algorithm below.

            The inputs to ``scan`` in the forward path are x_1, x_2, ..., x_T
            The outputs of ``scan`` in the forward path are y_1, y_2, ..., y_T, where
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

        """
        from torch._higher_order_ops.scan import scan

        joint_graph_x = ctx._joint_graph_x
        joint_graph_h = ctx._joint_graph_h
        # joint_graph_bwd_associative_scan = ctx._joint_graph_bwd_associative_scan
        dim = ctx._dim
        num_xs = ctx._num_xs
        scan_length = ctx._scan_length
        
        # import pdb
        # pdb.set_trace()
        flat_args = ctx.saved_tensors
        xs, outs = flat_args[:num_xs], flat_args[num_xs:]

        # pdb.set_trace()
        mapped_joint_graph_x = torch.vmap(joint_graph_x, dim, dim)
        mapped_joint_graph_h = torch.vmap(joint_graph_h, dim, dim)
        
        with torch._C._AutoDispatchBelowAutograd():
            
            # Compute the derivatives of the parts of the combine_fn
            ones = torch.unsqueeze(torch.ones_like(first_slice_copy(xs[0], dim)), dim)
            zeros = torch.unsqueeze(torch.zeros_like(first_slice_copy(xs[0], dim)), dim)
            
            # pdb.set_trace()
            # Form the gradient matrix
            def cumprod_and_prepad(val, size):
                return torch.concat([zeros] * max(size - 1 - val.shape[dim], 0.) + [ones] + [torch.cumprod(val, dim)], dim)
            
            # pdb.set_trace()
            grads_x = mapped_joint_graph_x(*flat_grads, *xs, *outs)
            grads_x = [torch.concat([aten.slice(torch.flip(g, [dim]), dim, 1, None, 1), ones], dim) for g in grads_x]
            
            # pdb.set_trace()
            grads_h = mapped_joint_graph_h(*flat_grads, *xs, *outs)
            grads_h = [torch.flip(g, [dim]) for g in grads_h]
            
            # pdb.set_trace()
            gradient_mat = [[cumprod_and_prepad(aten.slice(el, dim, n, -1, 1), scan_length)for n in range(0, scan_length, 1)]  for el in grads_h]
        
            # pdb.set_trace()
            grads = [torch.flip(torch.sum(torch.stack(g_mat, 0) * g_x, 0), [dim]) for g_mat, g_x in zip(gradient_mat, grads_x)]
            
            return *[None] * 5, *grads


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

    num_leaves_xs = len(xs)

    (
        fw_graph,
        joint_graph_x,
        joint_graph_h,
        # joint_graph_bwd_associative_scan,
    ) = create_fw_bw_graph_combinefn(combine_fn, xs, dim)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph_x, 
        joint_graph_h,
        # joint_graph_bwd_associative_scan,
        dim,
        num_leaves_xs,
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
