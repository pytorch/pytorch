# mypy: allow-untyped-defs
import functools
import itertools
from collections.abc import Sequence
from typing import Any, Callable, Optional

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.cond import create_bw_fn, materialize_as_graph
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    _set_compilation_env,
    check_meta_consistency,
    first_slice_copy,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


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


def get_tensor_mask(tensor_list: list[Any]) -> list[bool]:
    # Returns a mask whether a list element is a tensor or not
    return [True if isinstance(v, torch.Tensor) else False for v in tensor_list]


def mask_list(
    mask: list[bool], inp: list[Any], other: Optional[list[Any]] = None
) -> list[Any]:
    # Masks elements on an `inp` list.
    # If other is None, then the elements of the `inp` list where the mask is False are removed
    # If other is not None, then the elements of the `inp` list where the mask is False are
    # replaced with the elements of the `other` list
    if other is not None:
        return [i if m else o for m, i, o in zip(mask, inp, other)]
    else:
        return [i for m, i in zip(mask, inp) if m]


def first_slice_copy_with_grad(li):
    # First_slice_copy does not keep the original requires_grad flag,
    # but we need it for materialize_as_graph
    # in order to compute the correct gradients
    slc = [first_slice_copy(x).requires_grad_(x.requires_grad) for x in li]
    return slc


def split_into_chunks(iterable: Sequence[Any], chunk_sizes: list[int]) -> list[Any]:
    it = iter(iterable)
    assert sum(chunk_sizes) == len(
        iterable
    ), "the sum of all chunks needs to match the length of the iterable."
    return [list(itertools.islice(it, size)) for size in chunk_sizes]


class AssociativeScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("associative_scan")

    def __call__(self, combine_fn, xs, additional_inputs):
        # There is currently an issue that the ScanOp is sometimes called with
        # the additional_inputs being a list. See https://github.com/pytorch/pytorch/issues/145785
        # Once this issue is resolved, the assertion should only allow tuples
        # and the tuple cast should be removed
        assert isinstance(
            additional_inputs, (tuple, list)
        ), "additional_inputs must be a tuple."
        validate_subgraph_args_types(additional_inputs)
        return super().__call__(combine_fn, xs, additional_inputs)


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

    # TODO: Support lifted arguments in inductor for associative_scan
    # TODO: Support autograd for cases with lifted arguments for combine_mode=pointwise

    if not callable(combine_fn):
        raise ValueError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise ValueError("Dim must be an int, but got " + str(type(dim)))
    if combine_mode not in ["pointwise", "generic"]:
        raise ValueError(
            "Combine_mode must either 'pointwise' or 'generic', but got {combine_mode}"
        )

    if not torch.compiler.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(associative_scan, fullgraph=True, backend="eager")(
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
    if any(x.ndim <= dim for x in leaves):
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

    # Call the combine_fn with only a slice along the scan dim
    # and check whether the output leaves have the same slice dimensions
    sliced_leaves = [first_slice_copy(leaf) for leaf in leaves]

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
        x.shape != x_sliced.shape
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
        result_flat = generic_associative_scan(combine_fn, leaves, additional_inputs=())
    else:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=spec,
            num_leaves=len(leaves),
        )
        result_flat = associative_scan_op(combine_fn, leaves, additional_inputs=())

    if reverse:
        result_flat = [torch.flip(elem, [0]) for elem in result_flat]

    result_flat = [torch.movedim(elem, 0, orig_scan_dim) for elem in result_flat]

    return pytree.tree_unflatten(result_flat, spec)


def generic_associative_scan(operator, leaves, dim=0, additional_inputs=()):
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
        additional_inputs (Tuple of tensors): A tuple of lifted parameters from the global scope.
            This parameter will be populated internally.

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
            *additional_inputs,
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = operator(
                *[aten.slice(e, dim, 0, -1) for e in odd_elems],
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )
        else:
            even_elems = operator(
                *odd_elems,
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
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
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
):
    with disable_proxy_modes_tracing():
        sample_xs = [first_slice_copy(x) for x in itertools.chain(xs, xs)]
        combine_graph = reenter_make_fx(combine_fn)(*sample_xs, *additional_inputs)

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None
    assert len(outputs) == len(
        xs
    ), f"expected combine_fn to return {len(xs)} results but got {len(outputs)}"

    xs_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        first_slice_copy(x) for x in xs
    ]
    output_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        c.meta["val"] for c in outputs
    ]
    check_meta_consistency(xs_fake_tensors, output_fake_tensors, "init", "carry")

    _, combine_graph_name = unique_graph_id(
        proxy_mode, prefix="associative_scan_combine_graph"
    )

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, xs, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="associative_scan"
    )

    with disable_proxy_modes_tracing():
        out = tuple(aten.clone(x) for x in xs)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@associative_scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def associative_scan_op_dense(combine_fn, xs, additional_inputs):
    return generic_associative_scan(combine_fn, xs, additional_inputs=additional_inputs)


class AssociativeScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        combine_fn,
        num_xs,
        num_additional_inputs,
        *operands,
    ):
        ctx._num_xs = num_xs
        ctx._num_additional_inputs = num_additional_inputs
        xs, additional_inputs = split_into_chunks(
            operands, [num_xs, num_additional_inputs]
        )

        scan_length = xs[0].shape[0]
        ctx._scan_length = scan_length

        # First_slice_copy does not keep the original requires_grad flag,
        # but we need it here in order to compute the correct gradients
        xs_slices = [first_slice_copy(x) for x in itertools.chain(xs, xs)]

        ctx._combine_fn_bw = create_bw_fn(
            combine_fn,
            (*xs_slices, *additional_inputs),
        )

        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        with torch._C._AutoDispatchBelowAutograd():
            outs = associative_scan_op(combine_fn, xs, additional_inputs)
            save_tensors_and_symints_for_backward(ctx, list(operands) + list(outs))

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

            The ``combine_fn``, is the operator used during the forward associative_scan. For example
            def combine_fn(a: torch.Tensor, b: torch.Tensor):
                return a * b

            The ``ctx._combine_fn_bw``, used in the backward function, is the gradient of the ``combine_fn`` at a single step.
            It computes the gradients for x and y of ``combine_fn``. It is created utilizing ``create_bw_fn``, i.e.,
            ctx._combine_fn_bw = create_bw_fn(combine_fn, operands), where operands = (*xs_slices, *additional_inputs)
            It requires the primals (operands) followed by the tangents (upstream gradients) from a single step
            and produces the gradients of that step, i.e.,
            grad_h_t, grad_x_t = ctx._combine_fn_bw(xs_(t-1), xs_t, g_ys_t).
            While g_ys_t could be the upstream gradients directly, we leverage the ``ctx._combine_fn_bw`` to only produce
            instantaneous and unscaled gradients of the function ``combine_fn``, and thus we use g_ys_t = 1, i.e,
            grad_h_t, grad_x_t = ctx._combine_fn_bw(xs_(t-1), xs_t, 1),
            and scale the gradients accordingly at a later stage.
            Note: For simplicity, we omit the explicit 1 in the following description below.

            In other words, the first output of ``ctx._combine_fn_bw`` represents
            the gradient grad_h_t, that propagates information further back to previous inputs, e.g., xs_(t-1)
            and thus can be seen as a form of "hidden state", hence the name grad_h_t.
            The second output of ``ctx._combine_fn_bw`` represents the gradient grad_y_t,
            that can be seen as the instantaneous gradient for the input xs at setp t.

            These two quantities will be exploited in the algorithm below to compute the gradients of the input xs.

            The inputs to ``associative_scan`` in the forward path are xs_1, xs_2, ..., xs_T
            The outputs of ``associative_scan`` in the forward path are y_1, y_2, ..., y_T, where
            y_1 = xs_1
            y_2 = combine_fn(y_1, xs_2) = combine_fn(xs_1, xs_2)
            y_3 = combine_fn(y_2, xs_3) = combine_fn(combine_fn(y_1, xs_2), xs_2) = combine_fn(combine_fn(xs_1, xs_2), xs_2)
            ...
            y_T = combine_fn(y_(T-1), xs_T) = combine_fn(combine_fn(y_(T-2), xs_(T-1)), xs_T) = ...
            In the above expansion it also becomes aparent that the gradient with respect to the first element of the ``combine_fn``
            propagates information back to other steps.

            To understand the usage of ctx._combine_fn_bw better, a few examples may help:
            First, we want to derive the last output y_T with respect to particular elements of xs.
            dy_T/dx_T = dcombine_fn(y_{T-1}, xs_T)/dx_T -> second output of ctx._combine_fn_bw(y_{T-1}, xs_T)

            dy_T/dxs_{T-1} = dcombine_fn(y_{T-1}, xs_T)/dy_{T-1} . dcombine_fn(y__{T-2}, xs_{T-1})/dxs_{T-1}
                          -> first output of ctx._combine_fn_bw(y_{T-1}, xs_T)
                             . second output of ctx._combine_fn_bw(y_{T-2}, xs_{T-1})

            dy_T/dxs_{T-2} = dcombine_fn(y_{T-1}, xs_T)/dy_{T-1}
                            . dcombine_fn(y_{T-2}, xs_{T-1})/dy_{T-2}
                            . dcombine_fn(y_{T-3}, xs_{T-2})/dxs_{T-2}
                          ->  first output of ctx._combine_fn_bw(y_{T-1}, xs_T)
                            . first output of ctx._combine_fn_bw(y__{T-2}, xs_{T-1})
                            . second output of ctx._combine_fn_bw(y_{T-3}, xs_{T-2})

            A conceptually similar pattern can be observed, when deriving the output at various steps y_T, y_(T-1), etc.
            with respect to the input xs at a fixed step, i.e, xs_T.
            dy_{T-1}/dxs_T = 0

            dy_{T-1}/dxs_{T-1} = dcombine_fn(y_{T-2}, xs_{T-1})/dxs_{T-1} -> second output of ctx._combine_fn_bw(y_{T-2}, xs_{T-1})

            dy_{T-1}/dxs_{T-2} = dcombine_fn(y_{T-2}, xs_{T-1})/dy_{T-2} . dcombine_fn(y__{T-3}, xs_{T-2})/dxs_{T-2}
                              -> first output of ctx._combine_fn_bw(y_{T-2}, xs_{T-1})
                              . second output of ctx._combine_fn_bw(y_{T-3}, xs_{T-2})

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
            xs = torch.arange(1, 5) = [1, 2, 3, 4]
            y = torch.cumprod(xs) = [1, 2, 6, 24]

            The gradients of `y` with respect to `xs` can be computed as follows:
            Step 1.: Compute the gradients at every scan element with respect to y and x
            grad_h_0, grad_x_0 = [1, 1]
            grad_h_1, grad_x_1 = ctx._combine_fn_bw(xs_0, xs_1)
            ...
            grad_h_T, grad_y_T = ctx._combine_fn_bw(xs_(T-1), xs_T),
            which for the example above results in:
            grads_h = [1, 2, 3, 4]
            grads_x = [1, 1, 2, 6]

            Step 2.: Compute the gradient matrix
            h_mat_true = [[dy_0/dy_0, dy_1/dy_0, dy_2/dy_0, dy_3/dy_0],
                          [0        , dy_1/dy_1, dy_2/dy_1, dy_3/dy_2],
                          [0        , 0        , dy_2/dy_2, dy_3/dy_2],
                          [0        , 0        , 0        , dy_3/dy_3]]
            Note: In order to understand the derivation below, it is important realize that y_0 = xs_0

            2.1 Repeat the elements of gh to form the square matrix of derivatives
            h_mat = [[1, dy_1/dy_0, dy_2/dy_1, dy_3/dy_2],
                     [1, dy_1/dy_0, dy_2/dy_1, dy_3/dy_2],
                     [1, dy_1/dy_0, dy_2/dy_1, dy_3/dy_2],
                     [1, dy_1/dy_0, dy_2/dy_1, dy_3/dy_2]],

            which results in
            h_mat = [[1, 2, 3, 4],
                     [1, 2, 3, 4],
                     [1, 2, 3, 4],
                     [1, 2, 3, 4]].

            2.2 Fill the lower triangular part, including the diagonal, of the h_mat with 1s.
            I.e., use the ones_mask to fill with 1s.
            h_mat = [[1, dy_1/dy_0, dy_2/dy_1, dy_3/dy_2],
                     [1, 1        , dy_2/dy_1, dy_3/dy_2],
                     [1, 1        , 1        , dy_3/dy_2],
                     [1, 1        , 1        , dy_3/dy_2]],

            which results in
            h_mat = [[1, 2, 3, 4],
                     [1, 1, 3, 4],
                     [1, 1, 1, 4],
                     [1, 1, 1, 4]].

            # 2.3 Compute the cumulative products across dim + 1, i.e., the rows:
            This is required because of the chain rule
            For example, the desired matrix h_mat_true can be written as
            h_mat_true = [[dy_0/dy_0, dy_1/dy_0, dy_2/dy_0, dy_3/dy_0],
                          [0        , dy_1/dy_1, dy_2/dy_1, dy_3/dy_2],
                          [0        , 0        , dy_2/dy_2, dy_3/dy_2],
                          [0        , 0        , 0        , dy_3/dy_3]]

                       = [[dy_0/dy_0, dy_1/dy_0, dy_2/dy_1 dy_1/dy_0, dy_3/dy_0 dy_2/dy_1 dy_1/dy_0],
                          [0        , dy_1/dy_1, dy_2/dy_1          , dy_3/dy_2 dy_2/dy_1          ],
                          [0        , 0        , dy_2/dy_2          , dy_3/dy_2                    ],
                          [0        , 0        , 0                  , dy_3/dy_3                    ]],
            which can be done by applying the cumprod along the rows of the h_mat from step 2.2.
            In particular
            h_mat = cumprod([[1, dy_1/dy_0, dy_2/dy_1, dy_3/dy_2],
                             [1, 1        , dy_2/dy_1, dy_3/dy_2],
                             [1, 1        , 1        , dy_3/dy_2],
                             [1, 1        , 1        , dy_3/dy_2]],)

            h_mat = [[1, dy_1/dy_0, dy_2/dy_1 dy_1/dy_0, dy_3/dy_2 dy_2/dy_1 dy_1/dy_0],
                     [1, 1        , dy_2/dy_1          , dy_3/dy_2 dy_2/dy_1          ],
                     [1, 1        , 1                  , dy_3/dy_2                    ],
                     [1, 1        , 1                  , dy_3/dy_2                    ]],


            # 2.4 Fill the zeros_mask with 0s again
            This is the final step to arrive at the h_mat_true
            h_mat = h_mat_true = [[1, dy_1/dy_0, dy_2/dy_1 dy_1/dy_0, dy_3/dy_2 dy_2/dy_1 dy_1/dy_0],
                                  [0, 1        , dy_2/dy_1          , dy_3/dy_2 dy_2/dy_1          ],
                                  [0, 0        , 1                  , dy_3/dy_2                    ],
                                  [0, 0        , 0                  , dy_3/dy_2                    ]],

            which for the example above would be
            h_mat = [[1, 2, 6, 24],
                     [0, 1, 3, 12],
                     [0, 0, 1,  4],
                     [0, 0, 0,  1]]

            Step 3.: scale the matrix with the upstream gradients, e.g., dL_i/do_i
            scaled_h_mat = h_mat * dL_i/do_i
            Assuming all 1s for the upstream gradients this would result in:
            scaled_h_mat = [[1, 2, 6, 24],
                            [0, 1, 3, 12],
                            [0, 0, 1,  4],
                            [0, 0, 0,  1]]

            Step 4.: Reduce the matrix with sum along the columns to get the total contributions for x_i
            summed_h_mat = scaled_h_mat.sum(dim + 1),

            which would be
            summed_h_mat = [33, 16, 5, 1]

            Step 5.: Scale with the grads_x, e.g., do_i/dx_i, to obtain the final gradients
            grad_xs = summed_h_mat * grad_x

            grad_xs = [33, 16, 5, 1] * [1, 1, 2, 6]
            grad_xs = [33, 16, 10, 6]
            Which finally yields the gradients with respect to xs.
        """

        # The backward of associative_scan is always performed on the first dimension
        dim = 0
        scan_length = ctx._scan_length
        num_xs = ctx._num_xs
        num_additional_inputs = ctx._num_additional_inputs

        # Extract the inputs to the forward path and outputs from the forward path
        flat_args = saved_tensors_and_symints(ctx)
        xs, outs, additional_inputs = split_into_chunks(
            flat_args, [num_xs, num_xs, num_additional_inputs]
        )
        ndim = outs[0].ndim

        # First_slice_copy does not keep the original requires_grad flag,
        # but we need it here in order to compute the correcte gradients
        xs_slices = first_slice_copy_with_grad(itertools.chain(xs, xs))

        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        combine_fn_bw_gm = materialize_as_graph(
            ctx._combine_fn_bw,
            (
                *xs_slices,
                *additional_inputs,
                *[first_slice_copy(o) for o in outs],
            ),
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        # vmap joint graph over scan dimension to compute the individual
        # gradients for each time slice ``t`` in parallel.
        # This computation can be parallelized, as these are just the instantaneous gradients and not the full chain-rule
        mapped_combine_fn_bw_gm = torch.vmap(combine_fn_bw_gm, 0, 0)

        # Step 1.: Compute the gradients at every scan element ``t`` with respect to y and x
        # For the assoc. op f(x, y), get the derivatives df(x_i, y_{i-1})/dy_{i-1} and
        # df(x_i,y_{i-1})/dx_i for all i, with invalid index values giving derivatives equal to 1.
        dummy_upstream_grad = (torch.ones_like(x) for x in xs)
        grads = mapped_combine_fn_bw_gm(
            *(o.roll(1, dim) for o in outs), *xs, *dummy_upstream_grad
        )
        grad_h_t, grad_x_t = split_into_chunks(grads, [num_xs, num_xs])

        def compute_grad_h_mat(gh: torch.Tensor) -> torch.Tensor:
            # Prepare a ones and a zeros helper mask in order to easily compute the y_mat
            def compute_helper_tril_mask(diagonal):
                def expand_masks(mask):
                    for _ in range(ndim - 1):
                        mask = mask.unsqueeze(-1)
                    return mask

                tril_mask = torch.tril(
                    torch.ones(
                        scan_length, scan_length, device=gh.device, dtype=torch.bool
                    ),
                    diagonal=diagonal,
                )
                tril_mask = expand_masks(tril_mask)
                tril_mask = tril_mask.expand(-1, -1, *gh.shape[1:])
                return tril_mask

            # The ones mask is used to fill the main diagonal and all elements below it with 1s
            # The elements on the main diagonal are 1 because of do_0/dy_0 = do_1/dy_1 = ... = 1
            # and the elements below it are set to 1, in order for the cumprod can be computed properly.
            ones_mask = compute_helper_tril_mask(0)

            # The zero mask is used to set all elements below the main diagonal to 0, because do_0/dy_1 = do_0/dy_2 = ... = 0
            zeros_mask = compute_helper_tril_mask(-1)

            # 2.1 Repeat the elements of gh to form the square matrix of derivatives
            h_mat = gh.unsqueeze(dim).repeat_interleave(scan_length, dim)

            # 2.2 Fill the lower triangular part, including the diagonal,
            # of the h_mat with 1s. I.e., use the ones_mask to fill with 1s.
            h_mat.masked_fill_(ones_mask, 1.0)

            # 2.3 Compute the cumulative products across dim + 1
            h_mat = h_mat.cumprod(dim=dim + 1)

            # 2.4 Fill the zeros_mask with 0s again
            h_mat.masked_fill_(zeros_mask, 0.0)

            return h_mat

        def compute_grad(grad_x, grad_h, fg):
            # Set the i=0 component of df(x_i,y_{i-1})/dx_i to 1.0
            # i.e., the first gradient component is always 1.0
            torch.select(grad_x, dim, 0).fill_(1.0)

            # Step 2.: Compute the gradient matrix
            h_mat = compute_grad_h_mat(grad_h)

            # Step 3.: scale the matrix with the upstream gradients, e.g., dL_i/do_i
            scaled_h_mat = h_mat * fg

            # Step 4.: Reduce the matrix with sum along the columns to get the total contributions for x_i
            summed_h_mat = scaled_h_mat.sum(dim + 1)

            # Step 5.: Scale with the grads_x, e.g., do_i/dx_i, to obtain the final gradients
            grad_xs = summed_h_mat * grad_x

            return grad_xs

        # Stack all elements of the gradients along the first dimension.
        # This is useful as later the gradients of those elements can be computed in parallel.
        grad_x_stacked = torch.stack(grad_x_t)
        grad_h_stacked = torch.stack(grad_h_t)
        flat_grads_stacked = torch.stack(flat_grads)

        # The compute_grad function is parallelized across all individual elements of xs
        # as these gradients can be computed independently from each other
        compute_grad_mapped = torch.vmap(compute_grad, 0, 0)

        grads_xs = compute_grad_mapped(
            grad_x_stacked, grad_h_stacked, flat_grads_stacked
        )

        # TODO: Currently the gradients for the additional_inputs are not computed properly
        return *[None] * 3, *grads_xs, *[None] * num_additional_inputs


@associative_scan_op.py_impl(DispatchKey.Autograd)
def associative_scan_autograd(combine_fn, xs, additional_inputs):
    num_xs = len(xs)
    num_additional_inputs = len(additional_inputs)

    flat_out = AssociativeScanAutogradOp.apply(
        combine_fn, num_xs, num_additional_inputs, *(tuple(xs) + additional_inputs)
    )
    return (*flat_out,)


@associative_scan_op.py_impl(ProxyTorchDispatchMode)
def associative_scan_proxy_mode(mode, combine_fn, xs, additional_inputs):
    return trace_associative_scan(
        mode, associative_scan_op, combine_fn, xs, additional_inputs
    )


@associative_scan_op.py_impl(FakeTensorMode)
def assoiciative_scan_fake_tensor_mode(mode, combine_fn, xs, additional_inputs):
    with mode:
        return tuple(x.clone() for x in xs)


@associative_scan_op.py_functionalize_impl
def associative_scan_functionalize(ctx, combine_fn, xs, additional_inputs):
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        ret = associative_scan_op(
            functional_combine_fn,
            unwrapped_xs,
            unwrapped_additional_inputs,
        )
    return ctx.wrap_tensors(ret)


def _fake_associative_scan(combine_fn, xs, dim, reverse=False):
    inp_leaves, spec = pytree.tree_flatten(xs)
    result_flat: list[Any] = []
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
