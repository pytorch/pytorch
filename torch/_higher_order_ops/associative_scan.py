# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    _maybe_run_with_interpreter,
    autograd_not_implemented,
    check_input_alias_and_mutation_return_outputs,
    check_meta_consistency,
    first_slice_copy,
    reenter_make_fx,
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
    assert len(args) == 2 * num_leaves, (
        f"Combin_fn received wrong number of arguments, expected {2 * num_leaves}, but got {len(args)}"
    )
    lhs = pytree.tree_unflatten(args[:num_leaves], spec)
    rhs = pytree.tree_unflatten(args[num_leaves:], spec)
    return combine_fn(lhs, rhs)


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


class AssociativeScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("associative_scan")

    def __call__(self, combine_fn, xs, additional_inputs):
        # There is currently an issue that the ScanOp is sometimes called with
        # the additional_inputs being a list. See https://github.com/pytorch/pytorch/issues/145785
        # Once this issue is resolved, the assertion should only allow tuples
        # and the tuple cast should be removed
        assert isinstance(additional_inputs, (tuple, list)), (
            "additional_inputs must be a tuple."
        )
        additional_inputs = (
            tuple(additional_inputs)
            if isinstance(additional_inputs, list)
            else additional_inputs
        )
        validate_subgraph_args_types(additional_inputs)
        return super().__call__(combine_fn, xs, additional_inputs)

    def gen_schema(self, combine_fn, xs, additional_inputs):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        # For associative scan, we need two copies of xs for the combine function
        # The combine function takes two elements and returns one element
        xs_slice1 = [first_slice_copy(x) for x in xs]
        xs_slice2 = [first_slice_copy(x) for x in xs]
        all_inputs = tuple(xs_slice1 + xs_slice2 + list(additional_inputs))

        combine_gm: torch.fx.GraphModule = (
            combine_fn
            if isinstance(combine_fn, torch.fx.GraphModule)
            else materialize_as_graph(combine_fn, all_inputs)
        )

        example_inputs = [
            n.meta["val"] if "val" in n.meta else n.meta["example_value"]
            for n in combine_gm.graph.find_nodes(op="placeholder")
        ]

        (
            _,
            _,
            _,
            mutated_inputs,
            outputs,
        ) = check_input_alias_and_mutation_return_outputs(combine_gm, example_inputs)
        if len(mutated_inputs) > 0:
            raise RuntimeError(
                "For associative_scan, combine_fn cannot have in-place mutations but found "
                f"{mutated_inputs}-th inputs are mutated."
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("combine_fn", combine_gm)

        for idx, x in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", x)

        for idx, arg in enumerate(additional_inputs):
            schema_gen.add_arg(
                f"additional_input{idx}",
                arg,
            )

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(combine_fn, xs, additional_inputs)
        return schema_gen.gen_schema()


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
    # The reason we flatten xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    def _validate_input(cfn, lxs, d, r, cm):
        # Basic arguments check
        if not callable(cfn):
            raise ValueError("Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise ValueError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))
        if cm not in ["pointwise", "generic"]:
            raise ValueError(
                "Combine_mode must either 'pointwise' or 'generic', but got {cm}"
            )
        if cm == "pointwise" and not all(l.device.type == "cuda" for l in lxs):
            raise ValueError(
                "For combine_mode='pointwise', all input tensors need to be on CUDA"
            )

        # Checks for xs
        if len(lxs) == 0:
            raise ValueError("Expected at least 1 xs leaf")
        if any(not isinstance(x, torch.Tensor) for x in lxs):
            raise ValueError("xs leaves must be a Tensor")
        if any(x.is_sparse for x in lxs):
            raise ValueError(
                "xs leaves must dense Tensors, consider using `to_dense()`"
            )
        if any(x.ndim <= d for x in lxs):
            raise ValueError(
                "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
            )
        if any(x.shape[d] == 0 for x in lxs):
            raise ValueError(
                "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
            )

    ndim = leaves_xs_orig[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)

    _validate_input(combine_fn, leaves_xs_orig, dim, reverse, combine_mode)

    # Move scan dim to 0 and always perform scan on dim 0
    leaves_xs = [torch.movedim(elem, dim, 0) for elem in leaves_xs_orig]

    if reverse:
        leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

    # TODO: Support Autograd
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

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
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                    pytree.tree_unflatten([0] * len(leaves_xs), spec_xs),
                ),
                out_dims=0,
            ),
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )
        out = generic_associative_scan(combine_fn, leaves_xs, additional_inputs=())
        out = pytree.tree_unflatten(out, spec_xs)
    else:
        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec=spec_xs,
            num_leaves=len(leaves_xs),
        )

        def run_flattened_associative_scan(combine_fn, leaves_xs):
            return associative_scan_op(combine_fn, leaves_xs, additional_inputs=())

        out = _maybe_compile_and_run_fn(
            run_flattened_associative_scan,
            combine_fn,
            leaves_xs,
        )

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)

    out = pytree.tree_map(lambda elem: torch.movedim(elem, 0, dim), out)

    return out


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

    def call_operator(*args):
        return pytree.tree_leaves(operator(*args))

    def _scan(elems):
        """Perform the actual recursive scan on ``elems``."""
        num_elems = elems[0].shape[dim]

        if num_elems < 2:
            return elems

        reduced_elems = call_operator(
            *[aten.slice(elem, dim, 0, -1, 2) for elem in elems],
            *[aten.slice(elem, dim, 1, None, 2) for elem in elems],
            *additional_inputs,
        )

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = call_operator(
                *[aten.slice(e, dim, 0, -1) for e in odd_elems],
                *[aten.slice(e, dim, 2, None, 2) for e in elems],
                *additional_inputs,
            )
        else:
            even_elems = call_operator(
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
    from torch._dynamo.utils import clone_input

    with disable_proxy_modes_tracing():
        sample_xs = [first_slice_copy(x) for x in itertools.chain(xs, xs)]
        sample_additional_inputs = [
            clone_input(x) if isinstance(x, torch.Tensor) else x
            for x in additional_inputs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_xs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None
    outputs = pytree.tree_leaves(outputs)
    assert len(outputs) == len(xs), (
        f"expected combine_fn to return {len(xs)} results but got {len(outputs)}"
    )

    xs_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        first_slice_copy(x) for x in xs
    ]
    output_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        c.meta["val"] for c in outputs
    ]
    check_meta_consistency(
        xs_fake_tensors, output_fake_tensors, "init", "carry", include_contiguity=False
    )

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


""" associative_scan backward
    Example::
        xs = torch.arange(1, 5) = [1, 2, 3, 4]

        def combine_fn(a: torch.Tensor, b: torch.Tensor):
            return a * b

        ys = associative_scan(comine_fn, xs),
        which can be unpacked as:
        ys_0 = xs_0                                          = 1
        ys_1 = combine_fn(ys_0, xs_1) = combine_fn(1, 2)     = 2
        ...
        ys_T = combine_fn(ys_(T-1), xs_T) = combine_fn(6, 4) = 24
        ys = [1, 2, 6, 24]

        This creates a recursive data dependency structure where each output ys_t
        depends on all prior inputs xs_0 through xs_t. The dependency can be visualized as:

Level 0 (Input):    x0    x1    x2    x3    x4
                     \    /      |     |     |
                      \  /       |     |     |
Level 1:               y1 ───────┘     |     |
                        \              /     |
                         \            /      |
Level 2:                  y2 ─────────┘     |
                           \                /
                            \              /
Level 3:                     y3 ──────────┘
                              \
                               \
Level 4:                        y4


We could get the following backward gradient graph:


Level 0 (output):   gx0   gx1   gx2   gx3   gx4
                     \    /      |     |     |
                      \  /       |     |     |
Level 1:    gl_y1  ─> gy1  ──────┘     |     |
                        \              /     |
                         \            /      |
Level 2:    gl_y2     ─> gy2  ────────┘      |
                           \                 /
                            \               /
Level 3:    gl_y3        ─> gy3    ─────────┘
                              \
                               \
Level 4:    gl_y4           ─> gy4


, where gl_y1 is the gradient of the loss with respect to y1 and is the input of backward.


To calculate output, the chain rule suggests:

gx0 = gy1 * bw(y1, x0)
gx1 = gy1 * bw(y1, x1)
gx2 = gy1 * bw(y1, x2)
gx3 = gy2 * bw(y2, x3)
gx4 = gy3 * bw(y3, x4)

Noice the bw(...) is just the single step bw, whose formula can be computed from combine_fn.

Let's break down how to calculate gy_t by recursively substituting the unknowns:

gy1 = gl_y1 + gy2 * bw(y2, y1)
    = gl_y1 + (gl_y2  + gy3 * bw(y3, y2))* bw(y2, y1)
    = gl_y1 + gl_y2 * bw(y2, y1) + gy3 * bw(y3, y2) * bw(y2, y1)
    = gl_y1 + gl_y2 * bw(y2, y1) + gl_y3 * bw(y3, y2) * bw(y2, y1) + gy4 * bw(y4, y3) * bw(y3, y2) * bw(y2, y1)
    = gl_y1 + gl_y2 * bw(y2, y1) + gl_y3 * bw(y3, y2) * bw(y2, y1) + gl_y4 * bw(y4, y3) * bw(y3, y2) * bw(y2, y1)

Let's do the same for more terms:
gy2 = gl_y2 +  gl_y3 * bw(y3, y2) + gl_y4 * bw(y4, y3) * bw(y3, y2)
gy3 = gl_y3 + gl_y4 * bw(y4, y3)
gy4 = gl_y4

Notice that the above can be re-written as a matrix vector multiplication of y_mat and gy:

gy1      1, bw21, bw321, bw4321       gl_y1
gy2   =  0,  1  , bw321, bw4321   @   gl_y2
gy3      0,  0  ,   1  , bw4321       gl_y3
gy4      0,  0  ,   0  ,    1         gl_y4

, where bw4321 is an abreviation for bw(y4, y3) * bw(y3, y2) * bw(y2, y1) so on and so forth.

We could effectively compute the upper triangular matrix y_mat with:

cumprod([1, bw21, bw32, bw43]) then masking out the values as needed.


    Refences: https://justintchiu.com/blog/pscan_diff/

    NOTE: [associative_scan autograd implementation]


    The forward of associative_scan can be computed with the following steps:

    1.) Compute the forward output of the associative_scan
        ys = associative_scan(combine_fn, xs, additional_inputs)

    The backward of associative_scan can be computed with the following steps:

    2.) Prepare the backward graph
        We prepare the backward graph to be used in the backward function.
        We utilize ``create_bw_fn`` to generate the joint function:
        combine_fn_bw = create_bw_fn(combine_fn, operands)
        where operands = [ys_{t-1}, xs_t, additional_inputs]

    3.) Materialize the ``combine_fn_bw``
        This is required because torch.compile and torch.autograd.grad cannot trace through the joint backward function dynamically.

    4.) Compute the instantaneous gradients at every step t
        g_y_t, g_x_t = combine_fn_bw(y_{t-1}, x_t, 1.)
        Here we pass 1 as the upstream gradient to obtain the local partial derivatives.

        This gives:
            g_y = [g_y_0, g_y_1, ..., g_y_T] # i.e. (bw(y1, y0), bw(y2, y1)...)
            g_x = [g_x_0, g_x_1, ..., g_x_T] # i.e. (bw(y1, x0), bw(y1, x1)...)

    5.) Compute the gradient transition matrix

        As shown in the example above, each input xs_t affects all later outputs ys_i for i ≥ t.
        According to the chain rule, each such path contributes a product of local gradients g_ys_k.

        For example:
            ∂ys_T/∂xs_t = ∂ys_T/∂ys_{T-1} * ∂ys_{T-1}/∂ys_{T-2} * ... * ∂ys_{t+1}/∂ys_t * ∂ys_t/∂xs_t
                    = g_y_T * g_y_{T-1} * ... * g_y_{t+1} * g_x_t

        This motivates the use of a cumulative product over g_y to compute all such paths efficiently.

        We now construct the matrix of gradient transition paths:

        5.1 Repeat g_y values to form the base matrix
            y_mat = [[1, g_y_1, g_y_2, g_y_3],
                     [1, g_y_1, g_y_2, g_y_3],
                     [1, g_y_1, g_y_2, g_y_3],
                     [1, g_y_1, g_y_2, g_y_3]]

        5.2 Mask the lower triangle (inclusive) with 1s
            y_mat = [[1, g_y_1, g_y_2, g_y_3],
                     [1, 1    , g_y_2, g_y_3],
                     [1, 1    , 1    , g_y_3],
                     [1, 1    , 1    , 1    ]]

        5.3 Apply cumulative product row-wise
            y_mat = cumprod(y_mat, dim=1)
            Resulting in:
            y_mat = [[1, g_y_1, g_y_2 * g_y_1, g_y_3 * g_y_2 * g_y_1],
                    [1, 1     , g_y_2        , g_y_3 * g_y_2        ],
                    [1, 1     , 1            , g_y_3                ],
                    [1, 1     , 1            , 1                    ]]

        5.4 Zero out the lower triangle (exclusive)
            Final y_mat:
            y_mat = [[1, g_y_1, g_y_2 * g_y_1, g_y_3 * g_y_2 * g_y_1],
                    [0, 1    , g_y_2         , g_y_3 * g_y_2        ],
                    [0, 0    , 1             , g_y_3                ],
                    [0, 0    , 0             , 1                    ]]

    6.) Scale the y_mat with the upstream gradients g_ys
        scaled_y_mat = y_mat * g_ys
        Each entry now holds the full contribution of ∂L/∂y_j to ∂L/∂x_i via the path through y_j.

    7.) Reduce the scaled_y_mat with a row-wise sum
        summed_y_mat = scaled_y_mat.sum(dim=1)
        This accumulates all downstream contributions for each x_t.

    8.) Scale with the instantaneous input gradients g_x
        g_xs = summed_y_mat * g_x

        This gives the final input gradients:
            g_xs = [∂L/∂x_0, ∂L/∂x_1, ..., ∂L/∂x_T]

    NOTE: [scan partial grad handling]
        If any element of xs or of the outputs does not require gradients
        (i.e., requires_grad=False), then the corresponding gradients will be returned
        as tensors of zeros with the same shape as the element.
"""

associative_scan_op.py_autograd_impl(
    autograd_not_implemented(associative_scan_op, deferred_error=True)
)


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
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [
            first_slice_copy(inp) for inp in itertools.chain(unwrapped_xs, unwrapped_xs)
        ]
        sample_inputs = list(
            itertools.chain(
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        _check_alias_and_mutation(
            combine_fn, sample_inputs, "associative_scan", pre_dispatch
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
