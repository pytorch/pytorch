# mypy: allow-untyped-defs
import functools
import itertools
from collections.abc import Callable
from typing import Any

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    _maybe_run_with_interpreter,
    check_input_alias_and_mutation_return_outputs,
    check_meta_consistency,
    create_bw_fn,
    first_slice_copy,
    first_slice_copy_with_grad,
    materialize_as_graph,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    split_into_chunks,
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
    # pyrefly: ignore [unbound-name]
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

    # pyrefly: ignore [bad-override]
    def gen_schema(self, combine_fn, xs, additional_inputs):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        # For associative scan, we need two copies of xs for the combine function
        # The combine function takes two elements and returns one element
        xs_slice1 = [first_slice_copy(x) for x in xs]
        xs_slice2 = [first_slice_copy(x) for x in xs]
        all_inputs = tuple(xs_slice1 + xs_slice2 + list(additional_inputs))

        combine_gm: torch.fx.GraphModule = materialize_as_graph(combine_fn, all_inputs)
        (
            _,
            _,
            _,
            mutated_inputs,
            outputs,
        ) = check_input_alias_and_mutation_return_outputs(combine_gm)
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
    # TODO: Support lifted arguments in inductor for associative_scan
    # TODO: Support autograd for cases with lifted arguments for combine_mode=pointwise

    # The reason we flatten xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    def _validate_input(cfn, lxs, d, r, cm):
        # Basic arguments check
        if not callable(cfn):
            raise ValueError(f"Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise ValueError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))
        if cm not in ["pointwise", "generic"]:
            raise ValueError(
                f"Combine_mode must either 'pointwise' or 'generic', but got {cm}"
            )
        if cm == "pointwise" and not all(l.device.type in ("cuda", "xpu") for l in lxs):
            raise ValueError(
                "For combine_mode='pointwise', all input tensors need to be on CUDA or XPU"
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


class AssociativeScanAutogradOp(torch.autograd.Function):
    r""" associative_scan
        Example::
            xs = torch.arange(1, 5) = [1, 2, 3, 4]

            def combine_fn(a: torch.Tensor, b: torch.Tensor):
                return a * b

            ys = associative_scan(comine_fn, xs),
            which can be unpacked as:
            ys0 = xs0                                         = 1
            ys1 = combine_fn(ys0, xs1) = combine_fn(1, 2)     = 2
            ...
            ysT = combine_fn(ys(T-1), xsT) = combine_fn(6, 4) = 24
            ys = [1, 2, 6, 24]

            This creates a recursive data dependency structure where each output yst
            depends on all prior inputs xs0 through xst. The dependency can be visualized as:

    Level 0 (Input):    xs0    xs1    xs2    xs3    xs4
                        \    /       |      |      |
                         \  /        |      |      |
    Level 1:              ys1 ───────┘      |      |
                           \               /       |
                            \             /        |
    Level 2:                 ys2 ────────┘         |
                              \                   /
                               \                 /
    Level 3:                    ys3 ────────────┘
                                 \
                                  \
    Level 4:                        ys4


    We could get the following backward gradient graph:


    Level 0 (output):   g_xs0   g_xs1   g_xs2   g_xs3   g_xs4
                         \      /       |       |       |
                          \    /        |       |       |
    Level 1:    gl_ys1  ─> g_ys1  ──────┘       |       |
                            \                  /        |
                             \                /         |
    Level 2:    gl_ys2     ─> g_ys2  ────────┘          |
                               \                       /
                                \                    /
    Level 3:    gl_ys3        ─> g_ys3  ────────────┘
                                  \
                                   \
    Level 4:    gl_ys4           ─> g_ys4,

    where gl_y1 is the gradient of the loss with respect to ys1 and the input of backward.

    To calculate the gradients of the inputs, the chain rule suggests:

    g_xs0 = g_ys1
    g_xs1 = g_ys1 * bw(ys0, xs1) = g_ys1 * bwxs01
    g_xs2 = g_ys2 * bw(ys1, xs2) = g_ys2 * bwxs12
    g_xs3 = g_ys3 * bw(ys2, xs3) = g_ys3 * bwxs23
    g_xs4 = g_ys4 * bw(ys3, xs4) = g_ys4 * bwxs34

    Notice the bw(...) is just the single step bw (instantaneous gradients), whose formula can be computed from combine_fn.
    For example bw(ys3, xs4) (also abbreviated with bwxs34) computes the gradients ∂/∂xs4 combine_fn(ys3, xs4).
    Similarly, bw(ys4, ys3) (also abbreviated with bwys43) computes the gradients ∂/∂ys3 combine_fn(ys3, xs4).

    Let's break down how to calculate g_ys by recursively substituting the unknowns:

    g_ys1 = gl_ys1 + g_ys2 * bw(ys2, ys1)
          = gl_ys1 + (gl_ys2  + g_ys3 * bw(ys3, ys2)) * bw(ys2, ys1)
          = gl_ys1 + gl_ys2 * bw(ys2, ys1) + g_ys3 * bw(ys3, ys2) * bw(y2, y1)
          = gl_ys1 + gl_ys2 * bw(ys2, ys1) + gl_ys3 * bw(ys3, ys2) * bw(y2, y1) \
                   + g_ys4 * bw(ys4, ys3) * bw(ys3, ys2) * bw(ys2, ys1)
          = gl_ys1 + gl_ys2 * bw(ys2, ys1) + gl_ys3 * bw(ys3, ys2) * bw(y2, y1) \
                   + gl_ys4 * bw(ys4, ys3) * bw(ys3, ys2) * bw(ys2, ys1)

    Let's do the same for all the g_ys:
    g_ys2 = gl_ys2 + gl_ys3 * bw(ys3, ys2) + gl_y4 * bw(ys4, ys3) * bw(ys3, ys2)
    g_ys3 = gl_ys3 + gl_ys4 * bw(ys4, ys3)
    g_ys4 = gl_ys4

    Notice that the above can be re-written as columnwise multiplication of y_mat and gl_ys:

    g_ys1   1, bwys21, bwys321, bwys4321       gl_ys1
    g_ys2 = 0,    1  , bwys321, bwys4321   .   gl_ys2
    g_ys3   0,    0  ,     1  , bwys4321       gl_ys3
    g_ys4   0,    0  ,     0  ,        1       gl_ys4,

    where bwys21 is an abbreviation for bw(ys2, ys1),
    bwys321 is an abbreviation for bw(ys3, ys2) * bw(ys2, ys1) so on and so forth.

    We could effectively compute the upper triangular matrix y_mat with:
    cumprod([1, bwys21, bwys32, bwys43]) then masking out the values as needed.
    Thus, only [1, bwys21, bwys32, bwys43] are required to compute the y_mat.


        References: https://justintchiu.com/blog/pscan_diff/

        NOTE: [associative_scan autograd implementation]

        The forward of associative_scan can be computed with the following steps:

        1.) Compute the forward output of the associative_scan
            ys = associative_scan(combine_fn, xs, additional_inputs)

        The backward of associative_scan can be computed with the following steps:

        2.) Prepare the backward graph
            We prepare the backward graph to be used in the backward function.
            We utilize ``create_bw_fn`` to generate the joint function:
            combine_fn_bw = create_bw_fn(combine_fn, operands)
            where operands = [ys{t-1}, xst, additional_inputs]

        3.) Materialize the ``combine_fn_bw``
            This is required because torch.compile and torch.autograd.grad
            cannot trace through the joint backward function dynamically.

        4.) Compute the single step bw (instantaneous gradients) at every step t
            bwys{t-1}, bwxst = combine_fn_bw(ys{t-1}, xst, 1.)
            Here we pass 1 as the upstream gradient to obtain the local partial derivatives.

            This gives:
                bwys = [bw(ys1, ys0), bw(ys2, ys1), ..., bw(ysT, ys{T-1})]
                bwxs = [bw(ys1, xs0), bw(ys2, xs1), ..., bw(ys{T-1}, xsT)]

        5.) Compute the gradient transition matrix y_mat

            As shown in the example above, each input xst affects all later outputs ysi for i ≥ t.
            According to the chain rule, each such path contributes a product of local gradients g_ysk.

            For example:
                ∂ysT/∂xst = ∂ysT/∂ys{T-1} * ∂ys{T-1}/∂ys{T-2} * ... * ∂ys{t+1}/∂yst * ∂yst/∂xst
                        = bw(ysT, ys{T-1}) * bw(ys{T-1}, ys{T-2}) * ... * bw(ys{t+1}, yst) * bw(ys{t-1}, xst)

            This motivates the use of a cumulative product over bwys to compute all such paths efficiently.

            We now construct the matrix of gradient transition paths:

            5.1 Repeat g_y values to form the base matrix
                y_mat = [[1, bwys21, bwys32, bwys43],
                         [1, bwys21, bwys32, bwys43],
                         [1, bwys21, bwys32, bwys43],
                         [1, bwys21, bwys32, bwys43]]

            5.2 Mask the lower triangle (inclusive) with 1s
                y_mat = [[1, bwys21, bwys32, bwys43],
                         [1, 1     , bwys32, bwys43],
                         [1, 1     , 1     , bwys43],
                         [1, 1     , 1     , 1    ]]

            5.3 Apply cumulative product row-wise
                y_mat = cumprod(y_mat, dim=1)
                Resulting in:
                y_mat = [[1, bwys21, bwys32 * bwys21, bwys43 * bwys32 * bwys21],
                         [1, 1      , bwys32         , bwys43 * bwys32         ],
                         [1, 1      , 1              , bwys43                  ],
                         [1, 1      , 1              , 1                       ]]

            5.4 Zero out the lower triangle (exclusive)
                Final y_mat:
                y_mat = [[1, bwys21, bwys32 * bwys21, bwys43 * bwys32 * bwys21],
                         [0, 1      , bwys32         , bwys43 * bwys32         ],
                         [0, 0      , 1              , bwys43                  ],
                         [0, 0      , 0              , 1                       ]]

        6.) Scale the y_mat with the upstream gradients gl_ys
            scaled_y_mat = y_mat * gl_ys
            Each entry now holds the full contribution of ∂L/∂ysj to ∂L/∂xsi via the path through ysj.

        7.) Reduce the scaled_y_mat with a row-wise sum
            summed_y_mat = scaled_y_mat.sum(dim=1)
            This accumulates all downstream contributions for each xst.

        8.) Scale with the instantaneous input gradients bwxs
            g_xs = summed_y_mat * bwxs

            This gives the final input gradients:
                g_xs = [∂L/∂xs0, ∂L/∂xs1, ..., ∂L/∂xsT]

        NOTE: [scan partial grad handling]
            If any element of xs or of the outputs does not require gradients
            (i.e., requires_grad=False), then the corresponding gradients will be returned
            as tensors of zeros with the same shape as the element.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        combine_fn,
        num_xs,
        num_additional_inputs,
        *operands,
    ):
        ctx._num_xs = num_xs
        ctx._num_additional_inputs = num_additional_inputs
        ctx._combine_fn = combine_fn
        xs, additional_inputs = split_into_chunks(
            operands, [num_xs, num_additional_inputs]
        )

        scan_length = xs[0].shape[0]
        ctx._scan_length = scan_length

        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        with torch._C._AutoDispatchBelowAutograd():
            # 1.) Compute the forward output of the associative_scan
            ys = associative_scan_op(combine_fn, xs, additional_inputs)
            save_tensors_and_symints_for_backward(ctx, list(operands) + list(ys))

        return (*ys,)

    @staticmethod
    def backward(ctx, *gl_ys):
        r"""
        This function computes the gradients of the scan operation.
        For a detailed description see the document above.

        Args:
            flat_grads (torch.Tensor): The tensor of upstream gradients, or a nested pytree of tensors.
                                       E.g.: Gradient of the loss with respect to the forward output ys
        """

        # The backward of associative_scan is always performed on the first dimension
        dim = 0
        scan_length = ctx._scan_length
        num_xs = ctx._num_xs
        num_additional_inputs = ctx._num_additional_inputs

        # Extract the inputs to the forward path and outputs from the forward path
        flat_args = saved_tensors_and_symints(ctx)
        xs, additional_inputs, outs = split_into_chunks(
            flat_args, [num_xs, num_additional_inputs, num_xs]
        )
        ndim = outs[0].ndim

        # First_slice_copy does not keep the original requires_grad flag,
        # but we need it here in order to compute the correcte gradients
        xs_slices = first_slice_copy_with_grad(itertools.chain(xs, xs))

        # Construct the operands from the forward, fw_operands
        # and the operands for a single event t of the forward, fw_operands_slice
        fw_operands = (*xs, *additional_inputs)
        fw_operands_slice = (*xs_slices, *additional_inputs)

        # 2.) Prepare the backward graph
        combine_fn_bw = create_bw_fn(ctx._combine_fn, fw_operands_slice)

        # 3.) Materialize the ``combine_fn_bw``
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        combine_fn_bw_gm = materialize_as_graph(
            combine_fn_bw,
            (
                *fw_operands_slice,
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

        # 4.) Compute the single step bw (instantaneous gradients) at every step ``t``
        # Use a ones_like tensor in order not to scale the bwyst and bwxst,
        # with the upstream gradients yet.
        # Note: All bwyst and bwxst are computed in parallel, thus the tensors bwys and bwxs are the result.
        dummy_upstream_grad = (torch.ones_like(x) for x in gl_ys)
        grads = mapped_combine_fn_bw_gm(
            *(o.roll(1, dim) for o in outs), *fw_operands, *dummy_upstream_grad
        )
        bwys, bwxs = split_into_chunks(grads, [num_xs, num_xs])

        def compute_y_mat(bwys: torch.Tensor) -> torch.Tensor:
            # Prepare a ones and a zeros helper mask in order to easily compute the y_mat
            def compute_helper_tril_mask(diagonal):
                def expand_masks(mask):
                    for _ in range(ndim - 1):
                        mask = mask.unsqueeze(-1)
                    return mask

                tril_mask = torch.tril(
                    torch.ones(
                        scan_length, scan_length, device=bwys.device, dtype=torch.bool
                    ),
                    diagonal=diagonal,
                )
                tril_mask = expand_masks(tril_mask)
                tril_mask = tril_mask.expand(-1, -1, *bwys.shape[1:])
                return tril_mask

            # The ones mask is used to fill the main diagonal and all elements below it with 1s
            ones_mask = compute_helper_tril_mask(0)

            # The zero mask is used to set all elements below the main diagonal to 0
            zeros_mask = compute_helper_tril_mask(-1)

            # 5.1) Repeat the elements of bwys to form the square matrix
            y_mat = bwys.unsqueeze(dim).repeat_interleave(scan_length, dim)

            # 5.2) Fill the lower triangular part, including the diagonal,
            # of the h_mat with 1s. I.e., use the ones_mask to fill with 1s.
            y_mat.masked_fill_(ones_mask, 1.0)

            # 5.3) Compute the cumulative products across dim + 1
            y_mat = y_mat.cumprod(dim=dim + 1)

            # 5.4) Replace the elements we filled with 1s before with 0s
            y_mat.masked_fill_(zeros_mask, 0.0)

            return y_mat

        def compute_grad(bwxs, bwys, gl_ys):
            # Set the first gradient component of bwxs to 1.0, per definition.
            torch.select(bwxs, dim, 0).fill_(1.0)

            # 5.) Compute the gradient transition matrix
            y_mat = compute_y_mat(bwys)

            # 6.) scale the y_mat with the upstream gradients gl_ys
            scaled_y_mat = y_mat * gl_ys

            # 7.) Reduce the y_mat with sum along the columns to get the total contributions for xs_t
            summed_y_mat = scaled_y_mat.sum(dim + 1)

            # 8.) Scale with the bwxs to obtain the final gradients g_xs
            g_xs = summed_y_mat * bwxs

            return g_xs

        # Stack all leaves of the gradients along the first dimension.
        # This is useful as later the gradients of those leaves can be computed in parallel.
        bwxs_stacked_leaves = torch.stack(bwxs)
        bwys_stacked_leaves = torch.stack(bwys)
        gl_ys_stacked_leaves = torch.stack(gl_ys)

        # The compute_grad function is parallelized across all individual leaves of xs
        # as these gradients can be computed independently from each other
        # TODO: torch.vmap may create composability issues
        compute_grad_mapped = torch.vmap(compute_grad, 0, 0)

        g_xs = compute_grad_mapped(
            bwxs_stacked_leaves, bwys_stacked_leaves, gl_ys_stacked_leaves
        )

        # TODO: Currently the gradients for the additional_inputs are not computed properly
        return *[None] * 3, *g_xs, *[None] * num_additional_inputs


@associative_scan_op.py_autograd_impl
def associative_scan_autograd(combine_fn, xs, additional_inputs):
    num_xs = len(xs)
    num_additional_inputs = len(additional_inputs)

    if num_additional_inputs > 0:
        raise RuntimeError(
            "Associative_scan does currently not support gradients for lifted parameters!"
        )

    flat_out = AssociativeScanAutogradOp.apply(
        combine_fn,
        num_xs,
        num_additional_inputs,
        *(tuple(xs) + tuple(additional_inputs)),
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
