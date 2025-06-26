# mypy: allow-untyped-defs
import functools
import itertools
from collections.abc import Sequence
from typing import Any, Callable, Optional

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.cond import create_bw_fn
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    check_meta_consistency,
    first_slice_copy,
    materialize_as_graph,
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
from torch.utils._python_dispatch import _get_current_dispatch_mode


aten = torch._ops.ops.aten


def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
):
    assert len(args) == (num_init_leaves + num_inp_leaves), (
        f"Combin_fn received wrong number of arguments, expected {num_init_leaves + num_inp_leaves}, but got {len(args)}"
    )
    carry = pytree.tree_unflatten(args[:num_init_leaves], spec_init)
    xs = pytree.tree_unflatten(args[num_init_leaves:], spec_xs)
    return combine_fn(carry, xs)


def _extract_carry_and_out(flat_out: list[Any], num_carry: int):
    return split_into_chunks(flat_out, [num_carry, len(flat_out) - num_carry])


# We also do a clone with contiguous_format. This is to be consistent with
# eager semantic of scan, which stacks the outputs. The result is contiguous
# as a result of the stack operation.
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor:
    return (
        y.unsqueeze(0)
        .repeat(*([scan_length] + [1] * y.ndim))
        .clone(memory_format=torch.contiguous_format)
    )


# NOTE: These functions can be reused in associative_scan and eventually moved to
# torch._higher_order_ops.utils
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
    assert len(mask) == len(inp), (
        "The length of the mask needs to be identical to the length of the input"
    )
    if other is not None:
        assert len(inp) == len(other), (
            "If an input and an other list is provided, they need to have the same length"
        )
        return [i if m else o for m, i, o in zip(mask, inp, other)]
    else:
        return [i for m, i in zip(mask, inp) if m]


def first_slice_copy_with_grad(li: list[Any]) -> list[Any]:
    # First_slice_copy does not keep the original requires_grad flag,
    # but we need it for materialize_as_graph
    # in order to compute the correct gradients
    # The reason why first_slice_copy doesn't keep requires_grad flag is
    # because it's called in torch.autograd.Function.backward/forward.
    slc = [first_slice_copy(x).requires_grad_(x.requires_grad) for x in li]
    return slc


def split_into_chunks(iterable: Sequence[Any], chunk_sizes: list[int]) -> list[Any]:
    it = iter(iterable)
    assert sum(chunk_sizes) == len(iterable), (
        "the sum of all chunks needs to match the length of the iterable."
    )
    return [list(itertools.islice(it, size)) for size in chunk_sizes]


def call_operator(operator, *args):
    return pytree.tree_leaves(operator(*args))


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> tuple[pytree.PyTree, pytree.PyTree]:
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
            each tensor leaf is a stacked output along first dim, where each slice is the output of a scan iteration.

    Restrictions:
        - The combine_fn shouldn't have any aliasing between input-input, input-output, and output-output. E.g. return a view
            or the same tensor as input is not supported. As a workaround, can clone the output to avoid aliasing.

        - The combine_fn shoudn't mutate any inputs. We'll remove the mutation restriction for inference soon. Please file an issue
            if you input mutation support for training is needed.

        - The combine_fn's init carry should match the next_carry in pytree structure and in tensor metadata.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            # clone the output to avoid output-output aliasing
            return next_carry, y.clone()


        i0 = torch.zeros(1)
        xs = torch.arange(5)
        # returns torch.tensor([10.]), torch.tensor([[0], [1.], [3.], [6.], [10.]])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    # The reason we flatten init and xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    # Shortcut if no xs is provided
    if len(leaves_xs_orig) == 0:
        return init, []

    def _validate_input(cfn, lxs, linit, d, r):
        # Basic arguments check
        if not callable(cfn):
            raise RuntimeError("Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise RuntimeError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))

        # Checks for init
        if len(linit) == 0:
            raise RuntimeError("scan() operator requires init leaves.")
        for x in linit:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All init leaves must be a Tensor but got {x}")

        # Checks for xs
        for x in lxs:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All xs leaves must be a Tensor but got {x}")
        if any(x.ndim <= d for x in lxs):
            raise RuntimeError(
                "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
            )
        if any(x.shape[d] == 0 for x in lxs):
            raise RuntimeError(
                "All xs leaves must at least have 'dim' number of dimensions and scan dimension > 0"
            )

    ndim = leaves_xs_orig[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)

    _validate_input(combine_fn, leaves_xs_orig, leaves_init, dim, reverse)

    # Move scan dim to 0 and always perform scan on dim 0
    leaves_xs = []
    for elem in leaves_xs_orig:
        leaves_xs.append(torch.movedim(elem, dim, 0))

    if reverse:
        leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

    # TODO: Support _inductor lowering
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

    combine_fn = functools.partial(
        wrap_combine_fn_flat,
        combine_fn=combine_fn,
        spec_init=spec_init,
        spec_xs=spec_xs,
        num_init_leaves=len(leaves_init),
        num_inp_leaves=len(leaves_xs),
    )

    def run_flattened_scan(combine_fn, leaves_init, leaves_xs):
        return scan_op(combine_fn, leaves_init, leaves_xs, additional_inputs=())

    carry, out = _maybe_compile_and_run_fn(
        run_flattened_scan,
        combine_fn,
        leaves_init,
        leaves_xs,
    )

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)

    return carry, out


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, init, xs, additional_inputs):
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
        return super().__call__(combine_fn, init, xs, additional_inputs)


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, additional_inputs=()):
    def _scan(init, xs):
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(xs) == 0:
            return carry, []

        num_elems = xs[0].shape[dim]
        ind = 0

        # Compute dummy shapes for the pre-allocation
        num_init_leaves = len(init)
        dummy_carry, dummy_out = _extract_carry_and_out(
            call_operator(
                operator,
                *carry,
                *[first_slice_copy(elem, dim) for elem in xs],
                *additional_inputs,
            ),
            num_init_leaves,
        )

        out_tensor_mask = get_tensor_mask(dummy_out)
        dummy_out_masked = mask_list(out_tensor_mask, dummy_out)

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        # out: (num_elems, M, N, ...)
        # idx: (1, M, N)
        outs = [
            torch.zeros(
                [num_elems] + list(e.size()),
                dtype=e.dtype,
                device=e.device,
            )
            for i, e in enumerate(dummy_out_masked)
        ]
        idxs = [
            torch.ones_like(e, dtype=torch.int64).unsqueeze(0)
            for i, e in enumerate(dummy_out_masked)
        ]

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                # o: (num_elems, M, N ...)
                # x: (M, N, ...) -> (1, M, N)
                # ind * idx: (1, M, N,) with values to be ind
                # essentially: o[ind][n][k] = x[0][n][k]
                o.scatter_(0, ind * idx, x.unsqueeze(0))

        for i in range(num_elems):
            ind = i
            carry, out = _extract_carry_and_out(
                call_operator(
                    operator,
                    *carry,
                    *[elem.select(dim, ind) for elem in xs],
                    *additional_inputs,
                ),
                num_init_leaves,
            )

            # Store the inits in the outs matrix.
            store_out_in_outs(mask_list(out_tensor_mask, out), ind)

        # Expand outs with None depending on the tensor mask of the output
        outs_expanded = [outs.pop(0) if out_m else None for out_m in out_tensor_mask]

        return [*carry, *outs_expanded]

    scans = _scan(init, xs)
    return scans


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: list[torch.Tensor],
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
):
    from torch._dynamo.utils import clone_input

    with disable_proxy_modes_tracing():
        sample_inits = [clone_input(x_init) for x_init in init]
        sample_inputs = [first_slice_copy(x) for x in xs]
        sample_additional_inputs = [
            clone_input(x) if isinstance(x, torch.Tensor) else x
            for x in additional_inputs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_inits, *sample_inputs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            assert outputs is None
            assert len(node.args) == 1
            outputs = node.args[0]

    assert outputs is not None

    carry, output = _extract_carry_and_out(outputs, len(init))
    init_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        i.clone() for i in init
    ]
    carry_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        c.meta["val"] for c in carry
    ]
    check_meta_consistency(
        init_fake_tensors, carry_fake_tensors, "init", "carry", include_contiguity=False
    )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, init, xs, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="scan"
    )

    with disable_proxy_modes_tracing():
        scan_length = xs[0].shape[0]
        fake_carry, fake_outputs = _extract_carry_and_out(
            [o.meta["val"] for o in outputs], len(init)
        )
        out = (
            *fake_carry,
            *(stack_y(t, scan_length) for t in fake_outputs),
        )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, additional_inputs):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, init, xs, additional_inputs=additional_inputs)


class ScanAutogradOp(torch.autograd.Function):
    """
    Example ::

        def combine_fn(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x * y
            return next_carry, y

        The ``combine_fn_bw``, computing the gradients for x and y of ``combine_fn`` is computed as:
        def combine_fn_bw(x: torch.Tensor, y: torch.Tensor, g_carry: torch.Tensor, g_y: torch.Tensor):
            return g_y * y + g_carry * y, g_y * x + g_carry * x

        Note: In a real usecase of scan, there may be additional_inputs that participate in the
        forward as well as in the backward of the scan operator. For the sake of readability those inputs
        have been omitted in the following example, but are included in the subsequent detailed description below

        The forward output of scan is computed as:
        carry, ys = scan(combine_fn, init, xs).

        This computation can be unpacked as
        c_0, ys_0 = combine_fn(init, xs_0)
        c_1, ys_1 = combine_fn(carry_0, xs_1)
        c_2, ys_2 = combine_fn(carry_1, xs_2)
        ...
        c_T, ys_T = combine_fn(carry_(T-1), xs_T)

        We collect c_0, c_1, ..., c_T into a vector of carries that we save for the backward,
        but we only output (c_T, ys),
        where ys is the vector of all intermediate outputs [y_0, y_1, ..., y_T].

        Given the carries and the ys, the gradients for xs and for init can be computed as follows:
        We receive the upstream gradients in torch.autograd.Function, i.e., we get g_c_T and g_ys,
        where g_ys is the vector of all intermediate gradients of the outputs [g_ys_0, g_ys_1, ..., g_ys_T]

        We then proceed to compute the gradients for the init (g_init) and the xs (g_xs) by running a
        scan operation reverse over time. For example,

        g_c_(T-1), g_xs_T = combine_fn_bw(c_(T-1), xs_T, g_c_T, g_ys_T)
        g_c_(T-2), g_xs_(T-1) = combine_fn_bw(c_(T-2), xs_(T-1), g_c_(T-1), g_ys_(T-1))
        g_c_(T-3), g_xs_(T-2) = combine_fn_bw(c_(T-3), xs_(T-2), g_c_(T-2), g_ys_(T-2))
        ...
        g_init, g_xs_1 = combine_fn_bw(c_0, xs_1, g_c_0, g_ys_1)
        0     , g_xs_0 = combine_fn_bw(init, xs_0, g_init, g_ys_0),

        where combine_fn_bw takes the forward inputs of step t (i.e. c_(t-1), xs_t),
        the gradients of the carry of step t (i.e. g_c_t) and
        the upstream gradient of the output of step t (i.e. g_ys_T)
        and returns the gradient of xs_t -> g_xs_t, as well as the gradient for the carry of step t-1 -> g_c_(t-1).

        Through this procedure we end up with the
        gradients for the init -> g_init,
        the gradients for the xs -> g_xs.


    NOTE: [scan autograd implementation]

    The forward of scan can be computed as:
    1.) Prepare the forward graph wrapper ``combine_fn_with_carry_checkpoint``:
    To use a scan operation for the backward path as well, we need access to the carries from all steps.
    Thus, the function ``combine_fn`` is wrapped such that it returns all carries and not only the last carry.
    In particular, we define ``combine_fn_with_carry_checkpoint``:
    def combine_fn_with_carry_checkpoint(x: torch.Tensor, y: torch.Tensor):
        carry, y = combine_fn(x, y)
        return carry, (carry, y)

    The scan operator will stack all outputs along the scan dimension.
    Thus, by putting next_carry also into outputs of ``combine_fn_with_carry_checkpoint``,
    the carries from all steps will be stacked and hence gives us chekpointed_carries

    2.) Compute all carries, the last carry and all outputs using ``combine_fn_with_carry_checkpoint``:
    c_T, (carries, ys) = scan_op(combine_fn_with_carry_checkpoint, init, xs, additional_inputs),
    Where c_T (last carry) and ys (all outputs) are the original results of scan with the ``combine_fn``.
    However, carries are checkpointed carries from all steps.
    As a result of the forward, only the last carry c_T and the ys are returned,
    while all carries are saved for the backward.

    The backward of scan can be computed as:

    3.) Prepare the backward graph:
    We prepare the backward graph to be used in the backward function.
    We utilize ``create_bw_fn`` to generate the joint function, i.e.,
    ctx._combine_fn_bw = create_bw_fn(ctx._combine_fn, fw_operands), where fw_operands = [init, xs_0, additional_inputs]

    The ctx._combine_fn_bw requires the primals (operands)
    followed by the tangents (upstream gradients) from a single step
    and produces the gradients of that step, i.e.,
    g_c_(T-1), g_xs_T, g_additional_input_T = ctx._combine_fn_bw(c_(T-1), xs_T, additional_inputs, g_c_T, g_ys_T).

    4.) Create a wrapper of the ``combine_fn_bw``, i.e., ``combine_fn_bw_grad_accumulation``:
    In the forward, there may be additional inputs that participate in every forward step.
    The gradients for those additional inputs are also computed at every step and need to be accumulated over all steps,
    which is taken care of in this wrapper. For example:
    def combine_fn_bw_grad_accumulation(*args):
        carried_g_additional_input = args[:num_additional_inputs]
        inputs_bw_fn = args[num_additional_inputs:]
        g_c_(t-1), g_xs_t, g_additional_input_t = ctx._combine_fn_bw(*inputs_bw_fn)
        new_g_additional_inputs = carried_g_additional_input + g_additional_input_t
        # The ``new_g_additional_inputs`` and the ``g_c_t`` are encoded in the carry of the backward scan operator
        # The ``g_xs_t`` is encoded as the output of the backward scan operator
        return [*new_g_additional_inputs, *g_c_t, *g_xs_t]

    5.) Perform the backward scan as
    g_additional_inputs, g_init, g_xs = scan_op(combine_fn_bw_grad_accumulation, bw_init, bw_xs), where
    bw_init consists of the initial gradient carry for the additional_inputs (initialized with 0s):
    initial_g_additional_inputs, and the gradient of the last carry: g_c_T. Thus:
    bwd_init = [*initial_g_additional_inputs, *g_c_T].

    bw_xs consists of the combination of the upstream gradients g_ys,
    the forward carries prepended with the fw_init, i.e., bw_carries = concat([fw_init, fw_carries[:-1]]) and
    the fw_xs. In particular,
    bwd_xs = [*g_ys, *bw_carries, *fw_xs].

    Note: g_c_T and g_ys are provided through the torch.autograd.Function.backward's input

    As demonstrated in the Example above, this backward scan then yields the gradient for the init -> g_init
    and the gradient for the xs -> g_xs

    NOTE: [scan partial grad handling]
    If any element of init, of xs, of the outputs or of the additional_inputs does not require gradients,
    i.e., requires_grad=False, there will be still gradients returned for those elements,
    but those gradients will be a tensor filled with zeros of the same shape as the element itself.

    A special case are additional_inputs that are not tensors. Such inputs can occur for example with symbolic tracing,
    where the shape symbol (SymInt) becomes an additional_input.
    For such cases, we compute a ``additional_inputs_tensor_mask``, which is True for elements of additional_inputs
    that are tensors and False otherwise. Gradients of additional_inputs are only accumulated if this mask is True,
    otherwise, the value of initial_g_additional_inputs is passed, which is None for non-Tensor values.
    """

    @staticmethod
    def forward(
        ctx,
        combine_fn,
        num_leaves_init,
        num_leaves_xs,
        num_additional_inputs,
        *operands,
    ):
        ctx._num_leaves_init = num_leaves_init
        ctx._num_leaves_xs = num_leaves_xs
        ctx._num_additional_inputs = num_additional_inputs
        ctx._combine_fn = combine_fn
        init, xs, additional_inputs = split_into_chunks(
            operands, [num_leaves_init, num_leaves_xs, num_additional_inputs]
        )
        additional_inputs_tensor_mask = get_tensor_mask(additional_inputs)
        ctx._additional_inputs_tensor_mask = additional_inputs_tensor_mask

        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        # 1.) Prepare the forward graph wrapper ``combine_fn_with_carry_checkpoint``
        # The wrapper of the forward graph returns carries from all iterations,
        # not just from the last iteration. These are required in the backward path
        def combine_fn_with_carry_checkpoint(*args):
            carry, y = _extract_carry_and_out(combine_fn(*args), num_leaves_init)
            return [
                *carry,
                # We additionally checkpoint all the intemediate carry outputs for backward.
                *[
                    n_c.clone().detach() if isinstance(n_c, torch.Tensor) else n_c
                    for n_c in carry
                ],
                *y,
            ]

        with torch._C._AutoDispatchBelowAutograd():
            # 2.) Compute the all carries, the last carry and all outputs using ``combine_fn_with_carry_checkpoint``
            c_T, carries_ys = _extract_carry_and_out(
                scan_op(
                    combine_fn_with_carry_checkpoint,
                    init,
                    xs,
                    additional_inputs,
                ),
                num_leaves_init,
            )

            # Collect the carries for each time step from the outs
            # and save them for the backward path
            carries = list(carries_ys[:num_leaves_init])
            ys = list(carries_ys[num_leaves_init:])
            save_tensors_and_symints_for_backward(ctx, list(operands) + carries + ys)
            ctx._num_leaves_ys = len(ys)

            return (*c_T, *ys)

    @staticmethod
    def backward(ctx, *flat_grads):
        r"""
        This function computes the gradients of the scan operation.
        It does so by using a scan operator using all carries and the upstream gradients (see description above)

        Args:
            flat_grads (torch.Tensor): The tensor of flattened upstream gradients.
        """

        # Collect the saved items from the forward
        num_leaves_init = ctx._num_leaves_init
        num_leaves_xs = ctx._num_leaves_xs
        num_leaves_ys = ctx._num_leaves_ys
        num_additional_inputs = ctx._num_additional_inputs
        additional_inputs_tensor_mask = ctx._additional_inputs_tensor_mask

        def prepend_init_to_carries(init, carries):
            # Prepare the carries for the backward path.
            # This requires to concatenate the init and the carries
            return [
                torch.cat([torch.unsqueeze(i, 0), c[:-1]], dim=0)
                for i, c in zip(init, carries)
            ]

        def initialize_g_additional_inputs(
            additional_inputs,
        ):
            # The initial gradients for the additional_inputs are all zeros
            g_additional_inputs = [
                torch.zeros_like(ai) if ai_tm else None
                for ai_tm, ai in zip(additional_inputs_tensor_mask, additional_inputs)
            ]
            return g_additional_inputs

        # Retrieve the forward inputs and the forward outputs and dissect them
        flat_args = saved_tensors_and_symints(ctx)
        fw_init, fw_xs, additional_inputs, fw_carries, fw_ys = split_into_chunks(
            flat_args,
            [
                num_leaves_init,
                num_leaves_xs,
                num_additional_inputs,
                num_leaves_init,
                num_leaves_ys,
            ],
        )

        # 3.) Prepare the backward graph
        fw_operands = (
            *fw_init,
            *[first_slice_copy(xs) for xs in fw_xs],
            *additional_inputs,
        )
        ctx._combine_fn_bw = create_bw_fn(ctx._combine_fn, fw_operands)

        # 4.) Create the BW wrapper to accumulate the gradients for the additional_inputs
        def combine_fn_bw_grad_accumulation(*args):
            # Dissect args and re-order them for the ``ctx._combine_fn_bw``
            # The content of ``combine_fn_bw_tangents`` is [*carries_g, *outs_g]
            # The content of ``combine_fn_bw_primals`` is [*init, *xs, *additional_inputs]
            (
                carried_g_additional_input,
                combine_fn_bw_tangents,
                combine_fn_bw_primals,
            ) = split_into_chunks(
                args,
                [
                    num_additional_inputs,
                    num_leaves_init + num_leaves_ys,
                    num_leaves_init + num_leaves_xs + num_additional_inputs,
                ],
            )
            combine_fn_bw_args = (*combine_fn_bw_primals, *combine_fn_bw_tangents)

            g_c_t, g_xs_t, g_additional_inputs_t = split_into_chunks(
                ctx._combine_fn_bw(*combine_fn_bw_args),
                [num_leaves_init, num_leaves_xs, num_additional_inputs],
            )

            new_g_additional_inputs = [
                # If the additional inputs are ints or SymInts, those values are taken as is and no gradients are added
                carr_g + curr_g if add_inp_tm else carr_g
                for add_inp_tm, carr_g, curr_g in zip(
                    additional_inputs_tensor_mask,
                    carried_g_additional_input,
                    g_additional_inputs_t,
                )
            ]

            # The ``new_g_additional_inputs`` and the ``g_c_t`` are encoded in the carry of the backward scan operator
            # The ``g_xs_t`` is encoded as the output of the backward scan operator
            return [*new_g_additional_inputs, *g_c_t, *g_xs_t]

        # Materialize the ``combine_fn_bw_grad_accumulation``
        def construct_args_single_step_bw():
            # This function constructs the arguments for a single step of the backward scan.
            # In other words, it creates the arguments for ``combine_fn_bw_grad_accumulation``
            # The order of the arguments returned is identical to the order the backward scan
            # operations provides

            # The following arguments are used for the backward part of the joint graph
            # The first argument relates to the gradient accumulation of the additional inputs.
            # Because only tensor elements of additional inputs can have requires_grad=True,
            # the values for non-tensor elements of additional inputs are None
            masked_additional_inputs = [
                a.clone() if add_inp_tm else None
                for add_inp_tm, a in zip(
                    additional_inputs_tensor_mask, additional_inputs
                )
            ]

            # The second argument relates to the gradients of the carries.
            # Because the arguments are for a single step only,
            # only the first slice of the carries is used.
            sliced_carries = [first_slice_copy(c) for c in fw_carries]

            # The third argument relates to the gradients of the ys.
            # Because the arguments are for a single step only,
            # only the first slice of the ys is used.
            sliced_ys = [first_slice_copy(o) for o in fw_ys]

            # The following arguments are used for the forward part of the joint graph
            # The fourth argument relates to the init for the forward.
            # I.e., fw_init

            # The fifth argument relates to the xs for the forward.
            # Because the arguments are for a single step only,
            # only the first slice of the xs is used.
            # Note: It is important to preserve the requires_grad flag of xs
            # and thus we use the wrapper function ``first_slice_copy_with_grad``
            fw_xs_slice = first_slice_copy_with_grad(fw_xs)

            # The last argument relates to the additional inputs for the forward.
            # I.e., additional_inputs

            return (
                *masked_additional_inputs,
                *sliced_carries,
                *sliced_ys,
                *fw_init,
                *fw_xs_slice,
                *additional_inputs,
            )

        args_single_step_bw = construct_args_single_step_bw()

        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        combine_fn_bw_grad_accumulation_gm = materialize_as_graph(
            combine_fn_bw_grad_accumulation,
            args_single_step_bw,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        # Decompose the flat_grads into g_c_T, g_ys
        g_c_T, g_ys = split_into_chunks(flat_grads, [num_leaves_init, num_leaves_ys])

        # Initialize the g_additional_inputs with zero-tensors.
        # This step is necessary because the gradients of the additional inputs are accumulated in the
        # ``wrapper_bwd_combine_fn`` and thus need a zero-initialized starting point
        initial_g_additional_inputs = initialize_g_additional_inputs(additional_inputs)

        # Prepend the inits to the carries.
        # This is needed, because when computing the gradients, the last carry is not needed
        # but the first carry, the init, is required.
        bw_carries = prepend_init_to_carries(fw_init, fw_carries)

        # Prepare the xs for the backward scan.
        bwd_xs = [*g_ys, *bw_carries, *fw_xs]

        # The flipping of the ``bwd_xs`` is necessary because the scan_op in the backward is always performed in reverse
        bwd_xs = [torch.flip(elem, [0]) for elem in bwd_xs]

        # Prepare the bwd_init
        bwd_init = [*initial_g_additional_inputs, *g_c_T]

        # 5.) Perform the backwrad scan:
        # The ``combine_fn_bw_wrapped`` receives the
        # initial_g_additional_inputs and the last carry as the ``bwd_init`` and the
        # gradients of the outputs (g_ys), as well as the fw_carries and the fw_xs of the forward as the ``bwd_xs``
        gradients = scan_op(
            combine_fn_bw_grad_accumulation_gm,
            bwd_init,
            bwd_xs,
            additional_inputs,
        )

        # Unpack the computed gradients
        g_additional_inputs, g_init, g_xs = split_into_chunks(
            gradients, [num_additional_inputs, num_leaves_init, num_leaves_xs]
        )

        # The flipping back along the scan dimension is required to get the gradients in the right order for ``xs``
        g_xs = [torch.flip(elem, [0]) for elem in g_xs]

        return *[None] * 4, *g_init, *g_xs, *g_additional_inputs


@scan_op.py_autograd_impl
def scan_autograd(combine_fn, init, xs, additional_inputs):
    num_leaves_init = len(init)
    num_leaves_xs = len(xs)
    num_additional_inputs = len(additional_inputs)

    flat_out = ScanAutogradOp.apply(
        combine_fn,
        num_leaves_init,
        num_leaves_xs,
        num_additional_inputs,
        *(tuple(init) + tuple(xs) + additional_inputs),
    )
    return *flat_out[:num_leaves_init], *flat_out[num_leaves_init:]


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, init, xs, additional_inputs):
    return trace_scan(mode, scan_op, combine_fn, init, xs, additional_inputs)


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(mode, combine_fn, init, xs, additional_inputs):
    with mode:
        scan_length = xs[0].shape[0]
        carry, outputs = _extract_carry_and_out(
            combine_fn(
                *init,
                *[first_slice_copy(inp) for inp in xs],
                *additional_inputs,
            ),
            len(init),
        )
        out = (
            *carry,
            *(stack_y(t, scan_length) for t in outputs),
        )
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, additional_inputs):
    from torch._higher_order_ops.utils import (
        _check_alias_and_mutation,
        _maybe_run_with_interpreter,
    )

    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)

    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(
            _maybe_run_with_interpreter(combine_fn)
        )
        sample_unwrapped_xs_sliced = [first_slice_copy(inp) for inp in unwrapped_xs]
        sample_inputs = list(
            itertools.chain(
                unwrapped_init,
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        _check_alias_and_mutation(combine_fn, sample_inputs, "scan", pre_dispatch)
        ret = scan_op(
            functional_combine_fn,
            unwrapped_init,
            unwrapped_xs,
            unwrapped_additional_inputs,
        )
    return ctx.wrap_tensors(ret)


# dense implementation for scan. Used for testing only.
def _fake_scan(combine_fn, init, xs=None, dim=0, reverse=False):
    carry_leaves, carry_spec = pytree.tree_flatten(init)
    inp_leaves, inp_spec = pytree.tree_flatten(xs)
    if xs is None or len(inp_leaves) == 0:
        return init, []
    result_flat = []
    carry = carry_leaves
    op = reversed if reverse else lambda x: x

    dummy_carry, dummy_out = combine_fn(
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(
            [first_slice_copy(elem, dim) for elem in inp_leaves],
            inp_spec,
        ),
    )
    dummy_out_leaves, dummy_out_spec = pytree.tree_flatten(dummy_out)
    num_leaves = len(dummy_out_leaves)

    for ind in op(range(inp_leaves[0].size(dim))):
        xs = [elem.select(dim, ind) for elem in inp_leaves]

        carry, y = combine_fn(
            pytree.tree_unflatten(carry, carry_spec),
            pytree.tree_unflatten(xs, inp_spec),
        )
        carry, _ = pytree.tree_flatten(carry)
        y, _ = pytree.tree_flatten(y)
        result_flat.append(y)

    results = [
        torch.stack([e[leave_ind] for e in op(result_flat)])
        for leave_ind in range(num_leaves)
    ]
    return (
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(results, dummy_out_spec),
    )
