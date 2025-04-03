# mypy: allow-untyped-defs
import functools
import itertools
from collections.abc import Iterable
from typing import Any, Callable, Optional

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.cond import create_bw_fn, materialize_as_graph
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _maybe_compile_and_run_fn,
    check_meta_consistency,
    first_slice_copy,
    reenter_make_fx,
    save_tensors_and_symints_for_backward,
    saved_tensors_and_symints,
    unique_graph_id,
    UnsupportedAliasMutationException,
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
    assert len(args) == (
        num_init_leaves + num_inp_leaves
    ), f"Combin_fn received wrong number of arguments, expected {num_init_leaves + num_inp_leaves}, but got {len(args)}"
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


def split_into_chunks(iterable: Iterable[Any], chunk_sizes: list[int]) -> list[Any]:
    it = iter(iterable)
    assert sum(chunk_sizes) == len(
        iterable
    ), "the sum of all chunks needs to match the length of the iterable."
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

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            return next_carry, y

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
        assert isinstance(
            additional_inputs, (tuple, list)
        ), "additional_inputs must be a tuple."
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
    check_meta_consistency(init_fake_tensors, carry_fake_tensors, "init", "carry")

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
    NOTE: [scan autograd implementation]

    The ``combine_fn``, is the operator used during the scan. For example
    def combine_fn(x: torch.Tensor, y: torch.Tensor):
        next_carry = y = x * y
        return next_carry, y

    The ``combine_fn_bw``, used in the backward, is the joint function of the function ``combine_fn``.
    It receives the upstream gradients and the inputs of ``combine_fn`` and computes the gradients
    for x and y of ``combine_fn``. For example for the ``combine_fn`` above
    def combine_fn_bw(x: torch.Tensor, y: torch.Tensor, g_new_carry: torch.Tensor, g_y: torch.Tensor):
        return g_y * y + g_new_carry * y, g_y * x + g_new_carry * x

    To use a scan operation for the backward path as well, the function ``combine_fn`` is modified such that it
    returns all carries and not only the last one. In particular, we define ``combine_fn_with_carry_checkpoint``:
    def combine_fn_with_carry_checkpoint(x: torch.Tensor, y: torch.Tensor):
        next_carry, y = combine_fn(x, y)
        return next_carry, (next_carry, y)

    NOTE: [scan forward implementation]
    With the function defined as above, the forward output of scan works as follows:

    1.) Prepare the forward graph wrapper ``combine_fn_with_carry_checkpoint``:
    As mentioned above, we need the carries from all steps later for the backward path and thus we first prepare
    the wrapper ``combine_fn_with_carry_checkpoint``, which produces as outputs all carries, the last carry and all outputs.

    2.) Compute the all carries, the last carry and all outputs using ``combine_fn_with_carry_checkpoint``:
    Next we utilize the ``combine_fn_with_carry_checkpoint`` and compute all carries, the last carry and all outputs, i.e.,
    last_carry, (carries, outs) = scan_op(combine_fn_with_carry_checkpoint, init, xs, additional_inputs)

    3.) Prepare the backward graph:
    Finally, we prepare the backward graph to be used in the backward function.
    We utilize ``create_bw_fn`` to generate the joint function, i.e.,
    bw_fn = create_bw_fn(combine_fn, operands), where operands = [init, xs, additional_inputs]

    The bw_fn requires the primals (operands) followed by the tangents (upstream gradients) from a single step
    and produces the gradients of that step, i.e.,
    g_c_(T-1), g_xs_T, g_additional_input_T = bw_fn(c_(T-1), xs_T, additional_inputs, g_last_carry, g_ys_T).

    Because we utilize the ``bw_fn`` in combination with scan during the backward function which provides
    the arguments in the order tangents followed by primals, we need to wrap ``bw_fn`` into ``bw_fn_args_reordered``
    which corrects the order from tangents followed by primals to the expected order of primals followed by tangents.

    NOTE: [scan backward implementation]
    the backward of scan can be computed as:

    4.) Create a wrapper of the ``combine_fn_bw``, i.e., ``combine_fn_bw_grad_accumulation``:
    In the forward, there may be additional inputs that participate in every forward step.
    The gradients for those additional inputs are also computed at every step and need to be accumulated over all steps,
    which is taken care of in this wrapper.

    5.) Materialize the ``combine_fn_bw_grad_accumulation``:
    We need to materialize the bw graphs because dynamo is unable to
    trace through the joint function when torch.compile torch.autograd.grad.

    6.) Perform the backward scan as
    g_additional_inputs, g_init, g_xs = scan_op(combine_fn_bw_grad_accumulation, bw_init, bw_xs), where
    bw_init is the last carry from the forward, i.e., bwd_init = [*initial_g_additional_inputs, *g_last_carry] and
    bw_xs is the combination of the upstream gradients g_ys, the forward carries prepended with the fw_init and the fw_xs,
    i.e., bwd_xs = [*g_ys, *bw_carries, *fw_xs], with bw_carries = concat([fw_init, fw_carries[:-1]]).

    For ease of understanding the procedure of the gradient calculation is as follows:
    One starts from the last step with the init being the upstream gradient of initial_g_additional_inputs (all zeros) and g_last_carry.
    Then, in the first scan step, we compute the gradients g_c_T, g_xs_T and g_addititional_inputs_T with
    g_c_(T-1), g_xs_T and g_addititional_inputs_T = bw_fn(c_(T-1), xs_T, additional_inputs, g_last_carry, g_ys_T)
    We then accumulate g_addititional_inputs_T with the g_addititional_inputs_T and thus use as the init for the next step
    initial_g_additional_inputs + g_addititional_inputs_T, g_c_(T-1). Then, in the next step we compute the gradients for T-1, i.e.,
    g_c_(T-2), g_xs_(T-1) and g_addititional_inputs_(T-1) = bw_fn(c_(T-2), xs_(T-1), additional_inputs, g_c_(T-1), g_ys_(T-1)).
    We again accumulate the g_addititional_inputs_(T-1) and use it together with the g_c_(T-2) as the new init for the next step.
    This procedure continues until we arrive at the first step, i.e.,
    0, g_xs_0 and g_addititional_inputs_0 = bw_fn(init, xs_0, additional_inputs, g_init, g_ys_0).
    Through this procedure we end up with the
    gradients for the init -> g_init,
    the gradients for the xs -> g_xs and
    the gradients for the additional_inputs -> g_additional_inputs.

    As a last step, we mask the g_additional_inputs with Nones at places where the additional inputs are not tensors.

    Note: Because we start with the last step T and gradually progress backward in time to the first step,
    the elements of bw_xs are flipped along the scan dimension

    Note: g_last_carry and g_ys are provided through the upstream autograd infrastructure

    Note: The scan_op in the backward needs to operate always reverse over time, i.e., starting from the last time step and
    moving to the first. Therefore, the bwd_xs and the resulting gradients g_xs need to be flipped

    Note: If any element of init, of xs or of the outputs does not require gradients, i.e., requires_grad=False,
    there will be still gradients returned for those elements.
    However, those gradients will be a tensor of the same shape as the element, but the gradient will be filled with zeros.
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
        init, xs, additional_inputs = split_into_chunks(
            operands, [num_leaves_init, num_leaves_xs, num_additional_inputs]
        )
        additional_inputs_tensor_mask = get_tensor_mask(additional_inputs)
        ctx._additional_inputs_tensor_mask = additional_inputs_tensor_mask

        # 1.) Prepare the forward graph wrapper ``combine_fn_with_carry_checkpoint``
        # The wrapper of the forward graph returns carries from all iterations,
        # not just from the last iteration. These are required in the backward path
        def combine_fn_with_carry_checkpoint(*args):
            new_carry, y = _extract_carry_and_out(combine_fn(*args), num_leaves_init)
            return [
                *new_carry,
                # We additionally checkpoint all the intemediate carry outputs for backward.
                *[
                    n_c.clone().detach() if isinstance(n_c, torch.Tensor) else n_c
                    for n_c in new_carry
                ],
                *y,
            ]

        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        with torch._C._AutoDispatchBelowAutograd():
            # 2.) Compute the all carries, the last carry and all outputs using ``combine_fn_with_carry_checkpoint``
            carry, carries_outs = _extract_carry_and_out(
                scan_op(combine_fn_with_carry_checkpoint, init, xs, additional_inputs),
                num_leaves_init,
            )

            # Collect the carries for each time step from the outs
            # and save them for the backward path
            carries = list(carries_outs[:num_leaves_init])
            outs = list(carries_outs[num_leaves_init:])
            save_tensors_and_symints_for_backward(ctx, list(operands) + carries + outs)
            ctx._num_leaves_ys = len(outs)

            # 3.) Prepare the backward graph
            def combine_fn_bw(fn, ops):
                n_primals = len(ops)

                bw_fn = create_bw_fn(
                    fn,
                    ops,
                )

                def bw_fn_args_reordered(*args_and_grad_outs):
                    # Change the order of the primals and the tangents
                    # This is required because create_bw_fn creates the backward function such
                    # that it requires primals followed by tangents, while the backward scan provides
                    # tangents followed by primals. Therefore, this wrapper simply corrects this order mismatch.
                    tangents = args_and_grad_outs[: len(args_and_grad_outs) - n_primals]
                    primals = args_and_grad_outs[len(args_and_grad_outs) - n_primals :]
                    return bw_fn(*primals, *tangents)

                return bw_fn_args_reordered

            ctx._combine_fn_bw = combine_fn_bw(combine_fn, operands)

            return (*carry, *outs)

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

        # 4.) Create the BW wrapper to accumulate the gradients for the additional_inputs
        def combine_fn_bw_grad_accumulation(*args):
            # Separate off gradient accumulation for additional arguments from the arguments used for ``ctx._combine_fn_bw``
            # The content of ``combine_fn_bw_args`` is [*carries_g, *outs_g, *init, *xs, *additional_inputs]
            carried_g_additional_input, combine_fn_bw_args = split_into_chunks(
                args,
                [
                    num_additional_inputs,
                    num_leaves_init
                    + num_leaves_ys
                    + num_leaves_init
                    + num_leaves_xs
                    + num_additional_inputs,
                ],
            )

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

            return [*new_g_additional_inputs, *g_c_t, *g_xs_t]

        # 5.) Materialize the ``combine_fn_bw_grad_accumulation``
        def construct_args_single_step_bw():
            # This function constructs the arguments for a single step of the backward scan.
            # In other words, it creates the arguments for ``combine_fn_bw_grad_accumulation``
            # The order of the arguments returned is identical to the order the backward scan
            # operations provides

            # The following arguments are used for the backward part of the joint graph
            # The first argument relates to the gradients of the additional inputs.
            # Because only tensor elements of additional inputs can have requires_grad=True,
            # all non-tensor elements of additional inputs are masked
            masked_additional_inputs = [
                a.clone() if add_inp_tm else a
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
            (*args_single_step_bw,),
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        # Decompose the flat_grads into g_last_carry, g_ys
        g_last_carry, g_ys = split_into_chunks(
            flat_grads, [num_leaves_init, num_leaves_ys]
        )

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
        bwd_init = [*initial_g_additional_inputs, *g_last_carry]

        # 6.) Perform the backwrad scan:
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

        # The gradients for additional inputs that are not tensors are replaced with None.
        g_additional_inputs = mask_list(
            additional_inputs_tensor_mask,
            g_additional_inputs,
            [None] * num_additional_inputs,
        )

        return *[None] * 4, *g_init, *g_xs, *g_additional_inputs


@scan_op.py_impl(DispatchKey.Autograd)
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
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next():
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [first_slice_copy(inp) for inp in unwrapped_xs]
        sample_inputs = list(
            itertools.chain(
                unwrapped_init,
                sample_unwrapped_xs_sliced,
                unwrapped_additional_inputs,
            )
        )
        if _has_potential_branch_input_mutation(
            combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be modifying the input!"
            )
        if _has_potential_branch_input_alias(
            combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be aliasing the input!"
            )
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
