# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable, Optional

import torch
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _maybe_compile_and_run_fn,
    autograd_not_implemented,
    check_meta_consistency,
    first_slice_copy,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode

from .utils import _from_fun, _maybe_reenter_make_fx, create_fw_bw_graph


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


def call_operator(operator, *args):
    return pytree.tree_leaves(operator(*args))
    

def _extract_carry_and_out(flat_out: list[Any], num_carry: int):
    return flat_out[:num_carry], flat_out[num_carry:]


def get_gradient_mask(tensor_list: list[Any]) -> list[bool]:
    # Helper function to determine whether an element of a tensor_list requires gradients
    # Gradient masks are used during the backward path to
    # return gradients for elements with requires_grad=True and None for others
    return [
        True if isinstance(v, torch.Tensor) and v.requires_grad else False
        for v in tensor_list
    ]


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


def create_fw_bw_graph_combinefn(combine_fn, init, xs, additional_inputs):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    # Helper wrapper for the autograd forward.
    # This wrapper ensures that the forward returns all carries
    # instead of only the last one
    # The gradients of the carries forwarded to the output are
    # detached in order not to raise problems with the function aliasing outputs

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # TODO: do less re-computation with min-cut partitioner.
            num_init = len(init)
            num_additional_inputs = len(additional_inputs)

            fw_init = [pytree.tree_map(_from_fun, x) for x in init]
            fw_xs = [first_slice_copy(pytree.tree_map(_from_fun, x)) for x in xs]
            fw_additional_inputs = [
                pytree.tree_map(_from_fun, a) for a in additional_inputs
            ]
            bw_additional_inputs = [
                pytree.tree_map(_from_fun, a) for a in additional_inputs
            ]

            carry, outs = _extract_carry_and_out(
                call_operator(
                    combine_fn,
                    *fw_init,
                    *fw_xs,
                    *fw_additional_inputs,
                ),
                num_init,
            )

            bw_init, bw_xs = [pytree.tree_map(_from_fun, c) for c in carry], [
                pytree.tree_map(_from_fun, o) for o in outs
            ]

            # 1.) Create the forward and the joint graph of the combine_fn
            fw_graph, joint_graph = create_fw_bw_graph(
                lambda *args: call_operator(combine_fn, *args),
                False,
                (
                    *fw_init,
                    *fw_xs,
                    *fw_additional_inputs,
                ),
                (*bw_init, *bw_xs),
            )

            # 2.) Retrace the wrapper of the forward graph that returns carries from all iterations,
            # not just from the last iteration
            def wrapper_fwd_combine_fn(*args):
                new_carry, y = _extract_carry_and_out(fw_graph(*args), num_init)
                return [*new_carry, *[n_c.clone().detach() for n_c in new_carry], *y]

            new_fw_graph = _maybe_reenter_make_fx(wrapper_fwd_combine_fn)(
                *fw_init, *fw_xs, *fw_additional_inputs
            )

            # 3.) Obtain the gradients from the joint_graph and
            # compute the gradient mask.
            # The reason is that the combine_fn might contain ``with torch.no_grad()`` statements
            # Thus, even if the gradients of xs or additional_inputs should be tracked,
            # The ``torch.no_grad()`` statements may break the gradient tracking
            g_c, g_xs = _extract_carry_and_out(
                joint_graph(
                    *bw_init,
                    *bw_xs,
                    *fw_init,
                    *fw_xs,
                    *fw_additional_inputs,
                ),
                num_init,
            )
            init_mask = get_gradient_mask(g_c)
            xs_mask = get_gradient_mask(g_xs[: len(g_xs) - num_additional_inputs])
            additional_inputs_mask = get_gradient_mask(
                g_xs[len(g_xs) - num_additional_inputs :]
            )
            additional_inputs_tensor_mask = get_tensor_mask(additional_inputs)

            # 4.) Retrace the wrapper for the joint_graph
            def wrapper_bwd_combine_fn(*args):
                carried_g_additional_input = args[:num_additional_inputs]

                g_c, g_xs = _extract_carry_and_out(
                    joint_graph(*args[num_additional_inputs:]), num_init
                )
                current_g_additional_inputs = g_xs[len(g_xs) - num_additional_inputs :]

                new_g_additional_inputs = [
                    # The clone().detach() is required to avoid aliasing inputs
                    # In case of int and SymInts, those values are simply used
                    carr_g + curr_g
                    if add_inp_m
                    else (carr_g.clone().detach() if add_inp_tm else carr_g)
                    for add_inp_m, add_inp_tm, carr_g, curr_g in zip(
                        additional_inputs_mask,
                        additional_inputs_tensor_mask,
                        carried_g_additional_input,
                        current_g_additional_inputs,
                    )
                ]

                # Split off the parts of the additional inputs from the g_xs
                g_xs = g_xs[: len(g_xs) - num_additional_inputs]

                g_xs = mask_list(
                    xs_mask,
                    g_xs,
                    [
                        torch.zeros_like(el)
                        for el in args[
                            num_additional_inputs
                            + num_init : num_additional_inputs
                            + num_init
                            + len(g_xs)
                        ]
                    ],
                )
                g_c = mask_list(
                    init_mask,
                    g_c,
                    [
                        torch.zeros_like(el)
                        for el in args[
                            num_additional_inputs : num_additional_inputs + num_init
                        ]
                    ],
                )

                return [*new_g_additional_inputs, *g_c, *g_xs]

        new_joint_graph = _maybe_reenter_make_fx(wrapper_bwd_combine_fn)(
            *bw_additional_inputs,
            *bw_init,
            *bw_xs,
            *fw_init,
            *fw_xs,
            *fw_additional_inputs,
        )
        return (
            new_fw_graph,
            new_joint_graph,
            init_mask,
            xs_mask,
            additional_inputs_mask,
            additional_inputs_tensor_mask,
        )


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
            for i, e in enumerate(dummy_out)
        ]
        idxs = [
            torch.ones_like(e, dtype=torch.int64).unsqueeze(0)
            for i, e in enumerate(dummy_out)
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
            store_out_in_outs(out, ind)

        return [*carry, *list(outs)]

    scans = _scan(init, xs)
    return scans


# We also do a clone with contiguous_format. This is to be consistent with
# eager semantic of scan, which stacks the outputs. The result is contiguous
# as a result of the stack operation.
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor:
    return (
        y.unsqueeze(0)
        .repeat(*([scan_length] + [1] * y.ndim))
        .clone(memory_format=torch.contiguous_format)
    )


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
    @staticmethod
    def unpack_list(li, num_first_part, num_second_part):
        init = li[:num_first_part]
        xs = li[num_first_part : num_first_part + num_second_part]
        additional_inputs = li[num_first_part + num_second_part :]
        return init, xs, additional_inputs

    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph,
        num_leaves_init,
        num_leaves_xs,
        carry_mask,
        xs_mask,
        additional_inputs_mask,
        additional_inputs_tensor_mask,
        *flat_args,
    ):
        ctx._joint_graph = joint_graph
        ctx._num_leaves_init = num_leaves_init
        ctx._num_leaves_xs = num_leaves_xs
        init, xs, additional_inputs = ScanAutogradOp.unpack_list(
            list(flat_args), num_leaves_init, num_leaves_xs
        )
        ctx._num_additional_inputs = len(additional_inputs)

        ctx._carry_mask = carry_mask
        ctx._xs_mask = xs_mask
        ctx._additional_inputs_mask = additional_inputs_mask
        ctx._additional_inputs_tensor_mask = additional_inputs_tensor_mask
        ctx._nonTensor_additional_inputs = [
            add_inp
            for add_inp_mask, add_inp in zip(
                additional_inputs_tensor_mask, additional_inputs
            )
            if not add_inp_mask
        ]

        with torch._C._AutoDispatchBelowAutograd():
            carry, carries_outs = _extract_carry_and_out(
                scan_op(fw_graph, init, xs, additional_inputs),
                num_leaves_init,
            )

            # Collect the carries for each time step from the outs
            # and save them for the backward path
            carries = list(carries_outs[:num_leaves_init])
            outs = list(carries_outs[num_leaves_init:])
            add_inp_save = mask_list(additional_inputs_tensor_mask, additional_inputs)
            ctx.save_for_backward(*(init + xs + add_inp_save + carries))
            ctx._num_leaves_ys = len(outs)
            return (*carry, *outs)

    @staticmethod
    def backward(ctx, *flat_grads):
        r"""
        This function computes the gradients of the scan operation.
        It does so by constructing using an additional scan operator with the gradients

        Args:
            flat_grads (torch.Tensor): The tensor of flattened upstream gradients.

        Example::

            The ``fw_graph`` f(.,.), used in the forward function, is the operator used during the scan. For example
            def f(x: torch.Tensor, y: torch.Tensor):
                next_carry = y = x * y
                return next_carry, y

            The ``joint_graph`` g(.,.), used in the backward function, is the joint function of the function f(.,.).
            It receives the upstream gradients and the inputs of f and computes the gradients
            for x and y of f. For example for the function f above
            def g(g_new_carry: torch.Tensor, g_y: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
                return g_y * y + g_new_carry * y, g_y * x + g_new_carry * x

            To use a scan operation for the backward path as well, the function f is modified such that it
            returns all carries and not only the last one. In particular:
            def f_autograd(x: torch.Tensor, y: torch.Tensor):
                next_carry, y = f(x, y)
                return next_carry, (next_carry, y)

            The inputs to ``scan`` in the forward path are init; xs_1, xs_2, ..., xs_T
            With the modified function f, the outputs of ``scan`` in the forward path are (c_1, y_1), (c_2, y_2), ..., (c_T, y_T).
            The backward function receives gradients for c_T -> g_c_T and for y_1, y_2, ... y_T -> g_y_1, g_y_2, ... g_y_T = g_ys

            The gradients of init and xs can then be computed as
            xs_bwd = (*g_ys, *carries, *xs)
            g_init, g_xs = scan(joint_graph, g_c_T, xs_bwd, dim, True)

        """

        # Collect the saved items from the forward
        joint_graph = ctx._joint_graph
        num_leaves_init = ctx._num_leaves_init
        num_leaves_xs = ctx._num_leaves_xs
        num_leaves_ys = ctx._num_leaves_ys

        carry_mask = ctx._carry_mask
        xs_mask = ctx._xs_mask
        num_additional_inputs = ctx._num_additional_inputs
        additional_inputs_mask = ctx._additional_inputs_mask
        additional_inputs_tensor_mask = ctx._additional_inputs_tensor_mask
        nonTensor_additional_inputs = ctx._nonTensor_additional_inputs

        def prepare_carries_for_bwd(init, carries):
            # Prepare the carries for the backward path.
            # This requires to concatenate the init and the carries
            return [
                torch.cat([torch.unsqueeze(i, 0), c[:-1]], dim=0)
                for i, c in zip(init, carries)
            ]

        def prepare_initial_gradients(
            flat_grads,
            additional_inputs,
            additional_inputs_tensor_mask,
            num_leaves_init,
            num_leaves_ys,
        ):
            # Decompose the flat_grads into g_c_T, g_ys
            g_c_T, g_ys, _ = ScanAutogradOp.unpack_list(
                list(flat_grads), num_leaves_init, num_leaves_ys
            )

            # The initial gradients for the additional_inputs are all zeros
            g_additional_inputs = [
                torch.zeros_like(ai) if ai_tm else None
                for ai_tm, ai in zip(additional_inputs_tensor_mask, additional_inputs)
            ]
            return g_c_T, g_ys, g_additional_inputs

        def expand_list_with_value(real_grads, mask, values=None):
            g_list = []
            real_grads_cnt = 0
            values_cnt = 0
            for m in mask:
                if m:
                    g_list.append(real_grads[real_grads_cnt])
                    real_grads_cnt += 1
                else:
                    if values is None:
                        g_list.append(None)
                    else:
                        g_list.append(values[values_cnt])
                        values_cnt += 1
            return g_list

        # Retrieve the forward inputs and the forward outputs
        flat_args = ctx.saved_tensors
        carries = flat_args[-num_leaves_init:]
        (
            init,
            xs,
            additional_inputs_tensors,
        ) = ScanAutogradOp.unpack_list(
            list(flat_args[:-num_leaves_init]), num_leaves_init, num_leaves_xs
        )

        # Expand the additional_inputs, which are tensor values only,
        # with the non-Tensor values, e.g. SymInts or Ints
        additional_inputs = expand_list_with_value(
            additional_inputs_tensors,
            additional_inputs_tensor_mask,
            values=nonTensor_additional_inputs,
        )

        with torch._C._AutoDispatchBelowAutograd():
            # Prepare the initial gradients for the backward scan
            g_c_T, g_ys, g_additional_inputs = prepare_initial_gradients(
                list(flat_grads),
                additional_inputs,
                additional_inputs_tensor_mask,
                num_leaves_init,
                num_leaves_ys,
            )
            carries = prepare_carries_for_bwd(init, carries)
            
            bwd_scan_xs = [*g_ys, *carries, *xs]

            bwd_scan_xs = [torch.flip(elem, [0]) for elem in bwd_scan_xs]

            g_outs = scan_op(
                joint_graph,
                [*g_additional_inputs, *g_c_T],
                bwd_scan_xs,
                additional_inputs,
            )

            # Unpack the computed gradients
            (
                new_g_additional_inputs,
                g_init,
                g_xs,
            ) = ScanAutogradOp.unpack_list(
                g_outs, num_additional_inputs, num_leaves_init
            )

            g_xs = [torch.flip(elem, [0]) for elem in g_xs]

        new_g_additional_inputs = mask_list(
            additional_inputs_mask,
            new_g_additional_inputs,
            [
                torch.zeros_like(el) if m else None
                for m, el in zip(additional_inputs_tensor_mask, additional_inputs)
            ],
        )

        g_xs = mask_list(xs_mask, g_xs, [torch.zeros_like(x) for x in xs])
        g_init = mask_list(carry_mask, g_init, [torch.zeros_like(el) for el in g_init])
        return *[None] * 8, *g_init, *g_xs, *new_g_additional_inputs


@scan_op.py_impl(DispatchKey.Autograd)
def scan_autograd(combine_fn, init, xs, additional_inputs):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    # TODO: Figure out how to do this in dispatcher so that we don't have to do this check here
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad if isinstance(t, torch.Tensor) else True,  # type: ignore[union-attr]
        (init, xs, additional_inputs),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return scan_op(combine_fn, init, xs, additional_inputs)

    # TODO: The create_fw_bw is always invoked twice:
    # Once in the forward path and
    # once in the backward path, where it should only be invoked for the grad grad case.
    # We don't support this currently
    if not torch.is_grad_enabled():
        # This clause is hit in the case of double backward.
        # Currently scan does not support this and thus we just dummy call another scan
        with torch._C._AutoDispatchBelowAutograd():
            return scan_op(combine_fn, init, xs, additional_inputs)

    num_leaves_init = len(init)
    num_leaves_xs = len(xs)

    (
        fw_graph,
        joint_graph,
        carry_mask,
        xs_mask,
        additional_inputs_mask,
        additional_inputs_tensor_mask,
    ) = create_fw_bw_graph_combinefn(combine_fn, init, xs, additional_inputs)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph,
        num_leaves_init,
        num_leaves_xs,
        carry_mask,
        xs_mask,
        additional_inputs_mask,
        additional_inputs_tensor_mask,
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
