# mypy: allow-untyped-defs
import functools
import itertools
from typing import Any, Callable, List, Tuple, Optional

import torch
import torch._dynamo.variables
import torch._prims_common as utils
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dispatch.python import suspend_functionalization
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_alias,
    _has_potential_branch_input_mutation,
    _set_compilation_env,
    reenter_make_fx,
    unique_graph_id,
    UnsupportedAliasMutationException,
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

# Helper functions that are also used from other places
def first_slice_copy(t: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.select_copy(t, dim, 0)

def expand_tensor(t: torch.Tensor, dim: int, scan_length: int, memory_format: Optional[torch.memory_format] = None):
    if isinstance(t, torch.Tensor):
        return t.unsqueeze(dim).repeat(*([1] * dim + [scan_length] + [1] * (t.ndim - dim))).clone(memory_format=memory_format)
    return t

def _extract_carry_and_out(flat_out: List[Any], num_carry: int):
    return flat_out[:num_carry], flat_out[num_carry:]

# Internal functions for scan.py
def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
):
    assert len(args) == (num_init_leaves + num_inp_leaves)
    carry = pytree.tree_unflatten(args[:num_init_leaves], spec_init)
    xs = pytree.tree_unflatten(args[num_init_leaves:], spec_xs)
    carry, combined = combine_fn(carry, xs)
    carry_flat = pytree.tree_leaves(carry)
    combined_flat = pytree.tree_leaves(combined)
    assert num_init_leaves == len(carry_flat)
    return [*carry_flat, *combined_flat]

def create_fw_bw_graph_combinefn(combine_fn, init, input, dim):
    # See Note [HOP create fw_bw graph] in create_fw_bw_graph in utils.py

    # Helper wrapper for the autograd forward.
    # This wrapper ensures that the forward returns all carries
    # instead of only the last one
    # The gradients of the carries forwarded to the output are 
    # detached in order not to raise problems with the function aliasing outputs
    def wrapper_combine_fn(*args):
        new_carry, y = _extract_carry_and_out(combine_fn(*args), len(init))
        return [*new_carry, *[n_c.clone().detach() for n_c in new_carry], *y]

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            num_init = len(init)

            fw_init = [pytree.tree_map(_from_fun, x) for x in init]
            fw_input = [pytree.tree_map(_from_fun, x).select(dim, 0) for x in input]

            carry, outs = _extract_carry_and_out(
                wrapper_combine_fn(*fw_init, *fw_input), num_init
            )
            fw_carry, fw_outputs = [pytree.tree_map(_from_fun, c) for c in carry], [
                pytree.tree_map(_from_fun, o) for o in outs
            ]
            if any(carry.shape != ini.shape for carry, ini in zip(fw_carry, init)):
                raise RuntimeError(
                    "Expect carry produced by combine_fn to only contains tensors. "
                    f"Got types {[type(carry) for carry in fw_carry]}."
                )
            if any(not isinstance(carry, torch.Tensor) for carry in fw_carry):
                raise RuntimeError(
                    "Expect carry produced by combine_fn to only contains tensors. "
                    f"Got types {[type(carry) for carry in fw_carry]}."
                )
            if any(not isinstance(out, torch.Tensor) for out in fw_outputs):
                raise RuntimeError(
                    "Expect outputs produced by combine_fn to only contains tensors. "
                    f"Got types {[type(out) for out in fw_outputs]}."
                )

            # TODO: There is a major issue that the create_fw_bw in the higher_order_op is invoked twice:
            # Once in the forward path (as it should) and once in the backward path, where it shouldn't be called
            # If we can get rid of the second invokation, it would simplify this function
            
            # The forward graph needs to be constructed from a different combine_fn than the joint_graph
            fw_graph = _maybe_reenter_make_fx(wrapper_combine_fn)(*fw_init, *fw_input)
           
            _, joint_graph = create_fw_bw_graph(
                combine_fn,
                False,
                (*fw_init, *fw_input),
                (*fw_carry, *fw_outputs[num_init:]),
            )

        return fw_graph, joint_graph


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], Tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> Tuple[pytree.PyTree, pytree.PyTree]:
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
            each tensor leaf is a stacked output along dim, where each slice is the output of a scan iteration.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            return next_carry, y

        i0 = torch.zeros(1)
        xs = torch.arange(1, 5)
        # returns torch.tensor([10]), torch.tensor([1., 3., 6., 10.])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    if not callable(combine_fn):
        raise RuntimeError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise RuntimeError("Dim must be an int, but got " + str(type(dim)))
    if not isinstance(reverse, bool):
        raise RuntimeError("Reverse must be a bool, but got " + str(type(reverse)))

    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

    if not torch._dynamo.is_compiling():
        with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
            return torch.compile(scan, backend="eager", fullgraph=True)(
                combine_fn, init, xs, dim=dim, reverse=reverse
            )

    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs, spec_xs = pytree.tree_flatten(xs)

    if len(leaves_init) == 0:
        raise RuntimeError("Init tensors must be provided")
    if any(not isinstance(x, torch.Tensor) for x in leaves_init):
        raise RuntimeError("All init leaves must be a Tensor")
    if any(not isinstance(x, torch.Tensor) for x in leaves_xs):
        raise RuntimeError("All xs leaves must be a Tensor")
    if any(x.shape[dim] == 0 for x in leaves_xs):
        raise RuntimeError("All xs leaves must have a scan dimension > 0")

    if len(leaves_xs) > 0:
        shape = leaves_xs[0].shape
        ndim = len(shape)
        dim = utils.canonicalize_dim(ndim, dim)

        out = combine_fn(
            pytree.tree_unflatten(leaves_init, spec_init),
            pytree.tree_unflatten([elem.select(dim, 0) for elem in leaves_xs], spec_xs),
        )

        # The first output needs to have the same pytree as init
        carry_leaves = pytree.tree_leaves(out[0])
        if len(carry_leaves) != len(leaves_init):
            raise RuntimeError(
                "The number of leaves of the pytree of the new carry produced by the operator\
 needs to match the length of the pytree of the init"
            )
        if any(
            in_l.shape != out_l.shape for in_l, out_l in zip(leaves_init, carry_leaves)
        ):
            raise RuntimeError(
                "The pytree of the new carry produced by the operator needs to match the pytree of the init"
            )

        # There are no pytree restrictions on the second output of the operator
        _, tree_out = pytree.tree_flatten(out[1])

        combine_fn = functools.partial(
            wrap_combine_fn_flat,
            combine_fn=combine_fn,
            spec_init=spec_init,
            spec_xs=spec_xs,
            num_init_leaves=len(leaves_init),
            num_inp_leaves=len(leaves_xs),
        )

        result_carry, result_flat = _extract_carry_and_out(
            scan_op(
                combine_fn, leaves_init, leaves_xs, dim, reverse, additional_inputs=[]
            ),
            len(leaves_init),
        )

        return pytree.tree_unflatten(result_carry, spec_init), pytree.tree_unflatten(
            result_flat, tree_out
        )

    else:
        return pytree.tree_unflatten(leaves_init, spec_init), xs


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(self, combine_fn, init, xs, dim, reverse, additional_inputs):
        return super().__call__(combine_fn, init, xs, dim, reverse, additional_inputs)


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, reverse=False, additional_inputs=None):
    additional_inputs = additional_inputs if additional_inputs is not None else []

    def _scan(init, xs):
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(xs) == 0:
            return carry, []

        num_elems = xs[0].shape[dim]
        if reverse:
            ind = num_elems - 1
        else:
            ind = 0

        # Compute dummy shapes for the pre-allocation
        num_init_leaves = len(init)
        dummy_carry, dummy_out = _extract_carry_and_out(
            operator(
                *carry,
                *[first_slice_copy(elem, dim) for elem in xs],
                *additional_inputs,
            ),
            num_init_leaves,
        )

        # Pre-alocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        # out: (M, N, num_elems, ...)
        # idx: (M, N, 1, ...)
        outs, idxs = zip(
            *[
                [
                    torch.zeros(
                        list(e.size())[:dim] + [num_elems] + list(e.size())[dim:],
                        dtype=e.dtype,
                        device=e.device,
                    ),
                    torch.ones_like(e, dtype=torch.int64).unsqueeze(dim),
                ]
                for i, e in enumerate(dummy_out)
            ]
        )

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                # o: (M, N, num_elems, ...)
                # x: (M, N, ...) -> (M, N, 1, ...)
                # ind * idx: (M, N, 1, ...) with values to be ind
                o.scatter_(dim, ind * idx, x.unsqueeze(dim))

        for i in range(num_elems):
            ind = i if not reverse else num_elems - i - 1
            carry, out = _extract_carry_and_out(
                operator(
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


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: List[torch.Tensor],
    xs: List[torch.Tensor],
    dim: int,
    reverse: bool,
    additional_inputs: List[torch.Tensor],
):
    with disable_proxy_modes_tracing():
        sample_inits = [x_init.clone() for x_init in init]
        sample_inputs = [first_slice_copy(x, dim) for x in xs]
        sample_additional_inputs = [x.clone() for x in additional_inputs]
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

    for ini, ca in zip(init, carry):
        ini_meta = ini
        carry_meta = ca.meta["tensor_meta"]
        carry_val = ca.meta["val"]
        if (
            carry_val.device != ini_meta.device
            or carry_meta.dtype != ini_meta.dtype
            or carry_meta.shape != ini_meta.shape
        ):
            raise RuntimeError(
                f"Expected metadata of the combine_fn result {carry_meta} to be the same as "
                + f"the metadata of init with {ini_meta}"
            )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, init, xs, dim, reverse, additional_inputs)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="scan"
    )

    with disable_proxy_modes_tracing():
        scan_length = xs[0].shape[dim]
        expanded_outs = [
            pytree.tree_map(expand_tensor, t.meta["val"], dim, scan_length)
            for t in output
        ]
        out = [*init, *expanded_outs]

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, dim, reverse, additional_inputs):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return generic_scan(combine_fn, init, xs, dim, reverse, additional_inputs)


class ScanAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fw_graph,
        joint_graph,
        dim,
        reverse,
        num_leaves_init,
        additional_inputs,
        *ops,
    ):
        init = ops[:num_leaves_init]
        xs = ops[num_leaves_init:]

        ctx._joint_graph = joint_graph
        ctx._dim = dim
        ctx._reverse = reverse
        ctx._num_leaves_init = num_leaves_init
        ctx._num_leaves_xs = len(xs)

        with torch._C._AutoDispatchBelowAutograd():
            carry, carries_outs = _extract_carry_and_out(
                scan_op(fw_graph, init, xs, dim, reverse, additional_inputs),
                num_leaves_init,
            )

            # Collect the carries for each time step from the outs
            carries = carries_outs[:num_leaves_init]
            outs = carries_outs[num_leaves_init:]

            ctx.save_for_backward(
                *(init + xs + tuple(carries) + tuple(additional_inputs))
            )
            return (*carry, *outs)

    @staticmethod
    def backward(ctx, *flat_grads):
        r"""
        This function computes the gradients of the scan operation.
        It does so by constructing using an additional scan operator with the gradients
        
        Args:
            flat_grads (torch.Tensor): The tensor of upstream gradients, or a nested pytree of tensors.

        Example::

            The ``fw_graph`` f(.,.), used in the forward function, is the operator used during the scan. For example
            def f(x: torch.Tensor, y: torch.Tensor):
                next_carry = y = x + y
                return next_carry, y

            The ``joint_graph`` g(.,.), used in the backward function, is the gradient of the function f(.,.).
            It computes the gradients for x and y of f. For example for the function f above
            def g(x: torch.Tensor, y: torch.Tensor):
                return 1., 1.
                
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

        joint_graph = ctx._joint_graph
        dim = ctx._dim
        num_leaves_init = ctx._num_leaves_init
        num_leaves_xs = ctx._num_leaves_xs
        reverse = ctx._reverse

        # Retrieve the forward inputs and the forward outputs
        operands_outs = ctx.saved_tensors
        init = operands_outs[:num_leaves_init]
        xs = operands_outs[num_leaves_init : num_leaves_init + num_leaves_xs]
        carries = operands_outs[
            num_leaves_init + num_leaves_xs : 2 * num_leaves_init + num_leaves_xs
        ]
        additional_inputs = operands_outs[2 * num_leaves_init + num_leaves_xs :]

        with torch._C._AutoDispatchBelowAutograd():
            g_c_T = flat_grads[:num_leaves_init]
            g_ys = flat_grads[num_leaves_init:]

            if reverse:
                carries_augmented = [
                    torch.cat(
                        [torch.unsqueeze(i, dim), torch.flip(c[1:], [dim])], dim=dim
                    )
                    for i, c in zip(init, carries)
                ]
                xs = [torch.flip(x, [dim]) for x in xs]
            else:
                carries_augmented = [
                    torch.cat([torch.unsqueeze(i, dim), c[:-1]], dim=dim)
                    for i, c in zip(init, carries)
                ]

            xs_bwd = (*g_ys, *carries_augmented, *xs)
            g_init, g_xs = _extract_carry_and_out(
                scan_op(joint_graph, g_c_T, xs_bwd, dim, True, additional_inputs),
                num_leaves_init,
            )

            if reverse:
                g_xs = [torch.flip(g, [dim]) for g in g_xs]

        return None, None, None, None, None, None, *g_init, *g_xs


@scan_op.py_impl(DispatchKey.Autograd)
def scan_autograd(combine_fn, init, input, dim, reverse, additional_inputs):
    # A shortcut for the case where all inputs don't require gradient,
    # we skip tracing the forward and backward graph.
    if pytree.tree_all_only(
        torch.Tensor,
        lambda t: not t.requires_grad,  # type: ignore[union-attr]
        (init, input),
    ):
        with torch._C._AutoDispatchBelowAutograd():
            return scan_op(combine_fn, init, input, dim, reverse, additional_inputs)

    num_leaves_init = len(init)

    (
        fw_graph,
        joint_graph,
    ) = create_fw_bw_graph_combinefn(combine_fn, init, input, dim)

    flat_out = ScanAutogradOp.apply(
        fw_graph,
        joint_graph,
        dim,
        reverse,
        num_leaves_init,
        additional_inputs,
        *(init + input),
    )
    return *flat_out[:num_leaves_init], *flat_out[num_leaves_init:]


@scan_op.py_impl(ProxyTorchDispatchMode)
def scan_proxy_mode(mode, combine_fn, init, xs, dim, reverse, additional_inputs):
    return trace_scan(
        mode, scan_op, combine_fn, init, xs, dim, reverse, additional_inputs
    )


@scan_op.py_impl(FakeTensorMode)
def scan_fake_tensor_mode(mode, combine_fn, init, xs, dim, reverse, additional_inputs):
    with mode:
        scan_length = xs[0].shape[dim]
        carry, outputs = _extract_carry_and_out(
            combine_fn(
                *init,
                *[torch.select_copy(inp, dim, 0) for inp in xs],
                *additional_inputs,
            ),
            len(init),
        )
        out = (
            *carry,
            *tuple(
                expand_tensor(t, dim, scan_length)
                for t in outputs
            ),
        )
        return out


@scan_op.py_functionalize_impl
def scan_functionalize(ctx, combine_fn, init, xs, dim, reverse, additional_inputs):
    unwrapped_xs = ctx.unwrap_tensors(xs)
    unwrapped_init = ctx.unwrap_tensors(init)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    with ctx.redispatch_to_next() as m:
        functional_combine_fn = ctx.functionalize(combine_fn)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        sample_unwrapped_xs_sliced = [
            torch.select_copy(inp, dim, 0) for inp in unwrapped_xs
        ]
        sample_inputs = list(
            itertools.chain(
                unwrapped_init, sample_unwrapped_xs_sliced, unwrapped_additional_inputs
            )
        )
        if _has_potential_branch_input_mutation(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be modifying the input!"
            )
        if _has_potential_branch_input_alias(
            functional_combine_fn, sample_inputs, pre_dispatch=pre_dispatch
        ):
            raise UnsupportedAliasMutationException(
                "Combine_fn might be aliasing the input!"
            )
        ret = scan_op(
            functional_combine_fn,
            unwrapped_init,
            unwrapped_xs,
            dim,
            reverse,
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
        torch.stack([e[leave_ind] for e in op(result_flat)], dim)
        for leave_ind in range(num_leaves)
    ]
    return (
        pytree.tree_unflatten(carry, carry_spec),
        pytree.tree_unflatten(results, dummy_out_spec),
    )
