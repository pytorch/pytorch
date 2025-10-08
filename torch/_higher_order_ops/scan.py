# mypy: allow-untyped-defs
import enum
import functools
import itertools
import logging
from collections.abc import Callable
from typing import Any

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.partitioner import (
    _find_hop_subgraph_outputs,
    HopGraphMinCutPartitioner,
    HopPartitionedGraph,
)
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    check_input_alias_and_mutation_return_outputs,
    check_meta_consistency,
    fill_none_with_masks,
    filter_with_masks,
    first_slice_copy,
    get_tensor_mask,
    mask_list,
    materialize_as_graph,
    reenter_make_fx,
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
from torch.utils._python_dispatch import _get_current_dispatch_mode


logger: logging.Logger = logging.getLogger(__name__)
aten = torch._ops.ops.aten


def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
):
    assert len(args) == (num_init_leaves + num_inp_leaves), (
        f"combine_fn received wrong number of arguments, expected {num_init_leaves + num_inp_leaves}, but got {len(args)}"
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
        init (torch.Tensor or pytree with tensor leaves): The initial scan carry, a tensor, or nested pytree of tensors.
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

        - The combine_fn shouldn't mutate any inputs. We'll remove the mutation restriction for inference soon. Please file an issue
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
            raise RuntimeError(f"Combine_fn must be a callable, but got {cfn}")
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
        leaves_xs.append(torch.movedim(elem, dim, 0) if dim != 0 else elem)

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

    # pyrefly: ignore  # bad-override
    def gen_schema(self, combine_fn, init, xs, additional_inputs):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        all_inputs = tuple(
            list(init) + [first_slice_copy(x) for x in xs] + list(additional_inputs)
        )

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
                "For scan, combine_fn cannot have in-place mutations but found "
                f"{mutated_inputs}-th inputs are mutated."
            )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("combine_fn", combine_gm)

        for idx, arg in enumerate(init):
            schema_gen.add_arg(f"init{idx}", arg)

        for idx, arg in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", arg)

        for idx, arg in enumerate(additional_inputs):
            schema_gen.add_arg(f"additional_input{idx}", arg)

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(combine_fn, init, xs, additional_inputs)
        return schema_gen.gen_schema()


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
    # pyrefly: ignore  # bad-override
    def forward(
        ctx,
        hop_partitioned_graph,
        n_init,
        n_xs,
        n_additional_inputs,
        *operands,
    ):
        init, xs, additional_inputs = split_into_chunks(
            operands, [n_init, n_xs, n_additional_inputs]
        )
        ctx._scan_impl = ScanAutogradImpl(
            hop_partitioned_graph, init, xs, additional_inputs
        )
        with torch._C._AutoDispatchBelowAutograd():
            return ctx._scan_impl.call_forward()

    @staticmethod
    def backward(ctx, *grad_fw_outputs):
        return (
            None,
            None,
            None,
            None,
            *ctx._scan_impl.call_backward(*grad_fw_outputs),
        )


class ScanForwardIntermediatesHandlingPolicy(enum.Enum):
    """
    Partitioner can add interemdiates to the output of original graph.
    These intermediates fall into 4 categories and we want to have different policies for handling them by
    modifying the graph:

    CLONE: we clone the intermediate when it is a carried input (i.e. init). In this case, this carry will be
        replaced with new values at each forward step so we need to clone the carry as part of return (i.e. ys)
        so as to remove the aliasing and that each step's intermediate will be stacked together and saved in bacwkard.

    REMOVE_XS: we remove the intermediate from output when it is part of xs. Since xs is read-only, in this case,
        we can directly save them for backward to use.

    REMOVE_ADDITIONAL_INPUTS: we remove the intermediate from output when it is part of additinonal_inputs. additional_inputs
        are also read-only in each step, we can directly save them for bacwkard to use. We differentiate XS and ADDITIONAL_INPUTS
        so that we could have different treatment for them in backward. In backward, we need to put xs intermediates in carry but
        put additional_inputs as backward scan's additional_inputs.

    KEEP: this corresponds to a real intermediate tensor operations' output. It varies at each forward step, we could just keep
        it as part of ys.

    """

    KEEP = 0
    CLONE = 1
    REMOVE_XS = 2
    REMOVE_ADDITIONAL_INPUTS = 3


class ScanAutogradImpl:
    """
    Wraps over partitioned graph and encapsulates scan-specific implementation details
    """

    def __init__(
        self, hop_partitioned_graph: HopPartitionedGraph, init, xs, additional_inputs
    ):
        self.hop_partitioned_graph = hop_partitioned_graph
        self.init = init
        self.xs = xs
        self.additional_inputs = additional_inputs
        self.forward_intermediates_handling_policies: list[
            ScanForwardIntermediatesHandlingPolicy
        ] = []
        self.saved_fw_xs: list[Any] = []
        self.saved_fw_additional_inputs: list[Any] = []
        self.saved_intermediates: list[Any] = []
        self.fw_spec = pytree.tree_flatten((init, xs, additional_inputs))[1]
        self._optimize_forward_intermediates()

    def _insert_clone(
        self, need_copy_node: torch.fx.Node, output_node: torch.fx.Node
    ) -> torch.fx.Node:
        graph: torch.fx.Graph = output_node.graph
        with graph.inserting_before(output_node):
            clone_node = graph.call_function(
                torch.ops.aten.clone.default,
                args=(need_copy_node,),
            )
            clone_node.meta = (
                need_copy_node.meta.copy() if hasattr(need_copy_node, "meta") else {}
            )
        return clone_node

    def _optimize_forward_intermediates(self):
        """
        We optimize the forward intermediates by categorize forward intermediates into categories
        and construct a ScanForwardIntermediatesHandlingPolicy for them

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Need remove aliasing in fw_gm:\n%s",
                self.hop_partitioned_graph.fw_gm.print_readable(print_output=False),
            )

        fw_gm = self.hop_partitioned_graph.fw_gm
        fw_all_outputs = _find_hop_subgraph_outputs(fw_gm)
        phs = list(fw_gm.graph.find_nodes(op="placeholder"))
        fw_outputs = fw_all_outputs[: self.hop_partitioned_graph.n_fw_outputs]
        fw_intermediates = fw_all_outputs[self.hop_partitioned_graph.n_fw_outputs :]

        init_phs, xs_phs, additional_inputs_phs = pytree.tree_unflatten(
            phs, self.fw_spec
        )
        init_node_set, xs_node_set, addi_node_set = (
            set(init_phs),
            set(xs_phs),
            set(additional_inputs_phs),
        )

        assert len(self.forward_intermediates_handling_policies) == 0
        assert len(self.saved_fw_xs) == 0
        assert len(self.saved_fw_additional_inputs) == 0
        intermediate_idx_to_ph_idx = {}
        ph_idx = {ph: i for i, ph in enumerate(phs)}
        for i, out in enumerate(fw_intermediates):
            if out in init_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.CLONE
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            elif out in xs_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.REMOVE_XS
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            elif out in addi_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            else:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.KEEP
                )

        new_output_node = []
        real_graph_inputs = (
            list(self.init) + list(self.xs) + list(self.additional_inputs)
        )
        fw_output_node = next(iter(fw_gm.graph.find_nodes(op="output")))
        for intermediate_idx, (node, policy) in enumerate(
            zip(fw_intermediates, self.forward_intermediates_handling_policies)
        ):
            if policy == ScanForwardIntermediatesHandlingPolicy.CLONE:
                new_output_node.append(self._insert_clone(node, fw_output_node))
            elif policy == ScanForwardIntermediatesHandlingPolicy.REMOVE_XS:
                assert intermediate_idx in intermediate_idx_to_ph_idx
                inp_idx = intermediate_idx_to_ph_idx[intermediate_idx]
                self.saved_fw_xs.append(real_graph_inputs[inp_idx])
            elif (
                policy
                == ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
            ):
                assert intermediate_idx in intermediate_idx_to_ph_idx
                inp_idx = intermediate_idx_to_ph_idx[intermediate_idx]
                self.saved_fw_additional_inputs.append(real_graph_inputs[inp_idx])
            else:
                new_output_node.append(node)

        fw_output_node.args = (tuple(fw_outputs) + tuple(new_output_node),)
        fw_gm.graph.lint()
        fw_gm.recompile()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "after removing aliasing:\n%s", fw_gm.print_readable(print_output=False)
            )

    def call_forward(self):
        fw_outputs_and_intermediates: tuple[Any] = scan_op(
            self.hop_partitioned_graph.fw_gm, self.init, self.xs, self.additional_inputs
        )  # type: ignore[return-type]
        fw_outs = fw_outputs_and_intermediates[
            : self.hop_partitioned_graph.n_fw_outputs
        ]
        saved_intermediates = fw_outputs_and_intermediates[
            self.hop_partitioned_graph.n_fw_outputs :
        ]
        assert len(self.saved_intermediates) == 0
        self.saved_intermediates.extend(saved_intermediates)
        return tuple(fw_outs)

    def call_backward(self, *grad_fw_outputs):
        """
        Recall that fw_outputs = (*carry, *ys), bw_gm takes in (*fw_intermediates, *grad_carry, *grad_ys)
        and returns (*grad_init, *grad_xs, *grad_additional_inputs)
        The bacwkard is a reversed scan that can be constructed as follows:

          grad_additonal_inputs = torch.zeros_like(additional_inputs)
          bw_init = (grad_carry, grad_additional_inputs)
          bw_xs = (fw_intermediates, grad_ys)
          grad_init, grad_additional_inputs, grad_xs = scan(
            combine_fn,
            bw_init,
            bw_xs,
            reverse = True
          )
          , where combine_fn is defined as follows:

           def combine_fn(bw_init, bw_xs):
             grad_carry, grad_additional_inputs = bw_init
             fw_intermediates, grad_y = bw_xs
             nxt_grad_carry, grad_x, nxt_grad_additional_inputs = bw_gm(*fw_intermediates, *grad_carry, *grad_y)
             return (nxt_grad_carry, grad_additional_inputs + nxt_grad_additional_inputs), grad_x

          Note that grad_additional_inputs is accumulated with add, grad_carry is carried over to next iteration and
          grad_x is the ys output, which will be stacked together after the loop and will have the same shape as xs.
        """
        fw_policy = self.forward_intermediates_handling_policies
        saved_intermediates = self.saved_intermediates
        saved_fw_xs = self.saved_fw_xs
        saved_fw_additional_inputs = self.saved_fw_additional_inputs

        n_carry = len(self.init)

        grad_carry, grad_ys = grad_fw_outputs[:n_carry], grad_fw_outputs[n_carry:]
        additional_inputs_tensor_masks = [
            bool(isinstance(t, torch.Tensor)) for t in self.additional_inputs
        ]
        grad_additional_inputs = [
            torch.zeros_like(t)
            for t in filter_with_masks(
                self.additional_inputs, additional_inputs_tensor_masks
            )
        ]

        bw_init = [grad_carry, grad_additional_inputs]
        bw_xs = [
            grad_ys,
            saved_fw_xs,
            saved_intermediates,
        ]
        bw_additional_inputs = saved_fw_additional_inputs

        _, flat_spec = pytree.tree_flatten((bw_init, bw_xs, bw_additional_inputs))

        grad_spec = None

        def bw_single_step_wrapper(*args):
            bw_init, bw_xs, bw_additional_inputs = pytree.tree_unflatten(
                args, flat_spec
            )
            grad_carry, grad_additional_inputs = bw_init
            grad_y, saved_fw_xs, saved_intermediates = bw_xs
            saved_fw_additional_inputs = bw_additional_inputs

            fw_intermediates = []
            xs_it = iter(saved_fw_xs)
            carry_it = iter(saved_intermediates)
            addi_it = iter(saved_fw_additional_inputs)
            for policy in fw_policy:
                if policy in (
                    ScanForwardIntermediatesHandlingPolicy.CLONE,
                    ScanForwardIntermediatesHandlingPolicy.KEEP,
                ):
                    fw_intermediates.append(next(carry_it))
                elif policy == ScanForwardIntermediatesHandlingPolicy.REMOVE_XS:
                    fw_intermediates.append(next(xs_it))
                elif (
                    policy
                    == ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
                ):
                    fw_intermediates.append(next(addi_it))
                else:
                    raise RuntimeError(f"Unknown policy: {policy}")

            grad_fw_outputs = (*grad_carry, *grad_y)

            flat_out = self.hop_partitioned_graph.bw_gm(
                *fw_intermediates,
                *grad_fw_outputs,
            )

            next_grad_carry, grad_xs, grad_addi = split_into_chunks(
                flat_out,  # type: ignore[arg-type]
                [len(self.init), len(self.xs), len(self.additional_inputs)],
            )

            nonlocal grad_spec
            flat_grads, grad_spec = pytree.tree_flatten(
                (
                    next_grad_carry,
                    [
                        prev + cur
                        for prev, cur in zip(
                            grad_additional_inputs,
                            filter_with_masks(
                                grad_addi, additional_inputs_tensor_masks
                            ),
                        )
                    ],
                    grad_xs,
                )
            )
            return flat_grads

        single_step_bw_xs = pytree.tree_map(lambda t: t[0], bw_xs)
        bw_single_step_gm = materialize_as_graph(
            bw_single_step_wrapper,
            tuple(
                pytree.tree_flatten((bw_init, single_step_bw_xs, bw_additional_inputs))[
                    0
                ]
            ),
        )

        flat_grads = scan_op(
            bw_single_step_gm,
            pytree.tree_flatten(bw_init)[0],
            # TODO: torch.flip copies the tensor, we should optimize it away
            [torch.flip(x, (0,)) for x in pytree.tree_flatten(bw_xs)[0]],
            pytree.tree_flatten(bw_additional_inputs)[0],
        )
        assert grad_spec is not None
        grad_init, grad_additional_inputs, grad_xs = pytree.tree_unflatten(
            flat_grads, grad_spec
        )
        return (
            *grad_init,
            *[torch.flip(elem, (0,)) for elem in grad_xs],
            *fill_none_with_masks(
                grad_additional_inputs, additional_inputs_tensor_masks
            ),
        )


@scan_op.py_autograd_impl
def scan_autograd(combine_fn, init, xs, additional_inputs):
    with disable_proxy_modes_tracing():
        hop_partitioned_graph: HopPartitionedGraph = (
            HopGraphMinCutPartitioner.create_partitioned_graph(
                combine_fn,
                (*init, *[x[0] for x in xs], *additional_inputs),
                always_recompute_complex_exprs=True,
            )
        )

    return ScanAutogradOp.apply(
        hop_partitioned_graph,
        len(init),
        len(xs),
        len(additional_inputs),
        *init,
        *xs,
        *additional_inputs,
    )


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
