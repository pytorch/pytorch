# mypy: allow-untyped-defs
import enum
import functools
import itertools
from re import S
from typing import Any, Callable, Optional, Sequence, Union

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    check_input_alias_and_mutation_return_outputs,
    check_meta_consistency,
    create_bw_fn,
    fill_none_with_masks,
    filter_with_masks,
    first_slice_copy,
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
from torch.utils._python_dispatch import _get_current_dispatch_mode


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
    )  # type: ignore

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

def _partition(li, fn):
    true_list = []
    true_pos = []
    false_list = []
    false_pos =[]
    for i, val in enumerate(li):
        pred: bool = fn(val)
        if pred:
            true_pos.append(i)
            true_list.append(val)
        else:
            false_pos.append(i)
            false_list.append(val)
    return true_list, true_pos, false_list, false_pos

def _merge(true_list, true_pos, false_list, false_pos):
    assert len(true_list)  == len(true_pos) and len(false_list) == len(false_pos)
    if len(true_pos) == 0:
        return false_list

    if len(false_pos) == 0:
        return true_list

    l = max(true_pos[-1], false_pos[-1]) + 1
    res = [None] * l
    for pos, val in zip(true_pos, true_list):
        res[pos] = val
    for pos, val in zip(false_pos, false_list):
        res[pos] = val
    assert all(val is not None for val in res)
    return res


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
            hop_partitioned_graph,
            init,
            xs,
            additional_inputs
        )
        with torch._C._AutoDispatchBelowAutograd():
            return ctx._scan_impl.call_forward(ctx, init, xs, additional_inputs)

    @staticmethod
    def backward(ctx, *grad_fw_outputs):
        return None, None, None, None, *ctx._scan_impl.call_backward(ctx, *grad_fw_outputs)


def _find_hop_subgraph_outputs(gm: torch.fx.GraphModule) -> tuple[torch.fx.Node]:
    output_node_args = gm.graph.find_nodes(op="output")[0].args
    assert isinstance(output_node_args, tuple)
    return output_node_args[0]


class HopPartitionedGraph:
    def __init__(self,
        fw_gm: torch.fx.GraphModule,
        bw_gm: torch.fx.GraphModule,
        n_fw_outputs: int,
        n_checkpoints: int,
    ):
        self.fw_gm = fw_gm
        self.bw_gm = bw_gm
        self.n_fw_outputs = n_fw_outputs
        self.n_checkpoints =  n_checkpoints
        self._reorder_fw_output()

    def _reorder_fw_output(self):
        '''
        Before the pass, fw_gm returns (*fw_outputs, *intermediates1)
        and bw_gm takes (*intermediates2, *grad_fw_outputs) as input.
        intermediates1 and intermediates2 share the same node names but
        they might be in different order. E.g. this could happen if there
        are inputs that contain symints.

        To simplify downstream processing, this graph pass normalizes the output of fw_gm
        to be consistent with the bacwkard inputs:

        fw_gm:
          - input: fw_args
          - output: (*fw_outputs, *intermediates)

        bw_gm:
          - input: (*intermediates, *grad_fw_outputs)
          - output: grad_fw_args

        Exmaple:

        def fw_gm(x, y, z):
           a, b, c = f(x), g(y), k(z)
           return a, b, c, f_tmp, g_tmp, k_tmp

        , where a, b, c are fw_outputs, f_tmp, g_tmp, k_tmp are intermediates

        The corresponding bw_gm has the following signature:

        def bw_gm(f_tmp, g_tmp, k_tmp, grad_a, grad_b, grac):
          return grad_x, grad_y, grad_z
        '''
        # Initially the intermediates are the latter part of the fw outputs
        fw_gm_output_nodes = _find_hop_subgraph_outputs(self.fw_gm)
        fw_outputs_nodes = fw_gm_output_nodes[:self.n_fw_outputs]
        fw_intermediates_nodes = fw_gm_output_nodes[self.n_fw_outputs:]
        n_intermediates = len(fw_intermediates_nodes)
        if n_intermediates > 0:
            fw_intermediates_name_to_node = {n.name: n for n in fw_intermediates_nodes}

            # First n_intermediates placeholders
            bw_names: list[str] = [ph.name for ph in list(self.bw_gm.graph.find_nodes(op="placeholder"))[:n_intermediates]]
            new_fw_outputs = list(fw_outputs_nodes) + [fw_intermediates_name_to_node[name] for name in bw_names]

            output_node = self.fw_gm.graph.find_nodes(op="output")[0]
            output_node.args = (tuple(new_fw_outputs),)

            self.fw_gm.graph.lint()
            self.fw_gm.recompile()


class HopJointGraph:
    def __init__(self, joint_gm: torch.fx.GraphModule, n_primals: int, n_fw_outputs: int):
        self.joint_gm = joint_gm
        self.n_primals = n_primals
        self.n_fw_outputs= n_fw_outputs
        self._rename_phs()
        self._remove_redundant_sym_size_ops()
        self._mark_complex_exprs_as_must_recompute()

    def _rename_phs(self) -> None:
        self.n_tangents = 0
        for i, ph in enumerate(self.joint_gm.graph.find_nodes(op="placeholder")):
            if i < self.n_primals:
                ph.target = f"primals_{i}"
                ph.name = f"primals_{i}"
            else:
                self.n_tangents += 1
                ph.target = f"tangents_{i - self.n_primals}"
                ph.name = f"tangents_{i - self.n_primals}"

        self.joint_gm.graph.lint()
        self.joint_gm.compile()

    def _remove_redundant_sym_size_ops(self) -> None:
        """
        Graph pass that deletes torch.ops.sym_size.int operators whose output is a
        corresponding placeholder that holds the same symbol, and replace all uses
        of their output to be directly accessing the placeholders.
        """
        # Find all placeholders and their symbolic values
        placeholder_exprs = {}
        for node in self.joint_gm.graph.nodes:
            if isinstance(node, torch.fx.Node) and node.op == "placeholder" and hasattr(node, 'meta') and 'val' in node.meta:
                val = node.meta['val']
                if isinstance(val, torch.SymInt):
                    placeholder_exprs[val.node.expr] = node

        # Find sym_size nodes to remove
        nodes_to_remove = []
        # TODO: change to find_nodes
        for node in self.joint_gm.graph.find_nodes(op="call_function", target=torch.ops.aten.sym_size.int):
            assert hasattr(node, 'meta') and 'val' in node.meta, node
            # Check if we have a placeholder with the same symbol
            val = node.meta['val']
            expr = val.node.expr
            if expr in placeholder_exprs:
                placeholder_node = placeholder_exprs[expr]
                # Replace all uses of sym_size node with placeholder
                node.replace_all_uses_with(placeholder_node)
                nodes_to_remove.append(node)

        # Remove the redundant sym_size nodes
        for node in nodes_to_remove:
            self.joint_gm.graph.erase_node(node)

        # Clean up and recompile
        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def _mark_complex_exprs_as_must_recompute(self) -> None:
        def is_complex_expr(expr):
           return not expr.is_symbol

        for n in (node for node in self.joint_gm.graph.nodes if node.op =="call_function"):
            if not "val" in n.meta:
                continue
            val = n.meta["val"]
            if isinstance(val, torch.SymInt) and is_complex_expr(val.node.expr):
                assert n.meta.get("recompute", None) is None
                from torch._functorch.partitioners import CheckpointPolicy
                n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

        # Clean up and recompile
        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def partition(self, partition_fn: Callable) -> HopPartitionedGraph:

        print("before min_cut_partition:")
        self.joint_gm.print_readable()

        fw_gm, bw_gm = partition_fn(
            self.joint_gm,
            None,
            num_fwd_outputs=self.n_fw_outputs
        )

        print("after partition_fn:")
        print("fw:")
        fw_gm.print_readable()
        print("bw:")
        bw_gm.print_readable()

        n_checkpoints = len(_find_hop_subgraph_outputs(fw_gm)) - self.n_fw_outputs

        return HopPartitionedGraph(
            fw_gm,
            bw_gm,
            self.n_fw_outputs,
            n_checkpoints,
        )

class HopGraphPartitioner:
    @staticmethod
    def _create_joint_graph(fw_fn: Callable, fw_args: tuple[Union[torch.Tensor, torch.SymInt], ...]) -> HopJointGraph:
        fw_gm = materialize_as_graph(fw_fn, fw_args, force_enable_grad=True)
        fw_gm_output_nodes =  _find_hop_subgraph_outputs(fw_gm)

        assert all(isinstance(n, torch.fx.Node) and "val" in n.meta for n in fw_gm_output_nodes)
        fw_gm_output_vals = tuple(n.meta["val"] for n in fw_gm_output_nodes)  # type: ignore[arg-type]

        assert all(isinstance(val, torch.Tensor) for val in fw_gm_output_vals)
        example_grads = tuple(torch.zeros_like(val) for val in fw_gm_output_vals)

        joint_fn = create_bw_fn(fw_fn, fw_args, return_fw_outputs=True)
        # Need to first trace out the joint_fn with autograd info on
        # then functionalize the graph othewise the grad information is lost for functional tensor
        joint_gm = materialize_as_graph(joint_fn, fw_args + example_grads, force_enable_grad=True)
        joint_gm_functionalized = materialize_as_graph(torch.func.functionalize(joint_gm, remove="mutations_and_views"), fw_args + example_grads)

        return HopJointGraph(
            joint_gm_functionalized,
            len(fw_args),
            len(fw_gm_output_nodes),
        )

    @staticmethod
    def create(fw_fn: Callable, fw_args: tuple[Union[torch.Tensor, torch.SymInt], ...]) -> "HopPartitionedGraph":
        from torch._functorch.partitioners import min_cut_rematerialization_partition

        joint_graph: HopJointGraph = HopGraphPartitioner._create_joint_graph(fw_fn, fw_args)
        return joint_graph.partition(min_cut_rematerialization_partition)

class ScanOutputAliasPolicy(enum.Enum):
    """
    The partitioner can create aliasing between
    Enum for specifying the policy for handling the alias created by partitioner.

    1. if the output
    1. the output is the carry, in this case, we need to clone the carry and put the cloned
        result as part of return so that we can get a checkpoint in forward.
    2. the output is xs or
    """
    NONE = 0
    CLONE = 1
    REMOVE_XS = 2
    REMOVE_ADDITIONAL_INPUTS = 3


class ScanAutogradImpl:
    '''
    Wraps over paritioned graph and encapsulates scan-specific implementation details
    '''

    def __init__(self, hop_partitioned_graph: HopPartitionedGraph, init, xs, additional_inputs):
        self.hop_partitioned_graph = hop_partitioned_graph
        self.init = init
        self.xs = xs
        self.additional_inputs = additional_inputs
        self.output_policies = []
        self.xs_checkpoints = {}
        self.additional_inputs_checkpoints = {}
        self.fw_spec = pytree.tree_flatten((init, xs, additional_inputs))[1]
        self._remove_fw_gm_output_aliasing()

    def _insert_clone(self, need_copy_node: torch.fx.Node, output_node: torch.fx.Node) -> torch.fx.Node:
        graph: torch.fx.Graph = output_node.graph
        with graph.inserting_before(output_node):
            clone_node = graph.call_function(
                torch.ops.aten.clone.default,
                args=(need_copy_node,),
            )
            clone_node.meta = need_copy_node.meta.copy() if hasattr(need_copy_node, 'meta') else {}
        return clone_node


    def _remove_fw_gm_output_aliasing(self):
        from torch._higher_order_ops.utils import check_input_alias_and_mutation_return_outputs

        fw_gm = self.hop_partitioned_graph.fw_gm
        print("Need remove aliasing in fw_gm")
        fw_gm.print_readable()

        fw_all_outputs = _find_hop_subgraph_outputs(fw_gm)
        phs = list(fw_gm.graph.find_nodes(op="placeholder"))
        fw_outputs = fw_all_outputs[:self.hop_partitioned_graph.n_fw_outputs]
        fw_checkpoints = fw_all_outputs[self.hop_partitioned_graph.n_fw_outputs:]

        # create a handling policy for each checkpoints
        init_phs, xs_phs, additional_inputs_phs = pytree.tree_unflatten(phs, self.fw_spec)
        init_node_set, xs_node_set, addi_node_set = set(init_phs), set(xs_phs), set(additional_inputs_phs)

        assert len(self.output_policies) == 0
        assert len(self.xs_checkpoints) == 0
        assert len(self.additional_inputs_checkpoints) == 0
        reverse_alias_map = {}
        ph_idx = {ph: i for i, ph in enumerate(phs)}
        for i, out in enumerate(fw_checkpoints):
            if out in init_node_set:
                self.output_policies.append(ScanOutputAliasPolicy.CLONE)
                reverse_alias_map[i] = ph_idx[out]
            elif out in xs_node_set:
                self.output_policies.append(ScanOutputAliasPolicy.REMOVE_XS)
                reverse_alias_map[i] = ph_idx[out]
            elif out in addi_node_set:
                self.output_policies.append(ScanOutputAliasPolicy.REMOVE_ADDITIONAL_INPUTS)
                reverse_alias_map[i] = ph_idx[out]
            else:
               self.output_policies.append(ScanOutputAliasPolicy.NONE)


        new_output_node = []
        real_graph_inputs = list(self.init) + list(self.xs) + list(self.additional_inputs)
        fw_output_node = list(fw_gm.graph.find_nodes(op="output"))[0]
        for out_idx, (node, policy) in enumerate(zip(fw_checkpoints, self.output_policies)):
            if policy == ScanOutputAliasPolicy.CLONE:
                new_output_node.append(self._insert_clone(node, fw_output_node))
            elif policy == ScanOutputAliasPolicy.REMOVE_XS:
                assert out_idx in reverse_alias_map
                inp_idx = reverse_alias_map[out_idx]
                self.xs_checkpoints[out_idx] = real_graph_inputs[inp_idx]
            elif policy == ScanOutputAliasPolicy.REMOVE_ADDITIONAL_INPUTS:
                assert out_idx in reverse_alias_map
                inp_idx = reverse_alias_map[out_idx]
                self.additional_inputs_checkpoints[out_idx] = real_graph_inputs[inp_idx]
            else:
                new_output_node.append(node)

        fw_output_node.args = (tuple(fw_outputs) + tuple(new_output_node),)
        fw_gm.graph.lint()
        fw_gm.recompile()
        print("after removing aliasing")
        fw_gm.print_readable()


    def call_forward(self, ctx, init, xs, additional_inputs):
        fw_outputs_and_checkpoints: tuple[Any] = scan_op(
            self.hop_partitioned_graph.fw_gm,
            init,
            xs,
            additional_inputs
        )  # type: ignore[return-type]
        fw_outs = fw_outputs_and_checkpoints[:self.hop_partitioned_graph.n_fw_outputs]
        carry_checkpoints = fw_outputs_and_checkpoints[self.hop_partitioned_graph.n_fw_outputs:]
        carry_checkpoint_iter = iter(carry_checkpoints)

        # Put together the checkpoints
        checkpoints = []
        ctx._fw_policy = self.output_policies
        for i, policy in enumerate(ctx._fw_policy):
            if policy in (ScanOutputAliasPolicy.CLONE, ScanOutputAliasPolicy.NONE):
                checkpoints.append(next(carry_checkpoint_iter))
            elif policy == ScanOutputAliasPolicy.REMOVE_XS:
                checkpoints.append(self.xs_checkpoints[i])
            elif policy == ScanOutputAliasPolicy.REMOVE_ADDITIONAL_INPUTS:
                checkpoints.append(self.additional_inputs_checkpoints[i])
            else:
                raise RuntimeError(f"Unknown policy: {policy}")


        ctx._checkpoints = checkpoints
        return tuple(fw_outs)

    def call_backward(self, ctx, *grad_fw_outputs):
        '''
          Recall that fw_outputs = (*carry, *ys), bw_gm takes in (*fw_checkpoints, *grad_carry, *grad_ys)
          and returns (*grad_init, *grad_xs, *grad_additional_inputs)
          The bacwkard is a reversed scan that can be constructed as follows:

            grad_additonal_inputs = torch.zeros_like(additional_inputs)
            bw_init = (grad_carry, grad_additional_inputs)
            bw_xs = (fw_checkpoints, grad_ys)
            return scan(
              combine_fn,
              bw_init,
              bw_xs,
              reverse = True
            )
            , where combine_fn is defined as follows:

             def combine_fn(bw_init, bw_xs):
               grad_carry, grad_additional_inputs = bw_init
               fw_checkpoints, grad_y = bw_xs
               nxt_grad_carry, grad_x, nxt_grad_additional_inputs = bw_gm(*fw_checkpoints, *grad_carry, *grad_y)
               return (nxt_grad_carry, grad_additional_inputs + nxt_grad_additional_inputs), grad_x

            Note that grad_additional_inputs is accumulated with +, grad_carry is carried over to next iteration
            grad_x is outputed directly, which will be stacked together after the loop and will have the same shape as xs.
        '''
        fw_checkpoints = ctx._checkpoints
        fw_policy = ctx._fw_policy
        carry_checkpoints = []
        xs_checkpoints = []
        additional_inputs_checkpoints = []
        for t, policy in zip(fw_checkpoints, fw_policy):
            if policy in (ScanOutputAliasPolicy.CLONE, ScanOutputAliasPolicy.NONE):
                carry_checkpoints.append(t)
            elif policy == ScanOutputAliasPolicy.REMOVE_XS:
                xs_checkpoints.append(t)
            elif policy == ScanOutputAliasPolicy.REMOVE_ADDITIONAL_INPUTS:
                additional_inputs_checkpoints.append(t)
            else:
                raise RuntimeError(f"Unknown policy: {policy}")

        n_carry = len(self.init)

        grad_carry, grad_ys = grad_fw_outputs[:n_carry], grad_fw_outputs[n_carry:]
        additional_inputs_tensor_masks = [True if isinstance(t, torch.Tensor) else False for t in self.additional_inputs]
        grad_additional_inputs = [torch.zeros_like(t) for t in filter_with_masks(self.additional_inputs, additional_inputs_tensor_masks)]

        bw_init = [
            grad_carry,
            grad_additional_inputs
        ]
        bw_xs = [
            grad_ys,
            xs_checkpoints,
            carry_checkpoints,
        ]
        bw_additional_inputs = additional_inputs_checkpoints


        _, flat_spec = pytree.tree_flatten(
            (bw_init, bw_xs, bw_additional_inputs)
        )

        grad_spec = None

        def bw_single_step_wrapper(*args):
            bw_init, bw_xs, bw_additional_inputs = pytree.tree_unflatten(args, flat_spec)
            grad_carry, grad_additional_inputs = bw_init
            grad_y, xs_checkpoint, carry_checkpoint  = bw_xs
            additional_inputs_checkpoints = bw_additional_inputs

            fw_checkpoints = []
            xs_it = iter(xs_checkpoint)
            carry_it = iter(carry_checkpoint)
            addi_it = iter(additional_inputs_checkpoints)
            for policy in fw_policy:
                if policy in (ScanOutputAliasPolicy.CLONE, ScanOutputAliasPolicy.NONE):
                    fw_checkpoints.append(next(carry_it))
                elif policy == ScanOutputAliasPolicy.REMOVE_XS:
                    fw_checkpoints.append(next(xs_it))
                elif policy == ScanOutputAliasPolicy.REMOVE_ADDITIONAL_INPUTS:
                    fw_checkpoints.append(next(addi_it))
                else:
                    raise RuntimeError(f"Unknown policy: {policy}")

            grad_fw_outputs = (*grad_carry, *grad_y)

            flat_out = self.hop_partitioned_graph.bw_gm(
                *fw_checkpoints,
                *grad_fw_outputs,
            )

            next_grad_carry, grad_xs, grad_addi = split_into_chunks(
                flat_out,  # type: ignore[arg-type]
                [len(self.init), len(self.xs), len(self.additional_inputs)]
            )

            nonlocal grad_spec
            flat_grads, grad_spec = pytree.tree_flatten(
                (
                    next_grad_carry,
                    [
                        prev + cur
                        for prev, cur in zip(grad_additional_inputs, filter_with_masks(grad_addi, additional_inputs_tensor_masks))
                    ],
                    grad_xs,
                )
            )
            return flat_grads

        single_step_bw_xs = pytree.tree_map(lambda t: t[0], bw_xs)
        bw_single_step_gm = materialize_as_graph(
            bw_single_step_wrapper,
            tuple(pytree.tree_flatten(
                (bw_init, single_step_bw_xs, bw_additional_inputs)
            )[0])
        )

        flat_grads = scan_op(
            bw_single_step_gm,
            pytree.tree_flatten(bw_init)[0],
            # TODO: torch.flip copies the checkpoints, we should optimize it away
            [torch.flip(x, (0,)) for x in pytree.tree_flatten(bw_xs)[0]],
            pytree.tree_flatten(bw_additional_inputs)[0],
        )
        grad_init, grad_additional_inputs, grad_xs = pytree.tree_unflatten(flat_grads, grad_spec)  # type: ignore[arg-type]
        return *grad_init, *grad_xs, *fill_none_with_masks(grad_additional_inputs, additional_inputs_tensor_masks)


def _optimize_in_graph_checkpoint_nodes(fw_gm, fw_checkpoint_nodes, init, xs, additional_inputs):
    """
    Graph pass that clones/revmoes fw_checkpoint_nodes. Labels each fw_checkpoint_nodes as either:
    1. "need clone and return in fw_gm" - if it's a placeholder that is part of init
    2. "can save once as attribute ctx" - if it's a placeholder that is part of additional_inputs or xs

    The idea is that for init, we need to save it at every iteration so we want to keep it in the graph.
    For additional_inputs and xs, we can save it once and then re-use it for all iterations.

    The pass adds clones if they belong to category 1 and removes outputs if they belong to category 2
    """
    n_init = len(init)
    n_xs = len(xs)
    n_additional_inputs = len(additional_inputs)

    # Get the original forward output nodes (before checkpoints were added)
    output_node = fw_gm.graph.find_nodes(op="output")[0]
    old_args = list(output_node.args[0])
    fw_gm.print_readable()

    fw_placeholders = [n for n in fw_gm.graph.find_nodes(op="placeholder")]
    init_placeholders = set(fw_placeholders[:n_init])
    xs_ph_to_val = {ph: x for ph, x in zip(fw_placeholders[n_init:n_init + n_xs], xs)}
    additional_input_ph_to_val = {ph: t for ph, t in zip(fw_placeholders[n_init + n_xs:], additional_inputs)}

    # Categorize checkpoint nodes
    nodes_to_remove = [] # "can save once as attribute ctx"
    nodes_to_clone = []  # "need clone and return in fw_gm"

    output_node = fw_gm.graph.find_nodes(op="output")[0]
    output_args = list(output_node.args[0])

    for checkpoint_node in fw_checkpoint_nodes:
        if checkpoint_node in additional_input_ph_to_val or checkpoint_node in xs_ph_to_val:
            nodes_to_remove.append(checkpoint_node)
        elif checkpoint_node in init_placeholders:
            nodes_to_clone.append(checkpoint_node)

    repalcement_map = {}
    # Apply modifications to the graph
    # First handle cloning
    for node in nodes_to_clone:
        with fw_gm.graph.inserting_before(output_node):
            clone_node = fw_gm.graph.call_function(
                torch.ops.aten.clone.default,
                args=(node,),
            )
            clone_node.meta = node.meta.copy() if hasattr(node, 'meta') else {}
            repalcement_map[node] = clone_node

    new_output_args = []
    for node in output_args:
        if node in nodes_to_remove:
            continue
        if node in repalcement_map:
            new_output_args.append(repalcement_map[node])
        else:
            new_output_args.append(node)


    # Update the output node
    output_node.args = (tuple(new_output_args),)
    removed_node_to_pos = {node: i for i, node in enumerate(old_args[-len(fw_checkpoint_nodes):]) if node in nodes_to_remove}
    saved_xs = {removed_node_to_pos[node] : xs_ph_to_val[node] for node in nodes_to_remove if node in xs_ph_to_val}
    saved_additional_inputs = {removed_node_to_pos[node] : additional_input_ph_to_val[node] for node in nodes_to_remove if node in additional_input_ph_to_val}

    # Recompile the graph
    fw_gm.graph.lint()
    fw_gm.recompile()
    print("after _optimize", fw_checkpoint_nodes, nodes_to_remove)
    fw_gm.print_readable()
    return saved_xs, saved_additional_inputs


@scan_op.py_autograd_impl
def scan_autograd(combine_fn, init, xs, additional_inputs):

    with disable_proxy_modes_tracing():
        hop_partitioned_graph = HopGraphPartitioner.create(combine_fn, (*init, *[x[0] for x in xs], *additional_inputs))

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
