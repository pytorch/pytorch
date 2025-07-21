# mypy: allow-untyped-defs
import functools
import itertools
import logging
import operator
import typing
from collections import Counter
from collections.abc import Sequence
from typing import Any, Union

import torch
import torch._guards
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._inductor.constant_folding import ConstantFolder
from torch._inductor.fx_passes.dedupe_symint_uses import _SymHashingDict
from torch._inductor.utils import get_gpu_type
from torch.fx.experimental.symbolic_shapes import (
    guard_or_false,
    guard_or_true,
    statically_known_true,
)
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from .. import config
from ..pattern_matcher import (
    Arg,
    CallFunction,
    init_once_fakemode,
    KeywordArg,
    Match,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
    stable_topological_sort,
)
from .decompose_mem_bound_mm import check_device
from .replace_random import replace_random_passes


log = logging.getLogger(__name__)
patterns = PatternMatcherPass()
aten = torch.ops.aten
prims = torch.ops.prims

pass_patterns = [
    patterns,
    PatternMatcherPass(),
]


@init_once_fakemode
def lazy_init():
    from .fuse_attention import _sfdp_init
    from .misc_patterns import _misc_patterns_init
    from .pad_mm import _pad_mm_init

    _pad_mm_init()
    _sfdp_init()
    _misc_patterns_init()


def remove_no_ops(
    gm: torch.fx.GraphModule,
    zeros: OrderedSet[torch.fx.Node],
    ones: OrderedSet[torch.fx.Node],
):
    with torch.utils._python_dispatch._disable_current_modes():
        "Removes no-ops: (+ 0, - 0, * 1, / 1)"
        graph = gm.graph

        def fake_tensors_eq(t1, t2, fields=("shape", "dtype", "device")):
            if any(not isinstance(t, torch.Tensor) for t in (t1, t2)):
                return False
            for field in fields:
                if getattr(t1, field) != getattr(t2, field):
                    return False
            return True

        def replace_no_op(node, replace_input_index):
            replacement = node.args[replace_input_index]

            # https://github.com/pytorch/pytorch/issues/86128 causes
            # non-Tensor inputs even for ops with only Tensor inputs.
            # TODO - decompose/type promote to avoid this
            if not all(isinstance(arg, torch.fx.Node) for arg in node.args):
                return

            if not fake_tensors_eq(node.meta["val"], replacement.meta["val"]):
                if fake_tensors_eq(
                    node.meta["val"],
                    replacement.meta["val"],
                    ("shape", "device"),
                ):
                    with graph.inserting_after(node):
                        replacement = graph.call_function(
                            torch.ops.prims.convert_element_type.default,
                            args=(replacement, node.meta["val"].dtype),
                        )
                else:
                    return

            node.replace_all_uses_with(replacement)
            replacement.meta.update(node.meta)
            graph.erase_node(node)

        for node in graph.find_nodes(op="call_function", target=aten.add.Tensor):
            # TODO handle Tensor-Scalar adds, it's a different schema
            if len(node.args) == 2:
                if (
                    not any(e in zeros for e in node.args)
                    or node.kwargs.get("alpha", 1) != 1
                ):
                    continue

                replace_index = 1 if node.args[0] in zeros else 0
                replace_no_op(node, replace_index)

        for node in graph.find_nodes(op="call_function", target=aten.sub.Tensor):
            if len(node.args) == 2:
                if node.args[1] not in zeros or node.kwargs.get("alpha", 1) != 1:
                    continue

                replace_no_op(node, 0)

        for node in graph.find_nodes(op="call_function", target=aten.mul.Tensor):
            if len(node.args) == 2:
                if not any(e in ones for e in node.args):
                    continue

                replace_input_index = 1 if node.args[0] in ones else 0
                replace_no_op(node, replace_input_index)

        for node in graph.find_nodes(op="call_function", target=aten.div.Tensor):
            if len(node.args) == 2 and node.args[1] in ones:
                replace_no_op(node, 0)

        # meta tensors returned from the graph have no data and can be replaced with empty_strided
        for output_node in graph.find_nodes(op="output"):
            had_meta_return = False

            def visit(n):
                nonlocal had_meta_return
                val = n.meta.get("val")
                if isinstance(val, torch.Tensor) and val.device.type == "meta":
                    with graph.inserting_before(output_node):
                        n.replace_all_uses_with(
                            graph.call_function(
                                torch.ops.aten.empty_strided.default,
                                args=(val.size(), val.stride()),
                                kwargs={"dtype": val.dtype, "device": val.device},
                            )
                        )
                    had_meta_return = True

            torch.fx.map_arg(output_node.args, visit)
            if had_meta_return:
                graph.eliminate_dead_code()


def remove_redundant_views(gm: torch.fx.GraphModule):
    """
    Removes redundant views by reusing existing ones.
    """
    with torch.utils._python_dispatch._disable_current_modes():
        # A dictionary mapping a tensor to all aliased views.
        views: dict[torch.fx.Node, dict[torch.dtype, torch.fx.Node]] = {}
        graph = gm.graph

        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.view.dtype
        ):
            src = node.args[0]
            to_type = node.args[1]
            existing_views = views.get(src)
            is_needed = True

            if existing_views:
                # Replace the view with the an existing view if available.
                alias = existing_views.get(to_type)
                if alias:
                    is_needed = False
                    node.replace_all_uses_with(alias)
                    alias.meta.update(node.meta)
                    graph.erase_node(node)
            else:
                from_type = src.meta["val"].dtype
                existing_views = {from_type: src}
                views[src] = existing_views

            if is_needed:
                # Save the new alias but do not replace existing one.
                existing_views.setdefault(to_type, node)
                views[node] = existing_views

        # Clean up unused views.
        while True:
            unused_views = [alias for alias in views if not alias.users]
            if len(unused_views) == 0:
                break
            for unused in unused_views:
                views.pop(unused)
                graph.erase_node(unused)


class UniformValueConstantFolder(ConstantFolder):
    """
    Runs constant folding and replaces tensors that have a uniform value
    with a tensor constructor call: aten.full([shape], value, ...)
    """

    def __init__(self, gm, skip_constructors=False) -> None:
        super().__init__(gm, skip_constructors)
        self.node_storages_ptrs: dict[torch.fx.Node, int] = {}
        self.constant_data_ptrs: dict[torch.fx.Node, StorageWeakRef] = {}
        # we may constant fold a tensor which in the graph has a sym size
        # see: [constant folding refining of symints]
        self.node_replacements_shapes: dict[torch.fx.Node, list[int]] = {}

        # initialize symint -> node mapping so that we can
        # use symint nodes in full constructors
        self.symint_nodes = _SymHashingDict()
        for n in self.module.graph.nodes:  # type: ignore[union-attr]
            if "val" in n.meta and isinstance(n.meta["val"], torch.SymInt):
                self.symint_nodes[n.meta["val"]] = n

        # reference from torch/_funtorch/partitioners.py:get_default_op_list
        self.view_op_packets = [
            aten.squeeze,
            aten.unsqueeze,
            aten.alias,
            aten.view,
            aten.slice,
            aten.t,
            prims.broadcast_in_dim,
            aten.expand,
            aten.as_strided,
            aten.permute,
        ]

        self.indexing_op_packets = OrderedSet(
            [
                aten.slice,
            ]
        )

        self._add_peephole_patterns()

    def _add_peephole_patterns(self) -> None:
        """
        Add peephole patterns for nodes where we can infer constant value even if some inputs
        of the node are unknown.
        """
        for op in itertools.chain(
            self.module.graph.find_nodes(  # type: ignore[operator, union-attr]
                op="call_function", target=torch.ops.aten.mul.Tensor
            ),
            self.module.graph.find_nodes(  # type: ignore[operator, union-attr]
                op="call_function", target=torch.ops.aten.mul.Scalar
            ),
        ):
            tensor_val = op.meta.get("val", None)
            if not isinstance(tensor_val, torch.Tensor):
                continue

            def is_zero_int(arg: Any) -> bool:
                return isinstance(arg, int) and arg == 0

            if not any(is_zero_int(a) for a in op.args):
                continue

            t = torch.full(
                [1],  # shape
                0,  # value
                dtype=tensor_val.dtype,
                device=tensor_val.device,
                pin_memory=False,
            )
            self.add_node_replacement(op, t)

    def _support_dynamic_shape(self):
        return True

    def insertable_tensor_check(self, t: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor.flatten()[0].item()
        self.node_replacements_shapes[node] = node.meta["val"].shape
        self.constant_data_ptrs[node] = StorageWeakRef(tensor.untyped_storage())

    def insert_placerholder_values(self, env: dict[torch.fx.Node, Any]) -> None:
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            if "val" in n.meta and isinstance(n.meta["val"], torch.SymInt):
                env[n] = n.meta["val"]
            else:
                env[n] = self.unknown_value

    def _deduce_value(self, node: torch.fx.Node):
        # deduce value for full-like nodes
        # 1. for constructors, substitute value is a tensor of size [1]
        # 2. for view ops/indexing, substitute value is the same as the input
        # 3. for pointwise ops, run node to get the substitute value
        # 4. deal with some special ops
        # otherwise, stop deduce value and return unknown value

        # TODO: cat, more indexing
        # TODO - do on cpu to avoid syncs

        # single-elem attrs
        if node.op == "get_attr" or (
            node.op == "call_function"
            and node.target == torch.ops.aten.lift_fresh_copy.default
        ):
            out = super(ConstantFolder, self).run_node(node)
            if isinstance(out, torch.Tensor) and out.numel() == 1:
                return out

        # handle device_put op
        if node.target == prims.device_put.default:
            return super(ConstantFolder, self).run_node(node)

        # constructors ops
        if (
            node.op == "call_function"
            and node.target == aten.full.default
            and len(node.args) == 2
        ):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            value = args[1]
            # Don't specialize symbolic value.
            if not isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                new_args = [[1], value]
                return aten.full.default(*new_args, **node.kwargs)

        # handle before view ops because this changes value
        if node.target == aten.view.dtype:
            return super(ConstantFolder, self).run_node(node)

        # view ops, return input tensor, the first argument
        if hasattr(node.target, "overloadpacket") and (
            node.target.overloadpacket in self.view_op_packets
            or node.target.overloadpacket in self.indexing_op_packets
        ):
            assert isinstance(node.args[0], torch.fx.Node)
            return self.env[node.args[0]]

        # we don't want to return unknown value for symints so that we can
        # still constant fold through their use in constructors or views
        # if we see them in a pointwise node (e.g., tensor * symint)
        # we will bail
        if "val" in node.meta and isinstance(node.meta["val"], torch.SymInt):
            return node.meta["val"]

        # pointwise ops
        if isinstance(node.target, torch._ops.OpOverload) and (
            torch.Tag.pointwise in node.target.tags
            or node.target is torch.ops.aten.scalar_tensor.default
        ):
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

            if any(isinstance(inp, torch.SymInt) for inp in flattened_inputs):
                return self.unknown_value

            # we run the ops with dim 1, so remove memory_format to avoid error
            kwargs = dict(kwargs)
            kwargs.pop("memory_format", None)

            return node.target(*args, **kwargs)

        return self.unknown_value


def constant_fold_uniform_value(gm: torch.fx.GraphModule):
    with torch.utils._python_dispatch._disable_current_modes():
        "Runs constant folding and replaces constants which can be constructed with a single `full` call. Calls into remove_no_ops."
        aten = torch.ops.aten

        # Constant folding can leak memory, especially with repeated compilation, so we are only going to
        # remove constants which can be replaced with a constructor.
        cf = UniformValueConstantFolder(gm)
        cf.run()

        node_replacements = cf.node_replacements

        # note: [constant folding refining of symints]
        # constant folding will partially evaluate a graph such that values which have dependencies which
        # are entirely known at compile time may also become compile time constants. in some cases,
        # this will include symints which we had not yet previously deduced are guaranteed a
        # constant value and is then deduced in constant folding. an example is:
        # unbacked_symint_eq_11 = torch.full((), 11).item()
        # torch.full((unbacked_symint_eq_11,), 0)
        node_replacements_shapes = cf.node_replacements_shapes

        graph = gm.graph

        zeros = OrderedSet[Any]()
        ones = OrderedSet[Any]()

        # Got failures in `test_is_set_to_cuda` if we change aliasing on constants,
        # so just constant-ify if a Tensor is unaliased
        constant_data_ptr_count: typing.Counter[StorageWeakRef] = Counter()

        for node in cf.node_replacements:
            constant_data_ptr_count[cf.constant_data_ptrs[node]] += 1

        for node, value in node_replacements.items():
            # we dont have a functional way right now of instantiating a non-contiguous tensor with full/zeros/ones right now
            # hasn't shown up to be important yet
            if "val" not in node.meta:
                # This can only happen in AOTI
                continue

            fake_tensor = node.meta["val"]
            if not fake_tensor.is_contiguous(memory_format=torch.contiguous_format):
                continue

            # TODO - not sure about lossy uint->python value->uint conversions
            if fake_tensor.dtype in (
                torch.uint8,
                torch.uint16,
                torch.uint32,
                torch.uint64,
            ):
                continue

            if constant_data_ptr_count[cf.constant_data_ptrs[node]] > 1:
                continue

            with graph.inserting_after(node):
                # the conversion from tensor and back to value can be lossy, just use the original full ctor value
                if (
                    node.op == "call_function"
                    and node.target == aten.full.default
                    and len(node.args) == 2
                ):
                    value = node.args[1]

                # refines symints, see [constant folding refining of symints] above
                for runtime_size, compile_time_size in zip(
                    node_replacements_shapes[node], fake_tensor.shape
                ):
                    torch._check(runtime_size == compile_time_size)

                # replace SymInt as Node before creating a new full node
                # e.g. (1, s0) -> (1, arg0_1)
                node_shape = node_replacements_shapes[node]
                if not all(
                    not isinstance(s, torch.SymInt) or s in cf.symint_nodes
                    for s in node_shape
                ):
                    continue

                shapes = [
                    cf.symint_nodes[s] if isinstance(s, torch.SymInt) else s
                    for s in node_replacements_shapes[node]
                ]

                # zeros and ones just get traced into full, so we insert those
                new_node = graph.call_function(
                    aten.full.default,
                    args=(shapes, value),
                    kwargs={
                        "dtype": fake_tensor.dtype,
                        "layout": torch.strided,
                        "device": fake_tensor.device,
                        "pin_memory": False,
                    },
                )

                new_node.meta.update(node.meta)
                node.replace_all_uses_with(new_node)
                graph.erase_node(node)

                if value == 0:
                    zeros.add(new_node)
                elif value == 1:
                    ones.add(new_node)

        remove_no_ops(gm, zeros, ones)
        remove_redundant_views(gm)


def canonicalize_quant_mapping(gm: torch.fx.GraphModule):
    """


    torch.ops.higher_order.invoke_quant_packed(repeated_subgraph0, 'quant_invoke_0_0', (arg0_1, arg1_1));
    ->
    torch.ops.higher_order.invoke_quant(repeated_subgraph0, arg0_1, arg1_1, scheme = 'nf4');
    """
    graph = gm.graph
    invoke_quant_invocations = graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_quant_packed
    )
    for invoke_quant in invoke_quant_invocations:
        kwargs = dict(invoke_quant.kwargs)

        quant_options_node = kwargs.pop("quant_options", None)
        if quant_options_node is not None:
            assert isinstance(quant_options_node, torch.fx.Node)
            quant_options = torch._higher_order_ops.InvokeQuant(
                *invoke_quant.kwargs["quant_options"].args,
                **invoke_quant.kwargs["quant_options"].kwargs,
            )
        else:
            quant_options = torch._higher_order_ops.InvokeQuant()

        subgraph, *args = invoke_quant.args
        with gm.graph.inserting_before(invoke_quant):
            invoke_quant_replacement = graph.call_function(
                torch._higher_order_ops.invoke_quant,
                (subgraph, *args),
                kwargs,
            )
            invoke_quant_replacement.meta.update(subgraph.meta)
            invoke_quant_replacement.meta["quant_options"] = quant_options

            invoke_quant.replace_all_uses_with(invoke_quant_replacement)
            graph.erase_node(invoke_quant)

            if quant_options_node and len(quant_options_node.users) == 0:
                graph.erase_node(quant_options_node)

            first_user = next(iter(invoke_quant_replacement.users))

            if (
                len(invoke_quant_replacement.users) == 1
                and len(subgraph.users) == 1
                and first_user.target == operator.getitem
                and first_user.args[1] == 0
            ):
                subgraph_graph = getattr(gm, subgraph.target)
                output_node = torch._inductor.utils.output_node(subgraph_graph)
                assert (
                    isinstance(output_node.args[0], (list, tuple))
                    and len(output_node.args[0]) == 1
                )

                unpacked_output = output_node.args[0][0]
                output_node.args = (unpacked_output,)
                if "val" in output_node.meta:
                    output_node.meta["val"] = output_node.meta["val"][0]
                subgraph_graph.recompile()

                invoke_quant_replacement.meta.update(first_user.meta)
                first_user.replace_all_uses_with(invoke_quant_replacement)
                graph.erase_node(first_user)


def canonicalize_aten_ir_passes(gm: torch.fx.GraphModule):
    """
    Canonicalization passes that will run immediately after aot autograd
    tracing. Thsis must be run before all other graph passes.
    """
    canonicalize_quant_mapping(gm)


def joint_graph_passes(graph: torch.fx.GraphModule):
    """
    Run FX transformations on the joint forwards+backwards graph.
    """
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="joint_graph_passes",
    )

    lazy_init()
    count = 0

    # must occur before other passes
    canonicalize_aten_ir_passes(graph)

    if config.joint_custom_pre_pass is not None:
        GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
            config.joint_custom_pre_pass
        )
        count += 1

    from .post_grad import remove_noop_ops

    GraphTransformObserver(graph, "remove_noop_ops").apply_graph_pass(remove_noop_ops)

    if config.joint_graph_constant_folding:
        GraphTransformObserver(graph, "constant_fold_uniform_value").apply_gm_pass(
            constant_fold_uniform_value
        )

    if config.joint_custom_pre_pass is not None:
        GraphTransformObserver(graph, "joint_custom_pre_pass").apply_graph_pass(
            config.joint_custom_pre_pass
        )
        count += 1

    if config.pattern_matcher:
        for i, patterns in enumerate(pass_patterns):
            maybe_count = GraphTransformObserver(
                graph, f"pass_pattern_{i}"
            ).apply_graph_pass(patterns.apply)
            count += maybe_count if maybe_count is not None else 0

    if not config.fallback_random:
        # not trying into the bisector because decomps may have already affected rng reproducibility
        # we'll instead explicitly turn off the config
        count += replace_random_passes(graph)

    if config.joint_custom_post_pass is not None:
        GraphTransformObserver(graph, "joint_custom_post_pass").apply_graph_pass(
            config.joint_custom_post_pass
        )
        count += 1

    if count:
        stable_topological_sort(graph.graph)
        graph.graph.lint()
        graph.recompile()
    return graph


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.iota.default,
        KeywordArg("length"),
        start=KeywordArg("start"),
        step=KeywordArg("step"),
        dtype=KeywordArg("dtype"),
        device=KeywordArg("device"),
        requires_grad=KeywordArg("requires_grad"),
    ),
    pass_dict=patterns,
)
def fix_iota_device(match: Match, length, start, step, dtype, device, requires_grad):
    """
    Eager supports:

        aten.index(cuda_tensor, torch.arange(..., device="cpu"))

    But this results in an implicit host-device-copy and breaks cudagraphs.
    Rewrite the arange to use CUDA.
    """
    (node,) = match.nodes
    user_devices = OrderedSet[torch.device]()
    for user in node.users:
        if (
            user.op == "call_function"
            and user.target in (aten.index.Tensor, aten.index_put.default)
            and hasattr(user.meta.get("val"), "device")
        ):
            user_devices.add(user.meta["val"].device)  # type: ignore[union-attr]
        else:
            return  # bail out

    if len(user_devices) == 1 and "val" in node.meta:
        (user_device,) = user_devices
        if device.type != user_device.type:
            repl = match.graph.call_function(
                torch.ops.prims.iota.default,
                (length,),
                {
                    "start": start,
                    "step": step,
                    "dtype": dtype,
                    "device": user_device,
                    "requires_grad": requires_grad,
                },
            )
            repl.meta.update(node.meta)
            repl.meta["val"] = repl.meta["val"].to(user_device)
            node.replace_all_uses_with(repl)
            match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        CallFunction(
            torch.ops.prims.convert_element_type.default,
            KeywordArg("arg"),
            KeywordArg("dtype1"),
        ),
        KeywordArg("dtype2"),
    ),
    pass_dict=patterns,
)
def pointless_convert(match: Match, arg, dtype1: torch.dtype, dtype2: torch.dtype):
    """Remove chain of dtype conversions often created by AMP"""
    graph = match.graph
    node = match.output_node()
    allowed = torch.float16, torch.bfloat16, torch.float32, torch.float64
    if dtype1 in allowed and dtype2 in allowed:
        repl = graph.call_function(
            torch.ops.prims.convert_element_type.default, (arg, dtype2)
        )
        repl.meta.update(node.meta)
        node.replace_all_uses_with(repl)
        match.erase_nodes()


def definitely_equal(
    old_sizes: Sequence[Union[torch.SymInt, int]],
    new_sizes: Sequence[Union[torch.SymInt, torch.fx.Node, int]],
) -> bool:
    """
    Leverage guard_or_true/false to compare if two lists of int/symint are equal.
    Useful to compare sizes, strides etc.

    Can handle -1 in new_sizes which happens in the size arguments of a
    view op. old_sizes is supposed to be the tensor shape and should not
    contain -1.

    new_sizes can contains fx.Node when dynamic shape is enabled. In that
    case new_sizes[i].meta['val'] contains the real torch.SymInt.
    """

    num_neg1 = 0

    if len(old_sizes) != len(new_sizes):
        return False

    for lhs_item, rhs_item in zip(old_sizes, new_sizes):
        if isinstance(rhs_item, torch.fx.Node):
            rhs_item = rhs_item.meta["val"]

        assert isinstance(lhs_item, (int, torch.SymInt)), type(lhs_item)
        assert isinstance(rhs_item, (int, torch.SymInt)), type(rhs_item)

        # It still makes sense to call guard_or_true/false since lhs_item
        # rhs_item are torch.SymInt rather than sympy expressions when
        # dynamic shape is enabled.
        if guard_or_false(lhs_item == rhs_item):
            continue

        if guard_or_true(rhs_item != -1):
            return False

        num_neg1 += 1

        if num_neg1 > 1:
            return False
    return True


@register_graph_pattern(
    CallFunction(torch.ops.aten.view.default, KeywordArg("arg"), KeywordArg("size")),
    pass_dict=patterns,
)
def pointless_view(match: Match, arg, size):
    """Remove no-op view"""
    node = match.output_node()
    arg_size = list(node.args[0].meta["val"].shape)  # type: ignore[union-attr]
    if definitely_equal(arg_size, size):
        node.replace_all_uses_with(node.args[0])  # type: ignore[arg-type]
        match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        aten.view.default,
        CallFunction(aten.view.default, KeywordArg("arg"), KeywordArg("size1")),
        KeywordArg("size2"),
    ),
    pass_dict=patterns,
)
def pointless_view_pair(match: Match, arg, size1, size2):
    """
    Remove a pair of views that are pointless.
    """
    node = match.output_node()
    arg_size = list(arg.meta["val"].shape)
    if definitely_equal(arg_size, size2):
        node.replace_all_uses_with(arg)
        match.erase_nodes()
        counters["inductor"]["removed_pointless_view_pair"] += 1


@register_graph_pattern(
    CallFunction(
        aten.permute.default,
        CallFunction(aten.permute.default, KeywordArg("arg"), KeywordArg("perm1")),
        KeywordArg("perm2"),
    ),
    pass_dict=patterns,
)
def pointless_permute_pair(match: Match, arg, perm1, perm2):
    rank = len(perm1)
    assert len(perm2) == rank

    for i in range(rank):
        if perm1[perm2[i]] != i:
            return  # bail out
    node = match.output_node()
    node.replace_all_uses_with(arg)
    match.erase_nodes()


@register_graph_pattern(
    CallFunction(
        aten.bmm,
        Arg(),
        Arg(),
    ),
    pass_dict=patterns,
)
def bmm_to_mm(match: Match, mat1: torch.fx.Node, mat2: torch.fx.Node):
    """Convert bmm to mm when batch size is 1"""

    def repl(a, b):
        return torch.mm(a.squeeze(0), b.squeeze(0)).unsqueeze(0)

    if (
        check_device(mat1.meta["val"], mat2.meta["val"], get_gpu_type())
        and statically_known_true(mat1.meta["val"].shape[0] == 1)
        and statically_known_true(mat2.meta["val"].shape[0] == 1)
    ):
        match.replace_by_example(repl, [mat1, mat2])


# When softmax is used with temperature or other scaling, we get the pattern
#
#   scale(x) - scale(x).amax(dim, keepdim=True)
#
# which is expected to be at most zero, but we may end up with numerical
# discrepancies # between the recomputed values of scale(x) inside and out
# of the reduction, # depending on compiler optimizations, e.g. use of fma
# instructions.
#
# Here we replace it with the mathematically equivalent,
#
#   scale(x - x.amax(dim, keepdim=True))
#
# which is more stable as we only compute the scaling once.
#
# NOTE: This pattern must come after fused attention matching!


def _partial_softmax_pattern(linear_func, reverse=False, to_dtype=False):
    # Allow matching inp * other and other * input
    if reverse:
        scaled = CallFunction(
            linear_func, KeywordArg("other"), KeywordArg("inp"), _users=MULTIPLE
        )
    else:
        scaled = CallFunction(
            linear_func, KeywordArg("inp"), KeywordArg("other"), _users=MULTIPLE
        )
    if to_dtype:
        scaled = CallFunction(
            prims.convert_element_type, scaled, KeywordArg("dtype"), _users=MULTIPLE
        )
    amax = CallFunction(
        aten.amax.default, scaled, KeywordArg("dim"), KeywordArg("keepdim")
    )
    return CallFunction(aten.sub.Tensor, scaled, amax)


def _other_is_broadcasted_in_dim(match):
    # Check that the scaling factor is constant across the reduction dim,
    # so scaling doesn't change which index corresponds to the maximum value
    other = match.kwargs["other"]
    if isinstance(other, (int, float)):
        return True

    inp = match.kwargs["inp"]
    if not all(isinstance(x, torch.fx.Node) for x in (inp, other)):
        return False

    inp_example = inp.meta["val"]
    other_example = other.meta["val"]
    if isinstance(other_example, (torch.SymInt, torch.SymFloat)):
        return True

    if not all(isinstance(x, torch.Tensor) for x in (inp_example, other_example)):
        return False

    inp_ndim = inp_example.ndim
    other_shape = other_example.shape
    if inp_ndim < len(other_shape):
        return False

    # Pad other_shape to the same ndim as inp
    other_shape = [1] * (inp_ndim - len(other_shape)) + list(other_shape)

    dim = match.kwargs["dim"]
    if isinstance(dim, int):
        dim = (dim,)

    return all(statically_known_true(other_shape[d] == 1) for d in dim)


def mul_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    def repl(inp, other):
        if dtype is not None:
            inp = inp.to(dtype)

        sign: Union[int, float, torch.Tensor]
        if isinstance(other, (int, float, torch.SymInt, torch.SymFloat)):
            sign = 1 if other >= 0 else -1
        else:
            one = torch.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            sign = torch.where(other >= 0, one, -one)

        inp = inp * sign
        max_ = torch.amax(inp, dim=dim, keepdim=keepdim)
        return (inp - max_) * (sign * other)

    match.replace_by_example(repl, [inp, other])


for reverse, to_dtype in itertools.product((False, True), repeat=2):
    register_graph_pattern(
        _partial_softmax_pattern(aten.mul.Tensor, reverse=reverse, to_dtype=to_dtype),
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(mul_softmax_pattern)


def div_softmax_pattern(match: Match, *, inp, other, dim, keepdim, dtype=None):
    def repl(inp, other):
        if dtype is not None:
            inp = inp.to(dtype)

        sign: Union[int, float, torch.Tensor]
        if isinstance(other, (int, float, torch.SymInt, torch.SymFloat)):
            sign = 1 if other >= 0 else -1
        else:
            one = torch.scalar_tensor(1, dtype=inp.dtype, device=inp.device)
            sign = torch.where(other >= 0, one, -one)

        inp = inp * sign
        max_ = torch.amax(inp, dim=dim, keepdim=keepdim)
        return (inp - max_) / (sign * other)

    match.replace_by_example(repl, [inp, other])


for to_dtype in (False, True):
    register_graph_pattern(
        _partial_softmax_pattern(aten.div.Tensor, to_dtype=to_dtype),
        pass_dict=pass_patterns[1],
        extra_check=_other_is_broadcasted_in_dim,
    )(div_softmax_pattern)
