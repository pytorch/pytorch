# mypy: allow-untyped-defs
import copy
import functools
import hashlib
import heapq
import itertools
import logging
import math
import operator
import os
import os.path
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, TYPE_CHECKING, Union

import torch
import torch._inductor.inductor_prims
import torch.distributed
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters, is_node_meta_valid
from torch._functorch._activation_checkpointing.ac_logging_utils import (
    create_structured_trace_for_min_cut_info,
)
from torch._inductor import config as inductor_config
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.fx.experimental.symbolic_shapes import (
    find_symbol_binding_fx_nodes,
    free_symbols,
    hint_int,
    is_symbol_binding_fx_node,
    statically_known_false,
    statically_known_true,
)
from torch.fx.passes import graph_drawer
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy

from . import config
from ._activation_checkpointing.graph_info_provider import GraphInfoProvider
from ._activation_checkpointing.knapsack import (
    dp_knapsack,
    greedy_knapsack,
    ilp_knapsack,
)
from ._activation_checkpointing.knapsack_evaluator import KnapsackEvaluator
from ._aot_autograd.descriptors import AOTOutput, SavedForBackwardsAOTOutput
from ._aot_autograd.logging_utils import get_aot_graph_name
from ._aot_autograd.utils import get_cuda_generator_meta_val, is_with_effects
from .compile_utils import fx_graph_cse, get_aten_target, raise_getitems


if TYPE_CHECKING:
    import sympy


AOT_PARTITIONER_DEBUG: bool = config.debug_partitioner
log: logging.Logger = logging.getLogger(__name__)

aten = torch.ops.aten
prims = torch.ops.prims


@dataclass
class OpTypes:
    """Class for keeping track of different operator categories"""

    fusible_ops: OrderedSet[Callable]
    compute_intensive_ops: OrderedSet[Callable]
    random_ops: OrderedSet[Callable]
    view_ops: OrderedSet[Callable]
    recomputable_ops: OrderedSet[Callable]

    def is_fusible(self, node: fx.Node):
        return get_aten_target(node) in self.fusible_ops

    def is_compute_intensive(self, node: fx.Node):
        return get_aten_target(node) in self.compute_intensive_ops

    def is_random(self, node: fx.Node):
        return get_aten_target(node) in self.random_ops

    def is_view(self, node: fx.Node):
        return get_aten_target(node) in self.view_ops

    def is_recomputable(self, node: fx.Node):
        return get_aten_target(node) in self.recomputable_ops


@dataclass
class NodeInfo:
    # Be careful about iterating over these explicitly, as their order may not
    # be deterministic
    inputs: list[fx.Node]
    _required_fw_nodes: OrderedSet[fx.Node]
    required_bw_nodes: OrderedSet[fx.Node]
    unclaimed_nodes: OrderedSet[fx.Node]
    fw_order: dict[fx.Node, int]
    # Effectively maps to which of our primals are parameters
    static_lifetime_input_nodes: OrderedSet[fx.Node]

    @functools.cached_property
    def required_fw_nodes(self) -> list[fx.Node]:
        return sorted(
            (n for n in self._required_fw_nodes), key=lambda n: self.fw_order[n]
        )

    def is_required_fw(self, n: fx.Node) -> bool:
        return n in self._required_fw_nodes

    def is_required_bw(self, n: fx.Node) -> bool:
        return n in self.required_bw_nodes

    def is_unclaimed(self, n: fx.Node) -> bool:
        return n in self.unclaimed_nodes

    def get_fw_order(self, n: fx.Node) -> int:
        assert n in self._required_fw_nodes, f"Node {n} not in fw nodes!"
        return self.fw_order[n]


@dataclass
class MinCutOptions:
    ban_if_used_far_apart: bool
    ban_if_long_fusible_chains: bool
    ban_if_materialized_backward: bool
    ban_if_not_in_allowlist: bool
    ban_if_reduction: bool


def must_recompute(node: fx.Node) -> bool:
    return node.meta.get("recompute", None) in [
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
    ]


def has_recomputable_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if must_recompute(node):
            return True
    return False


def has_recomputable_rng_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if (
            must_recompute(node)
            and hasattr(node.target, "tags")
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return True
    return False


def sym_node_size(node: fx.Node) -> int:
    if isinstance(node.meta["val"], (torch.SymInt, torch.SymBool)):
        return 1
    assert isinstance(node.meta["val"], torch.SymFloat)
    return 4


class InvalidNodeBase:
    def __repr__(self):
        return "Invalid Node"


InvalidNode = InvalidNodeBase()


def _extract_graph_with_inputs_outputs(
    joint_graph: fx.Graph,
    inputs: list[fx.Node],
    outputs: list[fx.Node],
    outputs_descs: list[AOTOutput],
    subgraph: Optional[str] = None,
) -> fx.Graph:
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as
    inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
    new_graph = fx.Graph()
    env = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in inputs:
        new_node = new_graph.placeholder(node.name)
        # Can't use node_copy here as we may be turning previous call_function into placeholders
        new_node.meta = node.meta
        env[node] = new_node

    for node in joint_graph.nodes:
        if _must_be_in_backward(node) and subgraph != "backward":
            env[node] = InvalidNode  # type: ignore[assignment]
            continue

        if _must_be_in_forward(node) and subgraph != "forward":
            env[node] = InvalidNode  # type: ignore[assignment]
            continue

        if node in env:
            # Node must be one of our inputs. (Any member of env which wasn't an
            # input to start must have been created by this loop and won't be in
            # joint_graph.nodes).
            continue
        elif node.op == "placeholder":
            env[node] = InvalidNode  # type: ignore[assignment]
        elif node.op == "call_function":
            all_args = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            all_args = [
                isinstance(env[x], InvalidNodeBase)
                for x in all_args
                if isinstance(x, fx.Node)
            ]
            if any(all_args):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "get_attr":
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "output":
            pass
    output_values = []
    for x in outputs:
        if isinstance(x, fx.Node):
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            assert not isinstance(env[x], InvalidNodeBase), (
                f"Node {x} was invalid, but is output"
            )
            output_values.append(env[x])
        else:
            output_values.append(x)
    out = new_graph.output(tuple(output_values))
    out.meta["desc"] = outputs_descs

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph


def _is_primal(node: fx.Node) -> bool:
    return (
        node.op == "placeholder"
        and "tangents" not in str(node.target)
        and not _is_bwd_seed_offset(node)
        and not _is_fwd_seed_offset(node)
    )


def _is_tangent(node: fx.Node) -> bool:
    return node.op == "placeholder" and "tangents" in str(node.target)


def _is_bwd_seed_offset(node: fx.Node) -> bool:
    return node.op == "placeholder" and (
        "bwd_seed" in str(node.target) or "bwd_base_offset" in str(node.target)
    )


def _is_fwd_seed_offset(node: fx.Node) -> bool:
    return node.op == "placeholder" and (
        "fwd_seed" in str(node.target) or "fwd_base_offset" in str(node.target)
    )


def _is_backward_state(node: fx.Node) -> bool:
    return node.op == "placeholder" and isinstance(node.meta.get("val"), BackwardState)


def _has_tag_is_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "is_backward"


def _has_tag_must_be_in_forward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "must_be_in_forward"


def _has_tag_must_be_in_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "must_be_in_backward"


def _must_be_in_forward(node: fx.Node) -> bool:
    return _has_tag_must_be_in_forward(node)


def _must_be_in_backward(node: fx.Node) -> bool:
    return _has_tag_must_be_in_backward(node) or (
        _has_tag_is_backward(node) and is_with_effects(node)
    )


def _extract_fwd_bwd_outputs(
    joint_module: fx.GraphModule, *, num_fwd_outputs
) -> tuple[list[fx.Node], list[fx.Node], list[AOTOutput], list[AOTOutput]]:
    outputs = pytree.arg_tree_leaves(
        *(node.args for node in joint_module.graph.find_nodes(op="output"))
    )
    outputs_descs = pytree.arg_tree_leaves(
        next(iter(joint_module.graph.find_nodes(op="output"))).meta.get(
            "desc", [None] * len(outputs)
        )
    )
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    fwd_outputs_descs = outputs_descs[:num_fwd_outputs]
    bwd_outputs_descs = outputs_descs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs


def _remove_by_name(saved_values: list[fx.Node], name: str):
    for saved_value in saved_values:
        if saved_value.name == name:
            saved_values.remove(saved_value)
            break


def find_first_sym_node(
    fwd_module_outputs: Union[list[fx.Node], tuple[fx.Node]],
) -> int:
    idx = len(fwd_module_outputs)
    for i in range(len(fwd_module_outputs) - 1, -1, -1):
        if not is_sym_node(fwd_module_outputs[i]):
            idx = i + 1
            break
    return idx


def calculate_quantization_scaling(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    max: float = 57344.0,
    min: float = 1e-12,
):
    with graph.inserting_after(node):
        abs_node = graph.call_function(
            torch.ops.aten.abs.default,
            args=(node,),
        )
        abs_node.meta["val"] = torch.ops.aten.abs.default(node.meta["val"])
        abs_node.meta["tensor_meta"] = extract_tensor_metadata(abs_node.meta["val"])
    with graph.inserting_after(abs_node):
        amax_node = graph.call_function(
            torch.ops.aten.amax.default,
            args=(abs_node, [-1], True),
        )
        amax_node.meta["val"] = torch.ops.aten.amax.default(
            abs_node.meta["val"], [-1], True
        )
        amax_node.meta["tensor_meta"] = extract_tensor_metadata(amax_node.meta["val"])
    with graph.inserting_after(amax_node):
        amax_64_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(amax_node, torch.float64),
        )
        amax_64_node.meta["val"] = torch.ops.prims.convert_element_type.default(
            amax_node.meta["val"], torch.float64
        )
        amax_64_node.meta["tensor_meta"] = extract_tensor_metadata(
            amax_64_node.meta["val"]
        )
    with graph.inserting_after(amax_64_node):
        clamp_min_node = graph.call_function(
            torch.ops.aten.clamp_min.default,
            args=(amax_64_node, min),
        )
        clamp_min_node.meta["val"] = torch.ops.aten.clamp_min.default(
            amax_64_node.meta["val"], min
        )
        clamp_min_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_min_node.meta["val"]
        )
    with graph.inserting_after(clamp_min_node):
        reciprocal_node = graph.call_function(
            torch.ops.aten.reciprocal.default,
            args=(clamp_min_node,),
        )
        reciprocal_node.meta["val"] = torch.ops.aten.reciprocal.default(
            clamp_min_node.meta["val"]
        )
        reciprocal_node.meta["tensor_meta"] = extract_tensor_metadata(
            reciprocal_node.meta["val"]
        )
    with graph.inserting_after(reciprocal_node):
        mul_node = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(reciprocal_node, max),
        )
        mul_node.meta["val"] = torch.ops.aten.mul.Tensor(
            reciprocal_node.meta["val"], max
        )
        mul_node.meta["tensor_meta"] = extract_tensor_metadata(mul_node.meta["val"])
    with graph.inserting_after(mul_node):
        scale_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(mul_node, torch.float32),
            name="fp8_scale_" + str(node.name),
        )
        scale_node.meta["val"] = torch.ops.prims.convert_element_type.default(
            mul_node.meta["val"], torch.float32
        )
        scale_node.meta["tensor_meta"] = extract_tensor_metadata(scale_node.meta["val"])
    return scale_node


def perform_quantization(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    scale_node: torch.fx.Node,
    quant_type: torch.dtype,
    clamp_min: float,
    clamp_max: float,
) -> torch.fx.Node:
    with graph.inserting_after(scale_node):
        target_node_32 = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(node, torch.float32),
        )
        target_node_32.meta["val"] = torch.ops.prims.convert_element_type.default(
            node.meta["val"], torch.float32
        )
        target_node_32.meta["tensor_meta"] = extract_tensor_metadata(
            target_node_32.meta["val"]
        )
    with graph.inserting_after(target_node_32):
        scaled_target_node = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(target_node_32, scale_node),
        )
        scaled_target_node.meta["val"] = torch.ops.aten.mul.Tensor(
            target_node_32.meta["val"], scale_node.meta["val"]
        )
        scaled_target_node.meta["tensor_meta"] = extract_tensor_metadata(
            scaled_target_node.meta["val"]
        )
    with graph.inserting_after(scaled_target_node):
        clamp_min_scaled_node = graph.call_function(
            torch.ops.aten.clamp_min.default,
            args=(scaled_target_node, clamp_min),
        )
        clamp_min_scaled_node.meta["val"] = torch.ops.aten.clamp_min.default(
            scaled_target_node.meta["val"], clamp_min
        )
        clamp_min_scaled_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_min_scaled_node.meta["val"]
        )
    with graph.inserting_after(clamp_min_scaled_node):
        clamp_max_scaled_node = graph.call_function(
            torch.ops.aten.clamp_max.default,
            args=(clamp_min_scaled_node, clamp_max),
        )
        clamp_max_scaled_node.meta["val"] = torch.ops.aten.clamp_max.default(
            clamp_min_scaled_node.meta["val"], clamp_max
        )
        clamp_max_scaled_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_max_scaled_node.meta["val"]
        )
    with graph.inserting_after(clamp_max_scaled_node):
        quant_activation_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(clamp_max_scaled_node, quant_type),
            name="fp8_quant_" + str(node.name),
        )
        quant_activation_node.meta["val"] = (
            torch.ops.prims.convert_element_type.default(
                clamp_max_scaled_node.meta["val"], quant_type
            )
        )
        quant_activation_node.meta["tensor_meta"] = extract_tensor_metadata(
            quant_activation_node.meta["val"]
        )
    return quant_activation_node


def calculate_tensor_size(tensor: torch.Tensor) -> float:
    """
    Calculate the size of a PyTorch tensor in megabytes (MB).

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        float: Memory size in MB
    """
    # Get number of elements and size per element
    num_elements = tensor.numel()
    element_size = tensor.element_size()

    return (num_elements * element_size) / (1024 * 1024)


def get_allowed_dtypes() -> list[torch.dtype]:
    allowed_dtypes = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("allowed_dtypes", "torch.bfloat16")
    allowed_dtypes = [
        getattr(torch, dtype.split(".")[-1]) for dtype in allowed_dtypes.split(";")
    ]
    return allowed_dtypes


def should_quantize(node: torch.fx.Node) -> bool:
    allowed_dtypes = get_allowed_dtypes()
    if not is_node_meta_valid(node) or node.meta["val"].dtype not in allowed_dtypes:
        return False
    size_threshold = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("size_in_mb", 100)
    # calculate the size of the node
    size_in_mb = calculate_tensor_size(node.meta["val"])
    if not torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("skip_dynamo_guards", False):
        return size_in_mb >= size_threshold
    else:
        # case 1: we always quantize tensors with dynamic shapes
        if torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ].get("quantize_dynamic_shape", False):
            return statically_known_true(
                size_in_mb >= size_threshold
            ) or not statically_known_false(size_in_mb >= size_threshold)
        else:
            # case 2: we always not quantize tensors with dynamic shapes
            return statically_known_true(size_in_mb >= size_threshold)


def get_quant_type() -> torch.dtype:
    quant_type = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("quant_type", "torch.float8_e5m2")

    return getattr(torch, quant_type.split(".")[-1])


def calculate_range(dtype: torch.dtype) -> tuple:
    """
    Calculate the range of values for a given torch.dtype.
    Args:
        dtype (torch.dtype): The input dtype.
    Returns:
        tuple: A tuple containing the minimum and maximum values.
    """
    info = torch.finfo(dtype)
    return info.min, info.max


def quantize_activation_fw(graph: torch.fx.Graph) -> None:
    output = graph.find_nodes(op="output")[0]
    fwd_outputs = output.args[0]
    quant_type = get_quant_type()
    clamp_min, clamp_max = calculate_range(quant_type)
    node_to_quant = dict()
    tensor_scale_nodes, sym_scale_nodes = [], []
    for node in fwd_outputs:
        # check if the activation node is the node saved for quantization
        if node.meta.get("saved_for_quantization", False):
            # case: use scaling
            if torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ].get("use_scaling", True):
                # calculating the scale
                scale_node = calculate_quantization_scaling(
                    graph, node, clamp_max, 1e-12
                )
                # converting to fp8
                quant_node = perform_quantization(
                    graph, node, scale_node, quant_type, clamp_min, clamp_max
                )
                if not is_sym_node(scale_node):
                    tensor_scale_nodes.append(scale_node)
                else:
                    sym_scale_nodes.append(scale_node)
            else:
                # case: do not use scaling
                with graph.inserting_after(node):
                    quant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, quant_type),
                        name="fp8_quant_" + str(node.name),
                    )
                    quant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], quant_type
                        )
                    )
                    quant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        quant_node.meta["val"]
                    )
            node_to_quant[node] = quant_node
    # only update the return node args, and remain all other users unchanged
    output_updated_args = [
        node_to_quant[node] if node in node_to_quant else node for node in fwd_outputs
    ]
    # add the scale nodes to the output find the first sym_node in the output
    idx = find_first_sym_node(output_updated_args)
    scale_nodes = tensor_scale_nodes + sym_scale_nodes
    if scale_nodes:
        output_updated_args = (
            output_updated_args[:idx] + scale_nodes + output_updated_args[idx:]
        )

    output.update_arg(0, tuple(output_updated_args))
    counters["inductor"]["activation_quantization_fwd_aten_pass"] += 1


def quantize_activation_bw(graph: torch.fx.Graph) -> None:
    bw_inputs = [node for node in graph.nodes if node.op == "placeholder"]
    activation_node = None
    for node in bw_inputs:
        if node.meta.get("saved_for_quantization", False):
            node.meta.pop("saved_for_quantization")
            dequant_type = node.meta.pop("dequant_type")
            # dequantize the node
            if torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ].get("use_scaling", False):
                # case: use scaling
                with graph.inserting_after(node):
                    # find corresponding scale node
                    scale_name = "fp8_scale_" + node.name.replace("fp8_quant_", "")
                    scale_node = next(
                        bwd_input
                        for bwd_input in bw_inputs
                        if bwd_input.name == scale_name
                    )
                with graph.inserting_after(scale_node):
                    activation_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, dequant_type),
                    )
                    activation_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], dequant_type
                        )
                    )
                    activation_node.meta["tensor_meta"] = extract_tensor_metadata(
                        activation_node.meta["val"]
                    )
                with graph.inserting_after(activation_node):
                    divided_target_node_32 = graph.call_function(
                        torch.ops.aten.div.Tensor,
                        args=(activation_node, scale_node),
                    )
                    divided_target_node_32.meta["val"] = torch.ops.aten.div.Tensor(
                        activation_node.meta["val"], scale_node.meta["val"]
                    )
                    divided_target_node_32.meta["tensor_meta"] = (
                        extract_tensor_metadata(divided_target_node_32.meta["val"])
                    )
                with graph.inserting_after(divided_target_node_32):
                    dequant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(divided_target_node_32, dequant_type),
                    )
                    dequant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            divided_target_node_32.meta["val"], dequant_type
                        )
                    )
                    dequant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        dequant_node.meta["val"]
                    )
            else:
                with graph.inserting_after(node):
                    dequant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, dequant_type),
                        name="dequant_" + str(node.name),
                    )
                    dequant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], dequant_type
                        )
                    )
                    dequant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        dequant_node.meta["val"]
                    )
            # find the users of the node and replace them with the new node except the dequant_node
            for user in list(node.users.keys()):
                if user != dequant_node and user != activation_node:
                    user.replace_input_with(node, dequant_node)

    counters["inductor"]["activation_quantization_bwd_aten_pass"] += 1


def perform_fp8_activation_quantization(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    bwd_module_inputs: dict[str, fx.Node],
) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_activation_quantization_fwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: fwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    quantize_activation_fw(fwd_module.graph)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_activation_quantization_fwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: fwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_activation_quantization_bwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: bwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    quant_fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
    # update the corresponding bwd_inputs due to the fwd_outputs quantization
    for fwd_node in quant_fwd_module_outputs:
        if "fp8_quant_" in fwd_node.name:
            bwd_input = bwd_module_inputs[fwd_node.name.replace("fp8_quant_", "")]
            with bwd_module.graph.inserting_after(bwd_input):
                quant_bwd_input = bwd_module.graph.placeholder(name=fwd_node.name)
            dequant_type = bwd_input.meta["dequant_type"]
            quant_bwd_input.meta.update(fwd_node.meta)
            quant_bwd_input.meta["saved_for_quantization"] = True
            quant_bwd_input.meta["dequant_type"] = dequant_type
            bwd_input.replace_all_uses_with(quant_bwd_input)
            bwd_module.graph.erase_node(bwd_input)
    # update the bwd_inputs if quantization with scaling is used
    if torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("use_scaling", True):
        quant_bwd_module_inputs = list(bwd_module.graph.find_nodes(op="placeholder"))
        # update the corresponding bwd input nodes find the last non-tangent node
        bwd_input_loc = quant_bwd_module_inputs[-1]
        for bw_input in reversed(quant_bwd_module_inputs):
            if not _is_tangent(bw_input):
                bwd_input_loc = bw_input
                break

        scaled_fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
        for fwd_node in scaled_fwd_module_outputs:
            if "fp8_scale_" in fwd_node.name:
                # fwd node is a scale node
                with bwd_module.graph.inserting_after(bwd_input_loc):
                    scale_bwd_input = bwd_module.graph.placeholder(name=fwd_node.name)
                scale_bwd_input.meta.update(fwd_node.meta)
                bwd_input_loc = scale_bwd_input

    quantize_activation_bw(bwd_module.graph)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_activation_quantization_bwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: bwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )


def enable_activation_quantization(
    saved_values: list[fx.Node],
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    static_lifetime_input_nodes: Optional[OrderedSet[fx.Node]] = None,
) -> None:
    if (
        inductor_config.post_grad_fusion_options.get(
            "activation_quantization_aten_pass", None
        )
        is None
    ):
        return

    static_input_names = (
        [node.name for node in static_lifetime_input_nodes]
        if static_lifetime_input_nodes
        else []
    )
    saved_values_names = {node.name: node for node in saved_values}
    if torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("exclude_primals", False):
        saved_values_names = {
            node.name: node for node in saved_values if "primals" not in node.name
        }
    fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
    bwd_module_inputs = {
        node.name: node for node in bwd_module.graph.find_nodes(op="placeholder")
    }
    should_perform_fp8_quant = False
    for node in fwd_module_outputs:
        if node.name in saved_values_names and should_quantize(node):
            if node.name in static_input_names:
                log.debug("Skipping quantization of static input %s: ", node.name)
                continue
            node.meta["saved_for_quantization"] = True
            node.meta["dequant_type"] = node.meta["val"].dtype
            # some of the fwd outputs and bwd inputs are not share the same object
            bwd_module_inputs[node.name].meta["saved_for_quantization"] = True
            bwd_module_inputs[node.name].meta["dequant_type"] = node.meta["val"].dtype
            should_perform_fp8_quant = True

    if should_perform_fp8_quant:
        perform_fp8_activation_quantization(fwd_module, bwd_module, bwd_module_inputs)


def _extract_fwd_bwd_modules(
    joint_module: fx.GraphModule,
    saved_values: list[fx.Node],
    saved_sym_nodes: list[fx.Node],
    *,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: Optional[OrderedSet[fx.Node]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs = (
        _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    )
    placeholders = joint_module.graph.find_nodes(op="placeholder")
    primal_inputs = [*filter(_is_primal, placeholders)]
    tangent_inputs = [*filter(_is_tangent, placeholders)]
    fwd_seed_offset_inputs = [*filter(_is_fwd_seed_offset, placeholders)]
    bwd_seed_offset_inputs = [*filter(_is_bwd_seed_offset, placeholders)]
    backward_state_inputs = [*filter(_is_backward_state, placeholders)]

    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes + saved_values + tangent_inputs + bwd_seed_offset_inputs,
        bwd_outputs,
        bwd_outputs_descs,
        "backward",
    )

    distributed_enabled = torch.distributed.is_available()

    for node in bwd_graph.find_nodes(op="placeholder"):
        # This is to filter out saved values that don't actually end up being used by the backwards pass
        if not node.users:
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        # wait_tensor is a bit special: if we have a "dead activation" that is not used in the bw,
        # but this dead activation is actually a collective,
        # then the collective will generally by followed by a wait_tensor() call.
        # we need to peak one node further to see if this wait_tensor is dead as well.
        elif distributed_enabled and all(
            n.target is torch.ops._c10d_functional.wait_tensor.default
            and len(n.users) == 0
            for n in node.users
        ):
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
        elif _is_backward_state(node):
            # BackwardState is saved directly
            _remove_by_name(saved_values, node.name)
            assert backward_state_inputs

    # Now that we have the finalized list of saved values, we need to ensure
    # we propagate all symbols which are referenced by backwards inputs.
    # These are not directly used in the graph but are required for downstream
    # sizevar assignment
    saved_symbols: OrderedSet[sympy.Symbol] = OrderedSet()
    saved_sym_nodes_binding = []
    saved_sym_nodes_derived = []

    # Some symbols may already be bound in the directly saved_sym_nodes,
    # keep track of them so we don't re-bind them
    for node in saved_sym_nodes:
        symbol = is_symbol_binding_fx_node(node)
        if symbol:
            saved_symbols.add(symbol)
            saved_sym_nodes_binding.append(node)
        else:
            saved_sym_nodes_derived.append(node)

    # Now go through all of the prospective backward inputs and track any
    # other symbols we need to bind
    symbol_bindings = find_symbol_binding_fx_nodes(joint_module.graph)
    for node in itertools.chain(saved_sym_nodes_derived, saved_values, tangent_inputs):
        if "val" not in node.meta:
            continue
        new_symbols = free_symbols(node.meta["val"]) - saved_symbols
        # NB: Deterministic order please!
        for s in sorted(new_symbols, key=lambda s: s.name):
            # NB: For well formed graphs, the symbol should always be present,
            # but we also have ways to produce ill-formed graphs, e.g., direct
            # make_fx usages, so don't choke in this case
            if s not in symbol_bindings:
                continue
            saved_sym_nodes_binding.append(symbol_bindings[s])
        saved_symbols |= new_symbols

    # Update saved_sym_nodes that are now reordered to have all bindings at
    # front. This can also be used later on to figure out the position of saved
    # sym nodes in the output of fwd graph.
    saved_sym_nodes.clear()
    saved_sym_nodes.extend(saved_sym_nodes_binding + saved_sym_nodes_derived)

    # Now, we re-generate the fwd/bwd graphs.
    # NB: This might increase compilation time, but I doubt it matters
    fwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        primal_inputs + fwd_seed_offset_inputs,
        fwd_outputs + saved_values + saved_sym_nodes,
        fwd_outputs_descs
        + [
            SavedForBackwardsAOTOutput(i)
            for i in range(len(saved_values) + len(saved_sym_nodes))
        ],
        "forward",
    )
    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes
        + saved_values
        + tangent_inputs
        + bwd_seed_offset_inputs
        + backward_state_inputs,
        bwd_outputs,
        bwd_outputs_descs,
        "backward",
    )

    fwd_module = fx._lazy_graph_module._make_graph_module(joint_module, fwd_graph)
    bwd_module = fx._lazy_graph_module._make_graph_module(joint_module, bwd_graph)
    enable_activation_quantization(
        saved_values, fwd_module, bwd_module, static_lifetime_input_nodes
    )
    return fwd_module, bwd_module


def default_partition(
    joint_module: fx.GraphModule,
    _joint_inputs,
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: Optional[list[int]] = None,
    static_lifetime_input_nodes: Optional[OrderedSet[fx.Node]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the :attr:`joint_module` in a manner that closely resembles the
    behavior observed in the original ``.forward()`` and ``.backward()`` of the
    callable, i.e., the resulting forward graph contains those operators that
    are executed in the original ``.forward()`` callable passed to
    :func:`aot_function`.

    The default partitioner collects the operators that are between the forward
    inputs and the forward outputs. This helps in finding the tensors which have
    to be stashed for the backward pass. These stashed tensors become the output
    of the generated forward graph. The remaining operators are then placed in
    the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
    if has_recomputable_ops(joint_module):
        return min_cut_rematerialization_partition(
            joint_module,
            _joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
        )
    primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
    fwd_seed_offset_inputs = list(filter(_is_fwd_seed_offset, joint_module.graph.nodes))
    inputs = primal_inputs + fwd_seed_offset_inputs
    fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs = (
        _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    )
    forward_only_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph, inputs, fwd_outputs, fwd_outputs_descs, "forward"
    )
    forward_node_names = OrderedSet(
        node.name for node in forward_only_graph.nodes if node.op != "output"
    )
    saved_values = []
    saved_sym_nodes = []

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue
        if is_sym_node(node):
            # Symints must be kept separate from tensors so that PythonFunction only calls
            # save_for_backward on tensors and stashes symints in autograd .ctx
            saved_sym_nodes.append(node)
        elif "tensor_meta" not in node.meta and node.op == "call_function":
            # Since we can't save tuple of tensor values, we need to flatten out what we're saving
            users = node.users
            assert all(user.target == operator.getitem for user in users)
            saved_values.extend(users)
        else:
            backward_usages = [
                n for n in node.users if n.name not in forward_node_names
            ]
            if "tensor_meta" in node.meta and all(
                is_sym_node(n) for n in backward_usages
            ):
                # If we have a tensor in the forward, where only its sizes/strides are needed in the backward,
                # and not the actual tensor data,
                # then it will be a lot cheaper to save only the sizes/strides, and not the actual tensor.
                #
                # Note that saving the tensor could also cause compilation problems:
                # If the user mutated an input in the forward and uses its sizes/strides in the backward,
                # then we would be obligated to clone the input before saving it to appease autograd.
                # (This is how we originally found this bug).
                saved_sym_nodes.extend(backward_usages)
            else:
                saved_values.append(node)
    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())

    return _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_nodes=static_lifetime_input_nodes,
    )


INT_INF = int(1e6)


def _tensor_nbytes(numel: int, dtype) -> int:
    return numel * dtype.itemsize


def _size_of(node: fx.Node) -> int:
    def object_nbytes(x) -> int:
        if not isinstance(x, torch.Tensor):
            return 0
        return _tensor_nbytes(hint_int(x.numel(), fallback=4096), x.dtype)

    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, py_sym_types):
            return 1
        # NB: The fallback values here are meaningless, maybe we should respect
        # torch._inductor.config.unbacked_symint_fallback (but this is a
        # layering violation)
        elif isinstance(val, (list, tuple)):
            return sum(object_nbytes(n) for n in val)
        elif isinstance(val, dict):
            return sum(object_nbytes(n) for _, n in val.items())
        elif isinstance(val, torch.Tensor):
            return object_nbytes(val)

        raise RuntimeError(f"Unknown metadata type {type(val)} on node {node}")
    if node.op == "get_attr" or node.target is torch.ops.aten._assert_scalar.default:
        return 0
    raise RuntimeError(
        f"Node {node} didn't have `val` metadata; we should always have `val` metadata on the nodes."
    )


# Used for some investigative purposes
def _count_ops(graph: fx.Graph):
    from collections import defaultdict

    cnt: dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        if node.op == "call_function":
            cnt[node.target.__name__] += 1
    log.info("%s", sorted(cnt.items(), key=operator.itemgetter(1), reverse=True))


@functools.cache
def pointwise_ops():
    ops = []
    for attr_name in dir(torch.ops.aten):
        opoverloadpacket = getattr(torch.ops.aten, attr_name)
        if not isinstance(opoverloadpacket, torch._ops.OpOverloadPacket):
            continue

        for overload in opoverloadpacket.overloads():
            op_overload = getattr(opoverloadpacket, overload)
            if torch.Tag.pointwise in op_overload.tags:
                # currently aot autograd uses packet not overload
                ops.append(opoverloadpacket)
                break

    return ops


def sort_depths(args, depth_map: dict[fx.Node, int]) -> list[tuple[fx.Node, int]]:
    arg_depths = {
        arg: depth_map[arg] for arg in args if isinstance(arg, torch.fx.node.Node)
    }
    return sorted(arg_depths.items(), key=operator.itemgetter(1), reverse=True)


def reordering_to_mimic_autograd_engine(gm: fx.GraphModule) -> fx.GraphModule:
    """
    This pass finds the first bwd node in the graph (by looking at users of
    tangents) and then reorders the graph by walking from this node to all the
    way to the end of the graph. At each op in this traversal, we insert this op
    in a new graph and try to bring only the relevant subgraph from the other
    non-bwd edges relevant for this op. This closely mimics the behavior of
    autograd engine.

    Why is this pass required in the first place?

    This is an artifact of how partitioners work today. The starting point of
    partitioner is a joint graph, which is fwd and then bwd graph. In the case
    of checkpointing, we keep portions of fwd graph in their original place in
    the joint graph, while obtaining a bwd graph. As a result, the resulting bwd
    graph has copies of recomputed fwd subgraphs followed by the original bwd
    graph. If we run this naively, this leads to bad memory footprint, because
    the fwd subgraphs are live for way longer duration than necessary. This pass
    reorders the operations such that we prioritize the ops for the original bwd
    graph while only realizing those ops from the fwd graph that are necessary
    at any given point in the graph.
    """

    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    order = {node: idx for idx, node in enumerate(gm.graph.nodes)}

    def insert_node_in_graph(node):
        cur_nodes = [node]
        insertable_nodes: OrderedSet[fx.Node] = OrderedSet()
        while len(cur_nodes) > 0:
            node = cur_nodes.pop()
            if node in insertable_nodes or node in env:
                continue
            insertable_nodes.add(node)

            # Bias traversal towards the nodes that have higher depth - prioritizes
            # critical path first.
            cur_nodes += node.all_input_nodes

        insertable_nodes = sorted(insertable_nodes, key=lambda n: order[n])
        for node in insertable_nodes:
            env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Find first bwd node in the graph
    tangent_inputs = list(filter(_is_tangent, gm.graph.nodes))
    first_node_in_bwd = None
    minimum_order = math.inf
    for tangent in tangent_inputs:
        for user in tangent.users:
            if order[user] < minimum_order:
                minimum_order = order[user]
                first_node_in_bwd = user

    # If gradInp does not depend upon gradOut, we may not find any nodes in the "backwards pass"
    if first_node_in_bwd is None:
        return gm

    # Build the graph op-by-op by starting from the node all the way to the end
    # copy_ can be not using tangents at all, we must copy it.
    for node in list(gm.graph.nodes)[: order[first_node_in_bwd]]:
        if node.op == "call_function" and node.target == torch.ops.aten.copy_.default:
            insert_node_in_graph(node)

    for node in list(gm.graph.nodes)[order[first_node_in_bwd] :]:
        insert_node_in_graph(node)

    # The output node is already built by the traversal.
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm


def apply_graphsafe_rng_functionalization(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    fw_node: torch.fx.Node,
    bw_node: torch.fx.Node,
    device: torch.device,
    rng_count: int,
    last_fwd_input: torch.fx.Node,
    last_bwd_input: torch.fx.Node,
):
    """
    Note [CUDA Graph Safe RNG Functionalization]

    CUDA Graph capture doesn't work with get_rng_state and set_rng_state because these functions operate on CPU values,
    while CUDA Graph RNG capture uses on-device CUDA tensors. To solve this, we use graphsafe_set_state with a
    CUDA Generator registered to the CUDA Graph before capture begins. graphsafe_set_state updates the generator's pointer
    to reference a different GeneratorImpl, ensuring subsequent calls are correctly forwarded to the desired generator
    (and its cuda-tensor RNG state during graph capture).

    For each RNG operation's forward/backward pair:

    - We create two generators initialized with identical values
    - Each forward and backward call advances its respective generator equally
    - This keeps generators synchronized so forward and backward operations use matching RNG values

    When forward is called multiple times before backward (causing desynchronization):

    - We save the forward RNG state
    - We update the backward Generator's state before executing backward

    Before each CUDA Graph replay, replay_prologue updates captured RNG pointers with current states, ensuring backward Generator
    changes are reflected during replay.

    This function modifies both forward and backward computation graphs by:

    Creating RNG state placeholders for both passes
    Updating the forward node to use graph-safe RNG state
    Updating the backward node to use graph-safe RNG state

    For more details: https://github.com/pytorch/pytorch/issues/113541
    """
    device_idx = device.index
    assert device_idx is not None
    fw_graph = fw_module.graph
    bw_graph = bw_module.graph
    graphsafe_run_with_rng_state = torch._prims.rng_prims.graphsafe_run_with_rng_state

    # Handle forward pass

    # Note: [Generator arguments in AOTDispatcher]
    # Generator arguments in AOTDispatcher are added to support graphsafe rng
    # functionalization. See note above [CUDA Graph Safe RNG Functionalization]
    with fw_module.graph.inserting_after(last_fwd_input):
        fwd_rng_state = fw_module.graph.placeholder(f"fwd_rng_state_{rng_count}")
        fwd_rng_state.meta["val"] = get_cuda_generator_meta_val(device_idx)
        last_fwd_input = fwd_rng_state

    # Handle backward pass
    with bw_module.graph.inserting_after(last_bwd_input):
        bwd_rng_state = bw_module.graph.placeholder(f"bwd_rng_state_{rng_count}")
        # as above, clone so that meta val generator will not contain tensors
        bwd_rng_state.meta["val"] = get_cuda_generator_meta_val(device_idx)
        last_bwd_input = bwd_rng_state

    # Update forward node
    fw_kwargs = dict(fw_node.kwargs)
    fw_kwargs["rng_state"] = fwd_rng_state
    with fw_module.graph.inserting_after(fw_node):
        functional_fw_node = fw_graph.create_node(
            "call_function",
            graphsafe_run_with_rng_state,
            args=(fw_node.target, *fw_node.args),  # type: ignore[arg-type]
            kwargs=fw_kwargs,
        )
    fw_node.replace_all_uses_with(functional_fw_node)
    fw_graph.erase_node(fw_node)

    # Update backward node
    bwd_kwargs = dict(bw_node.kwargs)
    bwd_kwargs["rng_state"] = bwd_rng_state
    with bw_graph.inserting_before(bw_node):
        rng_output = bw_graph.create_node(
            "call_function",
            graphsafe_run_with_rng_state,
            args=(bw_node.target, *bw_node.args),  # type: ignore[arg-type]
            kwargs=bwd_kwargs,
        )
        bw_node.replace_all_uses_with(rng_output)
        bw_graph.erase_node(bw_node)

    return last_fwd_input, last_bwd_input


def functionalize_rng_ops(
    joint_module: fx.GraphModule,
    fw_module: fx.GraphModule,
    bw_module: fx.GraphModule,
    num_sym_nodes: int,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    # During user-driven activation checkpointing, we have to ensure that a rng
    # op in fwd yields the same output as the recomputed rng op in the bwd.  To
    # do this, we use functionalize wrappers to wrap the random ops and share
    # rng state between the fwd and bwd graphs.

    # There are 3 main steps to do this
    # Step 1 - Construct a mapping of rng node between the fwd and its counterpart in bwd.
    # Step 2 - Modify the fwd pass such that
    #   1) Replace rand with run_and_save_rng_state wrapper
    #   2) Replace the users of the original op with the output[1] of this op.
    #   3) Collect all the rng_state - output[0] of each op, and make them
    #   output nodes. Special care needs to be taken here because fwd outputs
    #   has symints at the very end.
    # Step 3 - Modify the bwd pass such that
    #   1) Add the input nodes just before the tangents for the stashed rng states
    #   2) Replace rand with run_with_save_rng_state wrappers
    #   3) Use the stashed states as inputs to these ops

    # Unique id to generate name
    uid = itertools.count()

    def get_rng_ops(gmod):
        random_nodes = {}
        for node in gmod.graph.nodes:
            if (
                node.op == "call_function"
                and hasattr(node.target, "tags")
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                random_nodes[node.name] = node
        return random_nodes

    def get_device(node) -> Optional[torch.device]:
        """
        Check the example value of the node outputs to find the device type.
        """
        if "val" not in node.meta:
            return None

        candidates = node.meta["val"]
        if not isinstance(candidates, tuple):
            candidates = (candidates,)

        for candidate in candidates:
            if isinstance(candidate, torch.Tensor):
                if candidate.device.type == "cuda":
                    return candidate.device

        return torch.device("cpu")

    def get_sample_rng_state(device: Optional[torch.device]):
        from torch._guards import detect_fake_mode  # noqa: F401

        fake_mode = detect_fake_mode()
        assert fake_mode is not None
        with fake_mode:
            if device is not None and device.type == "cuda":
                return fake_mode.from_tensor(torch.cuda.get_rng_state())
            return fake_mode.from_tensor(torch.get_rng_state())

    # Step 1 - Construct a mapping of rng node between the fwd and its counterpart in bwd.
    joint_graph_rng_ops = get_rng_ops(joint_module)
    fw_graph_rng_ops = get_rng_ops(fw_module)
    bw_graph_rng_ops = get_rng_ops(bw_module)
    recomputable_rng_ops_map = {}
    for node in joint_module.graph.nodes:
        if (
            must_recompute(node)
            and hasattr(node.target, "tags")
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            base_node = joint_graph_rng_ops[node.name]
            fw_node = fw_graph_rng_ops[node.name]
            bw_node = bw_graph_rng_ops[node.name]
            recomputable_rng_ops_map[base_node] = {"fwd": fw_node, "bwd": bw_node}

    run_and_save_rng = torch._prims.rng_prims.run_and_save_rng_state
    run_with_rng_state = torch._prims.rng_prims.run_with_rng_state

    bw_tangent_start_node = None
    for node in bw_module.graph.find_nodes(op="placeholder"):
        if "tangent" in node.name:
            bw_tangent_start_node = node
            break
    if bw_tangent_start_node is None:
        raise RuntimeError(
            "Couldn't find tangent node in graph inputs. This is unexpected, please file a bug if you see this"
        )

    fw_rng_state_outputs = []

    last_fwd_input = next(reversed(fw_module.graph.find_nodes(op="placeholder")))
    last_bwd_input = next(reversed(bw_module.graph.find_nodes(op="placeholder")))

    devices = OrderedSet(
        get_device(node_pair["fwd"]) for node_pair in recomputable_rng_ops_map.values()
    )
    devices.discard(torch.device("cpu"))
    # multiple cuda devices won't work with cudagraphs anyway,
    # fallback to non graphsafe rng checkpointing
    multi_cuda_devices = len(devices) > 1

    # this changes numerics, so if fallback_random is set we will not use it
    ind_config = torch._inductor.config
    use_rng_graphsafe_rng_functionalization = (
        config.graphsafe_rng_functionalization
        and not multi_cuda_devices
        and (
            not ind_config.fallback_random
            or ind_config.test_configs.graphsafe_rng_func_ignores_fallback_random
        )
    )

    for rng_count, (base_node, node_pair) in enumerate(
        recomputable_rng_ops_map.items()
    ):
        # Step 2 - Modify the fwd pass such that
        fw_node = node_pair["fwd"]
        bw_node = node_pair["bwd"]
        device = get_device(fw_node)

        fw_graph = fw_module.graph
        bw_graph = bw_module.graph

        if (
            use_rng_graphsafe_rng_functionalization
            and device is not None
            and device.type == "cuda"
        ):
            last_fwd_input, last_bwd_input = apply_graphsafe_rng_functionalization(
                fw_module,
                bw_module,
                fw_node,
                bw_node,
                device,
                rng_count,
                last_fwd_input,
                last_bwd_input,
            )
        else:
            with fw_graph.inserting_before(fw_node):
                functional_fw_node = fw_graph.create_node(
                    "call_function",
                    run_and_save_rng,
                    args=(fw_node.target, *fw_node.args),
                    kwargs=fw_node.kwargs,
                )
                state = fw_graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(functional_fw_node, 0),
                    kwargs={},
                )
                state.meta["val"] = get_sample_rng_state(device)

                rng_output = fw_graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(
                        functional_fw_node,
                        1,
                    ),
                    kwargs={},
                )
                # Copy the meta data from the original node
                rng_output.meta = copy.copy(fw_node.meta)

                fw_node.replace_all_uses_with(rng_output)
                fw_graph.erase_node(fw_node)
                fw_rng_state_outputs.append(state)

            # Step 3 - Modify the bwd pass such that
            with bw_graph.inserting_before(bw_tangent_start_node):
                state_name = f"rng_state_output_{next(uid)}"
                bw_rng_state_node = bw_graph.placeholder(state_name)
                bw_rng_state_node.meta["val"] = get_sample_rng_state(device)

            with bw_graph.inserting_before(bw_node):
                rng_output = bw_graph.create_node(
                    "call_function",
                    run_with_rng_state,
                    args=(bw_rng_state_node, bw_node.target, *bw_node.args),
                    kwargs=bw_node.kwargs,
                )

                bw_node.replace_all_uses_with(rng_output)
                bw_graph.erase_node(bw_node)

    # Add the rng states in the output of the fwd graph. AOT Autograd assumes
    # that symints are at the end of forward graph outputs. So, insert the new
    # rng states accordingly.
    if fw_rng_state_outputs:
        fw_output_node = next(iter(fw_module.graph.find_nodes(op="output")))
        fw_outputs = fw_output_node.args[0]
        sym_node_start_idx = len(fw_outputs) - num_sym_nodes
        outputs = (
            fw_outputs[:sym_node_start_idx]
            + tuple(fw_rng_state_outputs)
            + fw_outputs[sym_node_start_idx:]
        )
        fw_module.graph.output(outputs)
        fw_module.graph.erase_node(fw_output_node)
    fw_module.recompile()
    bw_module.recompile()
    return fw_module, bw_module


def force_save_collectives(joint_module: fx.GraphModule) -> None:
    """
    By default, the partitioner is not allowed to recompute collectives
    unless they come from a user-annotated AC region.
    See Note [Recomputing collectives in the partitioner]
    """
    for node in joint_module.graph.nodes:
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace == "_c10d_functional"
            and not must_recompute(node)
        ):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE


def force_save_bw_mutation_src(joint_module: fx.GraphModule) -> None:
    # If we have mutations of the same primal in forward and backward,
    # We must not recompute the source of mutation to not apply twice.
    has_mutation_in_bw: OrderedSet[torch.fx.Node] = OrderedSet()
    for node in reversed(joint_module.graph.nodes):
        if node.op == "output":
            continue

        is_copy_ = node.target == torch.ops.aten.copy_.default
        if is_copy_:
            if _has_tag_must_be_in_backward(node):
                has_mutation_in_bw.add(node.args[0])

            if _has_tag_must_be_in_forward(node) and node.args[0] in has_mutation_in_bw:
                node.args[1].meta["recompute"] = CheckpointPolicy.MUST_SAVE
        else:
            # We use invariant of aotdispatch joint graph,
            # That we emit copy_ only in the end of it.
            # We do not want to iterate through all the joint graph,
            # so break at the first non-output, non-copy_ node.
            break


def cleanup_recompute_tags(joint_module: fx.GraphModule) -> fx.GraphModule:
    """
    If there are two consecutive checkpointed blocks with no operator in
    between, we would still want to stash the tensor at the boundary of
    checkpointed blocks. The following pass makes the last output node
    non-recomputable to allow for that.
    """
    for node in joint_module.graph.nodes:
        if must_recompute(node):
            for user in node.users:
                if (
                    must_recompute(user)
                    and user.meta["ac_graph_id"] > node.meta["ac_graph_id"]
                ):
                    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            if node.meta.get("has_backward_hook", False) and not any(
                must_recompute(user) for user in node.users
            ):
                # If node is AC region output and has a backward hook on it, we intentionally choose to save it.
                # This is to work around circular dependencies in Traceable FSDP2+AC.
                # Example:
                # ```
                # out = fully_shard(utils.checkpoint(module))(x)
                # norm_out = layer_norm(out)
                # ```
                # Here there is a circular dependency:
                # 1. In backward, grad_input of layer_norm aka. `out_grad` is actually dependent on `out`.
                # 2. `out` depends on `out`'s backward hook created by FSDP2 (which does all-gather for `module` weights)
                #    in order to be recomputed.
                # 3. `out`'s backward hook, as is the case for all eager backward hooks, depends on `out_grad`
                #    -> circular dependency with (1)!
                #
                # Solution: check whether `out` has a backward hook, and if so, intentionally save `out`
                # in forward graph outputs. With this, we can break the above circular dependency.
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    return joint_module


def solve_min_cut(
    joint_graph: fx.Graph,
    node_info: NodeInfo,
    min_cut_options: MinCutOptions,
    dont_ban: Optional[OrderedSet[fx.Node]] = None,
):
    if dont_ban is None:
        dont_ban = OrderedSet()
    op_types = get_default_op_list()

    if AOT_PARTITIONER_DEBUG:
        joint_module_ops = OrderedSet(
            str(node.target._overloadpacket)
            for node in joint_graph.nodes
            if node.op == "call_function" and hasattr(node.target, "_overloadpacket")
        )
        ops_ignored = joint_module_ops - OrderedSet(
            str(i) for i in op_types.recomputable_ops
        )
        log.info("Ops banned from re-materialization: %s", ops_ignored)

    def can_fuse_into_auto_functionalized(a, b):
        if b.target != torch.ops.higher_order.auto_functionalized:
            return False
        mutable_op = b.args[0]
        (
            mutable_arg_names,
            _,
        ) = torch._higher_order_ops.auto_functionalize.get_mutable_args(mutable_op)
        for name in mutable_arg_names:
            arg = b.kwargs[name]
            if a is arg:
                return True
            if isinstance(arg, list):
                if a in arg:
                    return True
        return False

    def can_fuse_into_triton_kernel_wrapper_functional(a, b):
        if b.target != torch.ops.higher_order.triton_kernel_wrapper_functional:
            return False
        mutable_arg_names = b.kwargs["tensors_to_clone"]
        for name in mutable_arg_names:
            arg = b.kwargs["kwargs"][name]
            if a is arg:
                return True
        return False

    def is_fusible(a, b):
        # We can perform "memory fusion" into a cat, but cat cannot be a
        # producer to a fusion
        if get_aten_target(b) == aten.cat:
            return True
        if can_fuse_into_auto_functionalized(a, b):
            return True
        if can_fuse_into_triton_kernel_wrapper_functional(a, b):
            return True
        if (
            a.target is operator.getitem
            and a.args[0].target
            is torch.ops.higher_order.triton_kernel_wrapper_functional
        ):
            # if a is the output of a user triton kernel,
            # then (by default) we will not be able to fuse b into it
            return False
        return op_types.is_fusible(a) and op_types.is_fusible(b)

    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "Need networkx installed to perform smart recomputation heuristics"
        ) from e

    def is_materialized_backwards(node):
        if op_types.is_view(node):
            return False
        cur_nodes = OrderedSet([node])
        while len(cur_nodes) > 0:
            cur = cur_nodes.pop()
            for user in cur.users:
                if not node_info.is_required_fw(user) and not is_fusible(cur, user):
                    return True
                if op_types.is_view(user):
                    cur_nodes.add(user)

        return False

    def should_ban_recomputation(node):
        if node.op != "call_function":
            return False
        if node.target == operator.getitem:
            return False
        if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:
            return True
        if config.recompute_views and op_types.is_view(node):
            return False
        if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
            return False

        if min_cut_options.ban_if_not_in_allowlist:
            if not op_types.is_recomputable(node):
                return True
        else:
            if op_types.is_random(node) or op_types.is_compute_intensive(node):
                return True

        # If a node *must* be materialized in the backwards pass, then we
        # should never recompute it. This is a pretty subtle point.  In
        # general, the assumption we make is that recomputing a node in the
        # backwards pass is "free". However, if a node must be materialized
        # in the backwards pass, then recomputing it is never free.
        if min_cut_options.ban_if_materialized_backward and is_materialized_backwards(
            node
        ):
            log.debug("materialized backwards: %s %s", node, tuple(node.users))
            return True

        # Arbitrary hack that sometimes seems to help things. The above
        # modification appears to have made this heuristic a lot less critical
        # for performance.
        # NB: As of PR #121692, this hack no longer seems necessary.
        if node.dist_from_bw < 1000 and node.dist_from_bw > config.max_dist_from_bw:
            return True

        # If the output of an op is 4x smaller (arbitrary choice),
        # then we don't allow recomputation. The idea here is that for
        # things like reductions, saving the output of the reduction is very
        # cheap/small, and it makes sure we don't do things like recompute
        # normalizations in the backwards.
        if min_cut_options.ban_if_reduction:
            input_tensors_size = sum(
                _size_of(i) for i in node.args if isinstance(i, fx.Node)
            )
            output_size = _size_of(node)
            return output_size * 4 < input_tensors_size
        return False

    def is_materialized(node):
        if node.op == "placeholder":
            return True

        return not all(is_fusible(node, user) for user in node.users)

    def get_node_weight(node, static_lifetime_input_nodes) -> float:
        if (
            config.treat_parameters_as_free_to_save
            and node in static_lifetime_input_nodes
        ):
            return 0
        mem_sz = _size_of(node)
        if config.recompute_views and op_types.is_view(node):
            # If `config.recompute_views=True`, we don't save views. This is generally
            # a good idea since views are free to recompute, and it makes it a bit simpler
            # to analyze.
            # NB: If they're not free to recompute (e.g. nested tensors)... I
            # think we should modify checks for view_ops to `is_view` and check
            # that. Basically, with nested tensors, `aten.view` is not a "view
            # op".
            return math.inf

        if isinstance(node.meta["val"], py_sym_types):
            # We never want to save symfloats
            if not isinstance(node.meta["val"], torch.SymInt):
                return INT_INF

        # Heuristic to bias towards nodes closer to the backwards pass
        # Complete guess about current value
        mem_sz = int(mem_sz * (1.1 ** max(min(node.dist_from_bw, 100), 1)))
        if is_materialized(node):
            return mem_sz
        else:
            return mem_sz * 2

    nx_graph = nx.DiGraph()
    banned_nodes: OrderedSet[fx.Node] = OrderedSet()

    def ban_recomputation_if_allowed(node):
        if op_types.is_view(node):
            return False
        if node in dont_ban:
            # collectives are *always* banned from recompute, overriding `dont_ban`
            # (in particular, the activation memory budget logic is not allowed to recompute collectives)
            is_collective = (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace == "_c10d_functional"
            )
            if config.unsafe_allow_optimization_of_collectives or not is_collective:
                return False
        # This bans recomputation of the node unless we've been forced not to by
        # user annotation
        if must_recompute(node):
            return False

        if "val" in node.meta and isinstance(node.meta["val"], torch.SymFloat):
            return False
        banned_nodes.add(node)
        # A node will only ever be recomputed if there is a path from an
        # ancestor of this node to the backwards path through this node that
        # doesn't go through any saved value. If this node is saved, then that
        # condition is not possible.
        nx_graph.add_edge("source", node.name + "_in", capacity=math.inf)
        return True

    for node in joint_graph.nodes:
        if node.op == "output":
            continue

        if node in node_info.required_bw_nodes:
            if node not in node_info.inputs:
                nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
                continue
            # If someone saves a input for backward as-is and backward
            # returns that tensor as-is as a grad input, then the node x would
            # be both a required_bw_node and an input. In this case we
            # (1) connect x_in to to the source, (2) x_out to the sink, and
            # (3) assign the proper weight to the x_in-x_out edge, so that
            # x would be part of cut nodes. A case where this happens is if
            # NestedTensor saves a offset tensor as part of the singleton int
            # in sizes.
            nx_graph.add_edge(node.name + "_out", "sink", capacity=math.inf)

        if must_recompute(node):
            # If user explicitly says they want to recompute a node, we honor it
            # by adding an inf-capacity edge from X_in to the sink.
            # This way, X_in node is guaranteed to be part of the subgraph that contains "sink"
            # after the cut, thus guaranteeing that X op will be recomputed.
            nx_graph.add_edge(node.name + "_in", "sink", capacity=math.inf)
            continue

        if _is_primal(node) or _is_fwd_seed_offset(node):
            ban_recomputation_if_allowed(node)

        # If a node can't be recomputed (too expensive or involves randomness),
        # we prevent it from being recomputed by adding an inf edge to the source
        # We only need to ban nodes in the fw pass, as those are the only ones that would be recomputed.
        if node_info.is_required_fw(node) and should_ban_recomputation(node):
            ban_recomputation_if_allowed(node)

        # Checks if a node is actually a tuple. Can be simplified to just an isinstance check if we always use faketensors.
        is_non_tensor_node = (
            "val" not in node.meta and "tensor_meta" not in node.meta
        ) or ("val" in node.meta and not isinstance(node.meta["val"], torch.Tensor))

        if is_sym_node(node):
            weight = float(sym_node_size(node))
        elif is_non_tensor_node:
            weight = (
                0.0 if isinstance(node.meta.get("val"), BackwardState) else math.inf
            )
        else:
            weight = get_node_weight(node, node_info.static_lifetime_input_nodes)
        # Creates the weights on the "node" edge
        nx_graph.add_edge(node.name + "_in", node.name + "_out", capacity=weight)
        for user in node.users:
            nx_graph.add_edge(node.name + "_out", user.name + "_in", capacity=math.inf)

    # todo(chilli): This is the most questionable of the 3 heuristics for banning recompute.
    # Some example models to look at where this helps perf: poolformer_m36,
    # mixer_b16_224, cait_m36_384

    # The "rough" idea here is that if you have some node that is used by both a
    # node nearby downstream as well as a node far downstream, if we recompute
    # both of the downstream nodes, we're unlikely to be able to fuse both
    # downstream nodes together.

    # Thus, we shouldn't aim to recompute far downstream nodes that depend on
    # this node. That intuition of "far downstream" is captured by whether
    # there's an unfusible op along the chain somewhere

    # It could probably be improved by properly analyzing what's going on in the
    # backwards pass instead of only relying on whether it's unfusible in the
    # forwards.

    def find_first_unfusible(start_nodes: list[fx.Node], max_range: int) -> int:
        """
        Finds the first unfusible node in the chain of nodes starting from
        `start_nodes` and returns its position.
        """
        sorted_nodes: list[tuple[int, fx.Node, bool]] = []
        for n in start_nodes:
            heapq.heappush(sorted_nodes, (node_info.get_fw_order(n), n, True))

        while len(sorted_nodes) > 0:
            _, node, node_is_fusible = heapq.heappop(sorted_nodes)
            if not node_is_fusible:
                return node_info.get_fw_order(node)
            for user in node.users:
                if node_info.is_required_fw(user):
                    if node_info.get_fw_order(user) > max_range:
                        continue
                    val: tuple[int, fx.Node, bool] = (
                        node_info.get_fw_order(user),
                        user,
                        is_fusible(node, user),
                    )
                    if val not in sorted_nodes:
                        heapq.heappush(sorted_nodes, val)
        return max_range

    if min_cut_options.ban_if_used_far_apart:
        for used_node in node_info.required_fw_nodes:
            orders = [
                node_info.get_fw_order(user)
                for user in used_node.users
                if node_info.is_required_fw(user)
            ]
            fw_users = [
                user for user in used_node.users if node_info.is_required_fw(user)
            ]
            if len(orders) > 0:
                first_unfusible_use = find_first_unfusible(fw_users, max(orders))
                for user in tuple(used_node.users):
                    if (
                        node_info.is_required_fw(user)
                        and node_info.get_fw_order(user) > first_unfusible_use
                        and is_fusible(used_node, user)
                    ):
                        if user in banned_nodes:
                            continue
                        log.info(
                            "used above/below fusible %s:(%s) -> %s -> %s:(%s)",
                            used_node,
                            node_info.get_fw_order(used_node),
                            first_unfusible_use,
                            user,
                            node_info.get_fw_order(user),
                        )
                        ban_recomputation_if_allowed(user)

    # This heuristic is fairly straightforward. The idea is that although it is
    # cheap to recompute bandwidth-bound ops, we don't want to end up in a situation
    # where we have a long chain of pointwise ops from the beginning to the end
    # of the model (like say, residual connections)

    # todo: I'm not totally sure why this heuristic matters. It's possible that this is
    # working around Inductor fusion decisions, or that it's a patch over
    # suboptimal partitioning decisions

    # Some models it improves perf on are cait_m36_384, mixer_b16_224, poolformer_m36

    if min_cut_options.ban_if_long_fusible_chains:
        visited: OrderedSet[fx.Node] = OrderedSet()
        for start_node in joint_graph.nodes:
            if not node_info.is_required_fw(start_node):
                continue
            fusible: list[tuple[int, fx.Node]] = [
                (node_info.get_fw_order(start_node), start_node)
            ]
            start_order = node_info.get_fw_order(start_node)
            while len(fusible) > 0:
                _, cur = heapq.heappop(fusible)
                if cur in visited:
                    continue
                visited.add(cur)
                # 100 is arbitrary choice to try and prevent degenerate cases
                if (
                    node_info.get_fw_order(cur) > start_order + 100
                    and len(fusible) == 0
                ):
                    log.info(
                        "too long %s %s %s %s",
                        cur,
                        start_node,
                        node_info.get_fw_order(cur),
                        node_info.get_fw_order(start_node),
                    )
                    ban_recomputation_if_allowed(cur)
                    break

                for user in cur.users:
                    if (
                        node_info.is_required_fw(user)
                        and is_fusible(cur, user)
                        and user not in banned_nodes
                    ):
                        heapq.heappush(fusible, (node_info.get_fw_order(user), user))

    try:
        cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
    except Exception:
        log.info("Failed to compute min-cut on following graph:")
        log.info("\n".join(nx.readwrite.edgelist.generate_edgelist(nx_graph)))
        visualize_min_cut_graph(nx_graph)
        raise

    reachable, non_reachable = partition
    cutset: OrderedSet[tuple[str, str]] = OrderedSet()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes: OrderedSet[str] = OrderedSet()
    for node_in, node_out in cutset:
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)

    name_to_node = get_name_to_node(joint_graph)
    # To make this stuff deterministic
    node_idx = {node: idx for idx, node in enumerate(joint_graph.nodes)}
    saved_values = sorted(
        (name_to_node[node] for node in cut_nodes), key=lambda x: node_idx[x]
    )
    return saved_values, banned_nodes


def visualize_min_cut_graph(nx_graph):
    import networkx as nx
    import pydot

    dot_format = nx.nx_pydot.to_pydot(nx_graph).to_string()
    dot_graph = pydot.graph_from_dot_data(dot_format)[0]  # type: ignore[index]
    for edge in dot_graph.get_edges():
        weight = nx_graph[edge.get_source()][edge.get_destination()]["capacity"]
        # Set edge label to weight
        edge.set_label(str(weight))  # type: ignore[union-attr]
        # Color edges with weight 'inf' as red
        if weight == float("inf"):
            edge.set_color("red")  # type: ignore[union-attr]
    log.info("Visualizing the failed graph to min_cut_failed.svg")
    dot_graph.write_svg("min_cut_failed.svg")  # type: ignore[union-attr]


def get_default_op_list() -> OpTypes:
    default_recomputable_ops: list[Callable] = [
        aten.add,
        aten.sub,
        aten.div,
        aten.atan2,
        aten.mul,
        aten.max,
        aten.min,
        aten.pow,
        aten.remainder,
        aten.fmod,
        aten.__and__,
        aten.__or__,
        aten.__xor__,
        aten.__lshift__,
        aten.__rshift__,
        aten.eq,
        aten.ne,
        aten.ge,
        aten.gt,
        aten.le,
        aten.lt,
        aten.abs,
        aten.bitwise_not,
        aten.ceil,
        aten.floor,
        aten.frac,
        aten.neg,
        aten.relu,
        aten.round,
        aten.silu,
        aten.trunc,
        aten.log,
        aten.log10,
        aten.log1p,
        aten.log2,
        aten.lgamma,
        aten.exp,
        aten.expm1,
        aten.erf,
        aten.erfc,
        aten.cos,
        aten.acos,
        aten.cosh,
        aten.sin,
        aten.asin,
        aten.sinh,
        aten.tan,
        aten.atan,
        aten.tanh,
        aten.atanh,
        aten.sqrt,
        aten.rsqrt,
        aten.reciprocal,
        aten.sigmoid,
        aten.softplus,
        aten.threshold,
        aten.threshold_backward,
        aten.clamp,
        aten.where,
        aten.lerp,
        aten.addcmul,
        aten.gelu,
        aten.gelu_backward,
        aten.sum,
        aten.mean,
        aten._grad_sum_to_size,
        aten.sum_to_size,
        aten.amax,
        aten.to,
        aten.type_as,
        operator.getitem,
        aten.squeeze,
        aten.unsqueeze,
        aten.rsub,
        aten._to_copy,
    ]  # noqa: E501,B950
    recomputable_view_ops = [aten.squeeze, aten.unsqueeze, aten.alias]
    recomputable_view_ops += [
        aten.view,
        aten.slice,
        aten.t,
        prims.broadcast_in_dim,
        aten.expand,
        aten.as_strided,
        aten.permute,
        aten.select,
        aten.split,
    ]
    view_ops = recomputable_view_ops
    default_recomputable_ops += [
        prims.div,
        prims.convert_element_type,
        aten.clone,
        aten._to_copy,
        aten.full_like,
        prims.var,
        prims.sum,
        aten.var,
        aten.std,
        prims.broadcast_in_dim,
        aten.select,
        aten._unsafe_view,
        aten.view,
        aten.expand,
        aten.slice,
        aten.reshape,
        aten.broadcast_tensors,
        aten.scalar_tensor,
        aten.ones,
        aten.new_zeros,
        aten.lift_fresh_copy,
        aten.arange,
        aten.triu,
        aten.var_mean,
        aten.isinf,
        aten.any,
        aten.full,
        aten.as_strided,
        aten.zeros,
        aten.empty,
        aten.empty_like,
        aten.argmax,
        aten.maximum,
        prims.iota,
        prims._low_memory_max_pool_offsets_to_indices,
    ]  # noqa: E501,B950
    # Natalia said that we should allow recomputing indexing :)
    default_recomputable_ops += [aten.index, aten.gather]
    default_recomputable_ops += view_ops

    default_recomputable_ops += pointwise_ops()

    default_recomputable_ops += [
        aten.zeros_like,
    ]

    default_recomputable_ops += [method_to_operator(m) for m in magic_methods]
    recomputable_ops = OrderedSet(default_recomputable_ops)

    random_ops = OrderedSet[Callable[..., Any]](
        [aten.native_dropout, aten.rand_like, aten.randn_like]
    )
    compute_intensive_ops = [
        aten.mm,
        aten.convolution,
        aten.convolution_backward,
        aten.bmm,
        aten.addmm,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_efficient_attention,
        aten._flash_attention_forward,
        aten._efficient_attention_forward,
        aten.upsample_bilinear2d,
        aten._scaled_mm,
    ]  # noqa: E501,B950

    fusible_ops = recomputable_ops | random_ops
    return OpTypes(
        fusible_ops,
        OrderedSet(compute_intensive_ops),
        random_ops,
        OrderedSet(view_ops),
        recomputable_ops,
    )


def get_name_to_node(graph: fx.Graph):
    name_to_node = {}
    for node in graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def _optimize_runtime_with_given_memory(
    joint_graph: fx.Graph,
    memory: list[float],
    runtimes: list[float],
    max_memory: float,
    node_info: NodeInfo,
    all_recomputable_banned_nodes: list[fx.Node],
) -> tuple[float, list[int], list[int]]:
    SOLVER = config.activation_memory_budget_solver
    if SOLVER == "greedy":
        return greedy_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "ilp":
        return ilp_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "dp":
        return dp_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "dynamic_memory_budget_dp":
        log.warning(
            "dynamic_memory_budget_dp is an experimental solver. "
            "It does not guarantee performance improvements. "
            "Additionally, it is not guaranteed to be stable."
        )
        graph_info_provider = GraphInfoProvider.inialize_from_graph(
            joint_graph=joint_graph,
            all_recomputable_banned_nodes=all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=memory,
            recorded_knapsack_input_runtimes=runtimes,
        )
        return dp_knapsack(
            memory,
            runtimes,
            KnapsackEvaluator(
                graph_info_provider=graph_info_provider,
            ).get_knee_point_memory_budget(
                knapsack_algo=dp_knapsack,
                max_mem_budget=max_memory,
            ),
        )
    elif callable(SOLVER):
        saved_node_idx, recomp_node_idx = SOLVER(
            memory, joint_graph, max_memory, node_info, all_recomputable_banned_nodes
        )
        return (0.0, saved_node_idx, recomp_node_idx)
    else:
        raise RuntimeError(f"Not aware of memory budget knapsack solver: {SOLVER}")


from torch.utils._mode_utils import no_dispatch


# replace symbols in size and strides with their hints without guarding.
def _remove_symbols_without_guarding(x: torch.Tensor, fallback: int) -> torch.Tensor:
    shape = list(x.shape)

    def realize_symbol(d):
        return hint_int(d, fallback=fallback)

    shape = [realize_symbol(s) for s in shape]
    stride = [realize_symbol(s) for s in x.stride()]
    return x.new_empty_strided(shape, stride=stride)


def estimate_runtime(node):
    RUNTIME_MODE = config.activation_memory_budget_runtime_estimator

    def materialize_arg(x):
        if isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.Tensor):
            return _remove_symbols_without_guarding(x.meta["val"], fallback=4096)
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymInt):
            return hint_int(x.meta["val"], fallback=4096)
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymFloat):
            return 1.0
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymBool):
            return True
        else:
            return x

    if RUNTIME_MODE == "testing":
        return 1

    elif RUNTIME_MODE == "profile":
        with no_dispatch():
            from torch._inductor.runtime.benchmarking import benchmarker

            args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
            ms = benchmarker.benchmark_gpu(lambda: node.target(*args, **kwargs))
            return ms

    elif RUNTIME_MODE == "flops":
        # todo(chilli): Normalize this to also return ms
        from torch.utils.flop_counter import FlopCounterMode

        args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
        with FlopCounterMode(display=False) as mode:
            node.target(*args, **kwargs)
        counted_flops = mode.get_total_flops()
        return max(counted_flops, 1)
    else:
        raise RuntimeError(f"Not aware of runtime estimator: {RUNTIME_MODE}")


def choose_saved_values_set(
    joint_graph: fx.Graph,
    node_info: NodeInfo,
    memory_budget=1,
) -> list[fx.Node]:
    if memory_budget > 1 or memory_budget < 0:
        raise RuntimeError(
            f"The valid ranges for memory budget are 0 <= m <= 1. The provided value is {memory_budget}"
        )
    min_cut_options = MinCutOptions(
        ban_if_used_far_apart=config.ban_recompute_used_far_apart,
        ban_if_long_fusible_chains=config.ban_recompute_long_fusible_chains,
        ban_if_materialized_backward=config.ban_recompute_materialized_backward,
        ban_if_not_in_allowlist=config.ban_recompute_not_in_allowlist,
        ban_if_reduction=config.ban_recompute_reductions,
    )

    if config.aggressive_recomputation:
        min_cut_options = replace(
            min_cut_options,
            ban_if_used_far_apart=False,
            ban_if_long_fusible_chains=False,
            ban_if_materialized_backward=False,
            ban_if_not_in_allowlist=False,
        )
    if memory_budget == 0:
        return node_info.inputs

    runtime_optimized_saved_values, _ = solve_min_cut(
        joint_graph,
        node_info,
        min_cut_options,
    )
    # return runtime_optimized_saved_values
    if memory_budget == 1:
        return runtime_optimized_saved_values

    def estimate_activations_size(saved_values: list[fx.Node]) -> float:
        return sum(map(_size_of, saved_values)) / 1e9

    min_act_size = estimate_activations_size(node_info.inputs)
    max_act_size = estimate_activations_size(runtime_optimized_saved_values)
    # The optimized choice is smaller than the inputs anyways
    if max_act_size <= min_act_size:
        return runtime_optimized_saved_values

    def get_normalized_size(sz):
        return (sz / 1e9) / (max_act_size - min_act_size)

    def get_mem_ratio(activations: list[fx.Node]):
        return (estimate_activations_size(activations) - min_act_size) / (
            max_act_size - min_act_size
        )

    more_aggressive_options = replace(
        min_cut_options,
        ban_if_used_far_apart=False,
        ban_if_long_fusible_chains=False,
        ban_if_materialized_backward=False,
    )
    more_aggressive_saved_values, _ = solve_min_cut(
        joint_graph, node_info, more_aggressive_options
    )
    if get_mem_ratio(more_aggressive_saved_values) < memory_budget:
        return more_aggressive_saved_values

    aggressive_options = replace(
        more_aggressive_options,
        ban_if_not_in_allowlist=False,
    )
    aggressive_recomputation_saved_values, banned_nodes = solve_min_cut(
        joint_graph, node_info, aggressive_options
    )

    if get_mem_ratio(aggressive_recomputation_saved_values) < memory_budget:
        return aggressive_recomputation_saved_values

    from torch._inductor.fx_utils import get_node_storage

    input_storages = OrderedSet(get_node_storage(node) for node in node_info.inputs)

    def get_recomputable_banned_nodes(
        banned_nodes: OrderedSet[fx.Node],
    ) -> list[fx.Node]:
        return [
            i
            for i in banned_nodes
            if (
                # Only allow recomputing nodes that are actually required for BW
                i.dist_from_bw < int(1e9)  # type: ignore[attr-defined]
                and get_node_storage(i) not in input_storages
            )
        ]

    recomputable_banned_nodes = get_recomputable_banned_nodes(banned_nodes)
    must_save_nodes = [
        i
        for i in recomputable_banned_nodes
        if i.meta.get("recompute", False) == CheckpointPolicy.MUST_SAVE
    ]
    recomputable_banned_nodes = [
        i for i in recomputable_banned_nodes if i not in must_save_nodes
    ]

    # default: runtime_optimized_saved_values
    # more aggressive: more_aggressive_saved_values
    # full aggressive: aggressive_recomputation_saved_values

    all_recomputable_banned_nodes = sorted(
        recomputable_banned_nodes, key=_size_of, reverse=True
    )
    if len(all_recomputable_banned_nodes) == 0:
        return node_info.inputs + must_save_nodes
    memories_banned_nodes = [
        get_normalized_size(_size_of(i)) for i in all_recomputable_banned_nodes
    ]
    runtimes_banned_nodes = [
        estimate_runtime(node) for node in all_recomputable_banned_nodes
    ]
    from torch.utils._mode_utils import no_dispatch

    def get_saved_values_knapsack(memory_budget, node_info, joint_graph):
        with no_dispatch():
            (
                expected_runtime,
                saved_node_idxs,
                recomputable_node_idxs,
            ) = _optimize_runtime_with_given_memory(
                joint_graph,
                memories_banned_nodes,
                runtimes_banned_nodes,
                max(memory_budget, 0),
                node_info,
                all_recomputable_banned_nodes,
            )
        dont_ban: OrderedSet[fx.Node] = OrderedSet()
        for idx in recomputable_node_idxs:
            # if idx in all_recomputable_banned_nodes:
            try:
                dont_ban.add(all_recomputable_banned_nodes[idx])
            except BaseException:  # noqa: B036
                pass

        assert dont_ban.issubset(all_recomputable_banned_nodes)

        saved_values, _ = solve_min_cut(
            joint_graph,
            node_info,
            aggressive_options,
            dont_ban,
        )
        if AOT_PARTITIONER_DEBUG:
            create_structured_trace_for_min_cut_info(
                joint_graph=joint_graph,
                all_recomputable_banned_nodes=all_recomputable_banned_nodes,
                saved_node_idxs=saved_node_idxs,
                recomputable_node_idxs=recomputable_node_idxs,
                expected_runtime=expected_runtime,
                memories_banned_nodes=memories_banned_nodes,
                runtimes_banned_nodes=runtimes_banned_nodes,
                min_cut_saved_values=saved_values,
            )
        return saved_values, expected_runtime

    if config.visualize_memory_budget_pareto:

        def estimate_for_budget(b):
            saved_values, expected_runtime = get_saved_values_knapsack(
                b, node_info=node_info, joint_graph=joint_graph
            )
            return (
                b,
                sum(runtimes_banned_nodes) - expected_runtime,
                get_mem_ratio(saved_values),
            )

        options = [estimate_for_budget(0.0), estimate_for_budget(1.0)]

        if options[0][1:] != options[1][1:]:
            bisects = [(options[0], options[1])]
            while bisects:
                lhs, rhs = bisects.pop()
                if rhs[0] - lhs[0] < 1e-3:
                    options.append(lhs)
                    options.append(rhs)
                    continue
                mid = estimate_for_budget((lhs[0] + rhs[0]) / 2)
                if mid[1:] != lhs[1:]:
                    bisects.append((lhs, mid))
                if mid[1:] != rhs[1:]:
                    bisects.append((mid, rhs))
        options.sort()

        import matplotlib.pyplot as plt

        x_values = [item[2] for item in options]
        y_values = [item[1] for item in options]

        # Plotting the values with updated axis labels and chart title
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker="o")

        # Adding labels for each point
        for i, txt in enumerate(x_values):
            plt.annotate(
                f"{txt:.4f}",
                (txt, y_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        plt.xlabel("Memory Budget")
        plt.ylabel("Runtime of Recomputed Components")
        plt.title("Pareto Frontier of Memory Budget vs. Recomputation Runtime")
        plt.grid(True)
        fig = plt.gcf()
        plt.show()
        fig_dir = os.getcwd()
        if config.memory_budget_pareto_dir is not None:
            fig_dir = config.memory_budget_pareto_dir
            os.makedirs(fig_dir, exist_ok=True)
        rank_suffix = ""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank_suffix = f"_rank_{torch.distributed.get_rank()}"
        fig_name = os.path.join(
            fig_dir, f"memory_budget_pareto{rank_suffix}_{get_aot_graph_name()}.svg"
        )
        fig.savefig(fig_name)
        log.warning("Generated Pareto frontier curve at %s", fig_name)

    # todo(chilli): Estimated doesn't align exactly with actual - actual is
    # usually less memory than estimated. i'm guessing (actually quite
    # unsure about this) that's because estimated is just only including
    # tensors we actually banned from recompute, but there may be other
    # tensors that we choose to save.

    return get_saved_values_knapsack(
        memory_budget=memory_budget, node_info=node_info, joint_graph=joint_graph
    )[0]


def _sync_decision_cross_ranks(
    joint_graph: torch.fx.Graph, saved_values: list[torch.fx.Node]
):
    # use the same policy across different GPUs
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    def has_collectives(joint_graph):
        for node in joint_graph.nodes:
            if isinstance(
                node.target, torch._ops.OpOverload
            ) and node.target.namespace in {"_c10d_functional", "c10d_functional"}:
                return True
        return False

    def has_same_nodes(joint_graph):
        # proxy to check if the graph is the same across different GPUs.
        # We only consider the name and order of nodes. A more robust way
        # would be to check the hash of the whole graph (disregarding input shapes),
        # this is is a reasonable first-order approximation.
        node_str = "/".join(x.name for x in joint_graph.nodes)
        inputs = hashlib.sha256(node_str.encode("utf-8")).hexdigest()
        all_inputs = [None for _ in range(torch.distributed.get_world_size())]
        with no_dispatch(), unset_fake_temporarily():
            # TODO: maybe use a different process group?
            torch.distributed.all_gather_object(all_inputs, inputs)
        return all(all_inputs[0] == x for x in all_inputs)

    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
        and has_collectives(joint_graph)
        and has_same_nodes(joint_graph)
    ):
        with no_dispatch(), unset_fake_temporarily():
            objects = [[x.name for x in saved_values]]
            saved_ops_names_all_ranks: list[list[str]] = [
                [] for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(saved_ops_names_all_ranks, objects[0])
            name_to_node = get_name_to_node(joint_graph)
            saved_sizes: list[int] = []
            saved_ops_with_sizes: dict[str, int] = {}

            for idx, saved_ops_names in enumerate(saved_ops_names_all_ranks):
                saved_nodes = [name_to_node[op_name] for op_name in saved_ops_names]
                saved_size = 0
                for node in saved_nodes:
                    size_of_node = _size_of(node)
                    saved_size += size_of_node
                    if idx == torch.distributed.get_rank():
                        saved_ops_with_sizes[node.name] = size_of_node
                saved_ops_with_sizes["total size"] = saved_size
                saved_sizes.append(saved_size)

            saved_sizes_tensor = torch.tensor(
                saved_sizes,
                device=torch.distributed.distributed_c10d._get_object_coll_device(),
            )
            torch.distributed.all_reduce(
                saved_sizes_tensor, op=torch.distributed.distributed_c10d.ReduceOp.MAX
            )

            picked_rank_idx = int(torch.argmin(saved_sizes_tensor).item())
            sync_decision_cross_ranks_str = f"picked_rank_idx={picked_rank_idx}, saved_nodes of current rank={saved_ops_with_sizes}"
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_joint_graph_sync_decision_cross_ranks",
                    "encoding": "string",
                },
                payload_fn=lambda: sync_decision_cross_ranks_str,
            )

            saved_values = [
                name_to_node[n] for n in saved_ops_names_all_ranks[picked_rank_idx]
            ]

    return saved_values


def min_cut_rematerialization_partition(
    joint_module: fx.GraphModule,
    _joint_inputs,
    compiler="inductor",
    *,
    num_fwd_outputs,
    static_lifetime_input_indices: Optional[list[int]] = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the joint graph such that the backward recomputes the forward.
    Recomputing helps in trading off memory bandwidth with computation.

    To create the fwd and bwd graph, we copy the joint graph, manually set the
    outputs to just original forward or backward outputs. And then we run the
    resulting graphs through dead code elimination.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.
        _joint_inputs: The inputs to the joint graph. This is unused.
        compiler: This option determines the default set of recomputable ops.
            Currently, there are two options: ``nvfuser`` and ``inductor``.
        recomputable_ops: This is an optional set of recomputable ops. If this
            is not None, then this set of ops will be used instead of the
            default set of ops.
        num_fwd_outputs: The number of outputs from the forward graph.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """

    joint_module.graph.eliminate_dead_code()
    joint_module.recompile()

    fx_g = joint_module.graph

    #  add the CSE pass
    if config.cse:
        cse_graph = fx_graph_cse(fx_g)
        joint_module.graph = cse_graph
    joint_graph = joint_module.graph

    graph_has_recomputable_ops = has_recomputable_ops(joint_module)
    graph_has_recomputable_rng_ops = has_recomputable_rng_ops(joint_module)
    if graph_has_recomputable_ops:
        joint_module = cleanup_recompute_tags(joint_module)
    if not config.unsafe_allow_optimization_of_collectives:
        force_save_collectives(joint_module)
    force_save_bw_mutation_src(joint_module)

    def classify_nodes(joint_module, static_lifetime_input_indices):
        name_to_node = get_name_to_node(joint_module.graph)
        required_bw_nodes: OrderedSet[fx.Node] = OrderedSet()
        for node in joint_module.graph.nodes:
            if node.op == "placeholder" and "tangents" in node.target:
                required_bw_nodes.add(node)
            elif _must_be_in_backward(node):
                required_bw_nodes.add(node)

            if node in required_bw_nodes:
                required_bw_nodes.update(node.users)

        primal_inputs = list(filter(_is_primal, joint_module.graph.nodes))
        fwd_seed_offset_inputs = list(
            filter(_is_fwd_seed_offset, joint_module.graph.nodes)
        )
        inputs = primal_inputs + fwd_seed_offset_inputs
        fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs = (
            _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
        )
        required_bw_nodes.update(
            o for o in bwd_outputs if o is not None and o.op != "output"
        )
        forward_only_graph = _extract_graph_with_inputs_outputs(
            joint_module.graph, inputs, fwd_outputs, fwd_outputs_descs, "forward"
        )
        required_fw_nodes: OrderedSet[fx.Node] = OrderedSet(
            name_to_node[node.name]
            for node in forward_only_graph.nodes
            if node.op != "output"
        )
        unclaimed_nodes: OrderedSet[fx.Node] = OrderedSet(
            node
            for node in joint_module.graph.nodes
            if node not in required_fw_nodes and node not in required_bw_nodes
        )
        static_lifetime_input_nodes = OrderedSet(
            p for i, p in enumerate(primal_inputs) if i in static_lifetime_input_indices
        )
        fw_cnt = 0
        fw_order = {}
        for node in joint_module.graph.nodes:
            if node in required_fw_nodes:
                fw_order[node] = fw_cnt
                fw_cnt += 1
        return NodeInfo(
            inputs,
            required_fw_nodes,
            required_bw_nodes,
            unclaimed_nodes,
            fw_order,
            static_lifetime_input_nodes,
        )

    if static_lifetime_input_indices is None:
        static_lifetime_input_indices = []
    node_info = classify_nodes(joint_module, static_lifetime_input_indices)

    # networkx blows up on graphs with no required backward nodes
    # Since there's nothing to partition anyway, and the default partitioner can "handle"
    # this case, send our graph over to the default partitioner.
    if len(node_info.required_bw_nodes) == 0:
        return default_partition(
            joint_module,
            _joint_inputs,
            num_fwd_outputs=num_fwd_outputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
            static_lifetime_input_nodes=node_info.static_lifetime_input_nodes,
        )

    for node in reversed(joint_module.graph.nodes):
        if node.op == "output":
            node.dist_from_bw = int(1e9)
        elif not node_info.is_required_fw(node):
            node.dist_from_bw = 0
        else:
            node.dist_from_bw = int(1e9)
            for user in node.users:
                node.dist_from_bw = min(node.dist_from_bw, user.dist_from_bw + 1)

    memory_budget = config.activation_memory_budget
    for node in joint_graph.nodes:
        if isinstance(node.meta.get("memory_budget", None), float):
            memory_budget = node.meta["memory_budget"]
            break
    saved_values = choose_saved_values_set(
        joint_graph,
        node_info,
        memory_budget=memory_budget,
    )
    if config._sync_decision_cross_ranks:
        saved_values = _sync_decision_cross_ranks(joint_graph, saved_values)
    # save_for_backward on tensors and stashes symints in autograd .ctx
    saved_sym_nodes = list(filter(is_sym_node, saved_values))
    saved_values = list(filter(lambda n: not is_sym_node(n), saved_values))

    # NB: saved_sym_nodes will be mutated to reflect the actual saved symbols
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_nodes=node_info.static_lifetime_input_nodes,
    )
    if graph_has_recomputable_ops:
        if graph_has_recomputable_rng_ops:
            fw_module, bw_module = functionalize_rng_ops(
                joint_module, fw_module, bw_module, len(saved_sym_nodes)
            )
    bw_module = reordering_to_mimic_autograd_engine(bw_module)

    # raise all getitem ops to as early as possible
    # this is helpful for memory, especially in the case of aot_eager backend
    fw_module = raise_getitems(fw_module)
    bw_module = raise_getitems(bw_module)

    if AOT_PARTITIONER_DEBUG:
        # Calculate sorted sizes of saved values
        sorted_sizes = sorted([(_size_of(i), str(i)) for i in saved_values])

        # Log total theoretical activations stored
        total_activations_size_gb = sum(_size_of(i) for i in saved_values) / 1e9
        log.info("Theoretical Activations Stored: %.2f GB", total_activations_size_gb)

        # Log theoretical per activation storage sizes
        log.info("Theoretical Per Activation Storage Sizes: %s", sorted_sizes)
        fw_module_nodes = OrderedSet(
            node.name for node in fw_module.graph.nodes if node.op == "call_function"
        )
        bw_module_nodes = OrderedSet(
            node.name for node in bw_module.graph.nodes if node.op == "call_function"
        )
        remat_nodes = fw_module_nodes & bw_module_nodes

        counts: dict[str, int] = defaultdict(int)
        for node in fw_module.graph.nodes:
            if node.name in remat_nodes and hasattr(node.target, "_overloadpacket"):
                counts[str(node.target._overloadpacket)] += 1
        log.info(
            "# remat/fw/bw: %d/%d/%d",
            len(remat_nodes),
            len(fw_module_nodes),
            len(bw_module_nodes),
        )
        rematerialized_ops = sorted(
            counts.items(), key=operator.itemgetter(1), reverse=True
        )
        log.info("Count of Ops Rematerialized: %s", rematerialized_ops)
    return fw_module, bw_module


def draw_graph(
    traced: torch.fx.GraphModule,
    fname: str,
    figname: str = "fx_graph",
    clear_meta: bool = True,
    prog: Optional[Union[str, list[str]]] = None,
    parse_stack_trace: bool = False,
    dot_graph_shape: Optional[str] = None,
) -> None:
    if clear_meta:
        new_graph = copy.deepcopy(traced.graph)
        traced = fx.GraphModule(traced, new_graph)
        for node in traced.graph.nodes:
            node.meta = {}
    base, ext = os.path.splitext(fname)
    if not ext:
        ext = "." + config.torch_compile_graph_format
    log.info("Writing FX graph to file: %s%s", base, ext)
    g = graph_drawer.FxGraphDrawer(
        traced,
        figname,
        parse_stack_trace=parse_stack_trace,
        dot_graph_shape=dot_graph_shape,
    )
    x = g.get_main_dot_graph()
    write_method = getattr(x, "write_" + ext.lstrip("."))
    fname = f"{base}{ext}"
    if prog is None:
        write_method(fname)
    else:
        write_method(fname, prog=prog)
