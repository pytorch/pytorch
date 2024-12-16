import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.ao.ns.fx.utils import compute_sqnr
from torch.ao.quantization.pt2e.graph_utils import bfs_trace_with_node_process
from torch.export import ExportedProgram
from torch.fx import GraphModule, Node
from torch.nn import functional as F


NUMERIC_DEBUG_HANDLE_KEY = "numeric_debug_handle"
CUSTOM_KEY = "custom"

log = logging.getLogger(__name__)


def generate_numeric_debug_handle(ep: ExportedProgram) -> None:
    """
    Attach numeric_debug_handle_id for all nodes in the graph module of the given
    ExportedProgram, like conv2d, squeeze, conv1d, etc, except for placeholder.
    Notice that nodes like getattr are out of scope since they are not in the graph.

    The graph nodes of input exported program are modified inplace.

    Here's an example of using debug handle quantize flow::

        ep = export_for_training(eager_model, example_inputs)
        generate_numeric_debug_handle(ep)

        m = ep.module()
        quantizer = XNNPACKQuantizer()
        m = prepare_pt2e(m, quantizer)
        m = convert_pt2e(m)
    """

    # Sanity check the input data type
    if not isinstance(ep, ExportedProgram):
        raise ValueError(
            f"Expected ep to be ExportedProgram, got {type(ExportedProgram)}"
        )

    unique_id = 0

    def _find_max_id(node: torch.fx.Node) -> None:
        nonlocal unique_id
        unique_id = max(
            unique_id, node.meta.get(CUSTOM_KEY, {}).get(NUMERIC_DEBUG_HANDLE_KEY, 0)
        )

    def _assign_debug_handle(node: torch.fx.Node) -> None:
        nonlocal unique_id
        if CUSTOM_KEY not in node.meta:
            node.meta[CUSTOM_KEY] = {}

        if NUMERIC_DEBUG_HANDLE_KEY not in node.meta[CUSTOM_KEY]:
            node.meta[CUSTOM_KEY][NUMERIC_DEBUG_HANDLE_KEY] = unique_id
            unique_id += 1

    # Find the max ID that exists in the graph first, in case part of the graph
    # has already been annotated. This way we guarantee there are no duplicate
    # handle IDs.
    bfs_trace_with_node_process(ep, _find_max_id)

    unique_id += 1

    # Assign debug handles to all nodes in the graph that don't have one based on the
    # max ID found in the previous step.
    bfs_trace_with_node_process(ep, _assign_debug_handle)


class OutputLogger(torch.nn.Module):
    """
    Base class for capturing output values for nodes in a GraphModule, it only captures
    Tensor output currently, but we can extend it to work for other types of inputs later if needed
    """

    # Mark as impure so that calls to it will not be removed during DCE.
    _is_impure = True

    def __init__(
        self,
        debug_handle: int,
        node_name: Optional[str] = None,
        nn_module_stack: Optional[object] = None,
    ) -> None:
        super().__init__()
        self.node_name = node_name
        self.nn_module_stack = nn_module_stack
        self.debug_handle = debug_handle
        self.stats: List[torch.Tensor] = []

    def forward(self, x: object) -> object:
        if isinstance(x, torch.Tensor):
            self.stats.append(x.detach())
        return x

    def __extra_repr__(self) -> str:
        return (
            f"debug_handle={self.debug_handle}, node_name={self.node_name}, "
            "nn_module_stack={self.nn_module_stack}, num_stats={len(self.stats)})"
        )


def _insert_logger(model: GraphModule, node: Node, debug_handle: int) -> Node:
    """For a given node, adds an OutputLogger that observes the output of that node,
    and all its users use the OutputLogger output instead.
    The OutputLogger will contain the debug_handle which can be used to compare
    graphs after transforms"""

    # to avoid circular dep
    from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix

    # add a logger after the node
    with model.graph.inserting_after(node):
        get_new_attr_name = get_new_attr_name_with_prefix(f"{node.name}_logger")
        logger_name = get_new_attr_name(model)
        setattr(
            model,
            logger_name,
            OutputLogger(debug_handle, node.name, node.meta.get("nn_module_stack")),
        )
        logger_node = model.graph.call_module(logger_name, (node,), {})

    orig_users = list(node.users.keys())
    for user_node in orig_users:
        if user_node is logger_node:
            continue
        user_node.replace_input_with(node, logger_node)

    return logger_node


def prepare_for_propagation_comparison(model: GraphModule) -> GraphModule:
    """Add output loggers to node that has numeric_debug_handle

    Args:
        model (GraphModule): original model
    Returns:
        a model with output loggers for all nodes that has numeric_debug_handle_id
    """
    # don't change the original model
    model = copy.deepcopy(model)
    for n in model.graph.nodes:
        if (
            CUSTOM_KEY not in n.meta
            or NUMERIC_DEBUG_HANDLE_KEY not in n.meta[CUSTOM_KEY]
        ):
            continue
        numeric_debug_handle = n.meta[CUSTOM_KEY][NUMERIC_DEBUG_HANDLE_KEY]
        _insert_logger(model, n, numeric_debug_handle)

    model.recompile()
    return model


@dataclass(frozen=True)
class QuantizationComparisonResult:
    actual: torch.Tensor
    ref: torch.Tensor

    @property
    def mse_loss(self) -> torch.Tensor:
        return F.mse_loss(
            self.actual.to(dtype=torch.float32), self.ref.to(dtype=torch.float32)
        )

    @property
    def sqnr(self) -> torch.Tensor:
        return compute_sqnr(
            self.actual.to(dtype=torch.float32), self.ref.to(dtype=torch.float32)
        )

    def loss(
        self, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if self.actual.shape != self.ref.shape:
            raise ValueError(
                f"Cannot compare tensors with different shapes: {self.actual.shape} vs {self.ref.shape}"
            )
        return loss_function(
            self.actual.to(dtype=torch.float32), self.ref.to(dtype=torch.float32)
        )

    def __repr__(self) -> str:
        # Don't include the tensors themselves as they are quite large to print
        # out.
        return (
            f"QuantizationComparisonResult(mse_loss={self.mse_loss}, sqnr={self.sqnr})"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.actual, torch.Tensor):
            raise ValueError(
                f"`self.actual` value must be a Tensor, got: {self.actual}"
            )

        if not isinstance(self.ref, torch.Tensor):
            raise ValueError(f"`self.ref` value must be a Tensor, got: {self.ref}")
        if self.actual.shape != self.ref.shape:
            raise ValueError(
                f"Cannot compare tensors with different shapes: ref={self.ref.shape} vs actual={self.actual.shape}"
            )


@dataclass(frozen=True)
class NodeAccuracySummary:
    handle: int
    actual_node_name: str
    actual_module_stack: str
    ref_node_name: str
    ref_module_stack: str
    results: Sequence[QuantizationComparisonResult]


def _module_stack_to_str(module_stack: object) -> str:
    """Simplifies the stack from ("mod", "mod.foo", "mod.foo.0", "mod.foo.0.linear")
    to "mod.foo.0.linear"
    """
    if not isinstance(module_stack, dict):
        return str(module_stack)
    module_values_list = list(module_stack.values())
    if len(module_values_list) > 0:
        owning_module = module_values_list[-1][0]
        return str(owning_module)
    else:
        return str(module_stack)


def extract_results_from_loggers(
    model: GraphModule,
) -> Dict[int, Tuple[Optional[str], object, List[torch.Tensor]]]:
    """For a given model, extract the tensors stats and related information for each debug handle.

    Returns:
        A dict is keyed by the debug_handle id and the values are a list of Tensors recorded
        in loggers"""
    # Results maps debug handle to a tensor list for each model being compared.
    handles: Dict[int, Tuple[Optional[str], object, List[torch.Tensor]]] = {}
    for _name, module in model.named_children():
        if isinstance(module, OutputLogger) and len(module.stats) > 0:
            handles[module.debug_handle] = (
                module.node_name,
                module.nn_module_stack,
                module.stats,
            )

    return handles


def compare_results(
    ref_results: Dict[int, Tuple[Optional[str], object, List[torch.Tensor]]],
    actual_results: Dict[int, Tuple[Optional[str], object, List[torch.Tensor]]],
) -> Dict[int, NodeAccuracySummary]:
    """Given two dict mapping from `debug_handle_id` (int) to list of tensors
    return a map from `debug_handle_id` to `NodeAccuracySummary` that contains
    comparison information like SQNR, MSE etc.

    Args:
        ref_results (Dict[int, Tuple[str, object, List[torch.Tensor]]]): reference results for each debug_handle_id
        actual_results (Dict[int, Tuple[str, object, List[torch.Tensor]]]): actual results for each debug_handle_id

    Returns:
        Dict[int, NodeAccuracySummary]
    """
    comparisons = {}
    for debug_handle, (ref_name, ref_stack, ref_stats) in ref_results.items():
        if debug_handle not in actual_results:
            log.debug(
                "Cannot compare for handle %s because it wasn't found in the transformed model",
                debug_handle,
            )
            continue
        actual_name, actual_stack, actual_stats = actual_results[debug_handle]
        for a, b in zip(actual_stats, ref_stats):
            if a.shape != b.shape:
                log.warning(
                    f"Cannot compare tensors with different shapes: actual={a.shape} vs ref={b.shape}"
                )
        try:
            results = [
                QuantizationComparisonResult(actual=a, ref=b)
                for a, b in zip(actual_stats, ref_stats)
                if a.shape == b.shape
            ]
        except Exception as e:
            # Add extra information for an exception from QuantizationComparisonResult
            # if the shapes didn't match, to include the handle and the node names.
            raise ValueError(
                f"For numeric_debug_handle={debug_handle} from ref node {ref_name} and actual node {actual_name}"
            ) from e

        comparisons[debug_handle] = NodeAccuracySummary(
            handle=debug_handle,
            actual_node_name=actual_name or "",
            actual_module_stack=_module_stack_to_str(actual_stack),
            ref_node_name=ref_name or "",
            ref_module_stack=_module_stack_to_str(ref_stack),
            results=results,
        )

    return comparisons
