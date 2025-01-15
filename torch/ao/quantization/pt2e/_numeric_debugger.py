import copy
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

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


def _detach(x: object) -> object:
    detached: object = None
    if isinstance(x, torch.Tensor):
        detached = x.detach()
    elif isinstance(x, (list, tuple)):
        detached = type(x)([_detach(e) for e in x])
    elif isinstance(x, dict):
        detached = {k: _detach(e) for k, e in x.items()}
    else:
        detached = x
    return detached


def _tensor_shape_equals(x: object, y: object) -> bool:
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.shape == y.shape
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return all(_tensor_shape_equals(e1, e2) for e1, e2 in zip(x, y))
    elif isinstance(x, dict) and isinstance(y, dict):
        all_equal = True
        for k in x:
            all_equal = all_equal and k in y and (_tensor_shape_equals(x[k], y[k]))
        return all_equal
    else:
        log.debug("Comparing non Tensors: %s and %s, they must be equal", x, y)
        return type(x) == type(y) and x == y


def _loss_fn(
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: object, y: object
) -> object:
    """The returned loss will have the same structure as `x` and `y`, e.g.
    if both are Tensor, we'll return a Tensor
    if both are list, we'll return a list of Tensors
    if both are dict, we'll return a dict with the same key, and value being the loss between the
    two Tensors
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return loss(x.to(torch.float32), y.to(torch.float32))
    elif isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return type(x)([_loss_fn(loss, e1, e2) for e1, e2 in zip(x, y)])
    elif isinstance(x, dict) and isinstance(y, dict):
        return {k: _loss_fn(loss, e, y[k]) for k, e in x.items()}
    else:
        return None


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
        self.stats: List[object] = []

    def forward(self, x: object) -> object:
        self.stats.append(_detach(x))
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
    def mse_loss(self) -> object:
        return self.loss(F.mse_loss)

    @property
    def sqnr(self) -> object:
        return self.loss(compute_sqnr)

    def loss(
        self, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> object:
        return _loss_fn(loss_function, self.actual, self.ref)

    def __repr__(self) -> str:
        # Don't include the tensors themselves as they are quite large to print
        # out.
        return (
            f"QuantizationComparisonResult(mse_loss={self.mse_loss}, sqnr={self.sqnr})"
        )

    def __post_init__(self) -> None:
        if not isinstance(self.actual, (torch.Tensor, list, tuple, dict)):
            raise ValueError(
                f"`self.actual` value must be a Tensor, list, tuple or dict, got: {self.actual}"
            )

        if not isinstance(self.ref, (torch.Tensor, list, tuple, dict)):
            raise ValueError(
                f"`self.ref` value must be a Tensor, list, tuple or dict, got: {self.ref}"
            )

        if not _tensor_shape_equals(self.ref, self.actual):
            raise ValueError(
                f"Cannot compare tensors with different shapes: ref={self.ref} vs actual={self.actual}"
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
) -> Dict[int, tuple[Optional[str], object, List[object]]]:
    """For a given model, extract the tensors stats and related information for each debug handle.
    The reason we have a list of object, instead of Tensor is because the output of node may not be
    a Tensor, it could be (nested) list, tuple or dict as well.

    Returns:
        A dict is keyed by the debug_handle id and the values are a list of object recorded
        in loggers

    """
    # Results maps debug handle to a tensor list for each model being compared.
    handles: Dict[int, tuple[Optional[str], object, List[object]]] = {}
    for _name, module in model.named_children():
        if isinstance(module, OutputLogger) and len(module.stats) > 0:
            handles[module.debug_handle] = (
                module.node_name,
                module.nn_module_stack,
                module.stats,
            )

    return handles


def compare_results(
    ref_results: Dict[int, tuple[Optional[str], object, List[torch.Tensor]]],
    actual_results: Dict[int, tuple[Optional[str], object, List[torch.Tensor]]],
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
        try:
            results = [
                QuantizationComparisonResult(actual=a, ref=b)
                for a, b in zip(actual_stats, ref_stats)
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
