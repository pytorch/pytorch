# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import collections
import logging
import weakref
from typing import Any, cast, Deque, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter

from ._debug import map_debug_info


logger = logging.getLogger(__name__)


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Union[Node, None]:
    if t.requires_grad and t.grad_fn is None:
        # if no grad function (leaf tensors) we use view
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        if grad_fn is not None:
            return grad_fn.next_functions[0][0]
        else:
            raise RuntimeError("grad_fn after adding view is None")
    else:
        return t.grad_fn


def reverse_closure(
    roots: List[Node], target_nodes: Set[Node]
) -> Tuple[Set[Node], Set[Node]]:
    # Recurse until we reach a target node
    closure: Set[Node] = set()
    actual_target_nodes = set()
    q: Deque[Node] = collections.deque()
    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            q.append(node)
    while q:
        node = q.popleft()
        metadata = cast(Dict[str, List], node.metadata)
        reverse_edges = metadata.get("reverse_edges", [])
        for holder_ref, idx in reverse_edges:
            ref = holder_ref()
            if ref is None:
                # this reverse graph is no longer alive
                # raise RuntimeError("Reverse graph is no longer alive")
                continue
            fn = ref.node
            if fn in closure or fn is None:
                continue
            if fn in target_nodes:
                actual_target_nodes.add(fn)
                continue
            closure.add(fn)
            q.append(fn)
    return closure, actual_target_nodes


# Enable weak pointer
class Holder:
    def __init__(self, node: Node):
        self.node = node


# TODO: use weak references to avoid reference cycle
def construct_reverse_graph(roots: List[Node]) -> List[Holder]:
    q: Deque[Node] = collections.deque()
    root_seen: Set[Node] = set()
    reverse_graph_refs: List[Holder] = []
    for node in roots:
        if node is not None and node not in root_seen:
            q.append(node)
            root_seen.add(node)
    while q:
        node = q.popleft()
        for fn, idx in node.next_functions:
            if fn is not None:
                # Don't necessarily need to store on the graph
                metadata = cast(Dict[str, List], fn.metadata)
                reverse_edges = metadata.get("reverse_edges", [])
                if len(reverse_edges) == 0:
                    q.append(fn)
                holder = Holder(node)
                holder_ref = weakref.ref(holder)
                reverse_graph_refs.append(holder)
                reverse_edges.append((holder_ref, idx))
                metadata["reverse_edges"] = reverse_edges
    return reverse_graph_refs


def get_param_groups(inputs: List[Node], params: List[Node]) -> List[Dict[str, Any]]:
    inputs_closure, _ = reverse_closure(inputs, set())
    param_groups: Dict[Node, Dict[str, Set]] = dict()  # keyed on intermediates
    for i, param in enumerate(params):
        closure, intersected = reverse_closure([param], inputs_closure)
        param_group: Dict[str, Set] = {
            "params": {param},
            "intermediates": intersected,
        }
        for input_node in intersected:
            existing = param_groups.get(input_node, None)
            if existing is not None:
                existing["params"] = existing["params"].union(param_group["params"])
                existing["intermediates"] = existing["intermediates"].union(
                    param_group["intermediates"]
                )
                param_group = existing
            else:
                param_groups[input_node] = param_group

    # Sanity check: union of all param_groups params should be equal to all params
    union_params: Set[Node] = set()
    seen_ids: Set[int] = set()
    unique_param_groups = []
    for param_group in param_groups.values():
        if id(param_group) not in seen_ids:
            seen_ids.add(id(param_group))
            unique_param_groups.append(param_group)
            union_params = union_params.union(param_group["params"])

    # The assert will only be true if the input tensor requires gradients,
    # otherwise the autograd graph will miss the first layer of inputs
    # assert union_params == set(params)
    return unique_param_groups


def stage_backward_input(
    stage_outputs: List[torch.Tensor],
    output_grads: Optional[List[torch.Tensor]],
    stage_inputs: List[torch.Tensor],
    weights: Iterator[Parameter],
):
    """
    compute the gradients for only the stage inputs with respect to the stage outputs
    """
    stage_output_grad_fns: List[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, stage_outputs))
    )
    stage_input_grad_fns: List[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, stage_inputs))
    )
    weight_grad_fns: List[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, weights))
    )

    reverse_graph_refs = construct_reverse_graph(stage_output_grad_fns)
    param_groups = get_param_groups(stage_input_grad_fns, weight_grad_fns)
    del reverse_graph_refs

    for param_group in param_groups:
        for i, intermediate in enumerate(param_group["intermediates"]):

            def get_hook(param_group, i):
                def hook(grad_inputs):
                    if param_group.get("grads", None) is None:
                        param_group["grads"] = [None] * len(
                            param_group["intermediates"]
                        )
                    param_group["grads"][i] = grad_inputs

                return hook

            # These are always "split" nodes that we need to recompute, so
            # save their inputs.
            intermediate.register_prehook(get_hook(param_group, i))

    # Stage 0 inputs do not require grads? Should we skip in that case?
    if all(tensor.requires_grad for tensor in stage_inputs):
        if output_grads is None:
            # In case this is the loss and there are no output_grads, then we just use 1s
            output_grads = [
                torch.ones_like(stage_output) for stage_output in stage_outputs
            ]

        dinputs = torch.autograd.grad(
            stage_outputs,
            inputs=stage_inputs,
            grad_outputs=output_grads,
            retain_graph=True,
        )

        # update the gradients for inputs
        for i, inp in enumerate(stage_inputs):
            if inp.grad is None:
                inp.grad = dinputs[i]
            else:
                inp.grad += dinputs[i]
    else:
        dinputs = None
    return dinputs, param_groups


def stage_backward_weight(
    weights: Iterator[Parameter], param_groups: List[Dict[str, Any]]
):
    all_dweights = dict()
    for param_group in param_groups:
        # TODO: Handle case where intermediate can have multiple outputs
        intermediate_edges = tuple(
            GradientEdge(i, 0) for i in param_group["intermediates"]
        )
        weights_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])

        assert all(len(g) == 1 for g in param_group["grads"])
        # [NEW!] Able to pass a GradientEdge to autograd.grad as output
        # We do not need to retain_graph because... guarantee no overlap?
        # print("trying to execute: ", intermediate_edges, weights_edges)
        dweights = torch.autograd.grad(
            intermediate_edges,
            weights_edges,
            grad_outputs=sum(param_group["grads"], tuple()),
        )
        for w, dw in zip(param_group["params"], dweights):
            all_dweights[w] = dw
    # return grads in the original order weights were provided in
    out = []
    for w in weights:
        grad_acc = _get_grad_fn_or_grad_acc(w)
        dweight = all_dweights[grad_acc]
        out.append(dweight)
        if w.grad is None:
            w.grad = dweight
        else:
            w.grad += dweight
    return out


def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: Optional[List[int]] = None,  # deprecated, not used
):
    """
    This is a helper function to:
    1. compute the gradients for the stage inputs, and
    2. accumulate gradients for the stage module's parameters.

    Given the input value(s) and the corresponding gradient for the output
    value(s), compute and accumulate gradients for all parameter values (leaves
    in the autograd trace) as well as return a list of the gradients for the
    input values
    """
    if outputs_with_grads_idxs is not None:
        # Deprecated, not used in runtime calls, only exists in compiler
        stage_output = [stage_output[i] for i in outputs_with_grads_idxs]
        output_grads = [output_grads[i] for i in outputs_with_grads_idxs]

    try:
        # stage_output may be a composite datatype like dict. Extract all individual
        # tensor values here
        stage_output_tensors = []
        output_grad_tensors = []

        def extract_tensors_with_grads(output_val, grad_val):
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                assert isinstance(
                    grad_val, (torch.Tensor, type(None))
                ), f"Expected Tensor or None gradient but got {type(grad_val)}"
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                assert isinstance(
                    grad_val, (tuple, list)
                ), f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                assert len(output_val) == len(grad_val)
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(ov, gv)
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                assert isinstance(grad_val, dict)
                assert set(output_val.keys()) == set(grad_val.keys())
                for k in output_val.keys():
                    extract_tensors_with_grads(output_val[k], grad_val[k])
            else:
                # Output is a non-tensor type; just ignore it
                pass

        extract_tensors_with_grads(stage_output, output_grads)

        torch.autograd.backward(
            stage_output_tensors, grad_tensors=output_grad_tensors  # type: ignore[arg-type]
        )

        # Extract gradients wrt the input values
        grad_inputs = []
        for val in input_values:
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
            else:
                grad_inputs.append(None)

        # Alternative impl: `torch.autograd.grad`.
        # Note that `torch.autograd.grad` will not accumulate gradients into the
        # model's parameters.
        """
        inputs_with_grad = []
        for val in input_values:
            if isinstance(val, torch.Tensor) and val.requires_grad:
                inputs_with_grad.append(val)

        grad_inputs = torch.autograd.grad(
            stage_output_tensors, inputs_with_grad, output_grad_tensors,  # type: ignore[arg-type]
        )
        """

    except Exception as e:
        exc_msg = f"""
        Failed to run stage backward:
        Stage output: {map_debug_info(stage_output)}
        Output gradient: {map_debug_info(output_grads)}
        Input: {map_debug_info(input_values)}
        """
        raise RuntimeError(exc_msg) from e

    return grad_inputs


# TODO: handling requires_grad=False dynamically. Can we analyze this during initial
# IR emission?
def _null_coalesce_accumulate(lhs, rhs):
    """
    Coalesce two values, even if one of them is null, returning the non-null
    value.
    """
    if lhs is None:
        return rhs
    elif rhs is None:
        return lhs
    else:
        return torch.add(lhs, rhs)
