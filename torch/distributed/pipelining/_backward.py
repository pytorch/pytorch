# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import collections
import logging
from collections.abc import Iterator
from typing import Any, Optional, Union

import torch
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter

from ._debug import map_debug_info


logger = logging.getLogger(__name__)


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Union[Node, None]:
    """
    Get the grad function or grad accumulator for a tensor.

    Accumulate grad nodes are lazily created, so we need to a
    dummy view in order to trigger its creation.
    """
    if t.requires_grad and t.grad_fn is None:
        # if no grad function (leaf tensors) we use view
        viewed_t = t.view_as(t)
        grad_fn = viewed_t.grad_fn
        if grad_fn is not None:
            return grad_fn.next_functions[0][0]
        else:
            raise RuntimeError(
                "Attempted to get grad_fn, but got None."
                "Is this being created in a no-grad context?"
            )
    else:
        return t.grad_fn


def reverse_closure(
    roots: list[Node], target_nodes: set[Node], reverse_edges_dict
) -> tuple[set[Node], set[Node]]:
    """
    This function returns the reverse closure of the given roots,
    i.e. the set of nodes that can be reached from the roots by following the
    reverse edges of the graph. The target_nodes are the nodes that we want to
    include in the closure.
    """
    # Recurse until we reach a target node
    closure: set[Node] = set()
    visited_target_nodes = set()
    q: collections.deque[Node] = collections.deque()
    for node in roots:
        if node is not None and node not in closure:
            closure.add(node)
            q.append(node)
    while q:
        node = q.popleft()
        reverse_edges = reverse_edges_dict[node]
        for fn in reverse_edges:
            if fn in closure or fn is None:
                continue
            if fn in target_nodes:
                visited_target_nodes.add(fn)
                continue
            closure.add(fn)
            q.append(fn)
    return closure, visited_target_nodes


def construct_reverse_graph(roots: list[Node]) -> dict[Node, list[Node]]:
    q: collections.deque[Node] = collections.deque()
    root_seen: set[Node] = set()
    reverse_edges_dict: dict[Node, list[Node]] = collections.defaultdict(list)
    for node in roots:
        if node is not None and node not in root_seen:
            q.append(node)
            root_seen.add(node)
    while q:
        node = q.popleft()
        for fn, _ in node.next_functions:
            if fn is not None:
                if len(reverse_edges_dict[fn]) == 0:
                    q.append(fn)
                reverse_edges_dict[fn].append(node)
    return reverse_edges_dict


def get_param_groups(
    inputs: list[Node], params: list[Node], reverse_edges_dict
) -> list[dict[str, Any]]:
    """
    Given a list of inputs and a list of parameters, return a list of parameter
    groups, where each group contains the parameters and the intermediates that
    are connected to the parameters.

    The returned list of parameter groups is a list of dictionaries, where each
    dictionary contains the following keys:
    - "params": a set of parameters
    - "intermediates": a set of intermediates

    The returned list of parameter groups is a list of dictionaries,
    """
    # reverse graph that starts with inputs, and goes up to the dOutput or the loss,
    # but omits weights and any subgraphs connecting weights to this closure
    inputs_closure, _ = reverse_closure(inputs, set(), reverse_edges_dict)
    param_groups: dict[Node, dict[str, set]] = dict()  # keyed on intermediates
    for param in params:
        closure, intersected = reverse_closure(
            [param], inputs_closure, reverse_edges_dict
        )
        param_group: dict[str, set] = {
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
    union_params: set[Node] = set()
    seen_ids: set[int] = set()
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
    stage_outputs_or_loss: list[torch.Tensor],
    output_grads: Optional[list[torch.Tensor]],
    input_values: list[torch.Tensor],
    weights: Iterator[Parameter],
) -> tuple[tuple[Optional[torch.Tensor], ...], list[dict[str, Any]]]:
    """
    Compute the gradients for only the stage inputs with
    respect to the stage outputs (if non-last stage) or loss (if last stage)

    After computing input gradients, we save the intermediate nodes in `param_groups`
    for later use in stage_backward_weight. We don't need to save any other intermediate nodes
    that aren't needed for dW because when we do dW calculation, we start from saved intermediates.
    Detaching the stage_outputs_or_loss at the end of this function is important as
    it frees up the memory that the autograd graph is anticipating to be used later (but doesn't actually need).
    """
    stage_output_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, stage_outputs_or_loss))
    )
    stage_input_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, input_values))
    )
    weight_grad_fns: list[Node] = list(
        filter(None, map(_get_grad_fn_or_grad_acc, weights))
    )

    reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
    param_groups = get_param_groups(
        stage_input_grad_fns, weight_grad_fns, reverse_edges_dict
    )

    handles = []
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
            handle = intermediate.register_prehook(get_hook(param_group, i))
            handles.append(handle)

    if output_grads is None:
        # In case this is the loss and there are no output_grads, then we just use 1s
        output_grads = [
            torch.ones_like(stage_output) for stage_output in stage_outputs_or_loss
        ]

    # Some inputs may not be used or may not require gradients, so we filter them out
    input_values = [inp for inp in input_values if inp.requires_grad]
    dinputs = torch.autograd.grad(
        stage_outputs_or_loss,
        inputs=input_values,
        grad_outputs=output_grads,
        retain_graph=True,
    )
    # Update the gradients for inputs
    for inp, dinput in zip(input_values, dinputs):
        if inp.grad is None:
            inp.grad = dinput
        else:
            inp.grad += dinput

    # stage_outputs_or_loss are not used in backwards after this point, so we can safely remove it from the autograd graph
    # this allows autograd to clear up the graph dedicated for this tensor and free up significant memory
    for t in stage_outputs_or_loss:
        t.detach_()

    # hooks are no longer necessary, clean up for consistency
    for handle in handles:
        handle.remove()

    return dinputs, param_groups


def stage_backward_weight(
    weights: Iterator[Parameter], param_groups: list[dict[str, Any]], retain_graph=False
) -> tuple[Optional[torch.Tensor], ...]:
    # map weights to param_group_weights
    grad_acc_to_weight = {}
    weight_grads: list[Optional[torch.Tensor]] = []
    for index, weight in enumerate(weights):
        grad_acc = _get_grad_fn_or_grad_acc(weight)
        grad_acc_to_weight[grad_acc] = weight, index
        weight_grads.append(weight.grad)

    for param_group in param_groups:
        valid_edges = []
        valid_grad_outputs: list[torch.Tensor] = []

        for grads_tuple, intermediate in zip(
            param_group["grads"], param_group["intermediates"]
        ):
            non_none_grads = [g for g in grads_tuple if g is not None]
            if non_none_grads:
                summed_grad = sum(non_none_grads)
                valid_edges.append(GradientEdge(intermediate, 0))
                valid_grad_outputs.append(summed_grad)

        # Break a reference cycle caused inside stage_backward_input->get_hook->hook
        # The summarized cycle is:
        # `hook` -> cell -> param_group -> intermediates -> `hook`
        # because we install the hook function onto each of the intermediate autograd nodes.
        # We need to keep intermediates alive up until backward_weight, but we can free it now.
        del param_group["intermediates"]

        if valid_edges:  # Only call autograd.grad if we have valid gradients
            # [NEW!] Able to pass a GradientEdge to autograd.grad as output
            weights_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])
            dweights = torch.autograd.grad(
                valid_edges,
                weights_edges,
                grad_outputs=valid_grad_outputs,
                retain_graph=retain_graph,
            )

            # release grad memory early after use
            del param_group["grads"]

            for grad_acc, dw in zip(param_group["params"], dweights):
                weight, index = grad_acc_to_weight[grad_acc]
                if weight.grad is None:
                    weight.grad = dw
                else:
                    weight.grad += dw
    # return grads in the original order weights were provided in
    return tuple(weight_grads)


def stage_backward(
    stage_output,
    output_grads,
    input_values,
    outputs_with_grads_idxs: Optional[list[int]] = None,  # deprecated, not used
) -> tuple[Optional[torch.Tensor], ...]:
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
        stage_output_tensors: list[torch.Tensor] = []
        output_grad_tensors: list[Optional[torch.Tensor]] = []

        def extract_tensors_with_grads(
            output_val,
            grad_val,
            # Don't delete me- see [Note: ref cycle]
            extract_tensors_with_grads,
        ):
            if isinstance(output_val, torch.Tensor):
                if not output_val.requires_grad and output_val.grad_fn is None:
                    return
                assert isinstance(grad_val, (torch.Tensor, type(None))), (
                    f"Expected Tensor or None gradient but got {type(grad_val)}"
                )
                stage_output_tensors.append(output_val)
                output_grad_tensors.append(grad_val)
            elif isinstance(output_val, (tuple, list)):
                if grad_val is None:
                    return
                assert isinstance(grad_val, (tuple, list)), (
                    f"grad_value expected to have type {type(output_val)} but got {type(grad_val)}"
                )
                assert len(output_val) == len(grad_val)
                for ov, gv in zip(output_val, grad_val):
                    extract_tensors_with_grads(
                        ov,
                        gv,
                        extract_tensors_with_grads,
                    )
            elif isinstance(output_val, dict):
                if grad_val is None:
                    return
                assert isinstance(grad_val, dict)
                assert set(output_val.keys()) == set(grad_val.keys())
                for k in output_val.keys():
                    extract_tensors_with_grads(
                        output_val[k], grad_val[k], extract_tensors_with_grads
                    )
            else:
                # Output is a non-tensor type; just ignore it
                pass

        # Note: ref cycle
        # break a ref cycle that would keep tensors alive until GC runs
        # 1. extract_tensors_with_grads refers to a cell that holds refs to any vars defined in stage_backward
        #    and used in extract_tensors_with_grads
        # 2. extract_tensors_with_grads referred to both stage_output_tensors, output_grad_tensors,
        #    and to itself (extract_tensors_with_grads) since it makes a recursive call
        # 3. stage_output_tensors was kept alive by the above refcycle, and it holds activation tensors, which is bad
        # fix -> explicitly pass in the ref to the fn, so there is no gc cycle anymore
        extract_tensors_with_grads(
            stage_output, output_grads, extract_tensors_with_grads
        )

        torch.autograd.backward(
            stage_output_tensors,
            grad_tensors=output_grad_tensors,  # type: ignore[arg-type]
        )

        # Extract gradients wrt the input values
        grad_inputs: list[Optional[torch.Tensor]] = []
        for val in input_values:
            if isinstance(val, torch.Tensor):
                grad_inputs.append(val.grad)
                # Since gradients that will pass back to previous stages do not require gradient accumulation,
                # by decrementing the gradients' reference count at this point, the memory of gradients will be
                # returned to the allocator as soon as the next micro batch's get_bwd_send_ops comes and current
                # asynchronous send completes.
                # This prevents the gradients from persisting in GPU memory for the entire duration of step_microbatches
                # until clear_runtime_states() is called.
                val.grad = None
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

    return tuple(grad_inputs)


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
