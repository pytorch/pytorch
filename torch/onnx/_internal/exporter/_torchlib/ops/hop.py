"""Implementation for higher-order operators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.onnx._internal._lazy_import import onnx_ir as ir
from torch.onnx._internal.exporter._ir_builder import current_builder
from torch.onnx._internal.exporter._torchlib._torchlib_registry import onnx_impl


if TYPE_CHECKING:
    from collections.abc import Sequence


def call_op(
    op_type: str,
    *args: ir.Value | None,
    _num_outputs: int = 1,
    _domain: str = "",
    **kwargs: int | float | str | bool | ir.Graph | ir.TensorProtocol | Sequence[int],
) -> Sequence[ir.Value]:
    """Call an operator with the given arguments and keyword arguments.

    Arguments are always inputs, while keyword arguments are attributes.
    """
    # Use the scoped builder so subgraph nodes and ordinary native calls share
    # recording, optional-input trimming, and opset import handling.
    builder = current_builder()
    return builder.node(
        op_type,
        args,
        domain=_domain,
        version=builder.graph.opset_imports.get(_domain, 1)
        if _domain
        else builder.opset,
        attributes=kwargs,
        output_types=[None] * _num_outputs,
    )


@onnx_impl(torch.ops.higher_order.cond)
def higher_order_cond(
    cond: ir.Value,
    true_func: ir.Function,
    false_func: ir.Function,
    inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    then_node = ir.Node(
        true_func.domain, true_func.name, inputs, num_outputs=len(true_func.outputs)
    )
    else_node = ir.Node(
        false_func.domain, false_func.name, inputs, num_outputs=len(false_func.outputs)
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But there doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(true_func.outputs, then_node.outputs):
        out.name = f"{func_out.name}_{true_func.name}"
    for func_out, out in zip(false_func.outputs, else_node.outputs):
        out.name = f"{func_out.name}_{false_func.name}"

    return call_op(
        "If",
        cond,
        _num_outputs=len(true_func.outputs),
        then_branch=ir.Graph(
            (), then_node.outputs, nodes=[then_node], name=true_func.name
        ),
        else_branch=ir.Graph(
            (), else_node.outputs, nodes=[else_node], name=false_func.name
        ),
    )


@onnx_impl(torch.ops.higher_order.scan)
def higher_order_scan(
    body_func: ir.Function,
    scan_inits: Sequence[ir.Value],
    scan_inputs: Sequence[ir.Value],
    additional_inputs: Sequence[ir.Value] | None,
    reverse: bool = False,
) -> Sequence[ir.Value]:
    """https://github.com/pytorch/pytorch/blob/66ac724b56e6c37a534f3e066423ef2f41d7477f/torch/_higher_order_ops/scan.py#L109"""
    subgraph_inputs = [
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                shape=inp.shape,
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inits
        ],
        *[
            ir.Value(
                name=f"{inp.name}_{body_func.name}__subgraph_in",
                # The iterated element passed to the body subgraph does not have a sequence axis.
                # It will have a rank one less than the rank of the corresponding scan_input.
                shape=ir.Shape(inp.shape[1:]),  # type: ignore[index]
                type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
            )
            for inp in scan_inputs
        ],
    ]
    # The one and only node in the Scan subgraph that calls the body_func
    body_node = ir.Node(
        body_func.domain,
        body_func.name,
        [
            *subgraph_inputs,
            *(additional_inputs or []),
        ],
        num_outputs=len(body_func.outputs),
    )

    # ONNX Runtime complains about duplicate output names if we don't rename them.
    # But there doesn't seem to be an actual violation of SSA form without renaming.
    for func_out, out in zip(body_func.outputs, body_node.outputs):
        out.name = f"{func_out.name}_{body_func.name}"

    n_outputs = len(body_func.outputs) - len(scan_inits)
    return call_op(
        "Scan",
        *scan_inits,
        *scan_inputs,
        _num_outputs=len(body_func.outputs),
        body=ir.Graph(
            subgraph_inputs,
            body_node.outputs,
            nodes=[body_node],
            name=body_func.name,
        ),
        num_scan_inputs=len(scan_inputs),
        scan_input_directions=[(1 if reverse else 0) for _ in scan_inputs],
        scan_output_directions=[(1 if reverse else 0) for _ in range(n_outputs)],
    )


@onnx_impl(torch.ops.higher_order.while_loop)
def higher_order_while_loop(
    cond_func: ir.Function,
    body_func: ir.Function,
    carried_inputs: Sequence[ir.Value | int | float],
    additional_inputs: Sequence[ir.Value],
) -> Sequence[ir.Value]:
    """Implementation of while_loop using ONNX Loop operator.

    The ONNX Loop operator implements a generic looping construct with the signature:
    Loop(M, cond, v_initial) -> (v_final_and_scan_outputs)

    For while_loop, we use:
    - M: None (no trip count limit)
    - cond: initial condition value
    - v_initial: carried_inputs (loop-carried dependencies)

    The body subgraph takes:
    - iteration_num (int): current iteration number
    - condition_in (bool): loop continuation condition from previous iteration
    - loop_carried_dependencies: the carried values
    - additional_inputs: any additional inputs (constants/parameters)

    The body subgraph returns:
    - condition_out (bool): whether to continue looping
    - loop_carried_dependencies: updated carried values
    """

    # Create subgraph inputs for the Loop body
    # ONNX Loop body signature: (iter_num, cond_in, loop_carried_deps..., additional_inputs...)

    # Start subgraph construction
    subgraph_carried_inputs = []

    for i, inp in enumerate(carried_inputs):
        if isinstance(inp, ir.Value):
            subgraph_carried_inputs.append(
                ir.Value(
                    name=f"{inp.name}_{body_func.name}__subgraph_in",
                    shape=inp.shape,
                    type=ir.TensorType(inp.dtype),  # type: ignore[arg-type]
                )
            )
        elif isinstance(inp, int):
            subgraph_carried_inputs.append(
                ir.Value(
                    name=f"carried_input_{i}_{body_func.name}__subgraph_in",
                    shape=ir.Shape([]),
                    type=ir.TensorType(ir.DataType.INT64),
                )
            )
        elif isinstance(inp, float):
            subgraph_carried_inputs.append(
                ir.Value(
                    name=f"carried_input_{i}_{body_func.name}__subgraph_in",
                    shape=ir.Shape([]),
                    type=ir.TensorType(ir.DataType.FLOAT),
                )
            )
        else:
            raise NotImplementedError(
                f"Unsupported type for carried input: {type(inp)} ({inp}). "
                "Expected ir.Value, int, or float."
            )

    subgraph_inputs = [
        # Iteration number (int scalar, unused)
        ir.Value(
            name=f"iter_num_{body_func.name}",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.INT64),
        ),
        # Condition input (bool scalar, unused)
        ir.Value(
            name=f"cond_in_{body_func.name}",
            shape=ir.Shape([]),
            type=ir.TensorType(ir.DataType.BOOL),
        ),
        # Loop-carried dependencies
        *subgraph_carried_inputs,
    ]

    # Create the combined body function that handles both condition and body logic
    # First, call the body function with the same inputs
    body_node = ir.Node(
        body_func.domain,
        body_func.name,
        [
            *subgraph_carried_inputs,  # carried inputs
            *additional_inputs,
        ],
        num_outputs=len(body_func.outputs),  # carried inputs
    )

    # Then call the condition function with carried inputs + additional inputs
    cond_node = ir.Node(
        cond_func.domain,
        cond_func.name,
        [
            *body_node.outputs,  # updated carried inputs from body
            *additional_inputs,
        ],
        num_outputs=len(cond_func.outputs),
    )

    if len(cond_func.outputs) != 1:
        raise AssertionError("Condition function must return a single boolean value.")

    # ONNX Runtime complains about duplicate output names if we don't rename them
    for func_out, out in zip(body_func.outputs, body_node.outputs):
        out.name = f"{func_out.name}_{body_func.name}"
    for func_out, out in zip(cond_func.outputs, cond_node.outputs):
        out.name = f"{func_out.name}_{cond_func.name}"

    # The Loop body must return: (cond_out, loop_carried_deps...)
    # We use the condition output and the body outputs
    loop_body_outputs = [
        cond_node.outputs[0],  # condition output (bool)
        *body_node.outputs,  # updated carried inputs
    ]

    body_graph = ir.Graph(
        subgraph_inputs,
        loop_body_outputs,
        nodes=[body_node, cond_node],
        name=f"{body_func.name}_loop_body",
    )

    # End subgraph construction

    carried_inputs_values: list[ir.Value] = []
    for inp in carried_inputs:
        if isinstance(inp, ir.Value):
            carried_inputs_values.append(inp)
        elif isinstance(inp, int):
            const = current_builder().constant(inp, dtype=ir.DataType.INT64)
            carried_inputs_values.append(const)
        elif isinstance(inp, float):
            const = current_builder().constant(inp, dtype=ir.DataType.FLOAT)
            carried_inputs_values.append(const)
        else:
            raise NotImplementedError(
                f"Unsupported type for carried input: {type(inp)} ({inp}). "
                "Expected ir.Value, int, or float."
            )

    # Get initial condition by calling cond_func with initial inputs
    initial_outputs = current_builder().call_function(
        cond_func,
        *carried_inputs_values,
        *additional_inputs,
        version=1,
    )

    if len(initial_outputs) != 1:
        raise AssertionError("Condition function must return a single boolean value.")

    # Create the Loop operator call
    # Loop(M, cond, v_initial) where M is empty (no trip count limit)
    loop_outputs = call_op(
        "Loop",
        # M (trip count) - empty string means no limit
        None,
        # cond - initial condition
        initial_outputs[0],
        # v_initial - carried inputs (loop-carried dependencies)
        *carried_inputs_values,
        _num_outputs=len(carried_inputs_values),
        body=body_graph,
    )

    return loop_outputs


@onnx_impl(torch.ops.higher_order.invoke_subgraph)
def higher_order_invoke_subgraph(
    subgraph: ir.Function,
    identifier: str | None,
    *operands: ir.Value,
) -> Sequence[ir.Value]:
    """Export invoke_subgraph HOP by creating a direct function call.

    This preserves the function as a separate entity in the ONNX graph
    instead of inlining it, which is the purpose of invoke_subgraph.

    Note: The onnxscript optimizer should be configured to not inline functions
    created by invoke_subgraph to preserve the intended structure.

    Args:
        subgraph: The function to invoke
        identifier: Optional identifier for the subgraph (used for caching in PyTorch,
            not needed for ONNX export as the function reference provides all necessary information)
        *operands: Input values to pass to the function

    Returns:
        Sequence of output values from the function call
    """
    # This key can be used by downstream to avoid inlining
    subgraph.metadata_props["pkg.torch.ops.higher_order.invoke_subgraph.identifier"] = (
        str(identifier)
    )

    # Create the function call node
    return current_builder().call_function(
        subgraph,
        *operands,
        version=1,
    )
