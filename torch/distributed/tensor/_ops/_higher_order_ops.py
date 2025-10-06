# Copyright (c) Meta Platforms, Inc. and affiliates
# DTensor implementations for control flow operations like cond, while_loop, etc.

from typing import Any, Callable, Optional, Union

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.cond import cond_op
from torch.distributed.tensor._dtensor_spec import DTensorSpec


def analyze_branch_output(
    branch_fn: Callable[..., Any], operands: tuple[Any, ...]
) -> Union[DTensorSpec, list[Union[DTensorSpec, None]], None]:
    from torch.distributed.tensor import DTensor

    result = branch_fn(*operands)

    if isinstance(result, DTensor):
        return result._spec
    elif isinstance(result, (tuple, list)):
        specs: list[Optional[DTensorSpec]] = []
        for item in result:
            if isinstance(item, DTensor):
                specs.append(item._spec)
            else:
                specs.append(None)  #
        return specs
    else:
        return None  # Scalar or other non-DTensor result


def merge_dtensor_specs(
    true_spec: Union[DTensorSpec, list[Union[DTensorSpec, None]], None],
    false_spec: Union[DTensorSpec, list[Union[DTensorSpec, None]], None],
) -> Union[DTensorSpec, list[Union[DTensorSpec, None]], None]:
    if true_spec is None or false_spec is None:
        assert true_spec == false_spec, (
            f"Incompatible specs from two branches {true_spec} vs {false_spec}"
        )
        return true_spec

    # For now, implement a simple unification strategy:
    # If both specs are the same, use them. Otherwise, raise an error
    if isinstance(true_spec, DTensorSpec) and isinstance(false_spec, DTensorSpec):
        if (
            true_spec.placements == false_spec.placements
            and true_spec.mesh == false_spec.mesh
        ):
            return true_spec
        else:
            raise NotImplementedError(
                f"torch.cond expects two branches return the same placement "
                f"but one of true branch's return is {true_spec} vs false branch's return {false_spec}"
            )
    elif isinstance(true_spec, (tuple, list)) and isinstance(false_spec, (tuple, list)):
        merged_specs = []
        for t_spec, f_spec in zip(true_spec, false_spec):
            dtensor_spec = merge_dtensor_specs(t_spec, f_spec)
            assert isinstance(dtensor_spec, DTensorSpec) or dtensor_spec is None, (
                "Expect DTensorSpec or None"
            )
            merged_specs.append(dtensor_spec)
        return merged_specs
    else:
        # cond always normalize the output to be flat tuple
        raise NotImplementedError(
            f"Unsupported spec types: {type(true_spec)}, {type(false_spec)}"
        )


# Wrap result back to DTensor with the unified spec
def wrap_with_unified_spec(result: Any, spec: Any) -> Any:
    from torch.distributed.tensor import DTensor

    if isinstance(result, torch.Tensor) and isinstance(spec, DTensorSpec):
        return DTensor(
            result,
            spec,
            requires_grad=result.requires_grad
            if hasattr(result, "requires_grad")
            else False,
        )
    elif isinstance(result, (tuple, list)) and isinstance(spec, (tuple, list)):
        # Handle multiple outputs
        wrapped_results = []
        for item, item_spec in zip(result, spec):
            if isinstance(item, torch.Tensor) and isinstance(item_spec, DTensorSpec):
                wrapped_results.append(
                    DTensor(
                        item,
                        item_spec,
                        requires_grad=item.requires_grad
                        if hasattr(item, "requires_grad")
                        else False,
                    )
                )
            else:
                wrapped_results.append(item)
        return tuple(wrapped_results) if isinstance(result, tuple) else wrapped_results
    else:
        raise NotImplementedError(
            f"Unsupported result type: {type(result)} and spec type: {type(spec)}"
        )


# The real registration happens at torch.distributed.tensor.__init__.py:_register_dtensor_higher_order_ops
def cond_dtensor_handler(
    pred: Union[torch.Tensor, Any],
    true_fn: Callable[..., Any],
    false_fn: Callable[..., Any],
    operands: tuple[Any, ...],
) -> Any:
    """
    DTensor handler for torch.ops.higher_order.cond.

    This implementation:
    1. Performs sharding propagation for both branches
    2. Merges the output specs from both branches to find a compatible spec, raise error if the specs differ

    Returns:
        Result from executing cond with proper DTensor wrapping
    """
    from torch.distributed.tensor import DTensor

    assert isinstance(operands, tuple), "operands are expected to be a tuple or list."

    mesh = None
    for operand in (pred,) + operands:
        if isinstance(operand, DTensor):
            if mesh is None:
                mesh = operand.device_mesh
            assert mesh == operand.device_mesh, (
                f"All DTensor operands must have the same mesh but got {mesh} and {operand.device_mesh}"
            )

    assert mesh is not None, "No DTensor operands found in cond's inputs"

    true_output_spec = analyze_branch_output(true_fn, operands)
    false_output_spec = analyze_branch_output(false_fn, operands)

    merged_spec = merge_dtensor_specs(true_output_spec, false_output_spec)

    local_pred, local_operands = pytree.tree_map_only(
        DTensor, lambda x: x._local_tensor, (pred, operands)
    )

    local_result = cond_op(local_pred, true_fn, false_fn, local_operands)

    return wrap_with_unified_spec(local_result, merged_spec)
