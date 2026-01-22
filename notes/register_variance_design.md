# Design: Variance Tracking for Custom autograd.Function

## Context

In PR #169620, LTensor tracks `variant_dims` - which mesh axes a tensor varies along (i.e., has different values across ranks). The current design registers variance strategies for PyTorch ops via `register_variance_strategy`.

**Goal:** Allow custom `torch.autograd.Function` subclasses to specify how they affect variance tracking.

## Current System

1. **LTensor** tracks `variant_dims` - set of mesh axis names
2. **`_CUSTOM_VARIANCE_STRATEGY_MAP`** - maps functions to custom variance computation strategies
3. **`register_variance_strategy`** - decorator to register custom variance strategies for ops

When an op is called on LTensor:
1. `__torch_function__` computes union of variant dims from all inputs
2. If op is in `_CUSTOM_VARIANCE_STRATEGY_MAP`, uses that to compute output variance
3. Otherwise, output variance = union of input variances

## Design Proposal: `register_variance_for_function`

Based on existing patterns (`register_sharding`, `register_variance_strategy`):

```python
# torch/distributed/tensor/_ltensor.py (or new file)

def register_variance_for_function(func_class: type[torch.autograd.Function]):
    """
    Register variance tracking strategy for a custom autograd.Function.

    The decorated function receives:
      - input_variant_dims: set[str] - union of variant dims from all LTensor inputs
      - mesh: DeviceMesh - the mesh from input LTensors
      - *args, **kwargs: the original function arguments

    Returns:
      - output_variant_dims: set[str] - which axes the output varies along

    Example:
        class MyAllReduceOp(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, group_name):
                ctx.group_name = group_name
                return all_reduce(input, "sum", group_name)

            @staticmethod
            def backward(ctx, grad):
                return grad, None  # identity backward

        @register_variance_for_function(MyAllReduceOp)
        def _(input_variant_dims, mesh, input, group_name):
            # all_reduce makes output invariant on the reduced axis
            axis_name = _get_axis_name_from_group(mesh, group_name)
            return input_variant_dims - {axis_name}
    """
    def wrapper(variance_fn):
        # Register on the .apply method since that's what gets called
        _CUSTOM_VARIANCE_STRATEGY_MAP[func_class.apply] = variance_fn
        return variance_fn
    return wrapper
```

## Alternative: Class-based Registration

A more self-contained approach where variance is defined on the Function itself:

```python
class VarianceAwareFunction(torch.autograd.Function):
    """Mixin for Functions that declare variance behavior."""

    @staticmethod
    def variance_strategy(input_variant_dims: set[str], mesh: DeviceMesh, *args, **kwargs) -> set[str]:
        """Override to define output variance. Default: union of inputs."""
        return input_variant_dims


# Usage:
class MyAllReduceOp(VarianceAwareFunction):
    @staticmethod
    def forward(ctx, input, group_name):
        ...

    @staticmethod
    def backward(ctx, grad):
        ...

    @staticmethod
    def variance_strategy(input_variant_dims, mesh, input, group_name):
        axis_name = _get_axis_name_from_group(mesh, group_name)
        return input_variant_dims - {axis_name}
```

Then in `LTensor.__torch_function__`:
```python
# Check if it's a VarianceAwareFunction
if hasattr(func, '__self__') and hasattr(func.__self__, 'variance_strategy'):
    out_variant_dims = func.__self__.variance_strategy(out_variant_dims, mesh, *args, **kwargs)
elif func in _CUSTOM_VARIANCE_STRATEGY_MAP:
    out_variant_dims = _CUSTOM_VARIANCE_STRATEGY_MAP[func](out_variant_dims, mesh, *args, **kwargs)
```

## Recommendation

**Option 1** (decorator-based `register_variance_for_function`) is preferred because:
1. Matches `register_sharding` pattern already in codebase
2. Doesn't require inheritance
3. Can be applied to existing Function classes without modification
4. Registration is explicit and discoverable

## Implementation Checklist

- [ ] Add `register_variance_for_function` to `_ltensor.py`
- [ ] Update `__torch_function__` to pass `mesh` to variance strategies
- [ ] Add tests for custom Function variance registration
- [ ] Document in experimental API section
