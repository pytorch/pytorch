# LTensor Support for Custom Autograd Functions

## Problem Statement

When users write custom `autograd.Function` classes and use them with LTensor inputs, we need to track variance metadata through the forward pass. The challenge is that custom functions may contain:

1. **PyTorch ops** - These go through `__torch_function__` and LTensor tracks naturally
2. **Opaque kernels** (custom CUDA/C++) - These bypass `__torch_function__` and lose tracking

## Prior Art: How Functorch Solves This

Functorch intercepts `autograd.Function.apply` in Python:

```python
# torch/autograd/function.py
class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        if torch._C._are_functorch_transforms_active():
            return custom_function_call(cls, *args, **kwargs)
        return super().apply(*args, **kwargs)
```

## Design for LTensor

### Interception Point

Add an LTensor check in `Function.apply`:

```python
class Function:
    @classmethod
    def apply(cls, *args, **kwargs):
        if torch._C._are_functorch_transforms_active():
            return custom_function_call(cls, *args, **kwargs)

        # LTensor check
        if _has_ltensor_in_args(args, kwargs):
            return ltensor_function_call(cls, *args, **kwargs)

        return super().apply(*args, **kwargs)
```

### Shared Utilities

Static methods on `LTensor` for metadata operations:

```python
class LTensor(torch.Tensor):
    @staticmethod
    def _extract_metadata(args, kwargs=None):
        """Extract variant_dims, reduced_dims, mesh from LTensor inputs."""
        ...

    @staticmethod
    def _wrap(t, variant_dims, mesh, reduced_dims):
        """Wrap a plain tensor as LTensor if needed."""
        ...

    @staticmethod
    def _unwrap(t):
        """Unwrap LTensor to plain tensor."""
        ...
```

### Core Handler: `ltensor_function_call`

```python
def ltensor_function_call(autograd_function, *args, **kwargs):
    # 1. Extract metadata
    variant_dims, reduced_dims, mesh = LTensor._extract_metadata(args, kwargs)

    # 2. Unwrap and call forward
    unwrapped_args = tree_map(LTensor._unwrap, args)
    result = torch.autograd.Function.apply.__func__(
        autograd_function, *unwrapped_args, **unwrapped_kwargs
    )

    # 3. Apply custom or default strategy
    if autograd_function in _CUSTOM_VARIANCE_TRACKING_MAP:
        strategy_result = _CUSTOM_VARIANCE_TRACKING_MAP[autograd_function](...)
        # Handle tuple or single return
    else:
        out_variant_dims, out_reduced_dims = variant_dims, reduced_dims

    # 4. Wrap outputs
    return tree_map(lambda t: LTensor._wrap(t, out_variant_dims, mesh, out_reduced_dims), result)
```

## Unified Registration API

Single decorator `@register_variance_tracking_strategy` works for both ops and `autograd.Function`:

```python
_CUSTOM_VARIANCE_TRACKING_MAP: dict[Callable | type, Callable] = {}

def register_variance_tracking_strategy(target):
    """Register variance tracking strategy for ops or autograd.Function.

    Args:
        target: List of ops, single op, or autograd.Function class
    """
    def decorator(strategy_fn):
        if isinstance(target, list):
            for op in target:
                _CUSTOM_VARIANCE_TRACKING_MAP[op] = strategy_fn
        else:
            _CUSTOM_VARIANCE_TRACKING_MAP[target] = strategy_fn
        return strategy_fn
    return decorator
```

### Strategy Signature

```python
def strategy(input_variant_dims, input_reduced_dims, mesh, *args, **kwargs):
    # Return one of:
    return output_variant_dims                        # reduced uses default
    return (output_variant_dims, output_reduced_dims) # custom for both
```

## Usage Examples

### Default (No Registration)

```python
class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        return x + y * 2  # Just works - default propagation
```

### Custom Strategy for autograd.Function

```python
@register_variance_tracking_strategy(MyAllReduceOp)
def _(input_variant_dims, input_reduced_dims, mesh, input, group_name):
    dim_name = _get_dim_name_from_group(mesh, group_name)
    return input_variant_dims - {dim_name}
```

### Custom Strategy for Ops

```python
@register_variance_tracking_strategy([torch.ops._c10d_functional.all_reduce])
def _(input_variant_dims, input_reduced_dims, mesh, *args, **kwargs):
    _, _, group_name = args
    dim_name = _get_dim_name_from_group(mesh, group_name)
    return input_variant_dims - {dim_name}
```

### Returning Both Dims

```python
@register_variance_tracking_strategy(SomeFunction)
def _(input_variant_dims, input_reduced_dims, mesh, *args):
    return set(), {"dp"}  # No variance, reduced on dp
```

## Design Rationale

**Why a single unified decorator?**
- Same lookup mechanism (by key in one map)
- Consistent API for both ops and autograd.Function
- Simpler to learn and use

**Why default propagation?**
- Most functions just pass through variance/reduced dims
- Only opaque kernels need explicit registration
