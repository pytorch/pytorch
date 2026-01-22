# DTensor Dispatch Split Design: Moving Redistribution Above Autograd

## Problem Statement

Currently, when DTensor operations require input redistribution (e.g., an op needs
`Shard(0)` but receives `Replicate`), the redistribution happens **below autograd**
in `redistribute_local_args()`. This means:

1. Autograd doesn't see the redistribute call
2. No dedicated backward is generated for the redistribute
3. Forward and backward may have asymmetric redistribution behavior

## Current Architecture

### Call Chain

```
torch.add(dtensor_a, dtensor_b)
    │
    ▼
handle_torch_function_no_python_arg_parser()
    │                               [python_arg_parser.cpp:529]
    │
    ├─── TorchFunctionName::TorchFunction ──► __torch_function__ path
    │                                           │
    │                                           ▼
    │                                    DTensor.__torch_function__()
    │                                    - calls super().__torch_function__
    │                                    - registers grad placement hook
    │                                    - NO redistribution check
    │
    └─── TorchFunctionName::TorchDispatch ──► __torch_dispatch__ path
                                                │
                                                ▼
                                         dispatch_on_subclass()
                                                │     [python_arg_parser.cpp:316]
                                                │
                                                ▼
                                    if (!is_torch_function && is_dtensor(arg)):
                                                │     [python_arg_parser.cpp:378]
                                                │
                                                ▼
                                         dispatchDTensorOp()
                                                │     [python_variable.cpp:1401]
                                                │
                                                ├── sharding propagation
                                                ├── redistribute_local_args() ◄── PROBLEM!
                                                ├── local op execution
                                                └── wrap result
```

### Key Code Locations

| File | Line | Function | Role |
|------|------|----------|------|
| `torch/csrc/utils/python_arg_parser.cpp` | 529 | `handle_torch_function_no_python_arg_parser` | Entry point, determines TorchFunction vs TorchDispatch |
| `torch/csrc/utils/python_arg_parser.cpp` | 316 | `dispatch_on_subclass` | Dispatches to subclass handlers |
| `torch/csrc/utils/python_arg_parser.cpp` | 378 | (in dispatch_on_subclass) | DTensor special case check |
| `torch/csrc/autograd/python_variable.cpp` | 1401 | `dispatchDTensorOp` | Main DTensor dispatch logic |
| `torch/distributed/tensor/_dispatch.py` | 446 | `redistribute_local_args` | Performs redistribution (invisible to autograd) |
| `torch/distributed/tensor/_api.py` | 423 | `DTensor.__torch_function__` | Currently just registers hooks |
| `torch/distributed/tensor/_api.py` | 437 | `DTensor.__torch_dispatch__` | Raises NotImplementedError (C++ handles it) |

### The Problem in Detail

```python
# In _dispatch.py:446
def redistribute_local_args(op_info, suggested_schema, ...):
    for i, arg_spec in enumerate(op_info.flat_args_schema):
        if arg_spec != reshard_arg_spec:
            # This operates on LOCAL tensors, not DTensors!
            # Autograd never sees this.
            resharded_local = redistribute_local_tensor(
                local_tensor, arg_spec, reshard_arg_spec
            )
```

The `redistribute_local_tensor` function performs collectives on raw `torch.Tensor`,
completely bypassing the `Redistribute` autograd.Function that exists in `_redistribute.py:925`.

## Proposed Solution: Split Dispatch Between __torch_function__ and __torch_dispatch__

### New Architecture

```
torch.add(dtensor_a, dtensor_b)
    │
    ▼
handle_torch_function_no_python_arg_parser()
    │
    ├─── TorchFunctionName::TorchFunction ──► __torch_function__ path
    │                                           │
    │                                           ▼
    │                                    DTensor.__torch_function__()  ◄── NEW LOGIC HERE
    │                                           │
    │                                           ├── Quick sharding propagation
    │                                           │   (determine if redistribution needed)
    │                                           │
    │                                           ├── If redistribution needed:
    │                                           │   a_redist = Redistribute.apply(a, target)
    │                                           │              └──► VISIBLE TO AUTOGRAD!
    │                                           │
    │                                           └── Call op with redistributed inputs
    │                                               └──► continues to __torch_dispatch__
    │
    └─── TorchFunctionName::TorchDispatch ──► __torch_dispatch__ path
                                                │
                                                ▼
                                         dispatchDTensorOp()  ◄── SIMPLIFIED
                                                │
                                                ├── Inputs already have correct placements
                                                ├── Assert no redistribution needed
                                                ├── local op execution
                                                └── wrap result
```

### Why This Works

1. `__torch_function__` is called **BEFORE** autograd sees the operation
2. `Redistribute.apply()` goes through `torch.autograd.Function`
3. Autograd records the redistribute in its graph
4. Backward automatically gets `Redistribute.backward()`

### Implementation Options

#### Option A: Add C++ DTensor Handling in __torch_function__ Path

Modify `dispatch_on_subclass` to handle DTensor in the `__torch_function__` path:

```cpp
// python_arg_parser.cpp, in dispatch_on_subclass()

// NEW: Handle DTensor redistribution in __torch_function__ path
if (is_torch_function && is_dtensor(arg)) {
    ret = dispatchDTensorRedistributeIfNeeded(
        *opt_op, torch_api_function, args, kwargs, opt_stack);
    // This returns redistributed args, then continues to normal dispatch
}

// EXISTING: Handle DTensor compute in __torch_dispatch__ path
if (!is_torch_function && is_dtensor(arg)) {
    ret = dispatchDTensorOp(
        *opt_op, torch_api_function, args, kwargs, opt_stack);
}
```

#### Option B: Add Python Logic in DTensor.__torch_function__

```python
# _api.py

@classmethod
def __torch_function__(cls, func, types, args, kwargs=None):
    kwargs = kwargs or {}

    # NEW: Check and perform redistribution at DTensor level
    needs_redist, redist_schema = cls._check_redistribution_needed(func, args, kwargs)

    if needs_redist:
        # This goes through Redistribute autograd.Function!
        args, kwargs = cls._redistribute_inputs(args, kwargs, redist_schema)

    # Continue with normal dispatch (now with correct placements)
    out = super().__torch_function__(func, types, args, kwargs)

    # Existing hook registration
    def _register_grad_placement_hook(t):
        if isinstance(t, torch.Tensor) and t.grad_fn is not None:
            _ = _DTensorGradPlacementHook(t.grad_fn, args)
        return t

    tree_map(_register_grad_placement_hook, out)
    return out
```

#### Option C: Hybrid Approach (Recommended)

1. **C++ fast path**: In `__torch_function__`, do quick check if redistribution needed
   - If no redistribution: skip to `__torch_dispatch__` (fast path)
   - If redistribution needed: call Python `Redistribute.apply()`

2. **Python handles redistribution**: `Redistribute.apply()` is already autograd-aware

3. **C++ `__torch_dispatch__`**: Simplified - just unwrap, compute, wrap (no redistribution)

### New Function: dispatchDTensorRedistributeIfNeeded

```cpp
// python_variable.cpp

py::object dispatchDTensorRedistributeIfNeeded(
    const c10::OperatorHandle& op,
    py::handle py_op,
    py::handle args,
    py::handle kwargs,
    torch::jit::Stack* stack) {

    // 1. Quick sharding propagation (reuse existing logic)
    auto output_sharding = propagate_sharding_fast(op, stack);

    if (!output_sharding.needs_redistribute) {
        // Fast path: no redistribution needed
        // Return NotImplemented to continue normal dispatch
        return py::reinterpret_borrow<py::object>(Py_NotImplemented);
    }

    // 2. Redistribution needed - call Python Redistribute.apply()
    py::list new_args;
    for (size_t i = 0; i < PyTuple_Size(args.ptr()); i++) {
        py::object arg = py::reinterpret_borrow<py::object>(
            PyTuple_GetItem(args.ptr(), i));

        if (is_dtensor(arg.ptr())) {
            auto target_spec = output_sharding.redistribute_schema.args_schema[i];
            if (needs_redistribution(arg, target_spec)) {
                // Call Redistribute.apply() - this is autograd-visible!
                arg = call_redistribute_apply(arg, target_spec);
            }
        }
        new_args.append(arg);
    }

    // 3. Re-call the op with redistributed args
    //    This will go through __torch_function__ again, but now
    //    redistribution check will return false (inputs are correct)
    return PyObject_Call(py_op.ptr(), py::tuple(new_args).ptr(), kwargs.ptr());
}
```

### Changes Required

| File | Change |
|------|--------|
| `python_arg_parser.cpp:316` | Add DTensor check in `__torch_function__` path |
| `python_variable.cpp` | Add `dispatchDTensorRedistributeIfNeeded()` |
| `python_variable.cpp:1401` | Simplify `dispatchDTensorOp()` - remove redistribution |
| `_dispatch.py:446` | Remove `redistribute_local_args()` or assert never called |
| `_api.py:423` | (Option B) Add redistribution logic to `__torch_function__` |

### Performance Considerations

1. **Extra sharding propagation call?**
   - Can cache result between `__torch_function__` and `__torch_dispatch__`
   - Use thread-local storage keyed by (op, input_specs)

2. **Re-dispatch after redistribution?**
   - One extra `__torch_function__` call per op that needs redistribution
   - Most ops don't need redistribution, so fast path is common

3. **Python vs C++ redistribution check?**
   - Prefer C++ for the "does this need redistribution?" check
   - Python only invoked when redistribution is actually needed

### Backward Compatibility

The existing `Redistribute` autograd.Function in `_redistribute.py:925` already
handles forward/backward correctly:

```python
class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, device_mesh, placements, ...):
        ctx.current_spec = input._spec  # Save for backward
        # ... perform redistribution ...

    @staticmethod
    def backward(ctx, grad_output):
        previous_spec = ctx.current_spec
        # ... redistribute gradient back ...
```

By using this path instead of `redistribute_local_tensor`, we get correct
backward behavior automatically.

### Open Questions

1. **Should backward be symmetric or asymmetric?**
   - Current `Redistribute.backward()` has optimizations (skip replicate→partial)
   - May want to make this configurable

2. **How to handle ops that deliberately want asymmetric forward/backward?**
   - Could add opt-out mechanism
   - Or keep `redistribute_local_args` for specific ops

3. **What about `torch.compile`?**
   - Need to verify tracing captures redistributes correctly
   - May need special handling for AOTAutograd
