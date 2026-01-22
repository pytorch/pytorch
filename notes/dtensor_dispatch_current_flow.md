# How DTensor Dispatch Currently Works

## High-Level Call Chain

```
torch.add(dtensor_a, dtensor_b)
    │
    ▼ C++ arg parsing detects DTensor
handle_torch_function_no_python_arg_parser()  [python_arg_parser.cpp:529]
    │
    ├─── Determines: TorchFunctionName::TorchDispatch (not TorchFunction)
    │
    ▼
dispatch_on_subclass()  [python_arg_parser.cpp:316]
    │
    ├─── Iterates through overloaded args
    │
    ├─── Line 378: if (!is_torch_function && is_dtensor(arg))
    │         └──► Special DTensor handling triggered
    │
    ▼
dispatchDTensorOp()  [python_variable.cpp:1401]
    │
    ├── 1. Custom op handler check
    ├── 2. Sharding propagation (with caching)
    ├── 3. Get local results (includes redistribution!)
    ├── 4. Wrap results
    │
    └──► Return DTensor
```

## Why DTensor Goes Through __torch_dispatch__ Path

In `dispatch_on_subclass()` at line 378:

```cpp
if (!is_torch_function && is_dtensor(arg)) {
    // Special fast path for DTensor
    if (opt_op && opt_stack) {
        ret = dispatchDTensorOp(*opt_op, torch_api_function, args, kwargs, opt_stack);
    } else {
        // Slow path - reconstruct stack
        ret = dispatchDTensorOp(op_handle, torch_api_function, args, kwargs, &stack);
    }
}
```

Key observation: `!is_torch_function` means this **only** happens in the `__torch_dispatch__` path.

DTensor has a `__torch_dispatch__` implementation that raises NotImplementedError:

```python
# _api.py:437
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    # We just need to have an implementation here; the __torch_dispatch__ machinery
    # calls into a specific C++ fast path that doesn't call here.
    raise NotImplementedError("DTensor.__torch_dispatch__ should not actually get called")
```

The C++ code intercepts DTensor **before** calling the Python `__torch_dispatch__` method.

## Inside dispatchDTensorOp: Step-by-Step

### Step 1: Custom Op Handlers (Lines 1410-1425)

```cpp
const auto custom_op_handlers = op_dispatcher.attr("_custom_op_handlers");
PyObject* custom_op_handler = PyDict_GetItemWithError(custom_op_handlers.ptr(), py_op.ptr());
if (custom_op_handler) {
    auto result = checked_vectorcall(custom_op_handler, py_op.ptr(), args.ptr(), kwargs.ptr());
    stack->clear();
    return result;
}
```

Special ops like `convolution`, `argmin`, `argmax` have custom handlers that bypass normal dispatch.

### Step 2: Sharding Propagation with Caching (Lines 1428-1481)

```cpp
// Try C++ fast path first
auto opt_native_op_schema = create_native_op_schema(op, py_op, stack);
if (opt_native_op_schema.has_value()) {
    native_sharding_propagator_cache = &get_thread_local_native_sharding_propagator_cache();
    cached_sharding = native_sharding_propagator_cache->find(opt_native_op_schema->first);
}

py::object py_op_info;
if (!cached_sharding) {
    // Cache miss - compute sharding via Python
    py_op_info = checked_vectorcall(
        op_dispatcher.attr("unwrap_to_op_info").ptr(),
        py_op.ptr(), args.ptr(), kwargs.ptr());

    py::object sharding = checked_vectorcall(
        op_dispatcher.attr("_propagate_op_sharding_dispatch_slow_path").ptr(),
        py_op.ptr(), args.ptr(), kwargs.ptr(), py_op_info.ptr(),
        /*try_cache*/ !opt_native_op_schema.has_value() ? Py_True : Py_False);

    cached_sharding = sharding;

    // Cache the result
    if (opt_native_op_schema.has_value()) {
        native_sharding_propagator_cache->insert(
            std::move(opt_native_op_schema->first), std::move(sharding));
    }
}
```

**What gets computed:**
- `unwrap_to_op_info()` - Extracts DTensor specs and local tensors from args
- `_propagate_op_sharding_dispatch_slow_path()` - Determines output sharding and **whether redistribution is needed**

The result is an `OutputSharding` object containing:
- `output_spec` - What placements the output will have
- `needs_redistribute` - Boolean flag
- `redistribute_schema` - If redistribution needed, what target placements for inputs

### Step 3: Get Local Results - THE CRITICAL PART (Lines 1509-1525)

```cpp
const bool participating = !checked_vectorcall(
    compute_mesh.attr("get_coordinate").ptr()).is_none();

const bool local_results_success = get_local_results(
    op, cached_sharding, compute_mesh, participating, stack);

py::object py_local_results;
if (local_results_success) {
    py_local_results = torch::jit::createPyObjectForStack(std::move(*stack));
} else {
    get_py_op_info_if_needed();
    py_local_results = checked_vectorcall(
        get_dtensor_get_local_results_slow_path().ptr(),
        py_op.ptr(), args.ptr(), py_op_info.ptr());
}
```

This calls `_dispatch_get_local_results_slow_path()` in Python, which is where **redistribution happens below autograd**.

### Step 4: Wrap Results (Lines 1539-1614)

Fast path for common cases, or falls back to Python `_dispatch_fast_path_python_tail()`.

## The Redistribution Problem: Detailed Trace

### In Python: _dispatch.py:242

```python
def _dispatch_get_local_results_slow_path(self, op_call, args, op_info):
    output_sharding = op_info.output_sharding
    mesh = op_info.compute_mesh
    participating = mesh.get_coordinate() is not None
    local_results = None

    if participating:
        # *** THIS IS THE PROBLEM ***
        if output_sharding.needs_redistribute:
            assert output_sharding.redistribute_schema is not None
            self.redistribute_local_args(
                op_info,
                output_sharding.redistribute_schema,
                output_sharding.use_val_from_redistribute_schema,
            )

        # Run local op with potentially redistributed args
        local_tensor_args = pytree.tree_unflatten(
            cast(list[object], op_info.local_args),
            op_info.args_tree_spec,
        )
        local_results = op_call(*local_tensor_args, **op_info.local_kwargs)

    return local_results
```

### The redistribute_local_args Function: Lines 446-503

```python
@staticmethod
def redistribute_local_args(
    op_info: OpInfo,
    suggested_input_schema: OpSchema,
    use_val_from_redistribute_schema: bool,
) -> None:
    new_local_args: list[object] = []

    for i, arg_spec in enumerate(op_info.flat_args_schema):
        reshard_arg_spec = flatten_args_schema_to_reshard[i]

        if isinstance(arg_spec, DTensorSpec):
            local_tensor = cast(torch.Tensor, op_info.local_args[i])

            if arg_spec != reshard_arg_spec:  # Specs differ!
                # *** THIS IS WHERE REDISTRIBUTION HAPPENS ***
                # It operates on LOCAL TENSORS, not DTensors!
                # Autograd never sees this!
                resharded_local_tensor = redistribute_local_tensor(
                    local_tensor,      # Just a torch.Tensor
                    arg_spec,          # Current DTensorSpec
                    reshard_arg_spec,  # Target DTensorSpec
                )
                new_local_args.append(resharded_local_tensor)
            else:
                new_local_args.append(local_tensor)
        else:
            new_local_args.append(arg_spec)

    # Mutate op_info in place with redistributed local tensors
    op_info.local_args = tuple(new_local_args)
```

**Key issue**: `redistribute_local_tensor()` takes a raw `torch.Tensor` and performs collectives on it.

### redistribute_local_tensor: _redistribute.py:789

```python
def redistribute_local_tensor(
    local_tensor: torch.Tensor,  # NOT a DTensor!
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    use_graph_based_transform: bool | None = None,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """
    new_local_tensor = local_tensor

    for transform_info in transform_infos:
        i = transform_info.mesh_dim
        current, target = transform_info.src_dst_placements

        if target.is_replicate():
            if current.is_partial():
                # All-reduce
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_value(
                    local_tensor, device_mesh, i
                )
            elif current.is_shard():
                # All-gather
                current_placement = cast(Shard, current)
                new_local_tensor = current_placement._to_replicate_tensor(
                    local_tensor, device_mesh, i, transform_info.logical_shape
                )

        elif target.is_shard():
            if current.is_partial():
                # Reduce-scatter
                partial_spec = cast(Partial, current)
                new_local_tensor = partial_spec._reduce_shard_value(
                    local_tensor, device_mesh, i, target_placement
                )
            elif current.is_replicate():
                # Chunk (local op)
                new_local_tensor = target_placement._replicate_to_shard(
                    local_tensor, device_mesh, i, my_coordinate[i]
                )
            else:
                # All-to-all
                shard_spec = cast(Shard, current)
                new_local_tensor = shard_spec._to_new_shard_dim(
                    local_tensor, device_mesh, i,
                    transform_info.logical_shape, target_placement.dim,
                )

        # ... more cases ...
        local_tensor = new_local_tensor

    return new_local_tensor
```

These collectives (`_reduce_value`, `_to_replicate_tensor`, etc.) call into `torch.distributed._functional_collectives` which are **NOT autograd.Functions** - they're just raw collective operations.

## Why This Is Below Autograd

### The Autograd Graph Sees This:

```
Forward:
    Input: DTensor(local_tensor_a)
        ↓
    [redistribute happens HERE - invisible to autograd]
        ↓
    op_call(redistributed_local_a, local_b)  ← autograd sees this
        ↓
    Output: result
```

### Not This:

```
Forward (what we want):
    Input: DTensor(local_tensor_a)
        ↓
    Redistribute.apply(DTensor_a, target_spec)  ← autograd should see this
        ↓
    DTensor_a_redistributed
        ↓
    op_call(DTensor_a_redistributed, DTensor_b)
        ↓
    Output: result
```

### Colleague's Comment #1

> "The problem is that when DTensor implicitly redistributes, this happens below the level of autograd.
> So autograd doesn't see the redistribute call, and thus wouldn't generate a dedicated backward for the redistribute"

**Explanation**:
- `redistribute_local_tensor()` operates on `torch.Tensor` objects (local shards)
- These are the **unwrapped** tensors extracted from DTensor
- By the time we redistribute, we're already "inside" the DTensor operation
- Autograd only sees the outer `DTensor` → `DTensor` operation
- The redistribute happens on the local tensors between unwrap and local op execution

### Colleague's Comment #2

> "Well it's more intentional, I think- the redistribute in dispatch avoids recording a backwards specifically
> to allow backwards to behave differently. If we are aiming to change that the recording the redistribute
> seems like an obvious thing to try"

**Explanation**:

There IS an autograd.Function for redistribution - `Redistribute` in `_redistribute.py:925`:

```python
class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, device_mesh, placements, async_op=False, ...):
        ctx.current_spec = current_spec
        output = redistribute_local_tensor(local_tensor, current_spec, target_spec, async_op=async_op)
        return dtensor.DTensor(output, target_spec, requires_grad=input.requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        previous_spec = ctx.current_spec
        # Skip replicate → partial transformation in backward
        # This optimization is WHY redistribution was kept below autograd
        normalized_placements = []
        for current, target in zip(current_spec.placements, previous_spec.placements):
            if (current.is_shard() or current.is_replicate()) and target.is_partial():
                normalized_placements.append(Replicate())  # Optimization!
            else:
                normalized_placements.append(target)

        output = redistribute_local_tensor(local_tensor, current_spec, previous_spec, ...)
        return output_dtensor, None, None, None, None, None
```

The backward has special logic:
- Forward might go `Replicate → Partial` to prepare for a reduction
- Backward **skips** `Partial → Replicate` because it's more efficient to keep as `Replicate`

**This asymmetric forward/backward behavior is why redistribution was kept implicit** -
using the explicit `Redistribute.apply()` would force symmetric behavior unless the backward
has this special logic.

## The Explicit Redistribute Path (Unused by Dispatch)

When users call `dtensor.redistribute()` explicitly:

```python
# _api.py:617
def redistribute(
    self,
    device_mesh: DeviceMesh | None = None,
    placements: Sequence[Placement] | None = None,
    *,
    async_op: bool = False,
) -> DTensor:
    # This DOES go through autograd!
    return Redistribute.apply(
        self,
        device_mesh or self.device_mesh,
        placements or self.placements,
        async_op,
    )
```

This path IS visible to autograd because it uses `Redistribute.apply()`.

The problem is **implicit redistribution in dispatch** uses `redistribute_local_tensor()` directly,
bypassing the autograd.Function.

## Summary: The Flow Visualized

```
┌─────────────────────────────────────────────────────────────────┐
│ torch.add(DTensor_a, DTensor_b)                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ handle_torch_function_no_python_arg_parser                      │
│ - Determines: TorchDispatch (not TorchFunction)                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ dispatch_on_subclass                                            │
│ - Line 378: if (!is_torch_function && is_dtensor(arg))         │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ dispatchDTensorOp (C++)                                         │
│                                                                 │
│  1. Check custom handlers                                      │
│  2. Propagate sharding (with caching)                          │
│     ├─ unwrap_to_op_info(): DTensor → (spec, local_tensor)    │
│     └─ _propagate_op_sharding_dispatch_slow_path()            │
│         Returns: OutputSharding                                │
│           ├─ output_spec: Output placements                    │
│           ├─ needs_redistribute: bool                          │
│           └─ redistribute_schema: OpSchema (if needed)         │
│                                                                 │
│  3. Get local results                                          │
│     └─ _dispatch_get_local_results_slow_path() (Python)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ _dispatch_get_local_results_slow_path (Python)                 │
│                                                                 │
│  if output_sharding.needs_redistribute:                        │
│      redistribute_local_args(op_info, ...)  ◄── BELOW AUTOGRAD │
│          │                                                      │
│          └─► redistribute_local_tensor(                        │
│                  local_tensor,        ← torch.Tensor          │
│                  current_spec,                                 │
│                  target_spec)                                  │
│                  │                                              │
│                  └─► Performs collectives:                     │
│                      - all_gather                              │
│                      - all_reduce                              │
│                      - reduce_scatter                          │
│                      - all_to_all                              │
│                      (All invisible to autograd!)              │
│                                                                 │
│  local_results = op_call(*local_tensor_args)                   │
│                           ▲                                     │
│                           └─ Already redistributed!            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ dispatchDTensorOp (C++) - continued                            │
│                                                                 │
│  4. Wrap results                                               │
│     └─ OpDispatcher.wrap(local_results, output_spec)          │
│         Returns: DTensor                                       │
└─────────────────────────────────────────────────────────────────┘
```

## What Autograd Actually Sees

```python
# Autograd graph structure (simplified)
AddBackward0
  ├─ AccumulateGrad (for DTensor_a)
  └─ AccumulateGrad (for DTensor_b)

# Missing: RedistributeBackward (because it happened below autograd!)
```

## What We Want Autograd to See

```python
# Desired autograd graph structure
AddBackward0
  ├─ RedistributeBackward (for DTensor_a)  ◄── MISSING CURRENTLY
  │   └─ AccumulateGrad
  └─ AccumulateGrad (for DTensor_b)
```

This is why the solution is to move redistribution to `__torch_function__` where it happens
**before** the operation enters autograd, using `Redistribute.apply()` which creates the
`RedistributeBackward` node.
