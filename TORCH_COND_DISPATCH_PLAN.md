# torch.cond-based Dynamic Range Dispatch Implementation Plan

## Current Status (Understanding)

### Current "Hacky" Implementation
Based on my analysis of `/data/users/tianren/pytorch/torch/_inductor/kernel/custom_op.py`:

1. **Range-based Autotuning** (Lines 632-788 in `_range_based_lowering_fn`):
   - For each range (defined by split_points), autotune all implementations
   - Store winning implementation per range in `range_to_best_impl`
   - Merge consecutive ranges with identical implementations
   - If all ranges use same impl → directly inline (no dispatch needed)
   - Otherwise → create SubgraphBuffer with multi-range dispatch

2. **SubgraphBuffer Multi-Range Dispatch** (Lines 6993-7033 in `/data/users/tianren/pytorch/torch/_inductor/ir.py`):
   - Generates Python runtime if-else dispatch code
   - Each range gets its own subgraph compiled independently
   - At runtime, checks dimension size and calls appropriate subgraph
   - Example generated code:
     ```python
     def dispatch(args):
         arg0, arg1 = args
         dispatch_size = arg0.size(1)
         if 1 <= dispatch_size <= 512:
             return subgraph_range_1_512(args)
         elif 513 <= dispatch_size <= 2048:
             return subgraph_range_513_2048(args)
         else:
             return subgraph_range_2049_inf(args)
     ```

### Problems with Current Approach
- SubgraphBuffer is a "hack" - not elegant
- Each range is compiled separately, missing optimization opportunities
- Runtime dispatch in Python, not part of the traced graph
- Cannot benefit from dynamic shape compilation

## Proposed Solution: torch.cond-based Dispatch

### Key Idea
Instead of using SubgraphBuffer with Python if-else dispatch, create a dispatch function using `torch.cond` and compile it with `dynamic=True`:

```python
def dispatch_fn(x, weight):
    dim = x.shape[1]
    # Nested torch.cond for multiple ranges
    return torch.cond(
        dim <= 512,
        lambda: short_impl(x, weight),
        lambda: torch.cond(
            dim <= 2048,
            lambda: medium_impl(x, weight),
            lambda: long_impl(x, weight)
        )
    )

# Compile with dynamic shapes
compiled_dispatch = make_fx(dispatch_fn, dynamic=True)(example_inputs...)
# Lower to inductor
```

### Expected Benefits
1. **Elegant Graph Representation**: torch.cond is part of the traced graph
2. **Dynamic Shape Handling**: With `dynamic=True`, the compiler understands the dimension is dynamic
3. **Unified Kernel**: Single kernel with conditional logic instead of separate subgraphs
4. **Better Optimization**: Inductor can optimize across branches, share computations
5. **Range Merging**: Identical implementations can be merged into single cond predicate

### Implementation Strategy

#### Phase 1: Create Simple Test (Verify Feasibility)
Test that torch.cond + dynamic shapes works as expected:

```python
# Test 1: Simple dispatch with 3 implementations
def test_basic_torch_cond():
    def dispatch(x):
        dim = x.shape[1]
        return torch.cond(
            dim <= 100,
            lambda: x * 2,  # Simple impl
            lambda: torch.cond(
                dim <= 500,
                lambda: torch.tanh(x),  # Medium impl
                lambda: torch.relu(x)   # Long impl
            )
        )

    # Compile with dynamic=True
    compiled = torch.compile(dispatch, dynamic=True)

    # Test with multiple sizes - should create dynamic kernel
    x1 = torch.randn(2, 50, 128)   # Uses x*2
    x2 = torch.randn(2, 300, 128)  # Uses tanh
    x3 = torch.randn(2, 1000, 128) # Uses relu

    # Verify correct results and check generated code
    # Expected: Single kernel with torch.cond logic
```

**Key Questions to Answer:**
- ✅ Does torch.cond get traced properly with dynamic shapes?
- ✅ Does the lowered code show the conditional logic?
- ✅ Are the different implementations (sin/tanh/relu) visible in the kernel?
- ✅ Does dynamic shape trigger recompilation or does it handle all sizes?

#### Phase 2: Test make_fx Inside Inductor
Verify we can use make_fx with dynamic=True inside inductor to re-trace:

```python
# Inside _range_based_lowering_fn
with V.fake_mode:
    fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)

    # Create dispatch function
    def dispatch_fn(*tensors):
        x = tensors[0]
        dim = x.shape[dim_index]
        # Build nested torch.cond based on ranges
        return build_cond_dispatch(ranges, implementations, tensors)

    # Trace with dynamic shapes
    from torch.fx.experimental.proxy_tensor import make_fx
    dispatch_gm = make_fx(
        dispatch_fn,
        tracing_mode="symbolic",  # Use symbolic for dynamic shapes
        decomposition_table=select_decomp_table()
    )(*fake_inputs)
```

**Key Questions to Answer:**
- Can we access V.fake_mode inside lowering function?
- Does symbolic tracing preserve dynamic dimensions?
- Can we inline the resulting graph module to IR nodes?

#### Phase 3: Build Nested torch.cond from Ranges
Helper function to construct nested torch.cond based on merged ranges:

```python
def build_nested_torch_cond(
    ranges: list[tuple[tuple[int, int], Callable]],
    tensors: tuple[torch.Tensor, ...],
    dim_symbol: torch.SymInt,
    runtime_kwargs: dict
) -> torch.Tensor:
    """Build nested torch.cond for range dispatch.

    Args:
        ranges: List of ((start, end), impl_fn) tuples
        tensors: Input tensors
        dim_symbol: Symbolic dimension to dispatch on
        runtime_kwargs: Runtime kwargs to pass to impls

    Returns:
        Result tensor from appropriate implementation
    """
    if len(ranges) == 1:
        # Base case: single implementation
        (_, impl_fn) = ranges[0]
        return impl_fn(*tensors, **runtime_kwargs)

    # Recursive case: split at first range
    (start, end), impl_fn = ranges[0]
    rest_ranges = ranges[1:]

    return torch.cond(
        dim_symbol <= end,
        lambda: impl_fn(*tensors, **runtime_kwargs),
        lambda: build_nested_torch_cond(
            rest_ranges, tensors, dim_symbol, runtime_kwargs
        )
    )
```

#### Phase 4: Modify _range_based_lowering_fn
Replace SubgraphBuffer creation with torch.cond dispatch:

```python
def _range_based_lowering_fn(...):
    # ... existing autotuning code ...

    # After merging ranges
    merged_range_to_best_impl = _merge_identical_implementations(range_to_best_impl)

    # If single impl, inline directly (existing optimization)
    if len(merged_range_to_best_impl) == 1:
        return _lower_single_impl(...)

    # NEW: Create torch.cond dispatch instead of SubgraphBuffer
    from torch.fx.experimental.proxy_tensor import make_fx
    from ..decomposition import select_decomp_table

    # Build list of (range, impl, kwargs) sorted by range
    sorted_ranges = sorted(merged_range_to_best_impl.items())

    # Create dispatch function with torch.cond
    def cond_dispatch_fn(*tensors):
        dispatch_tensor = tensors[0]  # Assume first tensor for dispatch
        dim_value = dispatch_tensor.shape[dim_index]

        # Build nested cond
        return _build_torch_cond_dispatch(
            sorted_ranges, tensors, dim_value, runtime_kwargs
        )

    # Trace with symbolic shapes
    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()

        dispatch_gm = make_fx(
            cond_dispatch_fn,
            decomposition_table=decomposition_table,
            tracing_mode="symbolic",  # Critical for dynamic shapes
        )(*fake_inputs)

    # Inline the dispatch graph to IR nodes
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    result = inline_subgraph_to_ir_nodes(dispatch_gm, tensor_inputs, name)
    validate_ir(result)
    return result
```

### Range Merging Optimization
Already implemented in `_merge_identical_implementations` (lines 204-250):
- Merges consecutive ranges with same impl+kwargs
- This naturally reduces the number of torch.cond branches
- If all ranges merge to one, we skip dispatch entirely

Example:
```
Before merge:
  [1, 512] -> short_impl
  [513, 1024] -> short_impl
  [1025, 2048] -> medium_impl
  [2049, inf] -> long_impl

After merge:
  [1, 1024] -> short_impl
  [1025, 2048] -> medium_impl
  [2049, inf] -> long_impl

Generated cond:
  torch.cond(dim <= 1024,
             lambda: short_impl(...),
             lambda: torch.cond(dim <= 2048,
                                lambda: medium_impl(...),
                                lambda: long_impl(...)))
```

## Testing Plan

### Test 1: Basic torch.cond Compilation
File: `test_simple_cond.py`
- Simple dispatch with 3 different implementations (sin/tanh/relu)
- Compile with `dynamic=True`
- Test with multiple input sizes
- Verify generated code shows cond logic
- Check that all 3 implementations are in the kernel

### Test 2: make_fx with Dynamic Shapes Inside Inductor
File: `test_dynamic_fake_cond.py`
- Simulate inductor environment (V.fake_mode)
- Create symbolic shapes using torch.fx
- Trace dispatch function with make_fx(tracing_mode="symbolic")
- Verify symbolic dimensions are preserved
- Check that lowering works correctly

### Test 3: Full Integration Test
File: `test_new_torch_cond_impl.py` (already exists)
- Use actual custom_op autotuning
- Force different impls per range (for testing)
- Verify correct dispatch behavior
- Check generated code for torch.cond and multiple kernels

### Test 4: Same Implementation for All Ranges
- Verify optimization: should inline directly without torch.cond
- No conditional dispatch needed

## Expected Output Patterns

### With torch.cond (multiple implementations):
```python
# Triton kernel with conditional logic
@triton.jit
def kernel(...):
    # Load data
    # Conditional based on symbolic dim
    if symbolic_dim <= 512:
        # short_impl logic (sin)
    elif symbolic_dim <= 2048:
        # medium_impl logic (tanh)
    else:
        # long_impl logic (relu)
    # Store result
```

### Without torch.cond (single implementation after merge):
```python
# Simple inlined kernel
@triton.jit
def kernel(...):
    # Direct implementation, no conditionals
    # Just the winning impl logic
```

## Open Questions & Risks

### Questions to Investigate:
1. ✅ **V.fake_mode access**: Can we use V.fake_mode in _range_based_lowering_fn?
   - Based on code analysis: Yes, it's used in current implementation (line 737)

2. ✅ **Symbolic tracing**: Does `tracing_mode="symbolic"` preserve dynamic dims?
   - Need to test, but this is the standard way in inductor

3. **torch.cond lowering**: Does inductor properly lower torch.cond to conditional kernels?
   - Need to verify with test

4. **Performance**: Is torch.cond dispatch as efficient as separate kernels?
   - May have branch prediction overhead
   - But gains from fusion and single kernel might offset

5. **Multiple dispatch dimensions**: Current design assumes single dim for dispatch
   - Future extension could support multi-dimensional dispatch

### Potential Issues:
1. **Closure capture**: Lambda functions in torch.cond may not capture all needed vars
   - Solution: Use functools.partial or careful closure design

2. **Non-tensor arguments**: How to pass runtime_kwargs through torch.cond?
   - May need to bind kwargs before creating lambdas

3. **Graph inlining**: Can inline_subgraph_to_ir_nodes handle torch.cond graphs?
   - Need to verify compatibility

## Implementation Checklist

- [ ] Create test_simple_cond.py - basic torch.cond + dynamic compilation
- [ ] Run test and verify generated code shows cond logic
- [ ] Create test_dynamic_fake_cond.py - test make_fx in fake_mode
- [ ] Verify symbolic shapes are preserved
- [ ] Implement _build_torch_cond_dispatch helper function
- [ ] Modify _range_based_lowering_fn to use torch.cond instead of SubgraphBuffer
- [ ] Test with test_new_torch_cond_impl.py
- [ ] Verify correctness: all 3 ranges dispatch correctly
- [ ] Verify performance: compare to SubgraphBuffer approach
- [ ] Handle edge cases: single range, all same impl, etc.
- [ ] Add logging and diagnostics
- [ ] Update documentation

## Next Steps (Immediate)

1. **Create and run test_simple_cond.py** to verify basic feasibility
2. **Examine generated code** to understand how torch.cond is lowered
3. **Test make_fx with symbolic tracing** inside a fake_mode context
4. **Design the _build_torch_cond_dispatch helper** based on test results
5. **Implement the new _range_based_lowering_fn** incrementally
6. **Test and iterate** until all tests pass

## References
- Current implementation: `/data/users/tianren/pytorch/torch/_inductor/kernel/custom_op.py`
- SubgraphBuffer: `/data/users/tianren/pytorch/torch/_inductor/ir.py` lines 6902-7034
- Test file: `/data/users/tianren/pytorch/test_new_torch_cond_impl.py`
