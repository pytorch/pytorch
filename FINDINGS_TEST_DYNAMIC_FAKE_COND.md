# Findings from test_dynamic_fake_cond.py

## Date: 2024-11-19

## Summary
Tested make_fx with FakeTensorMode and symbolic tracing to understand how torch.cond behaves when traced with fake tensors. **Important finding**: torch.cond gets constant-folded away when dimensions are concrete at trace time.

## Test Results

### ✅ Test 1: Basic Fake Mode Usage
**Status: PASSED**

- Successfully created FakeTensorMode
- Created fake tensors on CUDA
- Tensors are properly marked as fake

**Note**: Dimensions are still concrete (int) not symbolic when created with default shape_env.

### ✅ Test 2: make_fx with FakeTensorMode + ShapeEnv
**Status: PASSED (with observations)**

Successfully traced with make_fx using `tracing_mode="symbolic"`.

**Key Observation**:
```python
dispatch_dim = x_fake.shape[1]  # This is 256 (concrete int)
```

When `dispatch_dim` is concrete:
- torch.cond evaluates at trace time: `256 <= 512` → `True`
- Compiler takes the `True` branch unconditionally
- **torch.cond is optimized away** - only sin implementation remains
- No symbolic comparison in traced graph

**Traced graph shows**:
```python
%sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default]
# No torch.cond, no tanh, no relu
```

### ✅ Test 3: Dynamic Shapes Configuration
**Status: PASSED (same behavior)**

Even with `assume_static_by_default = False`, the behavior is the same:
- Dimensions are still concrete at trace time
- torch.cond is constant-folded
- Only one branch (sin) in traced graph

### ✅ Test 4: Simulated Inductor Context
**Status: PASSED (same behavior)**

Simulated the lowering context with:
- FakeTensorMode + ShapeEnv
- Creating fake tensors as IR node placeholders
- Tracing dispatch function

**Result**: Same as above - torch.cond optimized away.

### ✅ Test 5: Execute Traced Graph with Real Tensors
**Status: PASSED**

The traced graph (with only sin branch) executes correctly with real tensors.

## Critical Finding: torch.cond Optimization

### Why torch.cond Disappeared

When we trace:
```python
x_fake = torch.randn(2, 256, 128, device="cuda")
dispatch_dim = x_fake.shape[1]  # = 256 (concrete)

torch.cond(
    dispatch_dim <= 512,  # = 256 <= 512 → True (at trace time!)
    lambda: short_impl(...),  # ← This is always taken
    lambda: nested_cond(...),   # ← Never executed
)
```

The compiler sees:
- Predicate is constant `True`
- Only one branch can ever execute
- Optimizes away the cond, inlines the True branch
- Result: just the sin operation

### This is Actually Correct Behavior!

For `torch.compile(f, dynamic=True)`:
- User function is traced with example inputs
- Dimensions become symbolic *during dynamo tracing*
- torch.cond with symbolic predicates is preserved

But for `make_fx(f, tracing_mode="symbolic")`:
- We manually create fake tensors with concrete shapes
- The shape[i] access returns concrete value
- torch.cond evaluates at trace time

## The Real Question: What Happens in Inductor V.fake_mode?

Inside inductor's `_range_based_lowering_fn()`:

```python
with V.fake_mode:
    fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
    # ← Are these tensors' dimensions symbolic or concrete?
```

**Key question**: Does `ir_node_to_tensor()` create fake tensors with **symbolic** dimensions?

Looking at the existing code in `custom_op.py` line 737:
```python
with V.fake_mode:
    fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
    # This is used for tracing implementations
```

The code already uses this pattern, so **V.fake_mode MUST provide symbolic shapes** for dynamic compilation to work.

## Implications for Our Implementation

### Option 1: V.fake_mode Already Provides Symbolic Shapes ✅ (Most Likely)

If inductor's V.fake_mode already marks dimensions as symbolic:

```python
with V.fake_mode:
    fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
    x = fake_inputs[0]
    dim = x.shape[1]  # ← Should be symbolic SymInt, not concrete int

    def dispatch_fn(*tensors):
        x = tensors[0]
        dim = x.shape[dim_index]  # ← Symbolic
        return torch.cond(
            dim <= 512,  # ← Symbolic comparison: Sym(s1 <= 512)
            lambda: short_impl(...),
            lambda: nested_cond(...)
        )

    traced_gm = make_fx(dispatch_fn, tracing_mode="symbolic")(*fake_inputs)
    # ↑ torch.cond should be preserved!
```

**Then our approach will work as planned!**

### Option 2: We Need to Mark Dimensions as Dynamic

If V.fake_mode doesn't automatically provide symbolic shapes, we may need to explicitly mark them:

```python
from torch._dynamo.source import LocalSource
from torch.fx.experimental.symbolic_shapes import DimDynamic

with V.fake_mode:
    # Mark specific dimensions as dynamic
    fake_inputs = ...
    # May need torch._dynamo.mark_dynamic(fake_inputs[0], 1)
```

## What test_simple_cond.py Showed Us

In `test_simple_cond.py`, we used `torch.compile(f, dynamic=True)`:
- Dynamo traces the user function
- **Dynamo automatically marks shapes as symbolic**
- torch.cond with symbolic predicates is preserved
- Multiple kernels are generated

This is different from manually using make_fx!

## Conclusion & Next Steps

### What We Learned

1. ✅ **make_fx with FakeTensorMode works** - tracing succeeds
2. ✅ **Graph can be traced and executed** - no errors
3. ⚠️  **torch.cond gets optimized away with concrete dimensions** - expected behavior
4. ❓ **Need to verify: Does V.fake_mode provide symbolic dimensions?**

### Critical Test Needed

We need to verify what `ir_node_to_tensor()` returns inside V.fake_mode:

```python
# Inside inductor, in a lowering function
with V.fake_mode:
    fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
    x = fake_inputs[0]
    dim = x.shape[1]

    print(f"dim type: {type(dim)}")
    print(f"Is symbolic: {isinstance(dim, torch.SymInt)}")
    print(f"dim value: {dim}")
```

If `dim` is a `SymInt`, then:
- ✅ Our torch.cond approach will work
- ✅ Symbolic comparisons will be preserved
- ✅ All branches will be traced

If `dim` is a concrete `int`, then:
- ❌ torch.cond will be optimized away
- ❌ Need to find another way to mark dimensions as dynamic
- ❌ May need a different approach

### Recommendation

**Look at how existing code handles this in custom_op.py:**

Line 737-756 already traces implementations with V.fake_mode:
```python
with V.fake_mode:
    fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)

    def impl_wrapper(*tensors):
        return impl_fn(*tensors, **{**runtime_kwargs, **impl_kwargs})

    impl_gm = make_fx(
        impl_wrapper,
        decomposition_table=decomposition_table,
        tracing_mode="symbolic",
    )(*fake_inputs)
```

This code works for tracing implementations. **The same pattern should work for our torch.cond dispatch!**

The difference is:
- Existing code: traces a single impl without conditionals
- Our code: traces multiple impls with torch.cond conditionals

But the tracing mechanism should be the same.

### Most Likely Scenario ✅

Based on the existing working code, **V.fake_mode very likely provides symbolic shapes**, and our approach will work:

1. Inside inductor, V.fake_mode is set up for symbolic tracing
2. `ir_node_to_tensor()` creates fake tensors with SymInt dimensions
3. When we extract `x.shape[1]`, we get a SymInt
4. torch.cond with SymInt predicates is preserved in the graph
5. All three branches (sin/tanh/relu) are traced
6. Multiple kernels are generated

### Next Step

**Proceed with implementation** of `_build_torch_cond_dispatch()` and modified `_range_based_lowering_fn()`, following the existing pattern in lines 737-756.

If torch.cond gets optimized away during actual testing, we can debug then. But based on code analysis, it should work!

## Files
- Test file: `/data/users/tianren/pytorch/test_dynamic_fake_cond.py`
- Plan document: `/data/users/tianren/pytorch/TORCH_COND_DISPATCH_PLAN.md`
- Previous findings: `/data/users/tianren/pytorch/FINDINGS_TEST_SIMPLE_COND.md`
