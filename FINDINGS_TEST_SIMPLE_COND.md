# Findings from test_simple_cond.py

## Date: 2024-11-19

## Summary
Successfully verified that torch.cond with dynamic=True compilation produces the desired behavior: multiple kernel implementations are generated and dispatched correctly based on runtime tensor dimensions.

## Test Results

### ✅ Test 1: Basic torch.cond + Dynamic Compilation
**Status: PASSED**

All three test cases passed:
- Short sequence (dim=256) correctly uses sin implementation
- Medium sequence (dim=1024) correctly uses tanh implementation
- Long sequence (dim=4096) correctly uses relu implementation

### ✅ Test 2: Dynamic Compilation Behavior
**Status: PASSED**

Multiple runs with different dimensions completed without errors or recompilation warnings.

## Key Findings

### 1. torch.cond is Properly Traced with Symbolic Shapes ✅

From the log (`log_simple_cond.txt`), the traced graph shows:
```python
# Symbolic dimension preserved
le: "Sym(s27 <= 512)" = arg1_1 <= 512

# Nested torch.cond structure preserved
cond = torch.ops.higher_order.cond(le, true_graph_0, false_graph_0, ...)
```

- Dimensions are symbolic (s27, s53, s77) ✅
- torch.cond predicates use symbolic comparison ✅
- Nested structure is preserved ✅

### 2. Three Different Triton Kernels are Generated ✅

Inductor generates **three separate Triton kernels**, one for each implementation:

**Kernel 1: sin implementation**
```python
def triton_poi_fused_mul_sin_view_0(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK):
    # ...
    tmp2 = tmp0 * tmp1
    tmp3 = tl_math.sin(tmp2)  # ← SIN operation
    tl.store(out_ptr0 + (x2), tmp3, xmask)
```

**Kernel 2: tanh implementation**
```python
def triton_poi_fused_mul_tanh_view_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK):
    # ...
    tmp2 = tmp0 * tmp1
    tmp3 = libdevice.tanh(tmp2)  # ← TANH operation
    tl.store(out_ptr0 + (x2), tmp3, xmask)
```

**Kernel 3: relu implementation**
```python
def triton_poi_fused_mul_relu_view_2(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK):
    # ...
    tmp2 = tmp0 * tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)  # ← RELU operation (max(0, x))
    tl.store(out_ptr0 + (x2), tmp4, xmask)
```

### 3. Dispatch Functions are Created for Each Branch ✅

Inductor creates wrapper functions for each conditional branch:
- `true_graph_0()` - for first branch (sin)
- `false_graph_0_true_graph_0()` - for nested true branch (tanh)
- `false_graph_0_false_graph_0()` - for nested false branch (relu)

Each wrapper:
- Takes symbolic size arguments (s27, s53, s77)
- Allocates output buffer with symbolic shapes
- Calls the appropriate Triton kernel
- Returns the result

### 4. Dynamic Shape Handling Works Correctly ✅

The kernels handle dynamic shapes via:
- Symbolic size variables passed as runtime arguments
- `xnumel = s27*s53*s77` - computed at runtime
- `assert_size_stride()` checks ensure shape consistency
- Same compiled kernel used for all input sizes

## Implications for Custom Op Range-Based Dispatch

### ✅ Feasibility Confirmed
The test demonstrates that the proposed torch.cond-based approach is **fully feasible**:

1. **torch.cond preserves symbolic shapes** - Critical for dynamic compilation
2. **Multiple implementations are compiled** - Each branch gets its own kernel
3. **Runtime dispatch works** - Correct kernel is selected based on dimension
4. **No recompilation needed** - Single compilation handles all sizes

### ✅ How it Works

When we compile:
```python
def dispatch(x, weight):
    dim_size = x.shape[1]  # This becomes symbolic: s27
    return torch.cond(
        dim_size <= 512,
        lambda: short_impl(x, weight),
        lambda: torch.cond(
            dim_size <= 2048,
            lambda: medium_impl(x, weight),
            lambda: long_impl(x, weight)
        )
    )
```

Inductor:
1. Traces the function with symbolic shapes
2. Preserves torch.cond operations in the graph
3. Compiles each branch (lambda) into a separate kernel
4. Generates runtime dispatch code that evaluates predicates
5. At runtime, checks the actual dimension value and calls appropriate kernel

### Comparison: torch.cond vs SubgraphBuffer

| Aspect | SubgraphBuffer (Current) | torch.cond (Proposed) |
|--------|-------------------------|----------------------|
| Graph representation | External Python dispatch | Part of traced graph ✅ |
| Dispatch code | Python if/else | torch.cond op ✅ |
| Symbolic shapes | Limited | Full support ✅ |
| Optimization potential | Each range separate | Cross-branch sharing ✅ |
| Code elegance | Hacky | Clean ✅ |
| Compilation | Per-range subgraphs | Unified with branches ✅ |

## Next Steps

### Phase 2: Test make_fx Inside Inductor Context

Now we need to verify that we can use `make_fx` with `tracing_mode="symbolic"` inside the inductor lowering function:

```python
def _range_based_lowering_fn(...):
    # Inside inductor lowering
    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)

        # Create torch.cond dispatch
        def dispatch_fn(*tensors):
            x = tensors[0]
            dim = x.shape[dim_index]  # Symbolic!
            return build_torch_cond(..., dim, ...)

        # Trace with symbolic shapes
        dispatch_gm = make_fx(
            dispatch_fn,
            tracing_mode="symbolic",
            decomposition_table=select_decomp_table()
        )(*fake_inputs)

    # Inline to IR
    result = inline_subgraph_to_ir_nodes(dispatch_gm, tensor_inputs, name)
```

**Key questions:**
- Does V.fake_mode provide symbolic shapes? ✅ (observed in existing code)
- Can make_fx trace torch.cond with symbolic predicates? (Need to test)
- Can inline_subgraph_to_ir_nodes handle the resulting graph? (Need to test)

### Phase 3: Implementation in custom_op.py

Once Phase 2 is verified, we can implement:
1. `_build_torch_cond_dispatch()` helper
2. Modify `_range_based_lowering_fn()` to use torch.cond instead of SubgraphBuffer
3. Test with actual custom op autotuning

## Conclusion

✅ **The torch.cond approach is viable and superior to SubgraphBuffer**

The test demonstrates that:
- torch.cond compiles correctly with dynamic=True
- Multiple implementations are preserved as separate kernels
- Runtime dispatch based on symbolic dimensions works
- The approach is cleaner and more PyTorch-native

We can proceed with confidence to implement the proposed solution in `custom_op.py`.

## Files
- Test file: `/data/users/tianren/pytorch/test_simple_cond.py`
- Log file: `/data/users/tianren/pytorch/log_simple_cond.txt`
- Plan document: `/data/users/tianren/pytorch/TORCH_COND_DISPATCH_PLAN.md`
