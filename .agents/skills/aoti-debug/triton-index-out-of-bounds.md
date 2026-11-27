# AOTI Triton Index Out of Bounds Debug Guide

This guide helps debug AOTI Triton kernel assertion errors with the `index out of bounds` pattern.

## Error Pattern

This guide applies when you see errors like:

```
/var/tmp/torchinductor_*/.../*.py:NN: unknown: block: [X,Y,Z], thread: [X,Y,Z]
Assertion `index out of bounds: 0 <= tmpN < ksM` failed.
```

### Key Information from Error

| Field | Value | Meaning |
|-------|-------|---------|
| File Path | `/var/tmp/torchinductor_*/*.py` | Generated Triton kernel file (runtime) |
| Line Number | `:NN` | Line in the generated kernel where assertion failed |
| Block/Thread | `[X,Y,Z]` | CUDA block and thread indices |
| Assertion | `0 <= tmpN < ksM` | Index `tmpN` must be within bounds `[0, ksM)` |

### Understanding the Assertion

- `tmpN`: A computed index value in the Triton kernel
- `ksM`: A dynamic kernel size parameter (runtime value)
- The assertion fails when `tmpN < 0` or `tmpN >= ksM`

---

## Step 1: Collect AOTI Package

You need access to the AOTI package that was compiled. This is typically a `.pt2` package or extracted archive containing a `wrapper.cpp` file.

**Key File**: `*.wrapper.cpp` contains:
- All Triton kernel source code (embedded as comments)
- Kernel launch configurations
- Input/output tensor mappings
- Dynamic shape variable definitions

---

## Step 2: Locate the Failing Kernel in C++ Wrapper

### Search for the Assertion Pattern

Extract the assertion pattern from the error (e.g., `tmp18 < ks0`) and search:

```bash
# Search for the specific assertion
grep -n "tmpN < ksM" /path/to/*.wrapper.cpp

# Get context around the assertion (80 lines before, 20 after)
grep -n -B80 -A20 "tmpN < ksM" /path/to/*.wrapper.cpp
```

### Find the Full Kernel Definition

The kernel is embedded as a Python docstring comment in the C++ wrapper:

```cpp
    /*
    async_compile.triton('triton_red_fused_...', '''
    import triton
    import triton.language as tl
    ...
    def triton_red_fused_...(in_ptr0, out_ptr1, ks0, xnumel, r0_numel, ...):
```

---

## Step 3: Understand the Kernel Logic

Analyze the code path leading to the assertion. Common patterns that cause index out of bounds:

### Pattern: Empty Tensor with ks0 = 0

When a dynamic shape `ks0 = 0`:
1. `tmp13 = (-1) + 0 = -1`
2. Index wrapping logic produces `-1`
3. Assertion `0 <= -1 < 0` fails

### Example Kernel Pattern

```python
tmp13 = (-1) + ks0           # ks0 - 1
tmp14 = tl.where(tmp12, tmp10, tmp13)  # if condition: use tmp10, else: ks0-1
tmp15 = ks0
tmp16 = tmp14 + tmp15        # wrap-around for negative indices
tmp17 = tmp14 < 0
tmp18 = tl.where(tmp17, tmp16, tmp14)  # if negative: add ks0

# ASSERTION: 0 <= tmp18 < ks0
tl.device_assert(((0 <= tmp18) & (tmp18 < ks0)), "index out of bounds")
```

---

## Step 4: Identify the Dynamic Shape Variable

### Find Where the Kernel is Called

```bash
grep -n "call_triton_KERNEL_NAME" /path/to/*.wrapper.cpp
```

### Example Output

```cpp
call_triton_red_fused_...(arg1415_1, buf696, s607, 1L, s13, ...);
```

### Parameter Mapping

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `in_ptr0` | `arg1415_1` | Input tensor |
| `out_ptr1` | `buf696` | Output buffer |
| `ks0` | `s607` | **Dynamic shape - this is the failing bound** |

### Find the Definition of the Shape Variable

```bash
grep -n "int64_t s607 = " /path/to/*.wrapper.cpp
```

This shows which input tensor dimension defines the shape:

```cpp
int64_t s607 = arg1416_1_size[0];
```

---

## Step 5: Trace Back to Model Input

### Find Input Index

Inputs are numbered sequentially. Find which input the argument corresponds to:

```bash
grep -n 'inputs_info_\[INDEX\].name = "argNNN_1"' /path/to/*.wrapper.cpp
```

### Check Input Constraints

```bash
grep -n "argNNN_1_size\[0\]" /path/to/*.wrapper.cpp
```

Look for guards like:
```cpp
if (arg_size[0] > 230400) {  // Upper bound check only - no lower bound!
```

**Common Issue**: Upper bound checks exist but no lower bound checks for `>= 1`.

---

## Step 6: Map to Model Code

### Use Source Node Comments

The C++ wrapper includes comments showing which PyTorch operations generated each kernel:

```bash
grep -n -B5 "call_triton_KERNEL_NAME" /path/to/*.wrapper.cpp | grep "Source Nodes"
```

### Example Output

```cpp
// Topologically Sorted Source Nodes: [slice_1, sub_89, cumsum, ge_231, where_2, index_copy]
```

### Map Operations to Python Code

| ATen Operation | Python Code Pattern |
|----------------|---------------------|
| `cumsum` | `torch.cumsum(tensor, dim=0)` |
| `sub` | `idx - 1` |
| `ge` | `idx >= 0` |
| `where` | `torch.where(condition, ...)` |
| `index_copy` | `tensor.index_copy(0, indices, source)` |

---

## Root Cause Analysis

### Common Root Causes

1. **Empty tensor at runtime**: A jagged/variable-length tensor has size 0 at runtime but wasn't tested during compilation
2. **Missing lower bound guards**: AOTI only generates upper bound checks, not lower bound checks
3. **Edge case not in sample inputs**: Sample inputs during AOTI export never included the edge case

---

## Fix Recommendations

### Option 1: Add Guard in Forward Method

```python
def forward(self, lengths: torch.Tensor, ...) -> torch.Tensor:
    if lengths.numel() == 0:
        device = lengths.device
        return torch.empty(0, self.output_dim, device=device)
    # ... rest of method
```

### Option 2: Fix the Specific Operation

Add handling for empty tensors in the problematic operation:

```python
def process_events(self, lengths: torch.Tensor, ...):
    if lengths.numel() == 0:
        return torch.empty(0, self.emb_dim, device=lengths.device)
    # ... rest of method
```

### Option 3: Include Edge Cases in AOTI Export

Ensure sample inputs during AOTI export include:
- Empty tensors (size 0)
- Minimum size tensors (size 1)
- Maximum expected sizes

---

## Useful Commands Summary

### Searching in AOTI Wrapper

```bash
# Find kernel by assertion pattern
grep -n "tmpN < ksM" *.wrapper.cpp

# Get full kernel context
grep -n -B80 -A20 "ASSERTION_PATTERN" *.wrapper.cpp

# Find kernel call site
grep -n "call_KERNEL_NAME" *.wrapper.cpp

# Find dynamic shape definition
grep -n "int64_t SHAPE_VAR = " *.wrapper.cpp

# Find input mapping
grep -n 'inputs_info_\[INDEX\].name' *.wrapper.cpp

# Find size constraints
grep -n "SHAPE_VAR_size\[0\]" *.wrapper.cpp
```

### Environment Variables for Debugging

```bash
# Enable debug output during torch.compile
export TORCH_COMPILE_DEBUG=1

# Save generated kernels to persistent location
export TORCHINDUCTOR_CACHE_DIR=/path/to/save/kernels

# Enable CUDA launch blocking for accurate stack traces
export CUDA_LAUNCH_BLOCKING=1
```
