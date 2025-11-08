# How to Use the Enhanced test_flex_determinism.py

## What It Does

The enhanced test now automatically:
1. **Records all Triton kernel calls** with their input/output hashes
2. **Compares runs kernel-by-kernel** to find exactly where non-determinism first appears
3. **Identifies the problematic kernel** and which tensor differs
4. **Points to the likely cause** if a previous kernel produced different output

## Running the Test

```bash
python test_flex_determinism.py
```

## Understanding the Output

### Scenario 1: Input Divergence (Previous Kernel Problem)

If you see:
```
Run 1: Comparing with run 0...
  ⚠️  grad_q differs

  Comparing 150 Triton kernels...

  ❌ FIRST DIVERGENCE FOUND!
     Kernel #45: triton_per_fused_softmax_2
     Grid: (64, 128, 1)
     Input argument: in_ptr1 (index 1)
     Run 0 hash: 12345.67
     Run 1 hash: 12389.45

     → Likely cause: Previous kernel #44: triton_red_fused_sum_delta_1
     → Its outputs: ['delta_ptr', 'sum_ptr']
```

**Interpretation:**
- Kernel #45 received different inputs between runs
- The difference came from kernel #44's output
- **Kernel #44 is the non-deterministic one**
- The problematic output is `delta_ptr` or `sum_ptr`

**Action:** Investigate kernel #44 (`triton_red_fused_sum_delta_1`)

### Scenario 2: Output Divergence (This Kernel Problem)

If you see:
```
Run 1: Comparing with run 0...
  ⚠️  grad_q differs

  Comparing 150 Triton kernels...

  ❌ NON-DETERMINISTIC KERNEL FOUND!
     Kernel #44: triton_red_fused_sum_delta_1
     Grid: (64, 128, 1)
     Output argument: delta_ptr (index 2)
     Run 0 hash: 12345.67
     Run 1 hash: 12389.45
     → This kernel produces different outputs with same inputs!
```

**Interpretation:**
- Kernel #44 received identical inputs both runs
- But it produced different outputs
- **This kernel itself is non-deterministic**

**Action:** Investigate kernel #44's implementation for:
- Non-deterministic reductions (order of operations)
- Race conditions
- Floating point accumulation order

### Scenario 3: All Kernels Match

If you see:
```
Run 1: Comparing with run 0...
  ✓ All 150 kernels have matching inputs and outputs
```

**Interpretation:**
- All Triton kernels are deterministic
- The non-determinism might be in:
  - PyTorch operations (outside Triton kernels)
  - Random number generation
  - Autotuning picking different kernel configurations

## Key Information in the Output

### Kernel Number
- `Kernel #44` - The sequential index of the kernel in execution order
- Helps you track down the kernel in generated code

### Kernel Name
- `triton_red_fused_sum_delta_1` - The generated kernel name
- Often indicates the operation (e.g., `red` = reduction, `poi` = pointwise)
- The fused operations are listed (e.g., `sum_delta`)

### Grid Size
- `Grid: (64, 128, 1)` - The (grid_x, grid_y, grid_z) launch configuration
- Helps identify the kernel in profiler/nsight

### Argument Name
- `delta_ptr` - The parameter name in the kernel
- `in_ptr0`, `in_ptr1` = inputs
- `out_ptr0`, `out_ptr1` = outputs
- Descriptive names like `delta_ptr`, `sum_ptr` indicate purpose

### Hash Values
- `Run 0 hash: 12345.67` - L1 norm of the tensor
- Different hashes = tensors have different values
- Used for fast comparison without storing full tensors

## Common Patterns

### Pattern 1: Delta Calculation Non-Determinism

```
Kernel: triton_red_fused_sum_delta_1
```
- This is mentioned in your comment (line 197-198)
- Delta calculation's reduction order may vary
- **Fix:** Use `force_filter_reduction_configs` config (line 202)

### Pattern 2: Reduction Order

Any kernel with `red_` prefix:
```
triton_red_fused_*
```
- Reductions can be non-deterministic if order isn't fixed
- **Fix:** Enable `torch._inductor.config.deterministic = True`

### Pattern 3: Autotuning

If the same kernel appears with different grids:
```
Run 0: Grid: (64, 128, 1)
Run 1: Grid: (128, 64, 1)
```
- Autotuning picked different configurations
- **Fix:** Use consistent benchmarking or disable autotuning

## Debugging Workflow

1. **Run the test** to identify the problematic kernel
   ```bash
   python test_flex_determinism.py
   ```

2. **Note the kernel name and number** from output
   ```
   Kernel #44: triton_red_fused_sum_delta_1
   ```

3. **Find generated kernel code**
   - Check `/tmp/torchinductor_$USER/` for generated `.py` files
   - Search for kernel name: `triton_red_fused_sum_delta_1`

4. **Look for known issues**:
   - Unordered reductions (`tl.sum` without fixed order)
   - Atomic operations
   - Shared memory race conditions
   - Grid-dependent calculations

5. **Try fixes**:
   - Enable deterministic config (line 202)
   - Force reduction configs (line 202)
   - Disable autotuning
   - Fix kernel implementation

## Example Session

```bash
$ python test_flex_determinism.py

==================== BITWISE DETERMINISM TEST ====================

--- Testing with dynamic=False, backend=inductor, mode=default ---
Testing Standard: 100%|████████| 1/1 [00:10<00:00, 10.5s/it]

Run 0: Recorded 150 Triton kernel calls

Run 1: Comparing with run 0...
  ⚠️  grad_q differs

  Comparing 150 Triton kernels...
  [Kernel 0-43: OK]

  ❌ FIRST DIVERGENCE FOUND!
     Kernel #44: triton_red_fused_sum_delta_1
     Grid: (64, 128, 1)
     Input argument: in_ptr1 (index 1)
     Run 0 hash: 45123.89
     Run 1 hash: 45127.34

     → Likely cause: Previous kernel #43: triton_poi_fused_mul_3
     → Its outputs: ['out_ptr0']

# Now you know:
# - The non-deterministic kernel is #43: triton_poi_fused_mul_3
# - Its output 'out_ptr0' differs between runs
# - This causes kernel #44 to receive different 'in_ptr1' input
```

## Tips

1. **Run with num_runs=2 first** - Faster to catch issues
2. **num_runs=3 confirms** - Rules out one-off flukes
3. **Check both forward and backward** - Note which fails
4. **Look at kernel names** - They often reveal the operation
5. **Grid size matters** - Same kernel, different grid = autotuning issue
6. **Save the output** - Hash values help track down issues

## Related Configuration

In your test (lines 202, 208):

```python
# Force consistent reduction order
inductor_config = {"test_configs.force_filter_reduction_configs": True}

# Enable full deterministic mode
inductor_config = {"deterministic": True}
```

Try these if you find non-deterministic reduction kernels!
