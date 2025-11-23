# Range-Based Autotuning Test Guide

## Task Goal
Implement range-based autotuning with implementation grouping:
- If all ranges select the **same implementation** → Generate **1 kernel** (no dispatch)
- If ranges select **different implementations** → Generate **N kernels** with dispatch

## Environment Setup
```bash
conda activate pytorch-3.12
```

## Test File
`/data/users/tianren/pytorch/test/inductor/test_dynamic_range_standalone.py`

## Run Test

### Clear Cache (REQUIRED before each test)
```bash
rm -rf /tmp/torchinductor_tianren/* /tmp/torch_inductor_range_dispatch/*
```

### Execute Test with Logging
```bash
cd /data/users/tianren/pytorch
TORCH_LOGS=+inductor,output_code python3 test/inductor/test_dynamic_range_standalone.py > log_test.txt 2>&1
```

## Verify Results

### 1. Check Key Log Messages
```bash
grep -E "(Found|unique|implementation|same|dispatch|Merging)" log_test.txt
```

**Expected for Same Implementation:**
```
Found 1 unique implementations across 3 ranges
All ranges use same implementation, generating single kernel
```

**Expected for Different Implementations:**
```
Found 3 unique implementations across 3 ranges
Creating 3-way dispatch with merged ranges
```

### 2. Check Generated Kernels
```bash
# Count Triton kernels generated
ls /tmp/torchinductor_tianren/*/*.py | wc -l

# View generated code
cat /tmp/torchinductor_tianren/fp/*.py | grep "triton_poi_fused" | wc -l
```

**Expected:**
- Same impl: 1 kernel
- Different impls: 3 kernels

### 3. Check Dispatch Function
```bash
cat /tmp/torch_inductor_range_dispatch/dynamic_range_autotuned_dispatch.py
```

Should show which implementation is selected for each range.

## Current Test Behavior
The test uses 3 implementations (einsum, chunked, broadcast) which currently get optimized to the **same code** by the compiler, resulting in **1 kernel** generation.

## To Test 3 Kernels
Modify implementations in test file to have genuinely different compute patterns that won't be optimized away.
