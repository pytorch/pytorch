---
name: inductor-debug-hanging-autotune
description: Debug PyTorch Inductor autotuning hangs during kernel compilation
triggers:
  - inductor hang
  - autotune stuck
  - autotune hanging
  - compilation stuck
  - triton compilation hang
  - precompilation stuck
---

# Debugging Inductor Autotuning Hangs

This skill guides you through debugging cases where PyTorch Inductor's autotuning gets stuck during kernel compilation.

## Step 1: Enable Detailed Logging and Diagnose

Ask the user to run their program with detailed Inductor logging:

```bash
TORCH_LOGS="+inductor" python your_script.py 2>&1 | tee inductor_log.txt
```

Tell the user to wait until the program hangs, then Ctrl+C and provide the log file or paste the relevant portions.

### Diagnose If This Skill Applies

When the user provides the log, first check if the hang is caused by kernel compilation. Look for these indicators:

**This skill IS applicable if:**
- Log contains "Submitted triton async compile" entries followed by some but not all "Precompiling benchmark choice...took" entries
- Log stops/hangs during the autotuning phase (after submissions, before all completions)
- The last log entries are about kernel compilation or precompilation
- User reports the hang occurs during `torch.compile()` warmup or first inference

**This skill is NOT applicable if:**
- Log shows the program completed compilation and hangs during runtime execution
- Log shows errors or exceptions before the hang (debug the error instead)
- Log shows the hang occurs during graph tracing (Dynamo), not Inductor compilation
- Log shows the hang occurs during AOTAutograd (use aoti-debug skill instead)
- No "Submitted triton async compile" entries exist (autotuning may not be enabled)

If this skill is not applicable, tell the user:
> "Based on the logs, the hang doesn't appear to be caused by Triton kernel compilation. The issue seems to be [describe what you see]. This skill is designed for debugging kernel compilation hangs during autotuning. You may need to investigate [suggest alternative approach]."

## Step 2: Analyze the Logs

When the user provides the log, look for two types of log entries:

### Submitted Kernels (V-level logs)
Look for lines matching this pattern:
```
V... torch/_inductor/select_algorithm.py:...] Submitted triton async compile for choice: TritonTemplateCaller(/path/to/kernel.py, ...)
```

**Log location (as of Feb 2025):** `torch/_inductor/select_algorithm.py` line 3478
```python
log.debug("Submitted triton async compile for choice: %s", c)
```

### Completed Kernels (I-level logs)
Look for lines matching this pattern:
```
I... torch/_inductor/select_algorithm.py:...] Precompiling benchmark choice TritonTemplateCaller(/path/to/kernel.py, ...) took X.XXs
```

**Log location (as of Feb 2025):** `torch/_inductor/select_algorithm.py` line 3518
```python
log.info("Precompiling benchmark choice %s took %.02fs", ...)
```

### Verify Logs Still Exist

If the user's logs don't contain these entries, first verify the log statements still exist in the codebase:
```bash
grep -n "Submitted triton async compile" torch/_inductor/select_algorithm.py
grep -n "Precompiling benchmark choice" torch/_inductor/select_algorithm.py
```

If the log statements have moved or been removed, search for alternative patterns or check the git history for changes to the logging.

### Find the Stuck Kernels

1. Extract all kernel file paths from "Submitted" logs
2. Extract all kernel file paths from "Precompiling...took" logs
3. The difference = stuck kernels that were submitted but never completed

Example analysis code:
```python
import re

# Parse submitted kernels
submitted_pattern = r'Submitted triton async compile for choice: TritonTemplateCaller\(([^,]+),'
submitted = set(re.findall(submitted_pattern, log_content))

# Parse completed kernels
completed_pattern = r'Precompiling benchmark choice TritonTemplateCaller\(([^,]+),'
completed = set(re.findall(completed_pattern, log_content))

# Find stuck kernels
stuck = submitted - completed
print(f"Stuck kernels ({len(stuck)}): {stuck}")
```

## Step 3: Analyze the Stuck Kernel

Once you identify the stuck kernel file path(s):

### 3.1 Read the Kernel File
Try to read the stuck kernel file using the Read tool (e.g., `/var/tmp/torchinductor_user/.../kernel.py`).

**If the file cannot be found or accessed:**
- The file may be on a different machine than where Claude is running
- The temp directory may have been cleaned up
- The path may have different user permissions

In this case, ask the user to provide the kernel file contents:
> "I cannot access the kernel file at [path]. Could you please run the following command and paste the output?"
> ```bash
> cat /path/to/stuck/kernel.py
> ```
> "Or if the file no longer exists, you may need to reproduce the hang and copy the file before it gets cleaned up."

### 3.2 Extract the Configuration
Look at the `@triton_heuristics.template` or `@triton_heuristics.pointwise` decorator and extract:
- Block sizes: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `XBLOCK`, `RBLOCK`, etc.
- `num_warps`, `num_stages`
- Hardware-specific options in `triton_meta`:
  - **AMD/ROCm**: `waves_per_eu`, `matrix_instr_nonkdim`, `kpack`
  - **NVIDIA**: `launch_cooperative_grid`, TMA options
  - **Intel XPU**: `generate_native_code`
- Device properties: `type` (cuda/hip/xpu), `cc` (compute capability)

### 3.3 Identify Potential Issues
Common causes of compilation hangs:

**General:**
- Extremely large block sizes causing register pressure
- Incompatible block size combinations
- Very complex kernel bodies with many operations

**AMD/ROCm specific:**
- High `waves_per_eu` with low `num_warps` and large `BLOCK_K`
- `matrix_instr_nonkdim` incompatible with block sizes
- Register pressure from MFMA instruction configurations

**NVIDIA specific:**
- Block sizes exceeding shared memory limits
- Incompatible TMA configurations
- Warp specialization issues on newer architectures

**Intel XPU specific:**
- Native code generation issues
- Incompatible block configurations

### 3.4 Write a Standalone Reproduction Script

Create a script that compiles just the stuck kernel to verify the hang:

```python
"""
Standalone script to reproduce the stuck Triton kernel compilation.
Run with: timeout 60 python reproduce_stuck_kernel.py
"""

import triton
import triton.language as tl
from triton.compiler import ASTSource, GPUTarget

# TODO: Copy the kernel code from the stuck file
@triton.jit
def stuck_kernel(...):
    # [Paste kernel body from the stuck file]
    pass

def compile_kernel():
    import time

    # TODO: Extract signature from stuck kernel's triton_meta
    signature = {
        0: "*bf16",  # Adjust based on actual types
        1: "*bf16",
        2: "*bf16",
        3: "i32",
    }

    # TODO: Extract constants from the kernel's constexpr values
    constants = {
        # Copy from kernel file
    }

    # TODO: Extract configs from triton_meta
    configs = [{}]

    ast_source = ASTSource(
        fn=stuck_kernel,
        signature=signature,
        constants=constants,
        configs=configs,
    )

    # TODO: Set target based on user's hardware
    # AMD: GPUTarget("hip", "gfx942", 64)  # MI300
    # AMD: GPUTarget("hip", "gfx90a", 64)  # MI200
    # NVIDIA: GPUTarget("cuda", 90, 32)   # H100
    # NVIDIA: GPUTarget("cuda", 80, 32)   # A100
    target = GPUTarget("hip", "gfx942", 64)

    # TODO: Extract options from kernel's decorator
    options = {
        "num_warps": 1,
        "num_stages": 2,
        "debug": False,
        "sanitize_overflow": False,
        # Add hardware-specific options from triton_meta
    }

    print(f"Triton version: {triton.__version__}")
    print(f"Target: {target}")
    print(f"Options: {options}")
    print(f"Constants: {constants}")

    start = time.time()
    try:
        binary = triton.compile(ast_source, target=target, options=options)
        print(f"Compilation succeeded in {time.time() - start:.2f}s")
        print(f"Binary hash: {binary.hash}")
    except Exception as e:
        print(f"Compilation failed after {time.time() - start:.2f}s: {e}")
        raise

if __name__ == "__main__":
    compile_kernel()
```

Tell the user to run:
```bash
timeout 60 python reproduce_stuck_kernel.py
```

If it times out, the kernel compilation hang is confirmed and reproducible.

## Step 4: Propose a Fix

Ask the user: "Would you like me to propose a fix to prevent this kernel configuration from being generated?"

If yes:

1. **Analyze the root cause**: Determine why this specific configuration causes the compiler to hang
2. **Find where configs are generated**: Look at relevant files:
   - `torch/_inductor/template_heuristics/triton.py` - Config generation and preprocessing
   - `torch/_inductor/runtime/coordinate_descent_tuner.py` - Runtime tuning
   - `torch/_inductor/runtime/triton_heuristics.py` - Compile options
3. **Propose a filter**: Add logic to skip the problematic configuration pattern, with a clear comment explaining why
4. **Consider scope**: The fix should be specific enough to avoid the hang but not overly broad to exclude valid configs

## Related Files

- `torch/_inductor/select_algorithm.py` - Autotuning logic and logging
- `torch/_inductor/template_heuristics/triton.py` - Config generation and hardware-specific heuristics
- `torch/_inductor/runtime/triton_heuristics.py` - Triton kernel compilation (`_precompile_config`)
- `torch/_inductor/runtime/coordinate_descent_tuner.py` - Coordinate descent tuning
- `torch/_inductor/kernel/mm.py` - Matrix multiplication templates
- `torch/_inductor/kernel/bmm.py` - Batched matrix multiplication templates
