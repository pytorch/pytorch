---
name: rocm-contributor
description: Guide AMD ROCm contributions to vLLM and PyTorch. Use when reviewing ROCm PRs, checking PR status, finding AMD GPU bugs, or submitting ROCm fixes. Triggers on: ROCm, HIP, AMD GPU, MI300X, MI350X, gfx942, gfx1151, Strix Halo, Wave64, Wave32, WARP_SIZE, FP8 fnuz, AITER, MLA, fused MoE, AMDSMI, vLLM PR, PyTorch PR, AMD CI, buildkite/amd-ci.
---

# ROCm Contributor

Comprehensive guide for contributing AMD ROCm fixes to vLLM and PyTorch.

## Quick Reference

### PR Status Commands

```bash
# vLLM
gh pr list --repo vllm-project/vllm --author @me --state open
gh pr checks <num> --repo vllm-project/vllm
gh api repos/vllm-project/vllm/pulls/<num>/reviews --jq '.[] | select(.state == "APPROVED")'

# PyTorch
gh pr list --repo pytorch/pytorch --author @me --state open
gh pr checks <num> --repo pytorch/pytorch
gh api repos/pytorch/pytorch/pulls/<num>/reviews --jq '.[] | select(.state == "APPROVED")'
```

### Key Maintainers

**vLLM:**
- @hongxiayang - ROCm lead
- @ganyi1996ppo - MLA/AITER expert
- @gshtras - ROCm maintainer

**PyTorch:**
- @jeffdaily - ROCm lead
- @jithunnair-amd - ROCm maintainer
- @malfet - Core maintainer (can approve merges)

## Common ROCm Bug Patterns

When reviewing or writing ROCm code, watch for these patterns:

| Pattern | Buggy Code | Fix |
|---------|------------|-----|
| List aliasing | `[[val] * n] * m` | `[[val] * n for _ in range(m)]` |
| Uninitialized registers | Missing `else` zeros | Add `else { Qlocal = {}; }` |
| Hardcoded device | `device="cuda"` | `device=tensor.device` |
| Broad exceptions | `except Exception:` | `except (ImportError, ValueError):` |
| Wrong WARP_SIZE | `constexpr int WARP_SIZE = 32` | Use `WARP_SIZE` from cuda_compat.h |
| FP8 fnuz max | Using 240.0 on ROCm | Use 224.0 for fnuz dtype |
| Wave64 assumptions | Assuming 32-lane warps | Check architecture WARP_SIZE |

## Instructions

### When Reviewing a ROCm PR

1. **Check root cause**: Does the fix address the actual bug, not just symptoms?

2. **Search for similar bugs**: Use grep to find other instances of the same pattern:
   ```bash
   # Example: Find other list aliasing bugs
   gh pr diff <num> --repo <repo>  # See the pattern
   grep -r "]] \* " --include="*.py" <path>  # Find similar
   ```

3. **Verify tests**: Does the PR include tests? Do existing tests still pass?

4. **Check CI status**:
   - vLLM: `buildkite/amd-ci` must pass (runs on MI300X hardware)
   - PyTorch: Watch for `Check labels` failures (needs maintainer labels)

5. **Look for duplicates**: Check if other PRs fix the same issue:
   ```bash
   gh pr list --repo <repo> --state open --json title,number
   ```

### When Checking PR Status

1. **List all open PRs**:
   ```bash
   gh pr list --repo vllm-project/vllm --author @me --state open
   gh pr list --repo pytorch/pytorch --author @me --state open
   ```

2. **Check CI for each**:
   ```bash
   gh pr checks <num> --repo <repo>
   ```

3. **Check approvals**:
   ```bash
   gh api repos/<repo>/pulls/<num>/reviews --jq '.[] | select(.state == "APPROVED") | .user.login'
   ```

4. **Identify blockers**:
   - vLLM: Needs maintainer approval + CI green
   - PyTorch: Needs Core Maintainer approval + labels + CI green

5. **Follow up on approved PRs**:
   ```bash
   gh pr comment <num> --repo <repo> --body "@maintainer All CI passing, ready to merge!"
   ```

### When Submitting a vLLM PR

1. **Title format**: `[Bugfix][Hardware][AMD] Description`
   - Categories: `[Bugfix]`, `[Feature]`, `[Refactor]`, `[Doc]`
   - Always include `[Hardware][AMD]` for ROCm changes

2. **Description must include**:
   - Summary of the bug/feature
   - Root cause analysis
   - The fix with before/after code
   - Hardware validation results:
     ```
     ## Hardware Validation
     - GPU: AMD Instinct MI300X (gfx942)
     - ROCm: 6.2
     - PyTorch: 2.5.1+rocm6.2
     - Test: [describe what you tested]
     ```

3. **Add tests** in `tests/` directory when possible

4. **Reference related PRs**: Link to related fixes

5. **After CI passes**, ping maintainers:
   ```bash
   gh pr comment <num> --repo vllm-project/vllm --body "@hongxiayang Ready for review - all CI passing."
   ```

### When Submitting a PyTorch PR

1. **Title format**: `[ROCm] Description`

2. **CC maintainers in body**:
   ```markdown
   cc @jeffdaily @jithunnair-amd @hongxiayang
   ```

3. **Request labels** (external contributors can't add directly):
   ```bash
   gh pr comment <num> --repo pytorch/pytorch --body '@pytorchbot label "topic: not user facing"'
   gh pr comment <num> --repo pytorch/pytorch --body '@pytorchbot label "release notes: not user facing"'
   ```

4. **Merge requires**:
   - Core Maintainer approval (not just any maintainer)
   - All required CI passing
   - Use `@pytorchbot merge` after approval

## Bug Pattern Details

### 1. Python List Aliasing

**Bug**: Using `*` to replicate lists creates references, not copies.

```python
# BUGGY: All inner lists are the same object
s_topk_ids_list = [[fake_expertid] * n] * max_num_tokens
s_topk_ids_list[0][0] = 999  # Changes ALL rows!

# FIXED: List comprehension creates independent lists
s_topk_ids_list = [[fake_expertid] * n for _ in range(max_num_tokens)]
```

**Where to look**: Python initialization code, especially in `__init__` methods.

### 2. Uninitialized GPU Registers

**Bug**: GPU registers contain garbage from previous kernel invocations.

```cpp
// BUGGY: Qlocal has garbage in lanes that don't load
if (lane16id < GQA_RATIO) {
  Qlocal[qkhe_depth] = *q_fetch_ptr_32B;
}
// Missing else branch!

// FIXED: Zero out unused lanes
if (lane16id < GQA_RATIO) {
  Qlocal[qkhe_depth] = *q_fetch_ptr_32B;
} else {
  Qlocal[qkhe_depth] = {};  // Zero initialization
}
```

**Where to look**: CUDA/HIP kernels with conditional loads, especially attention kernels.

### 3. Hardcoded Device Strings

**Bug**: Using `device="cuda"` instead of dynamic device.

```python
# BUGGY: Fails on multi-GPU or explicit device selection
tensor = torch.empty(shape, device="cuda")

# FIXED: Use input tensor's device
tensor = torch.empty(shape, device=input.device)

# OR: Accept device as parameter
def init_metadata(self, ..., device: int | str = "cuda"):
    tensor = torch.empty(shape, device=device)
```

**Where to look**: Tensor creation in ROCm-specific files.

### 4. Overly Broad Exception Handling

**Bug**: Catching `Exception` masks unexpected errors.

```python
# BUGGY: Hides real bugs like SyntaxError, MemoryError
try:
    result = some_function()
except Exception:
    result = fallback

# FIXED: Catch only expected exceptions
try:
    result = some_function()
except (ImportError, ModuleNotFoundError, ValueError, TypeError):
    result = fallback
```

**Where to look**: Feature detection code, optional imports.

### 5. Hardcoded WARP_SIZE

**Bug**: Assuming WARP_SIZE is always 32.

```cpp
// BUGGY: Wrong for AMD CDNA (Wave64)
constexpr int WARP_SIZE = 32;

// FIXED: Use dynamic macro from cuda_compat.h
#include "cuda_compat.h"
constexpr int kWarpSize = WARP_SIZE;  // 64 on CDNA, 32 on RDNA/NVIDIA
```

**Where to look**: CUDA kernels in `csrc/`, especially sampler and attention code.

### 6. Wrong FP8 Max Value

**Bug**: Using PyTorch's default finfo.max for fnuz dtype.

```python
# BUGGY: 240.0 causes accuracy issues on ROCm fnuz
fp8_max = torch.finfo(torch.float8_e4m3fnuz).max  # Returns 240.0

# FIXED: Use correct max for fnuz
if current_platform.is_fp8_fnuz() and dtype == torch.float8_e4m3fnuz:
    fp8_max = 224.0  # Correct value for fnuz
else:
    fp8_max = torch.finfo(dtype).max
```

**Where to look**: FP8 quantization code, especially dynamic quantization.

### 7. Tensor Slice Assignment vs .fill_()

**Bug**: Using `=` on tensor slices during CUDA graph capture.

```python
# POTENTIALLY BUGGY: May not capture correctly in CUDA graphs
self.buffer[start:end] = value

# SAFER: Explicit in-place operation
self.buffer[start:end].fill_(value)
```

**Where to look**: CUDA graph capture code, persistent buffer initialization.

## Hardware Reference

| Architecture | GPU | WARP_SIZE | FP8 Type | Notes |
|--------------|-----|-----------|----------|-------|
| gfx942 (CDNA3) | MI300X, MI300A | 64 | float8_e4m3fnuz | Wave64, fnuz max=224 |
| gfx950 (CDNA4) | MI350X | 64 | float8_e4m3fnuz | Wave64, fnuz max=224 |
| gfx1100 (RDNA3) | RX 7900 | 32 | float8_e4m3fn | Wave32 |
| gfx1151 (RDNA3.5) | Strix Halo | 32 | float8_e4m3fn | Wave32, APU |
| sm_* (NVIDIA) | All | 32 | float8_e4m3fn | Standard FP8 |

**Key differences:**
- **Wave64 vs Wave32**: CDNA uses 64-lane wavefronts, RDNA/NVIDIA use 32
- **fnuz vs fn**: ROCm CDNA uses fnuz dtype with max=224, others use fn with max=240
- **AITER support**: Only gfx9 (CDNA) architectures support AITER optimizations

## Repository Knowledge

### vLLM

**ROCm-specific paths:**
- `csrc/rocm/` - ROCm CUDA kernels (attention, etc.)
- `vllm/attention/ops/rocm_*` - ROCm attention ops
- `vllm/_aiter_ops.py` - AITER integration
- `vllm/model_executor/layers/fused_moe/rocm_*` - ROCm MoE
- `vllm/v1/attention/backends/mla/rocm_*` - ROCm MLA

**CI pipeline:**
- `DCO` - Developer Certificate of Origin (sign-off)
- `pre-commit` - Linting (ruff format, ruff check)
- `buildkite/amd-ci` - Hardware tests on MI300X
- `docs/readthedocs` - Documentation build

**Merge process:**
1. All CI must pass
2. At least one maintainer approval
3. Maintainer clicks merge (no bot)

### PyTorch

**ROCm-specific paths:**
- `torch/cuda/__init__.py` - AMDSMI integration
- `aten/src/ATen/hip/` - HIP backend
- `torch/utils/cpp_extension.py` - ROCm build support
- `torch/csrc/stable/c/shim.h` - C stable API

**CI pipeline:**
- `EasyCLA` - CLA signature
- `Check labels` - Required labels (needs maintainer)
- `trunk` - Main CI pipeline
- `inductor-build` - Inductor tests

**Required labels:**
- `module: rocm` - Auto-added based on path
- `topic: not user facing` - For internal changes
- `release notes: not user facing` - For changelog

**Merge process:**
1. All CI must pass
2. Core Maintainer approval required
3. Comment `@pytorchbot merge`

## Validation Checklist

Before marking a PR as ready:

- [ ] **Root cause**: Fix addresses actual bug, not symptoms
- [ ] **Completeness**: No other instances of bug in codebase
- [ ] **Tests**: New tests added or existing tests pass
- [ ] **CI green**: All checks pass, especially AMD CI
- [ ] **No duplicates**: No overlapping PRs for same issue
- [ ] **Hardware validation**: Tested on actual AMD GPU
- [ ] **Description**: Includes GPU model, ROCm version, test results

## Common Commands

```bash
# Check all your open vLLM PRs
for pr in $(gh pr list --repo vllm-project/vllm --author @me --state open --json number --jq '.[].number'); do
  echo "=== PR #$pr ==="
  gh pr checks $pr --repo vllm-project/vllm | head -5
done

# Find similar bug patterns
grep -rn "]] \* " --include="*.py" vllm/  # List aliasing
grep -rn 'device="cuda"' --include="*.py" vllm/  # Hardcoded device
grep -rn "except Exception" --include="*.py" vllm/  # Broad exception

# Check who approved a PR
gh api repos/vllm-project/vllm/pulls/12345/reviews --jq '.[] | select(.state == "APPROVED") | "\(.user.login): \(.body)"'

# Add follow-up comment after approval
gh pr comment 12345 --repo vllm-project/vllm --body "All CI passing, ready to merge!"
```

## Professional Ping Templates

### vLLM Merge Ping (Approved PRs)

```bash
gh pr comment <PR_NUM> --repo vllm-project/vllm --body "Hi @<maintainer>, all checks are passing and this has been hardware-verified on MI300X. Ready to be merged when you have a moment. Thanks!"
```

### PyTorch Core Maintainer Ping

```bash
gh pr comment <PR_NUM> --repo pytorch/pytorch --body "Hi @malfet, this PR has been approved by @<rocm_maintainer> (ROCm) and has full green CI on gfx942. It just needs a Core Maintainer's final sign-off. PTAL when you can!"
```

### Hardware Benchmark Template

Post this on PRs awaiting review to increase priority:

```markdown
### Hardware Verification (MI300X)

Verified on **AMD Instinct MI300X** (gfx942, ROCm 6.2).

| Metric | Before | After | Delta |
| :--- | :--- | :--- | :--- |
| **Latency** | X ms | Y ms | **Nx Speedup** |

**Validation:**
- Device: gfx942:sramecc+:xnack-
- Test: [describe test]
- Result: [pass/fail with details]
```

## Full PR Health Scan

Comprehensive audit commands for all open PRs:

```bash
# CI Deep-Dive
gh pr checks <PR_NUM> --repo <repo> --json name,state,bucket,link

# Mergeability Check
gh pr view <PR_NUM> --repo <repo> --json mergeable,mergeStateStatus,baseRefName

# Review Status
gh api repos/<owner>/<repo>/pulls/<PR_NUM>/reviews --jq '.[] | "\(.user.login): \(.state)"'

# Unresolved Comments
gh api repos/<owner>/<repo>/pulls/<PR_NUM>/comments --jq '.[] | "\(.user.login): \(.body | split("\n")[0])"'

# Issue Comments (bot messages, pings)
gh api repos/<owner>/<repo>/issues/<PR_NUM>/comments --jq '.[-5:] | .[] | "\(.user.login): \(.body | split("\n")[0])"'
```

## Contest Tracking

**AMD Strix Halo Contest Target:** 10 merged ROCm PRs

**Qualifying criteria:**
- Must be ROCm-specific (not typo fixes)
- Must be merged (not just approved)
- Shell scripts don't count
- PyTorch and vLLM both qualify

**Strategy:**
1. Focus on getting approved PRs merged first
2. Post hardware benchmarks to accelerate reviews
3. Ping maintainers politely after CI passes
4. Look for related bug patterns to find more issues

## Troubleshooting

### vLLM AMD CI Fails

1. Check buildkite logs: Click the buildkite/amd-ci link
2. Common issues:
   - Flaky infrastructure (retry with `/test rocm`)
   - Actual test failures (check MI300X-specific tests)
   - Pre-commit failures (run `ruff format` and `ruff check`)

### PyTorch "Check labels" Fails

External contributors cannot add labels. Request them:
```bash
gh pr comment <num> --repo pytorch/pytorch --body '@pytorchbot label "topic: not user facing"'
```

### PyTorch Merge Fails

Needs Core Maintainer approval, not just any maintainer. Check:
```bash
gh api repos/pytorch/pytorch/pulls/<num>/reviews --jq '.[] | "\(.user.login) (\(.state))"'
```

If approved by non-core maintainer, request core review:
```bash
gh pr comment <num> --repo pytorch/pytorch --body "@malfet Could you please approve this for merge?"
```
