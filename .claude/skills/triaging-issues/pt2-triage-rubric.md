# PT2 Triage Rubric

This rubric guides labeling decisions for PT2 oncall triage.

## 1. Component Isolation - Be Precise, Don't Over-Tag

### Dynamo vs Dynamic Shapes

| Signal | Label |
|--------|-------|
| `dynamic=False` fixes it | `module: dynamic shapes` only |
| Graph breaks, bytecode errors | `module: dynamo` |
| Guard failures, SymInt issues | `module: dynamic shapes` |
| Data-dependent operations (.item(), etc.) | `module: dynamic shapes` |

**Don't just slap `module: dynamo` on every torch.compile issue.**

---

## 2. Backend Isolation for Correctness Issues

When component isn't clear from the issue body:

1. **Check comments first** - often contains debugging info
2. **Look for information with backends:**
| Result | Label |
|--------|-------|
| Fails on `aot_eager`, not `eager` | `module: pt2-dispatcher` |
| Fails on `inductor`, not `aot_eager` | `module: inductor` |
| Fails during tracing (before backend) | `module: dynamo` |

---

**This is critical** when you have identified an issue as inductor, and the failing device is "cpu" ONLY, then this is a CPU inductor issue, and should be redirected to `oncall: cpu inductor`

## 3. Don't Over-Tag pt2-dispatcher

`module: pt2-dispatcher` is for bugs **IN** the dispatcher code, not just when it appears in a stack trace.

**Common mistake:** Seeing `_aot_autograd/` in a stack trace and assuming it's a pt2-dispatcher bug. The dispatcher code is on the call path for almost everything - that doesn't mean the bug is there.

**Only add pt2-dispatcher when:**
- The bug is clearly in AOT autograd logic (e.g., incorrect tensor metadata handling)
- The bug is in functionalization
- The bug is in FakeTensor implementation
- The bug is in custom operator registration/dispatch

**Don't add pt2-dispatcher when:**
- AOT autograd just happens to be on the stack trace
- The actual bug is in functorch transforms (use `module: functorch` instead)
- The actual bug is in inductor codegen (use `module: inductor` instead)
- You're not sure where the bug actually is

---

## 4. Don't Redirect When PT2 Owns the Code

**This is critical:** Don't redirect to another oncall just because their subsystem is *involved*. Only redirect when:
1. The bug is clearly **IN** their code, AND
2. PT2 code is not at fault

**Examples - DO NOT redirect:**

| Situation | Why NOT redirect |
|-----------|------------------|
| Export triggers a bug, but the bug is a leaked hook in AOT autograd | Bug is in PT2 code → PT2 owns it |
| DTensor has a bad error message under compile | Bug is in PT2's error handling → PT2 owns UX |
| Distributed training fails, but stack trace shows inductor issue | Bug is in inductor → PT2 owns it |

For PT2-D issues, you may also add `oncall: distributed` but DO NOT hand this off fully - keep the `oncall: pt2` label.

**Examples - DO redirect:**

| Situation | Why redirect |
|-----------|--------------|
| MKLDNN-specific codegen bug | `oncall: cpu inductor` owns MKLDNN |
| Export-only issue with no compile involvement | `oncall: export` owns it |
| Bug in DTensor's tensor subclass implementation | `oncall: distributed` owns DTensor internals |

**The test:** Ask "where would the fix need to be made?" If the fix is in PT2 code, PT2 owns it.

**Adding labels for visibility:** You CAN add domain labels (e.g., `module: dtensor`) so domain experts see the issue, but don't ADD the oncall redirect label unless you're actually handing it off.

**This is critical** when you have identified an issue as inductor, and the failing device is "cpu" ONLY, then this is a CPU inductor issue, and should be redirected to `oncall: cpu inductor`

---

## 5. Add Domain-Specific Labels for Visibility

Even when not redirecting, add labels so domain experts see the issue:

| Domain | Label |
|--------|-------|
| DTensor | `module: dtensor` |
| FSDP | `module: fsdp` |
| DDP | `module: ddp` |
| Flex attention | `module: flex attention` |

---

## 6. Use Feature-Specific Labels

Check for existing labels before inventing categories:

| Feature | Label |
|---------|-------|
| Caching issues | `compile-cache` |
| Determinism | `module: determinism` |
| Compile/startup time | `module: compile-time` |
| Numerical issues | `module: numerical-stability` |
| UX/error messages | `module: compile ux` |

---

## 7. functorch + compile

| Situation | Labels |
|-----------|--------|
| Compiling a functorch transform (vjp, grad, vmap) | `module: functorch`, `dynamo-functorch` |
| Only add `pt2-dispatcher` if stack trace shows AOT autograd | Check stack trace first |

---

## 8. High Priority Criteria

**This is critical** You should not explicitly add `high priority` - add `triage review` instead
so that it is reviewed at the next triage meeting by the oncall.

Mark `triage review` if ANY of these apply:

| Criteria | Example |
|----------|---------|
| **Crash** (segfault, illegal memory access) | Device-side assert, SIGSEGV |
| **Silently wrong results** | Output differs from eager with no error |
| **Regression** | "This used to work in version X" |
| **Flaky test** | Usually indicates regression |
| **Important model regressed** (>10% perf) | Common model, significant slowdown |
| **Important customer** | Huggingface, common usage patterns |

---

## 9. Fuzzer Issues

For `topic: fuzzer` issues:

1. Ensure rtol/atol are at default tolerances
2. Don't compare indices of max/min (avoids tolerance issues)
3. Use `torch._dynamo.utils.same` with `fp64_ref` for comparison
4. If criteria met and bug appears easy/common → triage normally
5. If complex and rare → add `low priority`

---

## 10. Quick Label Reference

### Core Components
- `module: dynamo` - Tracing, bytecode, graph breaks
- `module: inductor` - Codegen, Triton kernels
- `module: dynamic shapes` - Symbolic shapes, guards, data-dependent
- `module: pt2-dispatcher` - AOT autograd, functionalization, FakeTensor
- `module: cuda graphs` - CUDA graph capture/replay
- `module: flex attention` - Flex attention API

### Holistic Areas
- `module: compile ux` - Error messages, APIs, programming model
- `module: startup-compile-tracing time` - Compilation speed
- `module: performance` - Runtime performance
- `module: memory usage` - Memory issues

### Status Labels
- `triaged` - Done triaging
- `triage review` - Discuss at meeting
- `needs reproduction` - Blocked on repro
- `needs research` - Needs investigation
- `actionable` - Clear what to do

### Redirects
- `oncall: cpu inductor` - CPU/MKLDNN issues
- `oncall: export` - Export-specific issues
- `oncall: distributed` - Distributed training issues

### CPU Inductor Routing

Route to `oncall: cpu inductor` (not generic `oncall: pt2`) when the issue is specific to CPU backend in inductor:
- Title or body mentions `[CPU]`, `cpu`, or `MKLDNN`
- CPU-specific codegen bugs (e.g., float16 handling on CPU)
- Issues that only reproduce on CPU, not CUDA
- MKLDNN-specific kernel issues

Example: "[Inductor][CPU][float16] LayerNorm outputs NaN" → `oncall: cpu inductor`, NOT `oncall: pt2`
