---
name: torchscript-oncall
description: TorchScript oncall issue triage and resolution. Use when diagnosing TorchScript errors, triaging pytorch_jit oncall issues, debugging torch.jit.script/trace failures, resolving "failed in the TorchScript interpreter" errors, or advising on TorchScript workarounds and limitations.
---

# TorchScript Oncall Resolution

Resolve TorchScript oncall issues by triaging into the correct category, applying known workarounds, and redirecting misdirected issues. TorchScript is in **maintenance mode** — recommend migration to `torch.compile`/PT2 when feasible.

## Triage Workflow

1. **Classify the issue** using the decision tree below
2. **Look up known workarounds** in the relevant reference file
3. **Advise the user** with a workaround or redirect

### Decision Tree

Determine which category the issue falls into:

**Is the error "The following operation failed in the TorchScript interpreter"?**
→ Almost never a TorchScript bug (~15+ instances). Debug the model code, data pipeline, or custom op first. See "Common User Errors" in `references/common-errors.md`.

**Does the error occur at `torch.jit.script()` time?**
→ Likely a frontend/language limitation. See `references/frontend-limitations.md`.

**Does the error occur on the 2nd+ forward pass, or involve wrong devices/silent crashes?**
→ Likely an NNC/ProfilingGraphExecutor bug. See `references/runtime-issues.md`.

**Does `torch.jit.load` fail with "failed reading zip archive" or "Unknown builtin op"?**
→ Serialization issue. See `references/runtime-issues.md`.

**Is this actually an FX tracing, torch.compile, ONNX, or build system issue?**
→ Misdirected (~35% of all issues). Redirect to the correct oncall. See `references/misdirected-issues.md`.

### Issue Distribution

| Category | Share | Top Signal |
|---|---|---|
| Language/type system limitations | ~45% | Failure at `torch.jit.script()` time |
| Runtime/performance problems | ~20% | Failure during forward pass or latency spike |
| Misdirected (not TorchScript) | ~35% | FX, torch.compile, custom ops, build issues |

## General Triage Approach (2023-2025 era)

Triage and give high-level advice. Only provide fixes if the issue is clearly a TorchScript runtime bug or is so widespread/severe that it warrants a fix in the TorchScript layer.

1. **Frontend unsupported feature** → Recommend workaround (see references)
2. **"Failed in the TorchScript interpreter"** → Tell user to debug model code unless evidence shows the compiler is at fault
3. **Actual runtime error** → (a) Can model code work around it? (b) Can you disable ProfilingGraphExecutor/NNC/Static Runtime?

## Debugging Tips

- Set `PYTORCH_JIT_LOG_LEVEL` to a file name (e.g., `PYTORCH_JIT_LOG_LEVEL=">>profiling_graph_executor_impl"`) to dump TorchScript logging from that file to stderr.
- Use `torch.jit.last_executed_optimized_graph()` to inspect the optimized graph after execution.
- For NNC issues, disable fusers: `torch._C._jit_override_can_fuse_on_cpu(False)` and `torch._C._jit_override_can_fuse_on_gpu(False)`.
- For ProfilingGraphExecutor issues, disable it: `torch._C._get_graph_executor_optimize(False)`.

## Reference Files

Load only the reference file relevant to the issue:

- **`references/frontend-limitations.md`** — Unsupported Python constructs, type system limitations, module/compilation issues, and compatibility gaps. Read when the issue occurs at script-time.
- **`references/runtime-issues.md`** — ProfilingGraphExecutor, NNC/TensorExpr bugs, Static Runtime gaps, and serialization problems. Read when the issue occurs at runtime or during model loading.
- **`references/common-errors.md`** — "Failed in the TorchScript interpreter" diagnosis, configuration/build issues, and API misuse patterns. Read when the error message suggests a TorchScript failure but the root cause may be elsewhere.
- **`references/misdirected-issues.md`** — FX tracing, torch.compile, ONNX, Cinder/lazy imports, and build system issues that are not TorchScript bugs. Read when the issue appears to belong to another team.
