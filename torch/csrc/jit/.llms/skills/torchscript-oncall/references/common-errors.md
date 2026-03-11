# Common User Errors

## Table of Contents

1. ["Failed in the TorchScript interpreter"](#failed-in-the-torchscript-interpreter)
2. [Configuration and Build Issues](#configuration-and-build-issues)
3. [API Misuse Patterns](#api-misuse-patterns)

---

## "Failed in the TorchScript interpreter"

This is the **single most common error message** (~15+ instances) and is almost never a TorchScript bug. The TorchScript runtime executes torch ops; when an op fails, the error surfaces as "failed in the TorchScript interpreter" even though TorchScript is just the executor, not the cause.

### Actual root causes

| Root Cause | Example |
|---|---|
| Wrong tensor devices | CPU tensor passed where CUDA expected |
| Out-of-bounds index access | Embedding table index exceeds vocabulary size |
| Dtype mismatches | float32 vs float16 in conditional branches |
| Custom op failures | Thrift/RPC errors, FBGEMM failures |
| Remote execution failures | Errors surfacing through TorchScript as executor |
| Missing op registrations | Buck dependency issues |

### Diagnosis approach

1. Read the **full error message** — the actual op failure is usually printed below the "failed in the TorchScript interpreter" header
2. Check tensor devices: are all inputs on the expected device?
3. Check tensor shapes and dtypes: do they match what the model expects?
4. If a custom op is involved, debug the custom op directly
5. Only suspect TorchScript itself if the model code is verified correct and the error points to a graph pass modifying the graph incorrectly (this is rare)

---

## Configuration and Build Issues

### Missing Buck dependencies (5+ instances)

Custom ops require `torch.library.load_library` to be called, or the appropriate Buck target to be included. Without this, ops appear as "Unknown builtin op."

### split_config misconfiguration

`split_config` must properly exclude non-scriptable modules from the scripted portion. Misconfiguration causes script-time failures on modules that were never intended to be compiled.

### Model architecture changes

Shape/dimension mismatches at eval time are often caused by model architecture changes that weren't reflected in the saved model or inference config.

---

## API Misuse Patterns

### Calling trace_encoder() inside __init__

Tracing should happen outside of `__init__`. Calling it inside `__init__` can trigger "Can't redefine method: forward" errors.

### Not calling model.eval() before tracing

Failure to call `model.eval()` before `torch.jit.trace` causes non-deterministic behavior from dropout and batchnorm layers (they behave differently in train vs eval mode).

### Using x.type(self.dtype) instead of x.to(dtype=self.dtype)

During tracing, `x.type()` captures the concrete device, baking it into the graph. Use `x.to(dtype=...)` instead, which traces correctly.

### Frontend annotation tools

When users hit script-time issues, these annotations can help:

| Annotation | Effect |
|---|---|
| `@torch.jit.ignore` | Method is not compiled; calls back into Python at runtime |
| `@torch.jit.unused` | Method is not compiled; raises error if called at runtime |
| `@torch.jit.export` | Method is compiled and accessible from outside the module |
| `torch.jit._drop` | Stronger alternative to `ignore`/`unused` for stubborn cases |
