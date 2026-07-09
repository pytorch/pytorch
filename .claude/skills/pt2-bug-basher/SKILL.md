---
name: pt2-bug-basher
disable-model-invocation: true
description: Debug PyTorch 2 compiler stack failures including Dynamo graph breaks, Inductor codegen errors, AOTAutograd crashes, and accuracy mismatches. Use when encountering torch.compile errors, BackendCompilerFailed exceptions, recompilation issues, Triton kernel failures, FX graph problems, or when the user mentions debugging PT2, Dynamo, Inductor, or compiled model issues.
---

# PT2 Bug Basher

Debug test failures and runtime errors in the PyTorch 2 compiler stack (Dynamo, Inductor, AOTAutograd, FX graphs).

## Workflow Summary

1. **Environment check** -- Ask the user which conda environment to use. Verify it is active by checking `$CONDA_DEFAULT_ENV`. Then run `python -c "import torch; print(torch.__version__)"` to confirm torch is importable and report the version. If the environment is not active or torch cannot be imported, stop and ask the user to activate the correct environment before proceeding.
2. **Reproduce** -- Get a consistent reproduction of the failure
3. **Minimize** -- Reduce the repro to the smallest possible standalone case. Strip away unrelated model logic, use minimal tensor shapes, and isolate the specific op or pattern that triggers the bug.
4. **Add a unit test** -- **Do this BEFORE diving into code search or root cause investigation.** Add a failing test to the codebase that captures the bug. Place it in a specific, topic-appropriate test file (e.g., `test/dynamo/test_repros.py`, `test/inductor/test_torchinductor.py`, `test/export/test_export.py`). **Avoid `test/dynamo/test_misc.py`** — it is already oversized; find a more specific test file that matches the area of the bug. Use `torch.testing._internal.common_utils.TestCase` and `run_tests`. The test must fail before the fix and pass after. Having the test first keeps you grounded — you know exactly what "fixed" looks like before you start exploring the codebase.
5. **Validate on main** -- Use `EnterWorktree` to create a worktree checked out at `main`. Copy the new test file into the worktree and run the test there to confirm it **fails** on main. If the test passes on main, stop — the test may not be capturing the right bug, or the bug may already be fixed. Exit the worktree with `ExitWorktree` (action: remove) and return to the working branch before continuing.
6. **Gather logs** -- Run with appropriate `TORCH_LOGS` settings
7. **Classify** -- Use the [Error Triage](#error-triage) table to identify the category
8. **Inspect artifacts** -- Check FX graphs, IR, and generated code via `TORCH_COMPILE_DEBUG=1`
9. **Identify root cause** -- Trace from the error back through the compilation pipeline
10. **Fix** -- Apply the fix
11. **Verify** -- Run the new unit test AND nearby related existing tests (e.g., if you changed how `is_exporting` works, also run the existing `test_is_exporting` export test). Use `pytest -k` to quickly run related tests by name. The task is not complete until all pass.
12. **Self-review** -- Use the `/pr-review` skill to review your own changes before presenting them. Fix any issues it flags.
13. **Celebrate** -- Summarize the changes: explain the root cause, what was changed and why, and which tests were added/verified. Then tell the user the bug is squashed. Include a fun, varied motivational message or easter egg to keep spirits high (e.g., a pun, a quote, an ASCII art bug getting squashed). Keep it short and different each time.

## Investigation Strategy

### Prefer direct tools over meta_codesearch

Use `Grep`, `Glob`, and `Read` directly for code exploration. **Do not spawn `meta_codesearch` agents** — they are slow and expensive. The [Architectural Knowledge](#architectural-knowledge) and [Key Source Files](#key-source-files) sections below should give you enough context to know where to look. A targeted `Grep` for a function name is always faster.

### Know which compilation mode you're in

Before reading implementation code, determine the compilation mode. These share code but diverge in important ways:
- **`torch.compile`** -- Dynamo + Inductor. `tx.export=False`, no `_compiling_state_context()`.
- **`torch.export` (strict)** -- `tx.export=True`, `_compiling_state_context()` active.
- **`torch.export` (non-strict, **the default**)** -- Uses Dynamo via `fullgraph_capture` but `tx.export` may differ from strict. `_compiling_state_context()` active. Check `torch._export.config.use_new_tracer_experimental` — it changes which code path is used.

### Distinguish trace-time vs runtime

Many PT2 bugs come from confusing these two:
- **Trace-time**: Inside Dynamo's symbolic interpreter. Dynamo intercepts function calls and may constant-fold them (e.g., `is_exporting()` → `ConstantVariable(True)`).
- **Runtime**: Real tensors, real Python calls, module-level flags like `torch.compiler._is_exporting_flag`.

When debugging, add temporary `print()` statements directly in the source file rather than monkey-patching from outside — dispatch chains make monkey-patching unreliable.

## Gathering Information

Pick the right diagnostic tool based on the error category:

- **Quick overview**: `TORCH_LOGS="+dynamo,graph_breaks,recompiles" python your_script.py`
- **Full debug artifacts**: `TORCH_COMPILE_DEBUG=1 python your_script.py` — creates `torch_compile_debug/` with FX graphs, Inductor IR, and generated code
- **Generated code only**: `TORCH_LOGS="output_code" python your_script.py`
- **Structured tracing**: `TORCH_TRACE=/path/to/trace python your_script.py` then `tlparse /path/to/trace`
- **Single-threaded (for pdb)**: `TORCHINDUCTOR_COMPILE_THREADS=1 python your_script.py`

## Error Triage

Classify the failure using the error message and traceback:

| Error Pattern | Category | Jump To |
|---|---|---|
| `Unsupported: ...` or `graph break` in logs | Graph break | [Graph Breaks](#graph-breaks) |
| `BackendCompilerFailed` | Inductor/backend crash | [Backend Failures](#backend-compiler-failures) |
| `RecompileError` or `cache_size_limit` | Recompilation | [Recompilation](#recompilation-issues) |
| Accuracy mismatch / wrong numerical output | Accuracy | [Accuracy](#accuracy-issues) |
| `InternalTorchDynamoError` | Dynamo bug | [Internal Errors](#internal-dynamo-errors) |
| Segfault or CUDA IMA | Runtime crash | [Runtime Crashes](#runtime-crashes) |
| Triton assertion / index out of bounds | Triton kernel bug | [Triton Failures](#triton-kernel-failures) |

## Debugging by Category

### Graph Breaks

Graph breaks split the compiled graph into smaller subgraphs, often causing performance regressions or unexpected behavior.

**Diagnosis:**
```bash
TORCH_LOGS="graph_breaks" python your_script.py
```

**Key files:**
- `torch/_dynamo/exc.py` -- `Unsupported` exception class
- `torch/_dynamo/variables/` -- where most graph break decisions happen

**Common causes:**
- Unsupported Python constructs (data-dependent control flow, unsupported builtins)
- Tensor operations that can't be traced (in-place ops on inputs, unsupported dtypes)
- Calls to non-traceable functions

**Fix approach:**
1. Read the graph break message to identify the unsupported operation
2. Check if there's a decomposition or supported alternative
3. If the operation genuinely can't be traced, consider `torch._dynamo.allow_in_graph` or restructuring user code

### Backend Compiler Failures

`BackendCompilerFailed` means Inductor (or another backend) crashed during compilation.

**Diagnosis:**
```bash
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=2 python your_script.py
```

This generates `minifier_launcher.py` that isolates the minimal failing graph.

**Key files:**
- `torch/_dynamo/repro/after_aot.py` -- repro/minifier for post-AOT failures
- `torch/_inductor/` -- the backend itself

**Fix approach:**
1. Run the minifier to get a minimal reproduction
2. Inspect the FX graph (`TORCH_COMPILE_DEBUG=1`) to understand what ops are involved
3. Check if it's a lowering issue (`torch/_inductor/lowering.py`), scheduling issue, or codegen issue
4. Look at the generated output code if the error is in codegen

### Recompilation Issues

Excessive recompilation happens when guards are too specific, causing cache misses.

**Diagnosis:**
```bash
TORCH_LOGS="recompiles,recompiles_verbose,guards" python your_script.py
```

**Key config:**
- `torch._dynamo.config.recompile_limit` (default: 8)
- `torch._dynamo.config.fail_on_recompile_limit_hit` -- set to `True` to get a hard error

**Common causes:**
- Changing tensor shapes without marking them dynamic
- Python scalar values that change between calls
- Global state mutations between calls

**Fix approach:**
1. Read the recompilation reason from logs
2. Identify the failing guard
3. Either mark the relevant dimension as dynamic with `torch._dynamo.mark_dynamic()` or fix the source of guard instability

### Accuracy Issues

The compiled model produces different numerical results than eager mode.

**Diagnosis:**
```bash
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4 python your_script.py
```

This compares compiled vs. eager with an fp64 reference and dumps a repro if accuracy fails.

**Key utilities:**
- `torch/_dynamo/debug_utils.py` -- `same_two_models()`, `backend_accuracy_fails()`, `cast_to_fp64()`
- `torch._dynamo.config.repro_tolerance` (default: 1e-3)

**Fix approach:**
1. Get the minimal failing graph from the minifier
2. Compare eager vs. compiled output at fp64 precision
3. Binary search through ops to find the diverging operation
4. Check for known numerical issues (reduction order, fused kernels, dtype promotions)

### Internal Dynamo Errors

`InternalTorchDynamoError` indicates a bug in Dynamo itself.

**Diagnosis:**
```bash
TORCHDYNAMO_VERBOSE=1 python your_script.py
# or equivalently:
TORCH_LOGS="+dynamo" python your_script.py
```

**Key files:**
- `torch/_dynamo/symbolic_convert.py` -- bytecode interpreter
- `torch/_dynamo/variables/` -- variable tracking system
- `torch/_dynamo/guards.py` -- guard generation

**Fix approach:**
1. Get the full stack trace with `TORCHDYNAMO_VERBOSE=1`
2. Identify which bytecode instruction or variable type caused the crash
3. Create a minimal repro (the error message often includes a minifier path)
4. Debug with `TORCHINDUCTOR_COMPILE_THREADS=1` and pdb if needed

### Runtime Crashes

Segfaults and CUDA illegal memory access errors during execution of compiled code.

**Diagnosis (make crash deterministic):**
```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1 python your_script.py
```

**For CUDA IMA, add NaN checks:**
```bash
TORCHINDUCTOR_NAN_ASSERTS=1 python your_script.py
```

**For Inductor-level sync debugging:**
```python
torch._inductor.config.triton.debug_sync_kernel = True  # sync after every kernel
torch._inductor.config.triton.debug_sync_graph = True   # sync before/after graph
```

**Fix approach:**
1. Make the crash deterministic with `PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1`
2. Check if it's an input mismatch (shapes, devices, dtypes)
3. Inspect the generated kernel code with `TORCH_LOGS="output_code"`
4. Use `TORCHINDUCTOR_NAN_ASSERTS=1` to find the first kernel producing bad values
5. Check for dynamic shapes issues (historically a common source of IMA)

### Triton Kernel Failures

Triton assertion failures or index-out-of-bounds in generated kernels.

**Diagnosis:**
```bash
TORCH_LOGS="output_code,schedule" python your_script.py
```

**Key files:**
- `torch/_inductor/codegen/triton.py` -- Triton codegen
- `torch/_inductor/scheduler.py` -- kernel fusion decisions

**Fix approach:**
1. Get the generated Triton kernel from `output_code` logs
2. Check index computations for off-by-one or wrong stride calculations
3. Look at the IR (`TORCH_COMPILE_DEBUG=1`) to trace back to the FX op
4. Check if fusion decisions created invalid index combinations

## Key Source Files

| File | Purpose |
|---|---|
| `torch/_dynamo/exc.py` | Exception hierarchy and error formatting |
| `torch/_dynamo/debug_utils.py` | Minifier support, accuracy checking, input serialization |
| `torch/_dynamo/repro/after_dynamo.py` | Repro/minifier for Dynamo-stage failures |
| `torch/_dynamo/repro/after_aot.py` | Repro/minifier for post-AOTAutograd failures |
| `torch/_dynamo/repro/aoti.py` | Repro/minifier for AOTI failures |
| `torch/_dynamo/config.py` | Dynamo config (repro levels, recompile limits) |
| `torch/_dynamo/variables/torch.py` | Torch function handling, tracing state functions |
| `torch/_dynamo/variables/higher_order_ops.py` | HOP tracing (cond, map, etc.) |
| `torch/_dynamo/symbolic_convert.py` | Bytecode interpreter, InstructionTranslator |
| `torch/_dynamo/convert_frame.py` | Frame compilation, `fullgraph_capture` entry point |
| `torch/_dynamo/functional_export.py` | New export tracer (`_dynamo_graph_capture_for_export`) |
| `torch/_dynamo/eval_frame.py` | `torch._dynamo.export`, `optimize_assert` |
| `torch/_export/_trace.py` | Export pipeline (`_export`, `_strict_export`, `_non_strict_export`, `_export_to_aten_ir`) |
| `torch/_export/utils.py` | `_compiling_state_context()` |
| `torch/compiler/__init__.py` | `is_compiling()`, `is_exporting()`, runtime flags |
| `torch/_higher_order_ops/cond.py` | `torch.cond` implementation and proxy tracing |
| `torch/_higher_order_ops/utils.py` | `reenter_make_fx` for HOP branch tracing |
| `torch/_inductor/config.py` | Inductor config (debug flags, trace settings) |
| `torch/_inductor/debug.py` | DebugContext, graph visualization, IR logging |
| `torch/_logging/_registrations.py` | All registered log aliases and artifacts |

## Using the Minifier

The minifier reduces a failing graph to the smallest reproduction:

```bash
# Step 1: Generate the minifier launcher
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=2 python your_script.py

# Step 2: Run the minifier
python minifier_launcher.py minify

# Step 3: Run the minimized repro
python minifier_launcher.py run
```

For accuracy issues, use level 4:
```bash
TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4 python your_script.py
```
