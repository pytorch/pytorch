---
name: debug-graph-breaks
disable-model-invocation: true
description: Debug and fix torch.compile graph breaks to prioritize and eliminate graph-breaks, and optionally achieve fullgraph compilation

---

# debug-graph-breaks

## Description
Interactive walkthrough for diagnosing and eliminating `torch.compile` graph breaks. The skill helps users find graph breaks, understand them, prioritize which to fix, and apply fixes — whether the goal is `fullgraph=True` or simply reducing the most impactful breaks.

## Instructions for Claude

### Overview
When this skill is invoked, help the user find, understand, and fix graph breaks in their `torch.compile`d code. The workflow has three phases: **Detect**, **Diagnose**, **Fix**.

The graph break documentation website at https://meta-pytorch.org/compile-graph-break-site/ has detailed pages for each known graph break type. When the user encounters a specific graph break (e.g., GB0059), fetch the corresponding page (e.g., `https://meta-pytorch.org/compile-graph-break-site/gb/gb0059.html`) to get detailed context, examples, and fix suggestions specific to that break.

### Phase 1: Detect — Find the graph breaks

If the user provides a script path and no existing logs, run it with `TORCH_LOGS="graph_breaks"`:
```bash
TORCH_LOGS="graph_breaks" python your_script.py 2>&1
```
This prints each graph break with:
- The file and line in user code that triggered it
- The graph break reason (includes GB type, explanation, hints)
- A user code traceback
- A link to documentation for that specific graph break

If the output is ambiguous or a break's origin is unclear, re-run with verbose mode:
```bash
TORCH_LOGS="graph_breaks" TORCHDYNAMO_VERBOSE=1 python your_script.py 2>&1
```
The verbose mode adds the internal Dynamo stack trace and recent bytecode instructions, which can help when the standard output is ambiguous.

If the user provides existing `TORCH_LOGS` output, parse it directly and skip to Phase 2.

Other detection methods are available if the user requests them or provides the relevant artifacts:

#### Option B: tlparse report
If the user has a tlparse report (a structured HTML report from `TORCH_TRACE`), they can provide:
- A local directory path to the tlparse output
- A URL to the report

A tlparse report contains:
- **`index.html`**: Main page with a **stack trie** showing all compilations as a tree, color-coded by status (green = success, lime = graph break, red = error). Each compilation is identified by a **compile id** like `[0/0]` (frame 0, first compile) or `[1/0]` (frame 1, first compile — often a resume after graph break).
- **`failures_and_restarts.html`**: Table of all `RestartAnalysis` events (graph breaks) and compilation failures, with the full graph break reason for each.
- **Per-compilation directories** (e.g., `-_0_0_0/`, `-_1_0_1/`): Contain build products:
  - `dynamo_graph_break_reason_N.txt` — Full graph break details: user code location, reason, explanation, hints, user stack trace, and internal Dynamo traceback.
  - `dynamo_output_graph_N.txt` — The FX graph produced by that compilation.
  - `compilation_metrics_N.html` — Compile time, graph metrics (ops, nodes, inputs), restart reasons, guard count, cache metrics.
  - `*_ORIGINAL_BYTECODE_N.txt` / `*_MODIFIED_BYTECODE_N.txt` — Bytecode before/after Dynamo modification.

When reading a tlparse report:
1. Start with `failures_and_restarts.html` to get a quick summary of all graph breaks.
2. Read each `dynamo_graph_break_reason_N.txt` for full details.
3. Use `compilation_metrics_N.html` to understand compile time impact per subgraph.
4. The compile IDs with `_1` suffix (e.g., `[0/0_1]`) indicate a restart — the `_1` means Dynamo restarted analysis after discovering a graph break on attempt `_0`.

#### Option C: `fullgraph=True` (fix one break at a time)
If the user wants to fix breaks one at a time rather than all at once, use `fullgraph=True`:
```python
torch.compile(fn, fullgraph=True)(*args)
```
This raises an error on the first graph break encountered. Fix it, re-run, fix the next.

### Phase 2: Diagnose — Understand each graph break

For each graph break found, do the following:

1. **Parse the graph break reason.** Extract the GB type string (e.g., "Failed to trace builtin operator", "Unsupported Tensor.item() call with capture_scalar_outputs=False") and the GB documentation URL from the output.

2. **Fetch the graph break documentation page.** Use the URL from the log output (e.g., `https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0059.html`, alternatively internet may not be available and so the user should also be able to provide a local direcetory instead) to get detailed information, examples, and suggested fixes for that specific graph break type.

3. **Classify the graph break** using the hint categories from the log output:
   - **USER_ERROR**: The user's code has a bug that also fails in eager mode. Tell them to verify by running without `torch.compile`.
   - **FUNDAMENTAL**: Dynamo fundamentally cannot trace this pattern (e.g., generators, certain dynamic control flow). The user must restructure their code.
   - **SUPPORTABLE**: Dynamo could theoretically support this but doesn't yet. The user can file a PyTorch issue or find a workaround.
   - **CAUSED_BY_EARLIER_GRAPH_BREAK**: Fix earlier graph breaks first — this one may resolve itself.
   - **DYNAMO_BUG**: Likely a Dynamo bug. Suggest filing an issue.

4. **Identify the user code location.** Use the user stack trace to pinpoint the exact line in user code causing the break. Read that code directly to understand context.

5. **Read the user's script.** Always read the relevant code in the user's script to understand the surrounding context. Compare it to examples from the graph break documentation page to identify the exact pattern causing the break and suggest the most targeted fix.

6. **Prioritize.** Not all graph breaks are equal. Help the user decide which to fix:
   - **Easiest wins first**: `print()` calls, logging, simple `.item()` calls — these have straightforward fixes.
   - **Highest impact**: Graph breaks in hot loops or inside the core forward pass matter more than breaks in initialization or infrequent code paths.
   - **CAUSED_BY_EARLIER_GRAPH_BREAK last**: These may auto-resolve when earlier breaks are fixed.
   - **Cascading breaks**: A graph break inside a loop (GB7000) causes the entire frame to fall back to eager. Fixing the inner break eliminates both.
   - If the user does NOT need `fullgraph=True`, help them identify which breaks matter most for performance and which can be left alone.

### Phase 3: Fix — Eliminate graph breaks

For each graph break, apply a fix based on its type. Always read the user's code before making changes and then make changes directly.

#### Escape hatches (compile-safe alternatives)

These are APIs that let code run inside a compiled region without causing graph breaks:

**`torch._higher_order_ops.print(format_str, *args, **kwargs)`**
- Compile-safe replacement for `print()`. Uses Python format-string syntax.
- Example: `torch._higher_order_ops.print("Activated shape: {}", h.shape)`

**`torch._dynamo.config.reorderable_logging_functions`**
- Add logging functions (like `print`, `logging.info`, custom loggers) to this set. Dynamo will reorder them to avoid graph breaks while preserving the logging call.
- Example: `torch._dynamo.config.reorderable_logging_functions.add(print)`

**`torch._dynamo.config.ignore_logging_functions`**
- Add logging functions to this list to have Dynamo completely skip them during tracing (the calls are dropped).
- Example: `torch._dynamo.config.ignore_logging_functions = [print]`

**`@torch._dynamo.decorators.leaf_function`**
- Decorator that makes a function opaque to both Dynamo and AOT autograd. The function runs in eager mode at runtime. Requires a `@fn.register_fake` for shape inference at compile time.
- Use when: the function has side effects, calls non-compilable libraries, or its internals don't benefit from compilation.
```python
@torch._dynamo.decorators.leaf_function
def my_function(x):
    # This runs eagerly at runtime, no graph break
    ...

@my_function.register_fake
def my_function_fake(x):
    # Return a tensor with the correct shape/dtype for compile-time tracing
    return torch.empty_like(x)
```
- For functions that mutate arguments: `@leaf_function(mutates_args={"buf"})`

**`@torch._dynamo.decorators.nonstrict_trace`**
- Decorator that makes a function opaque to Dynamo but still traced by AOT autograd. The function body can contain graph-breaking code, but AOT autograd still traces through it so ops get compiled by Inductor.
- Use when: the function body has graph breaks but you still want Inductor optimization of the ops inside.
```python
@torch._dynamo.decorators.nonstrict_trace
def my_function(x):
    # Dynamo won't trace this, but AOT autograd will
    return x.relu()
```

**`torch.compiler.disable()`**
- Decorator/context manager that skips compilation for a function entirely. The function runs in eager mode. Causes a graph break at the call site.
- Use when: a function is fundamentally incompatible with compilation and doesn't need to be compiled.

**Custom ops (`torch.library`)**
- Register a Python function as a custom PyTorch operator. Dynamo treats custom ops as opaque, well-typed nodes in the graph — no graph break, and Inductor can call them at runtime.
- Use when: wrapping a third-party or non-traceable function that has well-defined tensor input/output shapes.
```python
import torch.library

@torch.library.custom_op("mylib::my_op", mutates_args=())
def my_op(x: torch.Tensor) -> torch.Tensor:
    # arbitrary non-traceable code here
    return x.numpy().copy()  # example

@my_op.register_fake
def my_op_fake(x):
    return torch.empty_like(x)
```
- Unlike `@leaf_function`, custom ops are part of the public `torch.library` API and show up as named nodes in the FX graph, making them more suitable when the op will be reused across multiple compiled functions.

#### Common fixes by graph break type

**`print()` / logging (GB0059: "Failed to trace builtin operator")**
- Fix options (in order of preference):
  1. `torch._dynamo.config.reorderable_logging_functions.add(print)` — keeps the print, no code change needed
  2. `torch._dynamo.config.ignore_logging_functions = [print]` — drops the print during compilation
  3. Replace with `torch._higher_order_ops.print("format {}", arg)` — compile-safe print HOP
  4. Remove the print entirely if it's debug-only

**Data-dependent branching (GB0035, GB0170)**
- `if tensor.item() > 0:` or `if tensor.sum():` — control flow depends on tensor values.
- Fix: Use `torch.cond()` for simple if/else, or restructure to avoid data-dependent branches. `torch.where()` works as a branchless replacement for simple cases.
- If the branch is on a value that's actually constant at trace time, use `torch._check()` or make it a Python constant.

**Tensor.item() / tolist() (GB0124, GB0109)**
- These extract scalar values from tensors, creating data dependencies.
- Fix options:
  1. Restructure to keep values as tensors (e.g., use `torch.norm(h)` instead of `h.norm().item()` followed by scalar math).
  2. Set `torch._dynamo.config.capture_scalar_outputs = True` to capture `.item()` in the graph.
  3. Move the `.item()` call outside the compiled region.

**Unsupported function call (GB0147)**
- A Python function that Dynamo can't trace through.
- Fix: Check if there's a PyTorch equivalent. If it's a third-party library call:
  1. Use `@leaf_function` with a `register_fake` if shape inference is possible.
  2. Use `@nonstrict_trace` if the function body contains ops that benefit from compilation.
  3. Use `torch.compiler.disable()` if the function doesn't need compilation.
  4. Move the call outside the compiled region.

**Attempt to trace generator (GB0003)**
- Generators can't be compiled.
- Fix: Convert generator to a list comprehension or regular loop, or call the generator from outside the compiled function.

**Unsupported context manager (GB0142)**
- Fix: Move the context manager outside the compiled region, or check if a Dynamo-compatible alternative exists (e.g., use `torch.no_grad()` instead of `torch.inference_mode()`).

**Module-level hooks (GB0083, GB0029)**
- Backward hooks on modules require compiled autograd.
- Fix: Either enable compiled autograd (`torch._dynamo.config.compiled_autograd=True`) or remove the hooks during compilation.

**Graph break in loop (GB7000)**
- A graph break inside a for/while loop causes the entire frame to be skipped.
- Fix: Fix the underlying graph break inside the loop first. This is always caused by another graph break.

#### General strategies

- **Move non-compilable code outside `torch.compile`:** Split your function so the compilable core is in one function and non-compilable setup/teardown is outside.
- **Replace Python builtins with torch equivalents:** e.g., `max()` -> `torch.max()`, `abs()` -> `torch.abs()`, `sorted()` -> `torch.argsort()`.
- **Avoid tensor-to-Python conversions inside compiled code:** `.item()`, `.tolist()`, `.numpy()`, `bool(tensor)`, `int(tensor)`.
- **Keep tensor operations as tensors:** Instead of extracting scalars and doing Python math, use torch ops that operate on tensors directly (e.g., `torch.where`, `torch.clamp`, `torch.norm`).

### Iterative workflow

After applying fixes, re-run the script with `TORCH_LOGS="graph_breaks"` (or `fullgraph=True`) to verify:
1. Check if the graph break is resolved.
2. If new graph breaks appear, repeat Phase 2-3.
4. Goal: either `fullgraph=True` runs without errors, or the user has eliminated the most impactful breaks.
5. Optional: If the user is a PyTorch developer (can be determined if torch is built locally) and the graph break message is unclear, suggest a change to the error message or to create an issue.

### Benchmarking improvements

After fixing graph breaks, help the user measure the improvement. Suggest these non-intrusive approaches:

**Measuring compile time:**
- Use `torch._dynamo.utils.CompileProfiler` as the backend to see per-graph compile times:
```python
with torch._dynamo.utils.CompileProfiler() as prof:
    compiled_fn = torch.compile(fn, backend=prof)
    compiled_fn(*args)
print(prof.report())
```
- Fewer subgraphs = less compile overhead. Each graph break adds a separate compilation.

**Measuring runtime performance:**
- Simple wall-clock timing (for CPU or CUDA):
```python
import time

# Warmup
for _ in range(3):
    compiled_fn(*args)
if torch.cuda.is_available():
    torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(100):
    compiled_fn(*args)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100
print(f"Average iteration: {elapsed*1000:.2f}ms")
```

**Comparing before/after:**
- Run the same benchmark before and after fixing graph breaks.
- Key metrics: number of subgraphs (fewer = better), total compile time, average runtime per iteration.
- The number of subgraphs can be checked via `TORCH_LOGS="graph_breaks"` output (count the "Graph break in user code" lines) or from a tlparse report's stack trie.

## Key References

- Graph break website: https://meta-pytorch.org/compile-graph-break-site/
- Graph break hints: `torch/_dynamo/graph_break_hints.py`
- Compile programming model: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html
- `leaf_function`: `torch/_dynamo/decorators.py`
- `nonstrict_trace`: `torch/_dynamo/decorators.py`
- `torch.compiler.disable()`: `torch/_dynamo/decorators.py`
- `error_on_graph_break()`: `torch/_dynamo/decorators.py`
- `torch._higher_order_ops.print`: `torch/_higher_order_ops/print.py`
- `CompileProfiler`: `torch/_dynamo/utils.py`
