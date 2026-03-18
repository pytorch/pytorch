---
name: debug-graph-breaks
description: Debug and fix torch.compile graph breaks to achieve fullgraph compilation

---

# debug-graph-breaks

## Description
Interactive walkthrough for diagnosing and eliminating `torch.compile` graph breaks, guiding users toward a fully capturable `fullgraph=True` program.

## Instructions for Claude

### Overview
When this skill is invoked, help the user find, understand, and fix graph breaks in their `torch.compile`d code. The workflow has three phases: **Detect**, **Diagnose**, **Fix**.

You have access to the graph break registry at `torch/_dynamo/graph_break_registry.json` (in the PyTorch repo). Use it to look up GB IDs and provide explanations and hints. Direct users to the graph break documentation website at https://meta-pytorch.org/compile-graph-break-site/ for detailed information on specific graph break types.

### Phase 1: Detect — Find the graph breaks

Ask the user which detection method they'd like to use (or recommend one based on context):

#### Option A: `TORCH_LOGS="graph_breaks"` (recommended for most users)
Tell the user to run their script with the environment variable:
```bash
TORCH_LOGS="graph_breaks" python your_script.py
```
This prints each graph break with:
- The file and line in user code that triggered it
- The graph break reason (includes GB type, explanation, hints)
- A user code traceback
- A link to documentation for that specific graph break

Ask the user to paste the output.

#### Option B: `torch._dynamo.explain()`
For programmatic inspection, the user wraps their function:
```python
explanation = torch._dynamo.explain(fn)(*args)
print(explanation)
```
This returns an `ExplainOutput` with `graph_count`, `graph_break_count`, `break_reasons` (each with a reason string and user stack), and `ops_per_graph`. Ask the user to paste the output.

#### Option C: `fullgraph=True` (for users who want to fix one break at a time)
```python
torch.compile(fn, fullgraph=True)(*args)
```
This raises an error on the first graph break encountered. Good for iterative fixing — fix one break, re-run, fix the next.

#### Option D: User already has graph break logs
If the user pastes graph break output directly, skip to Phase 2.

### Phase 2: Diagnose — Understand each graph break

For each graph break found, do the following:

1. **Parse the graph break reason.** Extract the GB type string (e.g., "Unsupported function call", "Data-dependent branching with non-constant __bool__").

2. **Look up the GB ID.** Search `torch/_dynamo/graph_break_registry.json` for the matching `Gb_type` to find the GB ID (e.g., GB0147), explanation, and hints.

3. **Classify the graph break** using the hint categories:
   - **USER_ERROR**: The user's code has a bug that also fails in eager mode. Tell them to verify by running without `torch.compile`.
   - **FUNDAMENTAL**: Dynamo fundamentally cannot trace this pattern (e.g., generators, certain dynamic control flow). The user must restructure their code.
   - **SUPPORTABLE**: Dynamo could theoretically support this but doesn't yet. The user can file a PyTorch issue or find a workaround.
   - **CAUSED_BY_EARLIER_GRAPH_BREAK**: Fix earlier graph breaks first — this one may resolve itself.
   - **DYNAMO_BUG**: Likely a Dynamo bug. Suggest filing an issue.

4. **Identify the user code location.** Use the user stack trace to pinpoint the exact line in user code causing the break.

5. **Link to documentation.** Provide the URL: `https://meta-pytorch.org/compile-graph-break-site/gb/gb{NNNN}.html` (where NNNN is the zero-padded number from the GB ID).

6. **Prioritize.** If there are multiple graph breaks, recommend tackling them in this order:
   - CAUSED_BY_EARLIER_GRAPH_BREAK last (they may auto-resolve)
   - USER_ERROR first (bugs in user code)
   - FUNDAMENTAL next (require code restructuring)
   - SUPPORTABLE last (may need workarounds or upstream fixes)

### Phase 3: Fix — Eliminate graph breaks

For each graph break, suggest a fix based on its type. Below are strategies for the most common graph break categories. Always read the user's code before suggesting changes.

#### Common fixes by graph break type

**Data-dependent branching (GB0035, GB0170)**
- `if tensor.item() > 0:` or `if tensor.sum():` — control flow depends on tensor values.
- Fix: Use `torch.cond()` for simple if/else, or restructure to avoid data-dependent branches. Sometimes `torch.where()` works as a branchless replacement.
- If the branch is on a value that's actually constant at trace time, ensure Dynamo can prove it constant (use `torch._check()` or make it a Python constant).

**Unsupported function call (GB0147)**
- A Python function that Dynamo can't trace through.
- Fix: Check if there's a PyTorch equivalent. If it's a third-party library call, move it outside the compiled region, use `torch.compiler.disable()` on that function, or use `@torch._dynamo.decorators.leaf_function` if shape inference is possible.

**Attempt to trace generator (GB0003)**
- Generators can't be compiled.
- Fix: Convert generator to a list comprehension or regular loop, or call the generator from outside the compiled function.

**Unsupported context manager (GB0142)**
- Fix: Move the context manager outside the compiled region, or check if a Dynamo-compatible alternative exists (e.g., use `torch.no_grad()` instead of `torch.inference_mode()`).

**Tensor.item() / tolist() (GB0124, GB0109)**
- These extract scalar values from tensors, creating data dependencies.
- Fix: If `capture_scalar_outputs=True` is acceptable, set it. Otherwise, restructure to keep values as tensors.

**Module-level hooks (GB0083, GB0029)**
- Backward hooks on modules require compiled autograd.
- Fix: Either enable compiled autograd (`torch._dynamo.config.compiled_autograd=True`) or remove the hooks during compilation.

**torch.compiler.disable()'d function (GB0098, GB0099)**
- A function is explicitly marked as not compilable.
- Fix: If you control the function, remove the `@torch.compiler.disable()` decorator and make it compilable. Otherwise, restructure so the disabled function is called outside the compiled region.

**Graph break in loop (GB7000)**
- A graph break inside a for/while loop causes the entire frame to be skipped.
- Fix: Fix the underlying graph break inside the loop first. This is always caused by another graph break.

#### General strategies

- **Move non-compilable code outside `torch.compile`:** Split your function so the compilable core is in one function and non-compilable setup/teardown is outside.
- **Use `torch.compiler.disable()`:** Wrap functions that don't need compilation and don't affect tensor computation.
- **Use `@leaf_function`:** For functions that should run eagerly but whose output shapes can be inferred. Register a `@fn.register_fake` to provide shape inference.
- **Replace Python builtins with torch equivalents:** e.g., `max()` -> `torch.max()`, `abs()` -> `torch.abs()`.
- **Avoid tensor-to-Python conversions inside compiled code:** `.item()`, `.tolist()`, `.numpy()`, `bool(tensor)`, `int(tensor)`.

### Iterative workflow

After suggesting fixes:
1. Ask the user to apply the fix and re-run with `TORCH_LOGS="graph_breaks"` (or `fullgraph=True`).
2. Check if the graph break is resolved.
3. If new graph breaks appear, repeat Phase 2-3.
4. Goal: `fullgraph=True` runs without errors (zero graph breaks).

### Verification

Once all graph breaks are resolved, suggest the user verify with:
```python
compiled_fn = torch.compile(fn, fullgraph=True)
result = compiled_fn(*args)
```

If this runs without error, the program is fully capturable.

## Example Session

User: "I'm getting graph breaks in my model, help me fix them"

1. Ask: "How would you like to detect graph breaks?" and recommend `TORCH_LOGS="graph_breaks"`.
2. User pastes output showing e.g.:
   ```
   Graph break in user code at model.py:42
   Graph Break Reason: Data-dependent branching with non-constant __bool__
     Explanation: ...
     Hint: ...
   ```
3. Look up GB0035 in the registry. Explain that the `if` on line 42 depends on a tensor value.
4. Read the user's code at that line. Suggest `torch.where()` or `torch.cond()` as a replacement.
5. User applies fix, re-runs. Repeat until `fullgraph=True` passes.

## Key References

- Graph break registry: `torch/_dynamo/graph_break_registry.json`
- Graph break website: https://meta-pytorch.org/compile-graph-break-site/
- Graph break hints: `torch/_dynamo/graph_break_hints.py`
- `torch._dynamo.explain()`: `torch/_dynamo/eval_frame.py`
- `ExplainOutput`: `torch/_dynamo/backends/debugging.py`
- Compile programming model: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html
- `leaf_function`: `torch/_dynamo/decorators.py`
- `torch.compiler.disable()`: `torch/_dynamo/decorators.py`
- `error_on_graph_break()`: `torch/_dynamo/decorators.py`
