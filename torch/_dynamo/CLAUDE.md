# torch/_dynamo

TorchDynamo is a Python-level JIT compiler that captures PyTorch programs into
FX graphs by symbolically executing Python bytecode. It hooks into CPython's
PEP 523 frame evaluation API to intercept execution, traces operations into an
FX graph, compiles the graph with a backend (e.g. Inductor), and generates new
bytecode that calls the compiled code.

## Architecture Overview

The compilation pipeline, in execution order:

1. **`eval_frame.py`** — Runtime entry point. `torch.compile()` wraps a
   function in an `OptimizedModule`. At runtime, the C extension
   (`torch._C._dynamo.eval_frame`) intercepts Python frames via PEP 523.
2. **`convert_frame.py`** — `ConvertFrameAssert.__call__` checks caches,
   handles recompilation limits, calls `_compile()` → `trace_frame()`.
3. **`symbolic_convert.py`** — The heart of Dynamo. `InstructionTranslator`
   symbolically executes bytecode instruction-by-instruction. Maintains a
   symbolic `stack` (list of `VariableTracker`s) and `symbolic_locals` (dict of
   name → `VariableTracker`). Opcodes are dispatched via a `dispatch_table`
   built by `BytecodeDispatchTableMeta`.
4. **`output_graph.py`** — `OutputGraph` owns the FX graph being built (via
   `SubgraphTracer`), the `SideEffects` tracker, guards, shape environment,
   and graph args. `compile_subgraph()` finalizes the graph, calls the backend,
   and generates output bytecode.
5. **`codegen.py`** — `PyCodegen` emits output bytecode: loads graph inputs
   (via `Source.reconstruct()`), calls the compiled graph, unpacks outputs, and
   replays side effects.
6. **`resume_execution.py`** — Generates continuation functions for execution
   after graph breaks.

## Key Abstractions

### VariableTracker (`variables/`)

Every Python value encountered during tracing is wrapped in a `VariableTracker`
subclass. Key interface: `as_python_constant()`, `as_proxy()`,
`call_function()`, `call_method()`, `var_getattr()`, `reconstruct()`.

Key fields: `source` (where the value came from, for guards) and
`mutation_type` (whether/how mutations are tracked).

**Factory**: `VariableTracker.build(tx, value, source=...)` dispatches to
`VariableBuilder` (sourced values needing guards) or `SourcelessBuilder`
(values created during tracing).

Key subclass families in `variables/`: `TensorVariable` / `SymNodeVariable`
(tensor.py), `ConstantVariable` (constant.py), `ListVariable` /
`TupleVariable` (lists.py), `ConstDictVariable` (dicts.py), `SetVariable` (sets.py),
`UserFunctionVariable` (functions.py), `BuiltinVariable` (builtin.py),
`NNModuleVariable` (nn_module.py), `UserDefinedObjectVariable`
(user_defined.py), `TorchHigherOrderOperatorVariable` (higher_order_ops.py),
`LazyVariableTracker` (lazy.py). `VariableBuilder` and `SourcelessBuilder` are
in builder.py.

### Source (`source.py`)

Tracks value provenance — how to access a value at runtime. Used for guard
generation (`source.make_guard(GuardBuilder.XXX)`) and bytecode reconstruction
(`source.reconstruct(codegen)`). Root sources: `LocalSource`, `GlobalSource`.
Chained sources: `AttrSource`, `GetItemSource`, `NNModuleSource`, etc.

### Guards (`guards.py`)

Runtime conditions that must hold for cached compiled code to be reused.
Install via `install_guard(source.make_guard(GuardBuilder.TYPE_MATCH))`.
Common types: `TYPE_MATCH`, `ID_MATCH`, `EQUALS_MATCH`, `TENSOR_MATCH`,
`SEQUENCE_LENGTH`. At finalization, `CheckFunctionManager` builds a tree of
C++ `GuardManager` objects for fast runtime checking.

### Side Effects (`side_effects.py`)

Tracks mutations during tracing (attribute stores, list mutations, cell
variable updates, tensor hooks) and replays them as bytecode after graph
execution. The `MutationType` system (`variables/base.py`) controls what
mutations are allowed: `ValueMutationNew/Existing`,
`AttributeMutationNew/Existing`, or `None` (immutable). The `scope` field
prevents cross-scope mutations inside higher-order operators.

### Other key files

- `trace_rules.py` — inline/skip/graph-break decisions per function
- `exc.py` — exception hierarchy: `Unsupported` (graph break), `RestartAnalysis`
  (restart tracing), `ObservedException` (user exceptions during tracing),
  `BackendCompilerFailed`
- `config.py` — configuration flags, supports `config.patch()` context
  manager/decorator
- `bytecode_transformation.py` / `bytecode_analysis.py` — low-level bytecode
  manipulation, liveness analysis
- `pgo.py` — profile-guided optimization for dynamic shapes
- `polyfills/` — traceable replacements for stdlib functions
- `repro/` — reproduction/minification tools

## Graph Breaks

Call `unimplemented()` (from `exc.py`) to trigger a graph break:

```python
from torch._dynamo.exc import unimplemented
from torch._dynamo import graph_break_hints

unimplemented(
    gb_type="short_category_name",
    context=f"dynamic details: {value}",
    explanation="Human-readable explanation of why this breaks the graph.",
    hints=[*graph_break_hints.SUPPORTABLE],
)
```

- `gb_type`: Context-free category (no dynamic strings).
- `context`: Developer-facing details (can be dynamic).
- `explanation`: User-facing explanation (can be dynamic).
- `hints`: Use constants from `graph_break_hints.py`: `SUPPORTABLE`,
  `FUNDAMENTAL`, `DIFFICULT`, `DYNAMO_BUG`, `USER_ERROR`,
  `CAUSED_BY_EARLIER_GRAPH_BREAK`.

The `break_graph_if_unsupported` decorator on instruction handlers catches
`Unsupported`, logs the graph break, updates the `SpeculationLog`, and restarts
analysis. On the second pass, the partial graph is compiled at the break point
and a resume function handles the rest.

## Testing

Tests live in `test/dynamo/`. Use `torch._dynamo.test_case.TestCase` as base
class — it calls `torch._dynamo.reset()` in setUp/tearDown and patches config
for strict error checking.

```bash
python test/dynamo/test_misc.py                       # whole file
python test/dynamo/test_misc.py MiscTests.test_foo    # single test
python test/dynamo/test_misc.py -k test_foo           # pattern match
```

### Common patterns

The default backend to `torch.compile()` is `backend="eager"`.

**CompileCounter** — count compilations and graph ops:
```python
cnt = torch._dynamo.testing.CompileCounter()

@torch.compile(backend=cnt)
def fn(x):
    return x + 1

fn(torch.randn(10))
self.assertEqual(cnt.frame_count, 1)
self.assertEqual(cnt.op_count, 1)
```

**fullgraph=True** — assert no graph breaks:
```python
torch.compile(fn, backend="eager", fullgraph=True)(x)
```

**EagerAndRecordGraphs** — inspect captured FX graphs:
```python
backend = torch._dynamo.testing.EagerAndRecordGraphs()
torch.compile(fn, backend=backend)(x)
graph = backend.graphs[0]
```

**normalize_gm + assertExpectedInline** — snapshot test graph output:
```python
from torch._dynamo.testing import normalize_gm
self.assertExpectedInline(
    normalize_gm(backend.graphs[0].print_readable(False)),
    """\
expected output here
""",
)
```

Call `torch._dynamo.reset()` within a test when testing multiple compilation
scenarios in a single test method. The base class handles setUp/tearDown reset
automatically.

## Debugging

### TORCH_LOGS

```bash
TORCH_LOGS="graph_breaks" python script.py       # see graph breaks
TORCH_LOGS="guards,recompiles" python script.py  # see guards and recompilation reasons
TORCH_LOGS="graph_code" python script.py         # see captured FX graph code
TORCH_LOGS="+dynamo" python script.py            # full debug logging
TORCH_LOGS="bytecode" python script.py           # see bytecode transformations
```

### Structured tracing (for production)

```bash
TORCH_TRACE=/path/to/dir python script.py  # explicit trace directory
```

Analyze with `tlparse`.

### Compile-time profiling

`TORCH_COMPILE_DYNAMO_PROFILER=1` prints per-function cumtime/tottime
(cProfile-style) showing where Dynamo spends time during tracing. Set to a
file path instead to save a profile loadable by `snakeviz`.

### comptime.breakpoint() (`comptime.py`)

Drops into pdb during **compilation** to inspect Dynamo state. Call
`comptime.breakpoint()` in user code; in the pdb session use `ctx`
(`ComptimeContext`) to call `print_locals()`, `print_bt()`, `print_graph()`,
or `get_local("x").as_fake()`.

### Bytecode Debugger (`bytecode_debugger.py`)

pdb-like debugger for stepping through Dynamo-generated bytecode. Useful for
debugging segfaults (no Python traceback) and codegen errors.

```python
with torch._dynamo.bytecode_debugger.debug():
    my_compiled_fn(x)
```

**Programmatic breakpoints** (no graph break): call
`torch._dynamo.bytecode_debugger.breakpoint()` in user code, or
`codegen.extend_output(create_breakpoint())` in codegen. Auto-activates
without an explicit `debug()` wrapper.

**Segfault debugging**: `v` (verbose) then `c` (continue) — every instruction
is printed with `flush=True` before execution, so the last line before a crash
is the culprit. On exceptions, the debugger stops at the faulting instruction
automatically.

## C++ Runtime (`torch/csrc/dynamo/`)

The C/C++ layer implements the PEP 523 frame evaluation hook, the cache, and
the guard evaluation tree. Performance-critical runtime on every Python frame.

### Frame Evaluation

**`eval_frame.c`** — Installs a custom frame evaluation function via
`_PyInterpreterState_SetEvalFrameFunc`. A thread-local callback controls
behavior: `None` (disabled), `Py_False` (run-only / cache lookup), or a
callable (full Dynamo).

**`eval_frame_cpp.cpp`** — `dynamo__custom_eval_frame` is called for every
frame: gets `ExtraState` from the code object, builds a `FrameLocalsMapping`
(O(1) access to locals without dict materialization), evaluates guards via
`run_root_guard_manager()` across all `CacheEntry`s (LRU ordered). On cache
hit, executes compiled code via a shadow frame (`dynamo_eval_custom_code_impl`
copies `localsplus` into a new frame with the compiled code object). On miss,
calls the Python callback to trigger compilation.

### Cache

**`extra_state.cpp/.h`** — `ExtraState` is attached per code object via
`_PyCode_SetExtra`. Contains a `cache_entry_list` (LRU linked list),
`frame_state` (dynamic shapes detection), and `FrameExecStrategy`.

**`cache_entry.cpp/.h`** — Each `CacheEntry` stores a `RootGuardManager*`
(raw C++ pointer for fast guard eval), the compiled code object, and the
backend.

### Guard Evaluation Tree (`guards.cpp`)

Guards are organized as a C++ tree (~7800 lines) mirroring the data access
pattern. `RootGuardManager` is the root, receiving a `FrameLocalsMapping`.
Each `GuardManager` node has leaf guards and child accessors.

**LeafGuard** subclasses: `TYPE_MATCH` (Py_TYPE pointer comparison),
`ID_MATCH`, `EQUALS_MATCH`, `TENSOR_MATCH` (dtype/device/shape/strides/dispatch
keys in C++), `DICT_VERSION`, `GLOBAL_STATE` (grad mode, autocast, etc.).

**GuardAccessor** subclasses define tree edges: `FrameLocalsGuardAccessor`
(O(1) index), `GetAttrGuardAccessor`, `DictGetItemGuardAccessor`,
`GlobalsGuardAccessor`, etc.

Key optimizations: fail-fast accessor reordering, dict version tag matching to
skip subtrees, `FrameLocalsMapping` avoids dict construction,
`check_nopybind()` avoids pybind11 overhead.

### Other C++ files

- `framelocals_mapping.cpp` — O(1) frame locals/cells/freevars access
- `cpython_defs.c` — copied CPython internals for frame manipulation
- `init.cpp` — `torch._C._dynamo` module and pybind11 bindings
- `debug_macros.h` — `DEBUG_TRACE`, `NULL_CHECK`, `INSPECT(...)` (drops into
  pdb from C)
- `compiled_autograd.cpp/.h` — compiled autograd engine
