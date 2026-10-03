# Native C++ SymNode: architecture

Flag: `CPP_SYMNODE=1`, which sets `torch._dynamo.config.use_cpp_symnode` (default off).

## 1. Problem

With dynamic shapes, every SymInt operation made during tracing (by meta kernels, decompositions,
dynamo, or user code) goes through Python:

```
C++ kernel: a.sym_size(0) * 4
  -> PythonSymNodeImpl::mul                 (GIL, pybind)
  -> SymNode.mul -> binary_magic_impl        (sym_node.py)
       hint math, get_proxy_mode(), _symop_cache lookup,
       sympy Mul(...) (lru_cached), pytype rules, new SymNode(...)
  -> back to C++ as a new PythonSymNodeImpl
```

On Qwen3-8B dynamic prefill that is about 8.6M node-level Python SymNode calls per compile. The
profile showed that sympy itself was cheap (3.7% of self-time, thanks to `lru_cache`). The cost
was the Python plumbing around each op: 26% in the SymNode layer, 5% in ShapeEnv, and 1-2% in the
`torch.SymInt` dunder wrappers. The goal was to make this whole per-op layer native, while
leaving Python fully authoritative for anything that records guards.

## 2. Design principles

1. **Python stays authoritative for every ShapeEnv mutation.** C++ never appends a guard, runtime
   assert, axiom, replacement or range refinement. When C++ cannot give a definite answer that
   Python would also give without mutating state, it delegates to Python.
2. **Native answers match Python's exactly.** The evaluator is a line-by-line port of Python's
   static evaluation (`_maybe_evaluate_static`). It only runs while Python's caches cannot be
   stale (the pristine gate, section 6).
3. **Native expressions are structurally identical to sympy's.** Every construction rule is
   ported line by line and differential-tested, so `to_sympy(native) == python_result` and
   `str()` matches byte for byte.
4. **Flag off is byte-identical.** Every edit outside `torch/csrc/symbolic/` sits behind
   `shape_env._native_env is not None` or `isinstance(x, _NativeSymNode)`.
5. **Unsupported means fallback, not approximation.** A step that needs machinery the port lacks
   throws `NativeUnsupported`. The op then reruns in Python and gives the exact Python result.

## 3. Component map

How to read this chart: boxes are grouped by where they live (Python, the bridge module, or
pure C++). Each box says what the piece is for, then its code name. The arrows show every way the
layers talk to each other. Most traffic stays inside the C++ core. Python is reached only through
the three labeled exits: handing an op or question to Python, recording an FX node, and replaying
C++ answers into Python's caches.

```mermaid
flowchart TB
    subgraph PY["Python side: owns all guard state"]
        USER["<b>Code that does shape math</b><br/>user code, dynamo, Python decompositions<br/><i>e.g. s0 * 4, s0 == s1, bool(s0 &gt; 1)</i>"]
        SYMINT["<b>The SymInt object users hold</b><br/>torch.SymInt / SymBool<br/><i>its operators are swapped for C versions</i>"]
        SE["<b>The source of truth for shapes</b><br/>ShapeEnv: guards, axioms, replacements, ranges<br/><i>only Python ever writes these</i>"]
        PYNODE["<b>The original slow path, kept as a safety net</b><br/>Python SymNode + sympy<br/><i>used whenever C++ gives up</i>"]
        PROXY["<b>FX graph recording</b><br/>make_fx proxy mode, handle_sym_dispatch<br/><i>turns SymInt ops into graph nodes</i>"]
    end

    subgraph BIND["Bridge: torch._C._symbolic, in libtorch_python"]
        GLUE["<b>Fast SymInt operators</b><br/>python_symint_glue.cpp<br/><i>runs s0 * 4 without entering Python</i>"]
        PYSYM["<b>Translator between the two worlds</b><br/>python_symbolic.cpp<br/><i>exposes C++ nodes to Python, converts expressions<br/>to and from sympy, builds the Python twin of a node</i>"]
    end

    subgraph CORE["Native core: torch/csrc/symbolic/, pure C++"]
        NODE["<b>A symbolic integer in C++</b><br/>NativeSymNodeImpl<br/><i>holds an expression, the current value (hint) and a type</i>"]
        NSE["<b>Read-only copy of ShapeEnv</b><br/>NativeShapeEnv<br/><i>symbols and ranges, cache of op results,<br/>answers guard questions that need no new guard</i>"]
        EXPR["<b>Expressions such as 4*s0 + 8</b><br/>Expr / ExprArena<br/><i>each distinct expression stored once, compared by pointer</i>"]
        RULES["<b>C++ re-implementation of the sympy rules PyTorch uses</b><br/>simplification, facts, comparisons, FloorDiv/Mod/Max,<br/>printing, range math"]
    end

    CPPK["<b>C++ meta kernels and FakeTensor</b><br/><i>use SymInts through the normal c10 interface;<br/>they cannot tell the node is native</i>"]

    USER -- "uses an operator" --> SYMINT -- "dispatches" --> GLUE --> NODE
    CPPK -- "e.g. size * 4" --> NODE
    NODE -- "asks a guard question,<br/>checks the op cache" --> NSE
    NODE -- "builds the result expression" --> EXPR
    NSE --> EXPR
    EXPR -- "applies" --> RULES
    SE -- "copies new symbols and ranges;<br/>reports every state change" --> NSE
    NODE -- "C++ gives up, or a guard must be recorded:<br/>hand over to Python" --> PYSYM
    PYSYM --> PYNODE -- "records the guard" --> SE
    NODE -- "tracing under make_fx:<br/>record an FX node" --> PYSYM
    PYSYM --> PROXY
    NSE -- "before Python uses its caches:<br/>replay C++ answers so they match" --> SE
```

### Files (about 12.6K lines of C++)

| layer | file | role |
|---|---|---|
| expressions | `Expr.h/.cpp` | `Expr` kinds, the hash-consing `ExprArena`, Integer/Rational/Float/oo/int_oo numbers, Symbol, Add/Mul/Pow construction |
| | `Assumptions.cpp` | three-valued facts (`is_integer`, `is_positive`, `is_nonnegative`, ...) per kind, ported from sympy's `_eval_is_*` |
| | `Relational.cpp` | Eq/Ne/Lt/Le/Gt/Ge evaluation, And/Or/Not, `canonicalize_bool_expr` |
| | `Functions.cpp` | every class in `torch/utils/_sympy/functions.py` (FloorDiv, Mod, PythonMod, Max, Min, CeilDiv, PowByNatural, ModularIndexing, Where, OpaqueUnaryFn_*, ...) |
| | `Expand.cpp`, `ExprTools.cpp`, `Sorting.cpp` | `safe_expand`, xreplace/free symbols/gcd, sympy `sort_key`/`ordered` |
| | `Printer.cpp` | sympy StrPrinter port, so `str(node)` needs no sympy |
| | `ValueRanges.h/.cpp` | `ValueRanges` and the `bound_sympy` handlers |
| state | `NativeShapeEnv.h/.cpp` | mirrored ShapeEnv state, symop memo, static evaluator, pristine gate, replay log |
| node | `NativeSymNodeImpl.h/.cpp` | the `c10::SymNodeImpl` subclass (about 57 overridden virtuals) |
| bridge | `PyFallback.h` | declarations of the only ways the core reaches Python: `materialize`, `python_impl`, `proxy_dispatch`, `proxy_mode`, `live_nodes_changed`, `native_config_is_default` |
| bindings | `python_symbolic.cpp` | `torch._C._symbolic` module: `_NativeSymNode`, `NativeShapeEnv`, `_Arena`, conversion, `PyFallback` definitions, stats and test hooks |
| | `python_symint_glue.cpp` | `_SymGlueMethod`: C implementations of the `torch.SymInt`/`SymBool` dunders for native nodes |

The core is compiled into libtorch_python (`build_variables.bzl`) so that incremental rebuilds
relink the smaller library and slow paths can call Python directly. The core includes no Python
headers, so moving it to libtorch_cpu is a bzl move. C++ kernels never link against the concrete
class; they only call `c10::SymNodeImpl` virtuals. No c10 headers were edited.

## 4. Lifecycle: how native nodes come into existence

How to read this chart: it follows one dynamic dimension (for example `seq_len = 128`) from the
start of a compile to the `SymInt` that dynamo hands to the rest of tracing. Python still does
all symbol creation; C++ only receives a copy and then decides whether it can take over the node.
Every "no" branch ends in exactly the upstream behavior.

```mermaid
flowchart TD
    A["<b>A new compile starts</b><br/><i>ShapeEnv.__init__</i>"] --> B{"<b>Should C++ be used?</b><br/>flag CPP_SYMNODE=1, and no config<br/>that C++ does not model<br/><i>e.g. translation validation, event recording,<br/>backed_size_oblivious, trace_asserts</i>"}
    B -- no --> PYONLY["<b>Everything stays Python</b><br/>identical to upstream<br/><i>_native_env = None</i>"]
    B -- yes --> C["<b>Set up the C++ side</b><br/>create the C++ copy of ShapeEnv;<br/>swap SymInt operators for C versions, once per process;<br/>make ShapeEnv's state containers report every write<br/><i>NativeShapeEnv, _install_native_glue, notifying containers</i>"]
    C --> D["<b>Dynamo sees an input with a dynamic dimension</b><br/>e.g. seq_len = 128<br/><i>create_symbol</i>"]
    D --> E["<b>Python creates the symbol s0 as usual</b><br/>picks the name, applies duck sizing and 0/1 specialization,<br/>sets its range, e.g. [2, inf]<br/><i>unchanged upstream code</i>"]
    E --> F["<b>Copy the symbol into C++</b><br/>name, current value 128, range [2, inf]<br/><i>mirror_symbol</i>"]
    F --> G["<b>Wrap the symbol in a node</b><br/>Python first builds its usual SymNode<br/><i>create_symintnode</i>"]
    G --> H{"<b>Can C++ represent it exactly?</b><br/>has a concrete value that fits in int64;<br/>every symbol in it was copied to C++;<br/>converting back gives the same sympy expression<br/><i>_maybe_native</i>"}
    H -- yes --> I["<b>SymInt backed by a C++ node</b><br/>all later math on it runs in C++"]
    H -- no --> J["<b>SymInt backed by a Python node</b><br/>e.g. unbacked (data-dependent) sizes, floats<br/>behaves exactly as upstream"]
```

What stays Python by design: symbol creation, unbacked symbols (`create_unbacked_*`), SymFloat
creation, and every guard write. Because unbacked symbols never go native, data-dependent errors
always come from Python.

## 5. Hot path: one SymInt operation

Example: `s0 * 4`, called either from Python (`SymInt.__mul__`) or from a C++ kernel
(`SymInt::operator*` -> `SymNodeImpl::mul`).

How to read this chart: it follows `s0 * 4` (with `s0 = 128`) from either entry point, a Python
operator on the left or a C++ kernel on the right, through the numbered steps of the C++ multiply.
The straight path down the middle never touches Python, and it is the path almost every op takes.
Each branch to "Hand over to Python" is a case where C++ cannot guarantee the same result as
Python, so Python does the op instead.

```mermaid
flowchart TD
    PYCALL["<b>Python code multiplies</b><br/>s0 * 4"] --> DESC{"<b>Simple enough for the C fast path?</b><br/>s0 is C++-backed; the other operand is a plain int/bool<br/>or also C++-backed; no keyword args; debug logging off<br/><i>_SymGlueMethod</i>"}
    DESC -- no --> ORIG["<b>Run the original Python operator</b><br/>exact upstream behavior<br/><i>_make_user_magic</i>"]
    DESC -- yes --> WRAP["<b>Turn 4 into a constant node</b><br/>and call the C++ multiply<br/><i>wrap_int</i>"]
    CCALL["<b>A C++ kernel multiplies</b><br/>e.g. computing a stride: size * 4"] --> MUL
    WRAP --> MUL["<b>C++ multiply starts</b><br/><i>NativeSymNodeImpl::mul</i>"]
    MUL --> HINT["<b>1. Compute the concrete result</b><br/>128 * 4 = 512, using Python's integer rules;<br/>on overflow or divide-by-zero, hand over to Python"]
    HINT --> PX{"<b>2. Is an FX graph being recorded?</b><br/>make_fx proxy mode is on"}
    PX -- yes --> PD["<b>Record the op in the graph</b><br/>Python's tracer adds a mul node,<br/>then the multiply itself runs in C++<br/><i>proxy_dispatch, handle_sym_dispatch</i>"]
    PX -- no --> SAME{"<b>3. Is the other operand a C++ node<br/>from the same ShapeEnv?</b><br/>a constant like 4 always is"}
    SAME -- no --> FB["<b>Hand over to Python</b><br/>build a Python twin of s0 and run Python's multiply;<br/>the result is a Python node from here on<br/><i>materialize, fallback</i>"]
    SAME -- yes --> MEMO{"<b>4. Was this exact op computed before?</b><br/>cache keyed on (op, left expression, right expression)<br/><i>symop memo</i>"}
    MEMO -- yes --> NEW
    MEMO -- no --> CONS["<b>5. Build the result expression</b><br/>apply the ported sympy rules, e.g.<br/>s0 * 4 becomes 4*s0, 4 * (s0 + 2) becomes 4*s0 + 8;<br/>pure C++, no Python"]
    CONS -- "needs a rule C++ does not implement" --> FB
    CONS -- ok --> NEW["<b>6-7. Create the result node</b><br/>expression 4*s0, value 512, type int;<br/>always a fresh node, as in Python"]
    NEW --> RET["<b>Return a SymInt to the caller</b><br/>built in C, no Python frame"]
    PD --> RET
    FB --> RET
```

Key points:
- **Expressions are interned.** Each `NativeShapeEnv` owns an `ExprArena`. Structurally equal
  expressions are the same `const Expr*`, so equality is a pointer compare and memo keys are pointer
  triples.
- **Nodes are immutable and freshly allocated,** matching Python's identity semantics.
  `clone()` returns `this`.
- **Locking:** one `std::mutex` per env guards the arena, memo and mirror. No Python call is made
  while it is held, to avoid a GIL/mutex deadlock.
- **Proxy dispatch runs before any fallback decision,** so the SymInts reaching
  `handle_sym_dispatch` carry the original native nodes and proxy-slot lookups hit.
- **Mixed native/Python operands** produce Python results, and the result stays Python.

## 6. Guard queries: static evaluation and delegation

Examples: `bool(s0 > 1)`, `guard_or_false(...)`, `statically_known_true(...)`.

How to read this chart: it follows a yes/no question about shapes, such as `if s0 > 1:`. C++
answers only when the answer can be proved from facts ShapeEnv already has, which means no new
guard is needed. Every other question goes to Python, which answers it using the current value
and records a guard. Before Python answers, it is caught up on the answers C++ already gave, so
its caches look exactly as if Python had answered everything itself.

```mermaid
flowchart TD
    Q["<b>Code asks a yes/no question about shapes</b><br/>e.g. if s0 &gt; 1<br/><i>guard_bool, guard_or_false, statically_known_true, ...</i>"] --> CFG{"<b>Are the shape configs at their defaults?</b><br/>C++ does not model non-default guard behavior<br/><i>backed_size_oblivious, aggressive_guard_free_semantics</i>"}
    CFG -- no --> DEL
    CFG -- yes --> NUM{"<b>Is the answer already a constant?</b><br/>the expression is just True, False or a number"}
    NUM -- yes --> ANS["<b>Answer in C++</b><br/>no guard needed"]
    NUM -- no --> PR{"<b>Has ShapeEnv learned anything since<br/>the symbols were created?</b><br/>any guard, runtime assert, axiom,<br/>replacement or narrowed range<br/><i>pristine gate</i>"}
    PR -- "yes: the C++ copy may be out of date" --> DEL
    PR -- "no: the C++ copy is complete" --> STATIC["<b>Try to prove it from known facts alone</b><br/>without looking at the current value;<br/>e.g. s0 &gt; 1 always holds when s0 is in [2, inf];<br/>same steps as Python's static evaluator<br/><i>_maybe_fast_eval_comparison, _maybe_evaluate_static</i>"]
    STATIC -- "proved true or false" --> LOG["<b>Note the question and its answer</b><br/>so Python's caches can be caught up later<br/><i>replay log</i>"]
    LOG --> ANS
    STATIC -- "cannot prove it" --> DEL["<b>Hand the question to Python</b><br/>build the Python twin of the node<br/>and call the same Python method<br/><i>delegate</i>"]
    DEL --> FLUSH["<b>First, catch Python up</b><br/>replay every answer C++ gave, in order,<br/>so Python's caches match a flag-off run<br/><i>replay log flush</i>"]
    FLUSH --> PYEVAL["<b>Python answers using the current value</b><br/>e.g. s0 = 128, so s0 &gt; 5 is True,<br/>and it records the guard s0 &gt; 5<br/>exactly as with the flag off"]
```

### The pristine gate

The native evaluator answers only while the ShapeEnv is **pristine**: no guards, deferred runtime
asserts, axioms, divisibility facts or replacements, and no range updates after symbol creation.
In that state a fresh native evaluation provably equals what Python computes, because Python's
version-keyed caches can only hold entries computed in the same state.

Python writes this state from many places, some of which bypass ShapeEnv methods (for example
`var_to_range[k] &= ...` in export). So, under a native env only, the seven state containers are
replaced by `list`/`dict`/`set` subclasses whose mutators call a pre-mutation hook
(`mark_not_pristine` or `mark_replacements`). After the first mutation:
- evaluation always delegates to Python (correct, but slow);
- arithmetic stays native while replacements are empty, because `expr == _expr` then;
- native `mod` answers only from the memo, since choosing Mod vs PythonMod reads ranges.

On Qwen the env stays pristine for the whole compile.

### The replay log

Native answers skip Python's LRU caches, and those caches do not key on guards or ranges. Left
alone, the flag-off path could later return a stale cached `None` (and guard) where flag-on
computed fresh. To prevent this, each env logs every distinct native query in last-use order.
**Before Python performs any evaluation or mutation on any env**, the logs of all live native envs
are replayed through the real Python methods, oldest first. The flush points are the
`evaluate_expr` entry, the `_lru_cache` wrapper, `guard_or_defer_runtime_assert`, the notifying
containers, `add_backed_var_to_val` and GraphPickler unpickle. Python's caches then end up with
exactly the flag-off entries in the flag-off order. An empty flush is one C call.

## 7. Materialization and fallback

`PyFallback::materialize(node)` builds a Python
`SymNode(expr=to_sympy(expr), shape_env, pytype, hint, constant, ...)` once per native node, with
the GIL held, and caches it on the binding side. The cache entry is erased in the node's destructor
if the node was ever materialized.
- `to_sympy` is cached per `Expr` id. Conversion is pure, so the cache never goes stale.
- Delegated ops call the Python method on the materialized node(s), and the result is a Python
  node.
- Every materialization and delegation is counted by reason (`torch._C._symbolic._stats()`).

About 180 `NativeUnsupported` sites exist. The main causes are eval steps needing general
`sympy.gcd`/`simplify`, `zoo`/`nan`/complex results, int64 overflow, non-53-bit Floats and sort-key
ties.

## 8. Python integration

- **`_NativeSymNode`** is the pybind class for `NativeSymNodeImpl`, registered as a subclass of
  `_SymNode`. It exposes the full Python `SymNode` surface: `expr` (lazy `to_sympy`), `hint`,
  `pytype`, `shape_env`, `constant`, `fx_node`, `free_symbols()`, every op method, `wrap_*`, the
  guard methods, `__reduce__`/`__deepcopy__` (which materialize) and `__repr__`. pybind's instance
  table keeps `symint.node` identity-stable while a Python reference exists, which `_SymNodeDict`
  proxy slots rely on.
- **`SymNodeTypes = (SymNode, _NativeSymNode)`** in `sym_node.py` replaces
  `isinstance(x, SymNode)` checks in sym_node, symbolic_shapes, proxy_tensor, partitioners,
  runtime_assert, sizevars, `_fake_tensor_utils` and `_graph_pickler`. A test forbids new plain
  `isinstance(..., SymNode)` checks.
- **Module-level helpers** in `symbolic_shapes.py` (`_guard_or`, `statically_known_true`,
  `expect_true`, `guard_bool/int`, `has_hint`, `free_symbols`, `has_free_symbols`,
  `guarding_hint_or_throw`, ...) dispatch straight into C++ for native nodes.
- **SymInt glue (`python_symint_glue.cpp`).** At the first native `ShapeEnv`, the hot attributes
  on `torch.SymInt`/`SymBool` (arithmetic, comparisons, `__bool__`, `__index__`, `__hash__`,
  `hint`, `sym_ite`, ...) are replaced by C `_SymGlueMethod` descriptors. Inside the fast-path
  domain they run the entire `_make_user_magic` logic in C++ and build the result object with
  `make_sym_object` (no Python frame). Outside it they vectorcall the saved original. This is what
  removed the roughly 1.9M `_make_user_magic` calls.
- **Casters.** `utils/pybind.cpp` and `utils/python_symnode.{h,cpp}` accept native nodes as
  operands and build `SymBool`/`SymInt` objects for them.
- **Lifetime.** Native nodes hold an `intrusive_ptr<NativeShapeEnv>`, so the arena outlives
  every node. The binding keeps a strong reference to the Python `ShapeEnv` only while live native
  nodes exist; an atomic count toggles it on the 0 <-> 1 transitions. A permanent reference would
  create an uncollectable cycle through `_native_env`.

## 9. Verification strategy

- **`test/test_native_symnode.py`** (about 5.7K lines, 19 classes). For every construction rule,
  fact, relational, function, printer case and value-range handler, the test builds the
  expression natively and in sympy, then asserts `to_sympy(native) == sympy` and
  `str(native) == str(sympy)`. Inputs include the 320-op Qwen corpus and seeded random expression
  trees.
- **Node level:** native vs Python `SymNode` for every virtual (hint, pytype, result expr, guard
  results, no extra guards recorded, overflow fallback, mixed operands), plus make_fx traces that
  must give identical graphs.
- **Evaluator:** the same query stream goes to both evaluators, in pristine and non-pristine
  states. The native answer must equal Python's or delegate, never differ.
- **Glue:** twin native ShapeEnvs run identical random op sequences through the descriptors and
  through the saved originals, comparing results, guards and replay logs.
- Each C++ semantic change was also cross-checked by a separate reviewer agent running its own
  differential fuzzers (often 0.5-1M checks per item).
- **Debug mode:** `TORCH_NATIVE_SYMNODE_CHECK=1` asserts at every flush that the mirror matches
  Python and that replayed results equal the native ones.
- **Crossing accounting** (Qwen harness, not checked in): a `sys.monitoring` hook
  attributes every Python SymNode, sympy and glue call during the compile to an allow-listed bucket
  or to NOT ALLOWED, and the harness fails unless NOT ALLOWED is empty.

## 10. Results (measured with C++ FakeTensor)

Qwen3-8B dynamic prefill, crossings during the compiled call:

| bucket | flag off | flag on |
|---|---|---|
| Python SymNode (node level) | 8,608,756 | 4 |
| SymInt wrappers (`_make_user_magic`) | 1,894,508 | 4 |
| sympy package | 3,066,423 | 146,528 |
| ShapeEnv (symbolic_shapes.py) | 1,890,516 | 765,913 |
| NOT ALLOWED | n/a | 0 |

What remains runs once per input, per produced size or per finished graph, not per op: symbol
creation, final guard production, post-trace passes and FX proxy bookkeeping.

Timing (aot_eager, first call / AOT-only, seconds):

| config | first call | aot_only |
|---|---|---|
| Python FakeTensor (with dispatch cache) | 13.25 | 6.22 |
| C++ FakeTensor | 14.37 | 7.08 |
| C++ FakeTensor + native SymNode | 10.39 | 4.29 |

Numerics were exact, the dynamo/AOT graphs and `shape_env` goldens were byte-identical to flag
off, and test_dynamic_shapes, test_sympy_utils, test_fake_tensor and test_proxy_tensor showed no new
failures with the flag on or off.

## 11. Known gaps

- **Pristine gate.** A workload that guards early (specialization, `torch._check`, recompiles)
  delegates every evaluation to Python after the first guard. Lifting this needs a port of
  ShapeEnv `simplify`/replacements, axioms and `_maybe_guard_rel`.
- **Unbacked symbols and SymFloat creation** stay Python. `guard_size_oblivious` delegates.
- **make_fx graph construction** (`handle_sym_dispatch`, `set_proxy_slot`) is Python by nature.
- **Location.** The core lives in libtorch_python, so C++ without Python loaded cannot create
  native nodes.
- **Python FakeTensor.** All numbers above were measured with C++ FakeTensor. Native SymNode
  under Python FakeTensor has not been tested yet.
