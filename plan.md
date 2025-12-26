# torch.compile Cold Compile Time Optimization Plan

Based on analysis of the codebase and inspired by [Abseil Performance Hints](https://abseil.io/fast/hints.html), this plan identifies optimization opportunities to reduce cold compile times.

---

## 1. Pre-compile Regex Patterns at Module Level

**Problem**: Regex patterns compiled at runtime instead of module load time.

**Files**:
- `torch/_inductor/codecache.py:615-616` - `re.sub()` patterns in `_reduce_graph_module()`
- `torch/_dynamo/codegen.py:409` - Character sanitization pattern
- `torch/_inductor/utils.py:260` - Pattern in profiling loop
- `torch/_inductor/graph.py:972-974` - `get_dtype()` regex match

**Changes**:
```python
# Before (runtime compilation)
code = re.sub(r"kernel_idx = \d+", "", code)

# After (module-level pre-compilation)
_KERNEL_IDX_PATTERN = re.compile(r"kernel_idx = \d+")
# ... in function:
code = _KERNEL_IDX_PATTERN.sub("", code)
```

- [x] Move `kernel_idx` and `constant_args_idx` patterns to module level in `codecache.py`
- [x] Pre-compile character sanitization pattern in `codegen.py`
- [x] Pre-compile `fused_abs_max_\d` pattern in `utils.py`
- [x] Pre-compile `as_strided|reinterpret_tensor` pattern in `graph.py`

**Test**:
```python
import time
import torch

def test_regex_precompile():
    """Verify regex patterns are pre-compiled at module level."""
    from torch._inductor import codecache
    assert hasattr(codecache, '_KERNEL_IDX_PATTERN'), "Pattern should be module-level"

    # Measure compile time improvement
    @torch.compile
    def f(x):
        return x + 1

    start = time.perf_counter()
    f(torch.randn(10))
    elapsed = time.perf_counter() - start
    print(f"First compile: {elapsed:.3f}s")
```

---

## 2. Reduce String Concatenation in Guard Generation

**Problem**: Guards are built using f-string concatenation in loops, creating many intermediate strings.

**Files**:
- `torch/_dynamo/guards.py:2849` - `.join()` inside f-string for C++ guards
- `torch/_dynamo/guards.py:2953-2960` - String building in TENSOR_MATCH loop
- `torch/_dynamo/guards.py:2833-2852` - Large multi-line f-string with `textwrap.dedent`

**Changes**:
```python
# Before
for term in terms:
    code.append(f"{tensor_name}.{term} == {real_value}")
result = " && ".join(code)

# After - use list accumulation with single join
code_parts = []
for term in terms:
    code_parts.append(f"{tensor_name}.{term} == {real_value}")
result = " && ".join(code_parts)
```

- [x] Replace string concatenation with list accumulation + join in guard code builders
- [x] Cache the C++ template header string instead of using `textwrap.dedent` each time
- [x] Use `io.StringIO` for large guard code string assembly

**Test**:
```python
import torch

def test_guard_string_building():
    """Verify guard generation doesn't create excessive intermediate strings."""
    import tracemalloc
    tracemalloc.start()

    @torch.compile
    def f(x, y, z):
        return x + y * z

    f(torch.randn(10), torch.randn(10), torch.randn(10))

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
    # Should see reduced peak memory from fewer intermediate strings
```

---

## 3. Cache Dict Key Lookups - Avoid Double Hashing

**Problem**: Dict keys are hashed/created twice - once for `in` check, once for access.

**Files**:
- `torch/_dynamo/variables/dicts.py:245-251` - `Hashable(vt)` created twice in `__contains__`
- `torch/_dynamo/variables/dicts.py:354-389` - Repeated key creation in `getitem_const()`

**Changes**:
```python
# Before
def __contains__(self, vt):
    return Hashable(vt) in self.items and not isinstance(self.items[Hashable(vt)], DeletedVariable)

# After
def __contains__(self, vt):
    key = Hashable(vt)
    return key in self.items and not isinstance(self.items[key], DeletedVariable)
```

- [x] Cache `Hashable(vt)` in local variable in `__contains__`
- [x] Cache key in `getitem_const()` and `maybe_getitem_const()`
- [x] Apply same pattern to `dict.get()` implementation

**Test**:
```python
import torch

def test_dict_key_caching():
    """Verify dict operations don't double-hash keys."""
    @torch.compile
    def f(d):
        return d.get('a', 0) + d.get('b', 0)

    result = f({'a': 1, 'b': 2})
    assert result == 3
    # Profile should show reduced Hashable instantiations
```

---

## 4. Reduce SideEffects Clone Overhead

**Problem**: `SideEffects.clone()` creates multiple nested dict copies on every speculation/rollback.

**Files**:
- `torch/_dynamo/side_effects.py:195-208` - Multiple `dict()` and `list()` copies

**Changes**:
```python
# Before
store_attr_mutations={k: dict(v) for k, v in self.store_attr_mutations.items()}

# After - use copy-on-write or shallow copy where safe
store_attr_mutations=self.store_attr_mutations.copy()  # If mutations are immutable
# Or use ChainMap for copy-on-write semantics
```

- [x] Analyze mutation patterns to determine if shallow copy is safe
- [x] Consider copy-on-write wrapper for `store_attr_mutations`
- [x] Use `list.copy()` instead of `list()` constructor

**Test**:
```python
import torch

def test_side_effects_clone():
    """Verify SideEffects clone is efficient."""
    @torch.compile
    def f(x):
        y = x.clone()
        y.add_(1)  # Triggers mutation tracking
        return y

    # Multiple compiles with mutations
    for _ in range(5):
        torch._dynamo.reset()
        f(torch.randn(10))
    # Should not see O(n^2) growth in clone operations
```

---

## 5. Lazy Load Polyfill Modules

**Problem**: All 11+ polyfill modules loaded at `torch._dynamo` import time.

**Files**:
- `torch/_dynamo/__init__.py:76` - Eager polyfill loader import
- `torch/_dynamo/polyfills/loader.py:34-47` - Immediate import of all submodules

**Changes**:
```python
# Before - loads all polyfills immediately
POLYFILLED_MODULES = tuple(
    importlib.import_module(f".{submodule}", package=polyfills.__name__)
    for submodule in POLYFILLED_MODULE_NAMES
)

# After - lazy loading on first use
_POLYFILLED_MODULES = None
def get_polyfilled_modules():
    global _POLYFILLED_MODULES
    if _POLYFILLED_MODULES is None:
        _POLYFILLED_MODULES = tuple(...)
    return _POLYFILLED_MODULES
```

- [x] Convert polyfill loader to lazy initialization pattern
- [x] Only load polyfills that are actually used during tracing
- [x] Defer trace_rules modification until first compilation

**Test**:
```python
import time

def test_lazy_polyfill_loading():
    """Verify polyfills are loaded lazily."""
    import sys
    # Clear any cached imports
    for mod in list(sys.modules.keys()):
        if 'polyfills' in mod:
            del sys.modules[mod]

    start = time.perf_counter()
    import torch._dynamo
    import_time = time.perf_counter() - start

    # Check polyfills not loaded yet
    polyfill_loaded = any('polyfills.builtins' in m for m in sys.modules)
    print(f"Import time: {import_time:.3f}s, Polyfills loaded: {polyfill_loaded}")
    # polyfill_loaded should be False until first compile
```

---

## 6. Combine Multiple Graph Passes

**Problem**: Multiple sequential passes over `graph.nodes` that could be combined.

**Files**:
- `torch/_inductor/fx_passes/reinplace.py:434-779` - Three separate passes
- `torch/_inductor/fx_passes/post_grad.py:1332-1361` - Double pass for get_attr cleanup
- `torch/_inductor/fx_passes/post_grad.py:1455-1470` - Multiple `find_nodes()` calls

**Changes**:
```python
# Before - three passes
for node in reversed(graph.nodes):  # Pass 1: build node_order
    node_order[node] = ...
for node in graph.nodes:  # Pass 2: handle inplacing
    if inplaceable_op := ...
for node, replacement in replace_dict.items():  # Pass 3: apply replacements
    node.replace_all_uses_with(replacement)

# After - combined pass
for i, node in enumerate(reversed(graph.nodes)):
    node_order[node] = len(graph.nodes) - i - 1
    # Also collect inplaceable ops here for later processing
    if is_inplaceable(node):
        inplaceable_nodes.append(node)
```

- [x] Combine node_order building with inplaceable op collection in `reinplace.py`
- [x] Combine get_attr collection and cleanup in `post_grad.py`
- [x] Replace multiple `find_nodes()` calls with single pass collecting all node types

**Test**:
```python
import torch

def test_combined_graph_passes():
    """Verify graph passes are combined efficiently."""
    @torch.compile
    def f(x):
        y = x.view(-1)
        z = y + 1
        return z.view(x.shape)

    # Enable pass timing
    import torch._inductor.config as config
    with config.patch({"trace.enabled": True}):
        f(torch.randn(2, 3))
    # Check logs for combined pass execution
```

---

## 7. Cache SymPy Simplification Results

**Problem**: Heavy SymPy operations during compilation without expression-level caching.

**Files**:
- `torch/fx/experimental/symbolic_shapes.py:6484-6560` - `simplify()` method
- `torch/fx/experimental/symbolic_shapes.py:6452-6470` - `replace()` method
- `torch/fx/experimental/symbolic_shapes.py:3170-3260` - Constraint solving

**Changes**:
```python
# Before
def simplify(self, expr):
    expr = safe_expand(expr)  # Expensive
    expr = self.replace(expr)  # More expensive
    # ... more operations

# After - add subexpression caching
@functools.lru_cache(maxsize=1024)
def _simplify_subexpr(self, expr_hash):
    # Cache simplified subexpressions
    ...

def simplify(self, expr):
    # Early exit for simple cases
    if expr.is_number:
        return expr
    # Use cached subexpression results
    ...
```

- [x] Add early exit for numeric/simple expressions in `simplify()`
- [x] Cache `safe_expand()` results for repeated expressions
- [x] Cache FloorDiv divisibility check results
- [x] Add subexpression deduplication before constraint solving

**Test**:
```python
import torch

def test_sympy_caching():
    """Verify SymPy operations are cached."""
    @torch.compile(dynamic=True)
    def f(x):
        return x.view(-1).sum()

    # Multiple calls with different shapes should reuse cached simplifications
    for size in [(2, 3), (3, 4), (4, 5)]:
        f(torch.randn(size))

    # Check cache hit rate in symbolic_shapes
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    # Inspect cache statistics
```

---

## 8. Cache Guard Manager Source Traversal

**Problem**: `get_guard_manager_from_source()` traverses complex source hierarchies repeatedly.

**Files**:
- `torch/_dynamo/guards.py:1296-1772` - Source traversal for guard managers
- `torch/_dynamo/source.py` - Source `.name` property recomputation

**Changes**:
```python
# Before
def get_guard_manager_from_source(self, source):
    # Complex traversal every time
    ...

# After
@functools.lru_cache(maxsize=256)
def _get_guard_manager_cached(self, source_name):
    # Cache based on source name
    ...

def get_guard_manager_from_source(self, source):
    return self._get_guard_manager_cached(source.name())
```

- [x] Add `@functools.lru_cache` to guard manager lookup
- [x] Cache `.name` property on Source objects as instance attribute
- [x] Pre-populate `_cached_guard_managers` dict during initialization

**Test**:
```python
import torch

def test_guard_manager_caching():
    """Verify guard manager lookups are cached."""
    @torch.compile
    def f(x, y):
        return x.t() @ y

    # Multiple tensor attributes should reuse cached managers
    f(torch.randn(3, 3), torch.randn(3, 3))

    # Check cache hit statistics
    from torch._dynamo.guards import GuardBuilder
    # Inspect _cached_guard_managers size
```

---

## 9. Optimize Dict Filtering with In-Place Mutation

**Problem**: Dict comprehensions create new dicts even when filtering removes few items.

**Files**:
- `torch/_dynamo/side_effects.py:657-662` - Dict filtering in `prune_dead_variables()`

**Changes**:
```python
# Before - creates new dict
self.id_to_variable = {k: v for k, v in self.id_to_variable.items() if is_live(v)}

# After - in-place removal for small number of dead items
dead_keys = [k for k, v in self.id_to_variable.items() if not is_live(v)]
for k in dead_keys:
    del self.id_to_variable[k]
```

- [x] Use in-place deletion when filtering removes minority of items
- [x] Apply same pattern to `store_attr_mutations` filtering
- [x] Consider using WeakValueDictionary for automatic cleanup

**Test**:
```python
import torch

def test_dict_inplace_filtering():
    """Verify dict filtering uses in-place mutation when efficient."""
    @torch.compile
    def f(x):
        y = x + 1
        del y  # Create dead variable
        return x * 2

    f(torch.randn(10))
    # Memory profile should show reduced dict allocations
```

---

## 10. Defer Compile-Time Instrumentation

**Problem**: Timing instrumentation (`dynamo_timed()`) has overhead from context manager creation.

**Files**:
- `torch/_dynamo/utils.py:692-835` - `dynamo_timed()` wrapper
- Multiple call sites creating context managers

**Changes**:
```python
# Before - always creates context managers
def dynamo_timed(key, ...):
    cx_mgrs = [compile_time_record_function(...)]
    if log_waitcounter:
        cx_mgrs.append(_WaitCounter(...))
    ...

# After - conditional instrumentation
def dynamo_timed(key, ...):
    if not _instrumentation_enabled():
        return nullcontext()
    # Only create managers when needed
    ...
```

- [x] Add fast path when instrumentation is disabled
- [x] Pre-compute event names instead of string concatenation
- [x] Use `__slots__` on timing context classes to reduce allocation

**Test**:
```python
import torch

def test_instrumentation_overhead():
    """Verify instrumentation has minimal overhead when disabled."""
    import torch._dynamo.config as config

    # Compile with instrumentation disabled
    with config.patch({"log_level": 0}):
        @torch.compile
        def f(x):
            return x + 1

        import time
        start = time.perf_counter()
        f(torch.randn(10))
        elapsed = time.perf_counter() - start
        print(f"Compile time (no instrumentation): {elapsed:.3f}s")
```

---

## 11. Cache Node Storage Lookups

**Problem**: `get_node_storage()` called multiple times for same node in reinplace logic.

**Files**:
- `torch/_inductor/fx_passes/reinplace.py:514-525` - Triple storage lookup

**Changes**:
```python
# Before
if get_node_storage(mutated_arg) is None:
    return False
shared_view_nodes = storage_to_nodes[get_node_storage(mutated_arg)]

# After
storage = get_node_storage(mutated_arg)
if storage is None:
    return False
shared_view_nodes = storage_to_nodes[storage]
```

- [x] Cache `get_node_storage()` result in local variable
- [x] Apply same pattern throughout `can_inplace()` function
- [x] Consider adding `@functools.lru_cache` to `get_node_storage()`

**Test**:
```python
import torch

def test_node_storage_caching():
    """Verify node storage lookups are cached."""
    @torch.compile
    def f(x):
        y = x.view(-1)
        y.add_(1)  # In-place op triggers storage lookups
        return y

    f(torch.randn(2, 3))
    # Profile should show single get_node_storage call per node
```

---

## 12. Optimize Graph Module Printing

**Problem**: `gm.print_readable()` called multiple times for debugging and cache keys.

**Files**:
- `torch/_inductor/compile_fx.py:1334-1349` - Multiple print_readable calls

**Changes**:
```python
# Before - print multiple times with different options
debug_str = gm.print_readable(print_output=False)
cache_key_str = gm.print_readable(print_output=False, fast_sympy_print=True)
log_str = gm.print_readable(print_output=False, include_stride=True)

# After - single print with caching
@functools.lru_cache(maxsize=1)
def _get_readable_str(gm_id):
    return gm.print_readable(print_output=False, include_stride=True, fast_sympy_print=True)

readable_str = _get_readable_str(id(gm))
```

- [x] Cache graph module string representation
- [x] Use compact binary format for cache keys instead of string
- [x] Lazy-generate debug strings only when logging enabled

**Test**:
```python
import torch

def test_graph_print_caching():
    """Verify graph printing is cached."""
    @torch.compile
    def f(x):
        return x.sin().cos().tan()

    # Check that print_readable called only once per graph
    import torch._inductor.compile_fx as compile_fx
    original_print = torch.fx.GraphModule.print_readable
    call_count = [0]

    def counting_print(*args, **kwargs):
        call_count[0] += 1
        return original_print(*args, **kwargs)

    torch.fx.GraphModule.print_readable = counting_print
    try:
        f(torch.randn(10))
        print(f"print_readable called {call_count[0]} times")
    finally:
        torch.fx.GraphModule.print_readable = original_print
```

---

## Priority Order

| Priority | Item | Impact | Effort | Risk |
|----------|------|--------|--------|------|
| P0 | 1. Pre-compile Regex | High | Low | Low |
| P0 | 3. Cache Dict Keys | High | Low | Low |
| P0 | 11. Cache Node Storage | High | Low | Low |
| P1 | 2. String Concatenation | High | Medium | Low |
| P1 | 7. SymPy Caching | High | Medium | Medium |
| P1 | 8. Guard Manager Cache | High | Medium | Low |
| P2 | 4. SideEffects Clone | Medium | Medium | Medium |
| P2 | 6. Combine Graph Passes | Medium | High | Medium |
| P2 | 9. Dict In-Place Filter | Medium | Low | Low |
| P3 | 5. Lazy Polyfills | Medium | Medium | High |
| P3 | 10. Instrumentation | Low | Medium | Low |
| P3 | 12. Graph Print Cache | Low | Low | Low |

---

## Measurement

Before and after each change, measure cold compile time:

```python
import torch
import time

def measure_cold_compile():
    torch._dynamo.reset()

    @torch.compile
    def f(x):
        return x.sin().cos().tan().exp().log()

    x = torch.randn(1024, 1024, device='cuda')

    start = time.perf_counter()
    f(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed

# Run multiple times, take median
times = [measure_cold_compile() for _ in range(5)]
print(f"Cold compile time: {sorted(times)[2]:.3f}s")
```

Target: **10-20% reduction** in cold compile time for typical models.
