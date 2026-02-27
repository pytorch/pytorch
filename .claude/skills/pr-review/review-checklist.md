# PR Review Checklist

This checklist covers areas that CI cannot check. Skip items related to linting, formatting, type checking, and import ordering.

## Code Quality

### Abstractions and Design

- [ ] **Clear abstractions** - State management is explicit; no dynamic attribute setting/getting
- [ ] **Match existing patterns** - Code follows architectural patterns already in the codebase
- [ ] **No over-engineering** - Only requested changes are made; no speculative features
- [ ] **No premature abstraction** - Helpers and utilities are only created when reused; three similar lines is better than a one-use helper
- [ ] **No trivial helpers** - Avoid 1-2 LOC helper functions used only once (unless significantly improves readability)

### API Design

When a PR introduces new API patterns, carefully evaluate the broader implications:

- [ ] **No flag-based internal access** - Reject patterns like `_internal=True` kwargs that gate internal functionality. These are confusing to reason about, impossible to document properly, and create BC headaches. Use a separate private function instead (e.g., `_my_internal_op()`)
- [ ] **Pattern already exists?** - Before accepting a new pattern, search the codebase to check if this pattern is already established. If not, the PR is introducing a new convention that needs stronger justification
- [ ] **Documentation implications** - Can this API be clearly documented? Flag-based access creates ambiguity about what is public vs private
- [ ] **BC implications going forward** - Will this pattern create future BC constraints?
- [ ] **Testing implications** - Does this pattern require awkward test patterns? Internal-only flags often lead to tests that use "forbidden" parameters
- [ ] **UX implications** - Is this pattern discoverable and understandable to users? Will it appear in autocomplete, type hints, or docs in confusing ways?

### Code Clarity

- [ ] **Self-explanatory code** - Variable and function names convey intent; minimal comments needed
- [ ] **Useful comments only** - Comments explain non-obvious context that cannot be inferred locally. For large comment use the `# Note [Good note title]` and `See Note [Good note title]` to write larger comments that can be referenced from multiple places in the codebase.
- [ ] **No backward-compatibility hacks** - Unused code is deleted completely, not renamed with underscores or marked with "removed" comments
- [ ] **Appropriate complexity** - Solutions are as simple as possible for the current requirements

### Common Issues to Flag

- Dynamic `setattr`/`getattr` for state management (prefer explicit class members)
- Unused imports, variables, or dead code paths
- Copy-pasted code that could be a shared helper
- Magic numbers without explanation
- Overly defensive error handling for impossible cases

## Testing

### Test Existence

- [ ] **Tests exist** - New functionality has corresponding tests
- [ ] **Tests are in the right place** - Tests should be added to an existing test file next to other related tests
- [ ] **New test file is rare** - New test file should only be added when new major features are added

### Test Patterns

- [ ] **Use OpInfo** - Any testing for an operator or a cross cutting feature must be done via OpInfo
- [ ] **Use TestCase** - Tests inherit from `torch.testing._internal.common_utils.TestCase`
- [ ] **Use run_tests** - Test file ends with `if __name__ == "__main__": run_tests()`
- [ ] **Use assertEqual for tensors** - Tensor comparisons use `assertEqual`, not raw assertions
- [ ] **Descriptive test names** - Test method names describe what is being tested
- [ ] **Device generic** - Any test checking compute result should happen in a Device-generic test class (taking device as an argument). Device-specific test should be very rare and in device-specific test files.

### Test Quality

- [ ] **Edge cases covered** - Tests include boundary conditions, empty inputs, error cases
- [ ] **Error conditions tested** - Expected exceptions are tested with `assertRaises` or `assertRaisesRegex`
- [ ] **No duplicated test logic** - Similar tests share a private helper method (e.g., `_test_foo(config)`) called from individual tests with different configs

**Example of good test structure:**
```python
def _test_feature_with_config(self, flag, expected_shape):
    """Shared test logic called by device-specific tests."""
    x = torch.randn(10)
    result = my_feature(x, flag)
    self.assertEqual(result.shape, expected_shape)

def test_feature_enabled(self):
    self._test_feature_with_config(True, (10, 10))

def test_feature_disabled(self):
    self._test_feature_with_config(False, (10, 5))
```

### Common Testing Issues

- Tests that only check the happy path without error cases
- Duplicated test code that should be a parameterized helper
- Tests that don't clean up resources (files, CUDA memory)
- Flaky tests (timing-dependent, order-dependent, golden value)
- Tests that skip without clear justification

## Security

### CI/CD and Workflow Security

When reviewing changes to workflows, build scripts, or CI configuration:

- [ ] **No secrets in workflow files** - PyTorch does not use repo secrets mechanism due to non-ephemeral runners; secrets can be compromised via reverse shell attacks
- [ ] **Ephemeral runners for sensitive jobs** - Binary builds, uploads, and merge actions must run on ephemeral runners only
- [ ] **No cache-dependent binaries in sensitive contexts** - sccache-backed builds are susceptible to cache corruption; these artifacts should not access sensitive info or be published for general use
- [ ] **Protected branch rules respected** - Changes to merge rules, release workflows, or deployment environments require extra scrutiny
- [ ] **Immutable artifact references** - Docker images use immutable tags; no overwriting of published artifacts

### PyTorch API Security

When reviewing changes to PyTorch APIs and user-facing code:

- [ ] **Model loading surfaces** - `torch.load` has a large attack surface; changes should not expand unsafe deserialization. Prefer safetensors for new serialization APIs
- [ ] **TorchScript security** - TorchScript models are executable code; introspection tools like `torch.utils.model_dump` can execute code from untrusted models and should not be used
- [ ] **Distributed primitives** - `torch.distributed`, RPC, and TCPStore have no auth/encryption and accept connections from anywhere; they are for internal networks only, not untrusted environments
- [ ] **No new pickle usage** - Avoid adding `pickle.load` or `torch.load` without `weights_only=True` on paths that could receive untrusted data

## Thread Safety & Concurrency

### Python Threading

- [ ] **No unprotected shared mutable state** - Shared data structures accessed from multiple threads are protected by locks or are inherently thread-safe
- [ ] **Lock ordering** - When multiple locks are acquired, ordering is consistent to avoid deadlocks
- [ ] **No GIL-reliant correctness** - Code that mutates shared state should not rely on the GIL for thread safety, since the GIL may not be present in free-threaded builds

### C++ Threading

- [ ] **No data races** - Shared mutable state is protected by mutexes or uses atomics with appropriate memory ordering
- [ ] **RAII lock guards** - Prefer `std::lock_guard` or `std::unique_lock` over manual `lock()`/`unlock()` to ensure exception-safe unlocking
- [ ] **No lock-order inversions** - When acquiring multiple locks, a consistent global ordering is followed
- [ ] **Correct atomic memory ordering** - `std::memory_order_relaxed` is only used when ordering with other operations is genuinely unnecessary; default to `seq_cst` or use `acquire`/`release` pairs

### CPython C API Thread Safety

This is particularly important for PyTorch's autograd, which has multi-threaded C++ code calling into the CPython C API.

- [ ] **GIL held for Python object access** - Any code that touches `PyObject*` (incref, decref, attribute access, container mutation) must hold the GIL. When releasing the GIL for long-running C++ work (`Py_BEGIN_ALLOW_THREADS`), verify no Python objects are accessed in that region
- [ ] **Borrowed references across GIL release** - Borrowed references (`PyTuple_GET_ITEM`, `PyList_GET_ITEM`) become unsafe if the GIL is released and reacquired, since another thread may have mutated the container
- [ ] **Decref-before-update hazard** - When replacing an item in a container (tuple, list, dict), update the container slot first, then `Py_DECREF` the old value. Decref can trigger `__del__` finalizers that re-enter and observe the container in an inconsistent state. Without the GIL (free-threaded builds), this is also a data race.

### Free-Threaded Python (NoGIL, PEP 703)

CPython 3.13t+ can run without the GIL. Code that was previously safe under the GIL may have races in free-threaded builds:

- [ ] **No implicit GIL serialization assumptions** - Code paths that assume only one thread can execute Python at a time are broken under NoGIL. Look for shared mutable state accessed from C extensions without explicit locking
- [ ] **Raw `PyTuple_SET_ITEM` / `PyList_SET_ITEM`** - These are raw slot writes with no memory ordering guarantees. In free-threaded builds, concurrent reads from other threads may see stale or torn values. Consider whether the data structure could be accessed concurrently and whether atomic operations or the thread-safe API alternatives are needed
- [ ] **Module-level mutable state in C extensions** - Global/static `PyObject*` variables or C-level caches accessed from multiple threads need synchronization in NoGIL builds

### PyTorch-Specific Concurrency

- [ ] **Autograd engine multi-threading** - The autograd engine runs node `apply()` methods from worker threads. Code in custom autograd node implementations must be safe for concurrent execution across different nodes, and must hold the GIL when accessing Python objects
- [ ] **CUDA stream synchronization** - Operations across different CUDA streams require explicit synchronization (`cudaStreamSynchronize`, `cudaEventRecord`/`cudaStreamWaitEvent`). Missing synchronization can cause silent data corruption
- [ ] **DataLoader worker safety** - Objects shared between the main process and DataLoader worker processes (or threads) must be fork-safe or use appropriate IPC mechanisms

## Performance

### Obvious Regressions

- [ ] **No unnecessary allocations** - Tensors are not repeatedly created in hot loops
- [ ] **Appropriate in-place operations** - Use in-place ops where possible in performance-critical paths
- [ ] **No Python loops over tensors** - Prefer vectorized operations over iterating tensor elements

### Device Handling

- [ ] **Device consistency** - Operations don't unexpectedly move tensors between devices
- [ ] **CUDA considerations** - CUDA-specific code handles synchronization appropriately
- [ ] **MPS compatibility** - Metal Performance Shaders are considered if applicable

### Memory Patterns

- [ ] **No memory leaks** - Temporary tensors are freed, no circular references
- [ ] **Efficient data structures** - Appropriate containers for access patterns
- [ ] **Gradient memory** - Proper use of `no_grad()`, `detach()` to avoid unnecessary graph retention

### Common Performance Issues

- Creating new tensors inside training loops instead of pre-allocating
- Synchronous CUDA operations where async would work
- Keeping computation graph alive longer than needed
- Redundant clones or copies
