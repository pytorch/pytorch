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
- [ ] **Useful comments only** - Comments explain non-obvious context that cannot be inferred locally
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
- [ ] **Tests are in the right place** - Test files match the convention (`test/test_*.py`)

### Test Patterns

- [ ] **Use TestCase** - Tests inherit from `torch.testing._internal.common_utils.TestCase`
- [ ] **Use run_tests** - Test file ends with `if __name__ == "__main__": run_tests()`
- [ ] **Use assertEqual for tensors** - Tensor comparisons use `assertEqual`, not raw assertions
- [ ] **Descriptive test names** - Test method names describe what is being tested

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

