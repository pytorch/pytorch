# PR Review Checklist

This checklist covers areas that CI cannot check. Skip items related to linting, formatting, type checking, and import ordering.

## Code Quality

### Abstractions and Design

- [ ] **Clear abstractions** - State management is explicit; no dynamic attribute setting/getting
- [ ] **No side-channel communication** - If behavior changes based on a hidden flag or dynamically-set attribute, the interface itself should change instead (different function signature, different class, different code path). Side-channel patterns (set a private flag in one place, check it in another via `getattr`) create undocumented behavioral modes
- [ ] **Proper interface, not on/off flags** - A private boolean that switches between two fundamentally different behaviors should be two separate code paths or a proper interface change, not a flag
- [ ] **Interface documentation** - New internal calling conventions, protocols, or contracts between components must have concrete documentation: what the caller provides, what the callee receives, what invariants hold, and cleanup responsibilities. Motivational comments ("this allows X") are not interface documentation
- [ ] **Match existing patterns in the same file** - Before accepting new code in a file, read how similar features are already implemented in that same file. If the file uses class attributes for boolean flags, new boolean flags must use class attributes. If the file uses a specific setter pattern, new setters must use the same pattern
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

### Public API Documentation

- [ ] **`__all__` requires docs** - Any callable added to a module's `__all__` must have a corresponding entry in the module's `.rst`/`.md` doc file (typically in an `autosummary` block). CI enforces this via `docs/source/conf.py`'s `coverage_post_process`
- [ ] **Never add to coverage ignore lists** - `coverage_ignore_functions` and `coverage_ignore_classes` in `docs/source/conf.py` are legacy allowlists. Never add new entries. Instead, either properly document the API or remove it from `__all__`

### Code Clarity

- [ ] **Self-explanatory code** - Variable and function names convey intent; minimal comments needed
- [ ] **Useful comments only** - Comments explain non-obvious context that cannot be inferred locally. For large comment use the `# Note [Good note title]` and `See Note [Good note title]` to write larger comments that can be referenced from multiple places in the codebase.
- [ ] **No backward-compatibility hacks** - Unused code is deleted completely, not renamed with underscores or marked with "removed" comments
- [ ] **Appropriate complexity** - Solutions are as simple as possible for the current requirements
- [ ] **Documentation shows correct patterns only** - Docs and markdown files should show the right way to do things directly, not anti-patterns followed by corrections. Code examples must have correct indentation, names, and syntax

### Initialization and Module Design

- [ ] **No fragile init ordering** - If multiple imports/calls must happen in a specific undocumented order, flag the design. Dependencies should be explicit or combined into a single entry point
- [ ] **Idempotent global state** - Registries and global lists that accumulate entries must handle multiple calls safely (no duplicate registration, clear cleanup story)

## PyTorch Infrastructure

When a PR touches code in the scope of any item below, **stop and investigate** whether the established infrastructure should be used.

### C++ Kernel Infrastructure

- [ ] **TensorIterator** — PR adds or modifies C++ kernel code that iterates over tensor data (raw pointers, `at::parallel_for`, manual contiguity checks, manual output reshape/resize)
- [ ] **DispatchStub** — PR adds C++ kernel code with manual `if (device_type == kCPU) ... else if (device_type == kCUDA)` dispatch instead of using `DECLARE_DISPATCH` / `DEFINE_DISPATCH` / `REGISTER_DISPATCH` from `aten/src/ATen/native/DispatchStub.h`
- [ ] **Structured Kernels** — PR adds a new ATen operator with separate hand-written functional, inplace, and out= variants instead of using `structured: True` + `structured_delegate` in `native_functions.yaml` to generate boilerplate
- [ ] **TORCH_CHECK variants** — PR uses generic `TORCH_CHECK` for conditions that have a more specific variant: `ValueError` → `TORCH_CHECK_VALUE`, `IndexError` → `TORCH_CHECK_INDEX`, `TypeError` → `TORCH_CHECK_TYPE`, `NotImplementedError` → `TORCH_CHECK_NOT_IMPLEMENTED`
- [ ] **AT_DISPATCH macros** — PR manually switches on `dtype` with `if (dtype == kFloat) ... else if (dtype == kDouble)` instead of using `AT_DISPATCH_FLOATING_TYPES`, `AT_DISPATCH_ALL_TYPES_AND`, or the `AT_DISPATCH_SWITCH` / `AT_DISPATCH_CASE` pattern from `aten/src/ATen/Dispatch.h`
- [ ] **Device guards (RAII)** — PR manually saves/restores device context (`cudaSetDevice` + try/catch) instead of using `DeviceGuard` or `OptionalDeviceGuard` from `c10/core/DeviceGuard.h`. **Note:** Operators registered in `native_functions.yaml` get automatic `DeviceGuard` insertion from codegen (controlled by `device_guard: True`, the default) — do NOT flag missing device guards for these ops unless they explicitly set `device_guard: False`
- [ ] **Memory format propagation** — PR allocates output tensors with `at::empty(shape, options)` (defaulting to contiguous) without calling `input.suggest_memory_format()` to preserve ChannelsLast or other input formats
- [ ] **Subclass-safe tensor allocation** — PR uses `at::empty(shape, input.options())` instead of `input.new_empty(shape)` or `at::empty_like(input)`, which don't propagate tensor subclass metadata
- [ ] **TORCH_LIBRARY operator registration** — PR registers operators using manual dispatcher calls instead of `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` macros from `torch/library.h`
- [ ] **TORCH_WARN_DEPRECATION** — PR uses `TORCH_WARN` for deprecation notices instead of `TORCH_WARN_DEPRECATION` which issues a proper `DeprecationWarning`

### CUDA & Device Management

- [ ] **C10_CUDA_CHECK** — PR calls raw CUDA APIs (`cudaMalloc`, `cudaMemcpy`, etc.) without wrapping in `C10_CUDA_CHECK()` from `c10/cuda/CUDAException.h`
- [ ] **C10_CUDA_KERNEL_LAUNCH_CHECK** — PR launches CUDA kernels with `<<<>>>` syntax but doesn't follow with `C10_CUDA_KERNEL_LAUNCH_CHECK()` immediately after to detect launch errors early
- [ ] **CUDAStreamGuard** — PR manually manages CUDA streams (`cudaStreamCreate`/`cudaStreamDestroy`) instead of using `CUDAStreamGuard` or getting streams from `at::cuda::getCurrentCUDAStream()` / `getStreamFromPool()`
- [ ] **CUDAEvent synchronization** — PR uses `cudaDeviceSynchronize()` or `cudaStreamSynchronize()` for cross-stream ordering instead of `CUDAEvent::record()` + `CUDAEvent::block()` which avoids unnecessary full synchronization
- [ ] **recordStream for allocator** — PR uses tensors on non-default CUDA streams without calling `c10::cuda::CUDACachingAllocator::recordStream()` to prevent premature memory reuse
- [ ] **CUDA graph compatibility** — PR adds host-GPU synchronization, unpinned memory transfers, or other graph-unsafe operations without checking `currentStreamCaptureStatusMayInitCtx()` to detect CUDA graph capture mode
- [ ] **AcceleratorHooksInterface** — PR adds device-specific `#ifdef USE_CUDA` blocks in generic code instead of using `AcceleratorHooksInterface` from `aten/src/ATen/detail/AcceleratorHooksInterface.h` for device-agnostic behavior
- [ ] **DeviceGuardImplInterface** — PR implements custom device management without going through `DeviceGuardImplInterface` from `c10/core/impl/DeviceGuardImplInterface.h`, bypassing the standard device abstraction layer

### Operator Registration & Codegen

- [ ] **native_functions.yaml** — PR adds a new ATen operator by writing manual C++ bindings and Python wrappers instead of declaring it in `aten/src/ATen/native/native_functions.yaml` and letting codegen produce the boilerplate
- [ ] **Operator tags** — PR adds an operator to `native_functions.yaml` without appropriate tags from `tags.yaml` (e.g., `pointwise`, `reduction`, `view_copy`, `core`, `pt2_compliant_tag`)
- [ ] **Missing Composite fallback** — PR adds a new operator to `native_functions.yaml` with only backend-specific dispatch keys (e.g., `CPU`, `CUDA`) but no `CompositeImplicitAutograd` or `CompositeExplicitAutograd` fallback. Without a Composite entry, the op will fail on all backends that don't have an explicit registration (XLA, MPS, HPU, PrivateUse1, etc.). Every new op should either have a Composite implementation or a clear justification for why it can only work on specific backends
- [ ] **Meta function registration** — PR adds a new operator without a meta (shape-only) implementation, blocking `torch.compile` and `torch.export`. Meta implementations can be registered in Python via `@register_meta` from `torch/_meta_registrations.py`, via `torch.library.impl(..., "Meta")`, or in C++ as a structured kernel with a `meta` dispatch key or via any `Composite` dispatch key in `native_functions.yaml` (since Composite kernels automatically work on Meta tensors)
- [ ] **Fake tensor implementation** — PR adds a custom op (registered via `torch.library`) without a fake implementation. Custom ops need `@register_fake` / `my_op.register_fake()` for `torch.compile` to trace through the op. For C++ ops registered via `native_functions.yaml`, the meta kernel serves this purpose. For Python `torch.library` custom ops, use `@torch.library.register_fake("mylib::my_op")` or `@my_op.register_fake` to provide a shape/dtype-only implementation. The fake impl receives `FakeImplCtx` with `ctx.new_dynamic_size()` for data-dependent output shapes
- [ ] **Schema annotations** — PR defines operator schemas without proper alias annotations (`Tensor(a)`, `Tensor(a!)`) for view and in-place ops, which breaks functionalization and autograd's alias tracking

### Autograd

- [ ] **derivatives.yaml** — PR writes a custom `autograd.Function` subclass for an operation that should have its backward formula registered in `tools/autograd/derivatives.yaml` (the centralized backward formula registry for ATen ops)
- [ ] **setup_context pattern** — PR writes `autograd.Function` with `forward(ctx, ...)` (legacy pattern) instead of separated `forward(...)` + `setup_context(ctx, inputs, output)` which is required for functorch compatibility (vmap, grad)
- [ ] **ctx.save_for_backward** — PR saves tensors in `autograd.Function` via `ctx.my_tensor = tensor` instead of `ctx.save_for_backward(tensor)`, causing memory leaks by keeping tensors alive longer than needed
- [ ] **gradcheck testing** — PR adds custom backward logic but doesn't test it with `torch.autograd.gradcheck()` / `gradgradcheck()` which verify numerical correctness of gradients via finite differences
- [ ] **Forward-mode AD** — PR adds a new differentiable op with backward formula in `derivatives.yaml` but doesn't add a `result:` entry for forward-mode AD (JVP). Can often use `auto_element_wise` or `auto_linear` for automatic generation
- [ ] **register_autograd for custom ops** — PR writes a full `autograd.Function` subclass for a custom op registered via `torch.library` instead of using the simpler `@my_op.register_autograd(backward, setup_context=...)` API
- [ ] **Vmap rule for custom ops** — PR adds a custom op or `autograd.Function` without a vmap rule (`generate_vmap_rule = True` or manual `vmap()` static method), breaking `torch.vmap` support

### Python Utilities

- [ ] **__torch_function__ support** — PR adds a new Python-level function that takes tensors but doesn't check `has_torch_function()` / call `handle_torch_function()`, breaking tensor subclass dispatch
- [ ] **Pytree registration** — PR manually flattens/unflattens custom container types (dataclasses, named tuples) instead of registering them with `torch.utils._pytree.register_pytree_node()` or `register_dataclass()`
- [ ] **tree_map** — PR manually walks nested structures of tensors with recursive functions instead of using `torch.utils._pytree.tree_map()`
- [ ] **_DecoratorContextManager** — PR implements a context manager that should also work as a decorator but doesn't inherit from `torch.utils._contextlib._DecoratorContextManager`
- [ ] **Deprecation utilities** — PR deprecates a function using ad-hoc `warnings.warn()` calls instead of PyTorch's deprecation infrastructure (`lazy_deprecated_import` for module-level, `TORCH_WARN_DEPRECATION` for C++)
- [ ] **No print statements** — PR adds `print()` calls for debugging or diagnostics. Use `torch._logging` utilities instead (`getArtifactLogger`, `LazyString`, `warning_once`). For the `torch.compile` stack specifically, use `trace_structured()` for structured artifacts that integrate with `tlparse` for production debugging. No bare `print()` should ever land in production code
- [ ] **torch.backends context** — PR manually saves/restores backend flags (`cudnn.deterministic`, etc.) instead of using the `torch.backends.cudnn.flags()` context manager

### nn Module Patterns

- [ ] **ModuleList / ModuleDict** — PR stores submodules in plain Python `list` or `dict` instead of `nn.ModuleList` or `nn.ModuleDict`, causing them to be invisible to `parameters()`, `to()`, `state_dict()`, etc.
- [ ] **nn.init methods** — PR manually initializes weights with `self.weight.data.normal_(0, 0.01)` instead of using `torch.nn.init.kaiming_uniform_()`, `xavier_uniform_()`, etc., which handle fan-in/fan-out calculations correctly
- [ ] **Parametrization framework** — PR implements custom weight reparameterization via forward pre-hooks (the deprecated pattern) instead of using `torch.nn.utils.parametrize.register_parametrization()`
- [ ] **_load_from_state_dict versioning** — PR changes a module's parameter layout without implementing `_load_from_state_dict()` for backward-compatible loading of old checkpoints (see BatchNorm's `_version = 2` pattern)
- [ ] **clip_grad_norm_** — PR manually computes gradient norms and clips in training loops instead of using `torch.nn.utils.clip_grad_norm_()` or `clip_grad_value_()`
- [ ] **LazyModule pattern** — PR implements deferred parameter initialization with manual shape inference in `forward()` instead of using `LazyModuleMixin` with `UninitializedParameter`

### Dynamo / Inductor / Compile

- [ ] **@register_lowering** — PR adds Inductor support for an op by modifying core lowering code instead of using `@register_lowering(aten.my_op)` from `torch/_inductor/lowering.py` with automatic type promotion and broadcasting
- [ ] **Inductor decompositions** — PR writes a full Inductor lowering for a complex op that can be decomposed into simpler already-lowered ops via `@register_decomposition` in `torch/_inductor/decomposition.py`
- [ ] **CustomGraphPass** — PR writes ad-hoc FX graph iteration for Inductor optimization instead of implementing `CustomGraphPass` (with `__call__` and `uuid()`) from `torch/_inductor/custom_graph_pass.py`
- [ ] **config.patch** — PR manually saves/restores Dynamo or Inductor config values in tests instead of using `torch._dynamo.config.patch()` as a decorator or context manager
- [ ] **Graph break hints** — PR calls `unimplemented()` in Dynamo without providing `gb_type`, `explanation`, or `hints` (like `SUPPORTABLE`, `FUNDAMENTAL`), making it hard for users to understand and fix graph breaks
- [ ] **Dynamo trace rules** — PR adds manual skip/inline logic in Dynamo variable tracking instead of updating `manual_torch_name_rule_map`, `MOD_INLINELIST`, or `MOD_SKIPLIST` in `torch/_dynamo/trace_rules.py`
- [ ] **torch.compile compatibility** — PR adds a new op or modifies an existing one without verifying it works under `torch.compile` (should test with `pt2_compliant_tag` and run opcheck)

### FX / Export

- [ ] **FX PassBase** — PR writes a custom FX graph transformation with manual graph walking instead of inheriting from `PassBase` (with `requires()`, `call()`, `ensures()`) from `torch/fx/passes/infra/pass_base.py`
- [ ] **FX PassManager** — PR manually orders and applies multiple FX passes instead of using `PassManager` with `this_before_that_pass_constraint` from `torch/fx/passes/infra/pass_manager.py`
- [ ] **FX Interpreter** — PR manually iterates FX graph nodes and tracks values in a dict instead of subclassing `torch.fx.Interpreter` which provides structured `run_node()` / `call_function()` / `call_module()` overrides
- [ ] **Subgraph rewriter** — PR manually matches and replaces graph patterns instead of using `replace_pattern()` from `torch/fx/subgraph_rewriter.py`
- [ ] **ShapeProp** — PR manually executes FX graphs to annotate shapes on nodes instead of using `ShapeProp(gm).propagate(*args)` from `torch/fx/passes/shape_prop.py`
- [ ] **torch.export dynamic shapes** — PR hard-codes tensor shapes in export constraints instead of using `Dim(name, min, max)` and `dims()` from `torch/export/dynamic_shapes.py`
- [ ] **make_fx** — PR manually initializes `torch.fx.Tracer` for proxy-based tracing instead of using `make_fx(f, tracing_mode="symbolic")` from `torch/fx/experimental/proxy_tensor.py`

### Type Promotion & Dtypes

- [ ] **elementwise_dtypes / TensorIterator** — PR manually implements type promotion logic for elementwise ops. In Python, use `elementwise_dtypes()` from `torch/_prims_common/` with the appropriate `ELEMENTWISE_TYPE_PROMOTION_KIND`. In C++, use `TensorIteratorConfig` which handles type promotion automatically: call `.promote_inputs_to_common_dtype(true)` and `.cast_common_dtype_to_outputs(true)` on the config builder, then `TensorIterator` computes `common_dtype()` for the kernel and handles all input/output casting. Kernels should operate on `iter.common_dtype()` via `AT_DISPATCH` rather than manually checking and promoting dtypes
- [ ] **result_type** — PR manually resolves output dtype from mixed-dtype inputs instead of using `torch.result_type()` (Python) or `at::result_type()` / `update_result_type_state()` (C++)
- [ ] **Complex dtype handling** — PR manually maps between complex and real dtypes (e.g., `complex64` to `float32`) instead of using `corresponding_real_dtype()` / `corresponding_complex_dtype()` from `torch/_prims_common/`
- [ ] **promoteTypes** — PR writes manual dtype promotion tables instead of using `c10::promoteTypes(a, b)` from `c10/core/ScalarType.h`

### Serialization

- [ ] **weights_only=False** — PR adds `torch.load(..., weights_only=False)`, explicitly opting out of safe deserialization. `weights_only=True` is already the default; setting it to `False` enables arbitrary code execution via pickle and is almost never the right thing to do. Flag this and ask the author to register safe globals via `torch.serialization.add_safe_globals()` instead
- [ ] **safe_globals** — PR adds new types to serialization that should be loadable with `weights_only=True` but doesn't register them via `torch.serialization.add_safe_globals()`
- [ ] **skip_data context** — PR implements metadata-only checkpoint inspection by reading full tensors instead of using `torch.serialization.skip_data()` context manager

### Distributed

- [ ] **DeviceMesh** — PR manually creates multiple `ProcessGroup`s for multi-dimensional parallelism (TP + DP) instead of using `DeviceMesh` from `torch/distributed/device_mesh.py` which manages this automatically
- [ ] **Distributed testing** — PR spawns multiple real processes for distributed unit tests instead of using `MultiThreadedPG` from `torch/testing/_internal/distributed/multi_threaded_pg.py` for single-process testing

### Tensor Subclasses

- [ ] **_make_wrapper_subclass** — PR creates tensor subclasses by calling `torch.Tensor.__new__()` directly instead of using `torch.Tensor._make_wrapper_subclass()` which properly sets up the subclass wrapper
- [ ] **__tensor_flatten__ / __tensor_unflatten__** — PR adds a tensor subclass without implementing `__tensor_flatten__()` and `__tensor_unflatten__()`, breaking serialization and `torch.compile` support

### Miscellaneous

- [ ] **torch._check** — PR uses `assert` or `if not cond: raise` in Python op implementations instead of `torch._check()` / `torch._check_is_size()` which work correctly with meta tensors and symbolic shapes
- [ ] **C++ extension building** — PR uses raw `setuptools` or `distutils` for building C++ extensions instead of `torch.utils.cpp_extension.CppExtension` / `CUDAExtension` / `load_inline()` which handle compiler flags, ABI, and includes
- [ ] **register_package for custom devices** — PR adds custom device serialization handling by monkey-patching `torch.save`/`torch.load` instead of using `torch.serialization.register_package()` to register location tag and restore functions
- [ ] **@register_backend** — PR adds a `torch.compile` backend by manually modifying internal dispatch tables instead of using `@torch._dynamo.backends.registry.register_backend(name=...)`

## Testing

### Test Existence

- [ ] **Tests exist** - New functionality has corresponding tests
- [ ] **Tests are in the right place** - Tests should be added to an existing test file next to other related tests
- [ ] **New test file is rare** - New test file should only be added when new major features are added

### Test Patterns

- [ ] **Proper module ownership** - Test files must have a real `# Owner(s): ["module: ..."]` label, not `"module: unknown"`. The author should create a new module label if needed and add themselves as owner
- [ ] **Use OpInfo** - Any testing for an operator or a cross-cutting feature must be done via OpInfo. Flag manual tests (e.g., `assertEqual(a + b, expected)`) for operators that already have OpInfo entries — these are redundant and will rot. When a PR adds dtype/device support to an operator, the testing should come from existing OpInfo infrastructure automatically (e.g., by adding the dtype to the operator's OpInfo `dtypes`), not from new manual tests. Likewise, a test checking a specific behavior for a single operator should not be a standalone test — the OpInfo infrastructure for that test category should be updated to cover the behavior across all applicable operators
- [ ] **Use ModuleInfo** - Manual forward/backward tests for `nn.Module` subclasses should use `ModuleInfo` from `torch/testing/_internal/common_modules.py` and the `@modules` decorator instead of hand-written per-module tests
- [ ] **Use TestCase** - Tests inherit from `torch.testing._internal.common_utils.TestCase`
- [ ] **Use run_tests** - Test file ends with `if __name__ == "__main__": run_tests()`
- [ ] **Use assertEqual for tensors** - Tensor comparisons use `assertEqual`, not raw assertions or `torch.allclose`
- [ ] **Device generic** - Any test checking compute result should happen in a device-generic test class (taking device as an argument) via `instantiate_device_type_tests`. Device-specific tests should be very rare and in device-specific test files
- [ ] **Use @dtypes** - PR writes separate test methods per dtype or manual `for dtype in [...]` loops instead of using the `@dtypes(...)` decorator from `common_device_type.py`
- [ ] **Use @parametrize** - PR duplicates test methods that differ only in a parameter instead of using `@parametrize` from `common_utils.py`
- [ ] **Use @ops for operator tests** - PR writes manual per-operator test iterations instead of using the `@ops(op_db)` decorator which automatically parametrizes tests over OpInfo entries
- [ ] **Use make_tensor** - PR creates test tensors with `torch.rand(shape)` (implicit CPU, implicit dtype) instead of `make_tensor(shape, device=device, dtype=dtype)` from `torch.testing` which enforces explicit device/dtype
- [ ] **Use common dtype groups** - PR manually lists dtypes like `[torch.float32, torch.float64]` instead of using helpers like `floating_types()`, `all_types_and_complex()`, etc. from `common_dtype.py`
- [ ] **Use toleranceOverride** - PR hard-codes tolerance values in individual assertions instead of using `@toleranceOverride` / `@precisionOverride` decorators which set per-dtype tolerances
- [ ] **Use DecorateInfo for OpInfo skips** - PR adds `@skipIf` conditionals inside OpInfo test methods instead of using `DecorateInfo` in the OpInfo's `skips` or `decorators` tuple
- [ ] **Use largeTensorTest** - PR manually checks free memory before large-tensor tests instead of using `@largeTensorTest("4 GB")` decorator from `common_device_type.py`
- [ ] **Descriptive test names** - Test method names describe what is being tested

### Test Quality

- [ ] **Edge cases covered** - Tests include boundary conditions, empty inputs, error cases
- [ ] **Error conditions tested** - Expected exceptions are tested with `assertRaisesRegex`, not bare `assertRaises`. `assertRaisesRegex` verifies both the exception type and message, catching cases where the right exception is raised for the wrong reason. Bare `assertRaises` should be flagged — always require a message pattern match
- [ ] **No duplicated test logic** - Similar tests share a private helper method called from individual tests with different configs
- [ ] **Use weakref for lifetime testing** - PR uses `sys.getrefcount()` to test whether objects are kept alive. Use `weakref.ref()` instead — create a weak reference, delete the strong references, then check if the weakref is dead (`wr() is None`). `sys.getrefcount` is a CPython implementation detail that varies across versions and is fragile

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
- [ ] **Decref-before-update hazard** - When replacing an item in a container (tuple, list, dict), update the container slot first, then `Py_DECREF` the old value. Decref can trigger `__del__` finalizers that re-enter and observe the container in an inconsistent state. Without the GIL (free-threaded builds), this is also a data race. This is **always** a must-fix — even if "safe in practice" because of refcount guarantees, the pattern is wrong and breaks under NoGIL. The correct pattern costs nothing extra

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

### Profiling & Benchmarking

- [ ] **Use torch.profiler** - PR adds manual `time.time()` instrumentation instead of using `torch.profiler.profile()` context manager with `schedule()` and `tensorboard_trace_handler()`
- [ ] **Use torch.utils.benchmark.Timer** - PR benchmarks with `time.time()` loops instead of `torch.utils.benchmark.Timer` which handles warmup, statistics, and proper CUDA synchronization
