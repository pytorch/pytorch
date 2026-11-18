# `torch/headeronly` review guide

`torch/headeronly` contains APIs that third-party extensions compile into their
own binaries without linking LibTorch. Prioritize correctness parity between the old and the migrated API, and additionally review changes for source compatibility,
symbol isolation, and independence from LibTorch.

For migrations from another namespace (e.g., c10:: or at::), the change should move an implementation into
`torch/headeronly` and leave the old API as a delegating compatibility shim.
Common failures include silently rewriting the implementation, changing the old
API's meaning, and testing through a target that links LibTorch.

## Compare migrations mechanically

Use the base and head established by the enclosing PR review. Do not infer a
different base from `main`. Compare the old and new implementations directly,
ignoring formatting noise:

```bash
diff -uw \
  <(git show <base>:<old/path>) \
  <(git show <head>:torch/headeronly/<new/path>)
```

Any change in implementation needs explicit justification in the PR comment and code
comments and should be tested by targeted coverage. When a semantic migration is not
possible, the code change should introduce a headeronly version of an API that shares
common implementation. In this case, there should be clear documentation on the
concrete differences between the two APIs.

In particular:
- Treat migrations as ideally byte-identical moves, not rewrites. Compare the implementations mechanically; any semantic change needs separate justification and targeted tests.
- Preserve the complete public surface, including overloads, supported types, signatures, and compile-time availability.
- Verify equivalent behavior across supported compilation modes, including CUDA host/device passes and ROCm versions. Also confirm that the new path participates in the required build, export, and hipify rules.


## Preserve the old API exactly

For every moved symbol, find callers of the old spelling and verify that the API still has the same implementation and meaning:

```bash
git grep -n '<symbol>' -- aten torch test
```

- The old header must still declare the symbol and delegate to the new
  implementation. Merely including the new header and relying on argument
  dependent lookup is not sufficient.
- Do not leave independent implementations in both the old and new headers. They
  should share as much functionality as possible, with the implementation in the new torch/headeronly file.
- Preserve the overload set. Combining multiple `enable_if` overloads into one
  overload with a disjunction can break explicit template arguments and
  function pointers even when ordinary calls still compile.
- Preserve namespace placement. A symbol that was intentionally at global
  scope, such as an unqualified CUDA or ROCm helper, must remain available at
  global scope.

Check which in-tree path now exercises each moved implementation. A migration
is incomplete when only one specialization delegates to the new header while
other copied paths have no in-tree caller.

## Isolate emitted symbols

Definitions in the `torch::headeronly` namespace must use the visibility macros
from `torch/headeronly/macros/Macros.h`:

```cpp
HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
// ...
HIDDEN_NAMESPACE_END(torch, headeronly)
```

These macros keep a header implementation local to the extension that compiled
it. Without hidden visibility, two extensions built against different PyTorch
versions can preempt each other's definitions at load time.

Do not add a bare `namespace torch::headeronly`, even for aliases or forward
declarations. If backward compatibility requires a definition at global scope,
keep it global and require a comment explaining why it cannot use the hidden
namespace.

## Keep public headers self-contained

Every include and macro in this directory reaches third-party translation
units.

- Follow the include closure and verify it remains independent of LibTorch.
- Prefix new public macros with `TORCH_HEADERONLY_` or an established project
  prefix such as `C10_`.
- `#undef` implementation-only macros after their last use. Public API macros
  may remain defined, but their names and behavior are part of the API.
- Define compatibility shims in one header and make legacy headers include it;
  do not maintain duplicate definitions.
- Check transitive dependencies before adding backend-specific includes.

## Verify the independent tests

New APIs must be listed in `torch/header_only_apis.txt` and exercised under `test/cpp/aoti_abi_check/`. These tests are not expected to cover every behavioral edge case. They should demonstrate that the public entry points and supported call forms remain accessible without linking LibTorch. Comprehensive correctness testing belongs in the API’s owning subsystem.

- Tests must call the `torch::headeronly` API, not the ATen or c10 compatibility
  shim.
- `test_aoti_abi_check` must not link `torch`, `torch_cpu`, `torch_cuda`, or
  `c10`. Runtime-only dependencies such as `gtest_main`, `sleef`, and
  `torch::cudart` are acceptable; verify any new dependency is likewise
  runtime-only.
- Confirm `tools/linter/adapters/header_only_linter.py`'s `CPP_TEST_GLOBS`
  matches the test's path and extension. A new subdirectory needs an explicit glob.
- For each device branch, verify that the test target compiles an architecture
  that selects it and that CI runs the test on a compatible GPU shard. Code
  removed during preprocessing, or never run on compatible hardware, is not
  covered.
- A CUDA-only CMake gate provides no ROCm coverage. Call this out when the
  migrated implementation has ROCm-specific behavior.
