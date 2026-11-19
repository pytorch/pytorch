# Reviewing test changes under `test/`

Tests should be decoupled from specific hardware as much as possible, so that
generic tests naturally work on all accelerators, including new accelerators
added in the future. Each accelerator runs only the tests it genuinely needs,
reducing duplicated execution and maximizing test reuse.

## When the rules apply

**Scenario 1: Test-case refactoring**

A refactoring of a test class may be split into multiple PRs, so the
following rules may be too strict; developers can explain in the PR
comments.

- Conditions:
  - All changed files are under `test/` or `torch/testing/`.
  - **And** the PR title and description do not indicate that it is not a
    test-case refactoring.
- Rules:
  - Rule 1: One responsibility per test class
  - Rule 2: Correct `hw_classification` on every test class
  - Rule 3: Requirements for `GENERIC` classes
  - Rule 4: Requirements for `ACCELERATOR` classes
  - Rule 5: Requirements for `CPU`/`CUDA`/`MPS`/`XPU` classes
  - Rule 6: Preserve test coverage
- Scope: the test classes the change touches

**Scenario 2: Other**

This is the everyday development case; to reduce impact on developers, apply
the rules only to files not listed in
`tools/linter/adapters/test_linter_allowlist.json`:

- Conditions:
  - The PR also changes files outside `test/` or `torch/testing/`.
  - **Or** the PR title or description indicates that it is not a test-case
    refactoring.
- Case 1: the changed file is in `tools/linter/adapters/test_linter_allowlist.json`
  - Rules: none (not yet migrated; skip checking)
- Case 2: the changed file is not in the `tools/linter/adapters/test_linter_allowlist.json`
  - Rules:
    - Rule 1: One responsibility per test class
    - Rule 2: Correct `hw_classification` on every test class
    - Rule 3: Requirements for `GENERIC` classes
    - Rule 4: Requirements for `ACCELERATOR` classes
    - Rule 5: Requirements for `CPU`/`CUDA`/`MPS`/`XPU` classes
    - Rule 6: Preserve test coverage
    - Rule 7: Requirements for `tools/linter/adapters/test_linter_allowlist.json`
  - Scope: the whole file, not just the modified parts

Notes:

- The rules referenced below are defined in the following sections of this
  document; apply the rules listed for a case to the files named in its
  Scope.
- The scenario is decided by the PR's changed files and its stated purpose
  in the title or description, as specified in the `Conditions` above.

## Rule 1: One responsibility per test class

A test class (any class that defines a `test*` method) focuses on exactly one
responsibility; a class with mixed responsibilities must be split, and every
test must live in the class that matches it.

A class's responsibility is one of three kinds:

- **Generic**: all tests in the class are independent of any accelerator
  device; all accelerators share the same logic, and running them once on
  a cpu-only machine guarantees their quality.
- **Accelerator-agnostic**: all tests in the class can run on multiple
  accelerators, and every accelerator should run them to ensure quality.
- **Accelerator-specific**: all tests in the class are only applicable to
  one particular device, e.g. `CUDA`, `MPS`, `XPU`, `CPU`. Running them on
  other accelerators will fail.

Notes:

- When splitting a class with mixed responsibilities, decide where each test
  belongs using both the test name and what the test actually verifies:
  logic shared by all accelerators (Generic), behavior every accelerator
  should run (Accelerator-agnostic), or features specific to one accelerator
  (Accelerator-specific).

## Rule 2: Correct `hw_classification` on every test class

Every single-responsibility test class (any class that defines a `test*`
method) must declare exactly one:

    hw_classification = HardwareClassification.<MEMBER>

Valid values of `<MEMBER>` are `GENERIC`, `ACCELERATOR`, `CPU`, `CUDA`,
`MPS`, `XPU`.

The `hw_classification` value must match the class's single responsibility;
responsibilities map to HardwareClassification values as follows:

- **Generic** <-> `GENERIC`
- **Accelerator-agnostic** <-> `ACCELERATOR`
- **Accelerator-specific** <-> `CPU`/`CUDA`/`MPS`/`XPU`

## Rule 3: Requirements for `GENERIC` classes

- Test methods take no `device`/`devices` parameter.
- Any content in this class should be device-independent.
  - No device-related decorators:
    - `@skipCPUIf`, `@skipCUDAIf`, `@skipXPUIf`, `@skipGPUIf`,
      `@skipMPSIf`, `@skipPRIVATEUSE1If`
    - `@onlyCPU`, `@onlyCUDA`, `@onlyMPS`, `@onlyXPU`, `@onlyOn`,
      `@onlyAccelerator`, `@onlyPRIVATEUSE1`, `@onlyNativeDeviceTypes`
    - `@dtypesIfCPU`, `@dtypesIfCUDA`, `@dtypesIfXPU`, `@dtypesIfMPS`,
      `@dtypesIfPRIVATEUSE1`
  - No accelerator APIs:
    - `torch.accelerator.*`
    - `torch.cuda.*`
    - `torch.mps.*`
    - `torch.xpu.*`
    - `torch.mtia.*`
- Test class form:
  - May be a plain class
  - May be instantiated via `@instantiate_parametrized_tests` (when the
    test class uses `@parametrize`/`@ops`/`@dtypes`)
  - Must never be instantiated via `instantiate_device_type_tests`

## Rule 4: Requirements for `ACCELERATOR` classes

- Every test method takes a `device` or `devices` parameter:
  - Must use that parameter for the test's device-dependent operations
    (e.g. creating Tensors on it, or mapping it to the corresponding
    distributed backend and so on).
  - Merely accepting the parameter is not enough.
- The test class must be generic across all accelerators:
  - No `@only*` decorators (except `@onlyAccelerator`, which is only for
    tests that genuinely cannot run on CPU, e.g. tests of CPU-accelerator
    interaction):
    - `@onlyCPU`
    - `@onlyCUDA`
    - `@onlyMPS`
    - `@onlyXPU`
    - `@onlyPRIVATEUSE1`
    - `@onlyNativeDeviceTypes`
    - `@onlyNativeDeviceTypesAnd`
  - No hardcoded devices; use the `device` or `devices` parameter instead:
    - `.cuda()`
    - `.to("cuda")`
    - `device="cuda"`
    - the same forms with other device names (`"mps"`, `"xpu"`, `"hpu"`)
  - No accelerator-specific APIs (except `torch.accelerator.*`,
    `torch.Event`, `torch.Stream`):
    - `torch.cuda.*`
    - `torch.mps.*`
    - `torch.mtia.*`
    - `torch.xpu.*`
  - Allowed device-specific skip decorators:
    - `@skipCPUIf` and its variants
    - `@skipCUDAIf` and its variants
    - `@skipXPUIf` and its variants
    - `@skipMPSIf` and its variants
    - `@skipIf` (only the accelerator-type-aware version from
      `common_device_type`)
  - Allowed dtype decorators:
    - `@dtypesIfCPU`
    - `@dtypesIfCUDA`
    - `@dtypesIfMPS`
    - `@dtypesIfXPU`
    - `@dtypes`
- Test class form:
  - Must be instantiated via `instantiate_device_type_tests` (which also
    expands `@parametrize`/`@ops`/`@dtypes`)
    - `only_for` must not be used
    - `except_for` may be used to exclude unsupported devices

## Rule 5: Requirements for `CPU`/`CUDA`/`MPS`/`XPU` classes

No additional requirements; keeping the existing form or migrating to
`instantiate_device_type_tests` instantiation are both acceptable.

## Rule 6: Preserve test coverage

Modifying tests must not reduce their existing coverage without an explicit
explanation. Any coverage reduction must be documented in the PR description
with the reason for the change; an undocumented reduction is a finding.

## Rule 7: Requirements for `tools/linter/adapters/test_linter_allowlist.json`

`tools/linter/adapters/test_linter_allowlist.json` is the list of test files
that do not yet meet these rules. A test-refactoring PR should delete the
corresponding entries from this file (advisory rather than mandatory,
because one file may be refactored across several PRs).

## Before approving

Be able to state:

1. which scenario the PR falls into;
2. what the single responsibility of each touched test class is and why its
   `hw_classification` matches;
3. that each touched test class meets the requirements of its classification
   (Rule 3/4/5);
4. whether and why `tools/linter/adapters/test_linter_allowlist.json`
   changed.
