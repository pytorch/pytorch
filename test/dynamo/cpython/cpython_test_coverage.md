# CPython Dynamo Coverage Plan

Status: current gate is G3.

Goal: improve `PYTORCH_TEST_WITH_DYNAMO=1` coverage for CPython tests by first
using one file as a tight sandbox:

```
test/dynamo/cpython/3_13/test_dict.py
```

The broader CPython Dynamo passrate is about 45%. Starting with one file keeps
the agent loop fast and lets us group work by root-cause category instead of
scattering effort across many files.

Operational loop instructions live in `test/dynamo/cpython/agent_manager.md`.
This file is the source of truth for active gate, scope, exit criteria, test
commands, and measured progress.

## Ground Rules

- Do not relax gate exit criteria during an implementation cycle.
- Do not mark a gate complete without measured evidence from the current tree.
- Do not edit vendored CPython tests under `test/dynamo/cpython/3_13` unless the
  human explicitly asks for a CPython import/update.
- Do not add new expected-failure or skip sentinels without human approval.
- Use `agent_space/` for scratch files, temporary reports, JUnit XML, and
  sentinel backups.
- For real Dynamo behavior fixes, add focused regression tests in the normal
  Dynamo test suite and remove only the CPython sentinels proven fixed.

Sentinel directories:

```
test/dynamo_expected_failures/
test/dynamo_skips/
```

Example key:

```
CPython313-test_dict-DictTest.test_update
```

## Current `test_dict.py` Baseline

Measured with:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest --collect-only -q \
  test/dynamo/cpython/3_13/test_dict.py
```

Current counts:

| File | Collected | Xfail sentinels | Skip sentinels |
| --- | ---: | ---: | ---: |
| `test_dict.py` | 112 | 26 | 7 |

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/object_protocol.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/symbolic_convert.py
```

Current skip sentinels:

```
DictTest.test_container_iterator
DictTest.test_dict_items_result_gc
DictTest.test_dict_items_result_gc_reversed
DictTest.test_free_after_iterating
DictTest.test_track_dynamic
DictTest.test_track_literals
DictTest.test_track_subtypes
```

## Measurement Commands

Run `test_dict.py`:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py
```

Run one test:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
```

Write JUnit XML:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  --junitxml=agent_space/cpython_dynamo/test_dict.xml \
  test/dynamo/cpython/3_13/test_dict.py
```

Count current `test_dict.py` expected failures and skips:

```
find test/dynamo_expected_failures -maxdepth 1 -type f \
  -name 'CPython313-test_dict-*' | wc -l
find test/dynamo_skips -maxdepth 1 -type f \
  -name 'CPython313-test_dict-*' | wc -l
```

Use Dynamo logs only for opaque single-test repros:

```
TORCH_LOGS="+dynamo,graph_breaks" TORCHDYNAMO_VERBOSE=1 \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
```

## Gates

### G0: Baseline and Histogram

Status: complete.

Purpose: make the first autonomous loop evidence-driven before source changes.

Exit criteria:

- Record git SHA, Python version, exact command, and JUnit XML under
  `agent_space/cpython_dynamo/`.
- Verify the current counts: 112 collected tests, 45 expected-failure sentinels,
  and 7 skip sentinels.
- Build a normalized failure histogram for all 45 expected failures.
- Record at least 10 actionable single-test repros in "Work Queue" with exact
  test names, normalized failure reason, and likely source file.
- Run at least one single-test repro with its sentinel temporarily bypassed and
  restored.
- Leave source files and sentinels unchanged.

### G1: Update, Fromkeys, and Merge

Status: complete.

Target sentinels:

```
DictTest.test_update
GeneralMappingTests.test_update
SubclassMappingTests.test_update
DictTest.test_fromkeys
DictTest.test_fromkeys_operator_modifying_dict_operand
DictTest.test_fromkeys_operator_modifying_set_operand
DictTest.test_merge_operator
DictTest.test_merge_and_mutate
```

Exit criteria:

- Remove at least 5 target sentinels.
- Add focused Dynamo regression tests for non-trivial semantic fixes.
- `test_dict.py` has no unexpected successes or new real failures.
- Remaining target sentinels have normalized failure reasons recorded.

### G2: Item Access and Mutating Methods

Status: complete.

Target sentinels:

```
CAPITest.test_getitem_knownhash
DictTest.test_getitem
DictTest.test_bad_key
DictTest.test_invalid_keyword_arguments
DictTest.test_setdefault_atomic
DictTest.test_setitem_atomic_at_resize
DictTest.test_resize2
```

Exit criteria:

- Remove at least 4 target sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `test_dict.py` has no unexpected successes or new real failures.
- Remaining target sentinels have normalized failure reasons recorded.

### G3: Split-Dict and Non-String Key Edges

Status: current.

Target sentinels:

```
DictTest.test_object_set_item_single_instance_non_str_key
DictTest.test_splittable_del
DictTest.test_splittable_pop
DictTest.test_splittable_popitem
DictTest.test_splittable_setdefault
DictTest.test_splittable_to_generic_combinedtable
DictTest.test_str_nonstr
```

Progress note: `DictTest.test_str_nonstr` was removed during the G2 cycle from
the same rich-comparison fix and is not counted as a new G3 removal. Current G3
cycle progress is three additional removals:
`DictTest.test_object_set_item_single_instance_non_str_key`,
`DictTest.test_splittable_pop`, and
`DictTest.test_splittable_to_generic_combinedtable`.

Exit criteria:

- Remove at least 4 target sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `test_dict.py` has no unexpected successes or new real failures.
- Remaining target sentinels are categorized as Dynamo semantic gaps or CPython
  implementation-detail boundaries.

### G4: Dict Views and Set-Like Operations

Status: pending.

Target sentinels:

```
DictTest.test_views_mapping
DictTest.test_items_symmetric_difference
DictTest.test_errors_in_view_containment_check
DictTest.test_dictitems_contains_use_after_free
```

Exit criteria:

- Remove at least 3 target sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `test_dict.py` has no unexpected successes or new real failures.
- Remaining target sentinels have normalized failure reasons recorded.

### G5: Side Effects and Reentrancy

Status: pending.

Target sentinels:

```
DictTest.test_mutating_lookup
DictTest.test_equal_operator_modifying_operand
DictTest.test_reentrant_insertion
DictTest.test_dict_contain_use_after_free
DictTest.test_init_use_after_free
DictTest.test_eq
```

Exit criteria:

- Remove at least 3 target sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `test_dict.py` has no unexpected successes or new real failures.
- Remaining target sentinels are categorized by side-effect mechanism.

### G6: Iterators, Reversed, and Reconstruction

Status: pending.

Target sentinels:

```
DictTest.test_iterator_pickling
DictTest.test_itemiterator_pickling
DictTest.test_valuesiterator_pickling
DictTest.test_reverseiterator_pickling
DictTest.test_reverseitemiterator_pickling
DictTest.test_reversevaluesiterator_pickling
DictTest.test_reversed
DictTest.test_oob_indexing_dictiter_iternextitem
```

Exit criteria:

- Remove at least 3 target sentinels, or prove with evidence that the remaining
  targets are CPython pickle/iterator implementation details.
- Add focused Dynamo regression tests for user-visible iterator or
  reconstruction fixes.
- `test_dict.py` has no unexpected successes or new real failures.
- Remaining target sentinels are categorized.

### G7: Repr, Copy, GC, and Boundary Triage

Status: pending.

Target sentinels:

```
DictTest.test_repr
DictTest.test_repr_deep
DictTest.test_copy_fuzz
DictTest.test_copy_maintains_tracking
DictTest.test_literal_constructor
```

Also triage current skip sentinels:

```
DictTest.test_container_iterator
DictTest.test_dict_items_result_gc
DictTest.test_dict_items_result_gc_reversed
DictTest.test_free_after_iterating
DictTest.test_track_dynamic
DictTest.test_track_literals
DictTest.test_track_subtypes
```

Exit criteria:

- Remove at least 2 target expected-failure sentinels, unless the baseline
  evidence shows fewer than 2 user-visible fixes remain in this category.
- Add focused Dynamo regression tests for user-visible fixes.
- Every remaining `test_dict.py` expected failure and skip is categorized as:
  fixed, stdlib/C boundary, CPython implementation detail, side-effect semantic
  gap, reconstruction gap, unsupported protocol, or untriaged.
- No `test_dict.py` sentinel remains untriaged.

### G8: `test_dict.py` Rebaseline and Next File

Status: pending.

Purpose: close the dict sandbox and choose the next file.

Exit criteria:

- Run `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short
  test/dynamo/cpython/3_13/test_dict.py`.
- Record pass/skip/fail/unexpected-success counts.
- Record final expected-failure and skip sentinel counts for `test_dict.py`.
- Compute sentinel removals by gate and total.
- Propose the next file and category gates under `Proposed Gate Changes Awaiting
  Human Approval`.

## Work Queue

G0 baseline evidence from current tree:

- Metadata: `agent_space/cpython_dynamo/metadata.json`
- Counts: `agent_space/cpython_dynamo/counts.json`
- Exact commands: `agent_space/cpython_dynamo/commands.txt`
- JUnit XML: `agent_space/cpython_dynamo/test_dict.xml`
- Histogram: `agent_space/cpython_dynamo/failure_histogram.md`
- Single-test repro outputs: `agent_space/cpython_dynamo/single_repros/`

Measured actionable repros:

1. `DictTest.test_update`: `torch._dynamo.exc.Unsupported: Unsupported method
   call; call_method ConstDictVariable() update [] {}`. Likely source:
   `torch/_dynamo/variables/dicts.py`.
2. `GeneralMappingTests.test_update`: `torch._dynamo.exc.Unsupported:
   Unsupported method call; call_method ConstDictVariable() update [] {}`.
   Likely source: `torch/_dynamo/variables/dicts.py`.
3. `SubclassMappingTests.test_update`: `torch._dynamo.exc.Unsupported:
   Unsupported method call; call_method ConstDictVariable() update [] {}`.
   Likely source: `torch/_dynamo/variables/dicts.py`.
4. `DictTest.test_merge_operator`: `torch._dynamo.exc.Unsupported: Length
   mismatch when unpacking object for UNPACK_SEQUENCE; expected length: 2,
   actual: 1`. Likely source: `torch/_dynamo/polyfills/__init__.py`.
5. `DictTest.test_object_set_item_single_instance_non_str_key`:
   `AssertionError: Expected str key, got <class 'int'>`. Likely source:
   `torch/_dynamo/variables/dicts.py`.
6. `DictTest.test_splittable_to_generic_combinedtable`: `AssertionError:
   Expected str key, got <class 'int'>`. Likely source:
   `torch/_dynamo/variables/dicts.py`.
7. `DictTest.test_getitem`: `torch._dynamo.exc.Unsupported: Read uninitialized
   cell`. Likely source: `torch/_dynamo/side_effects.py`.
8. `DictTest.test_mutating_lookup`: `torch._dynamo.exc.Unsupported: Read
   uninitialized cell`. Likely source: `torch/_dynamo/side_effects.py`.
9. `DictTest.test_reentrant_insertion`: `torch._dynamo.exc.Unsupported: Read
   uninitialized cell`. Likely source: `torch/_dynamo/side_effects.py`.
10. `DictTest.test_items_symmetric_difference`:
    `torch._dynamo.exc.InternalTorchDynamoError: IndentationError: expected an
    indented block after function definition on line N (<eval_with_key>.0, line
    N)`. Likely source: `torch/fx/graph_module.py`.
11. `DictTest.test_views_mapping`: `torch._dynamo.exc.Unsupported: builtin
    isinstance() cannot determine type of argument;
    isinstance(GetAttrVariable(DictKeysVariable(), mapping),
    UserDefinedClassVariable(<class 'mappingproxy'>))`. Likely source:
    `torch/_dynamo/variables/dicts.py`.
12. `DictTest.test_reversed`: `torch._dynamo.exc.Unsupported: Observed
    exception; raised exception AssertionError('StopIteration not raised by
    next')`. Likely source: `torch/_dynamo/symbolic_convert.py`.
13. `DictTest.test_repr`: `torch._dynamo.exc.InternalTorchDynamoError:
    RecursionError: maximum recursion depth exceeded`. Likely source:
    `torch/_dynamo/variables/dicts.py`.
14. `DictTest.test_splittable_del`: `torch._dynamo.exc.Unsupported: Attempted
    to call function marked as skipped; module: sys, qualname: getsizeof, skip
    reason: cannot determine source file for sys (likely a C extension or
    builtin)`. Likely source: `torch/_dynamo/trace_rules.py`.
15. `DictTest.test_iterator_pickling`: `torch._dynamo.exc.Unsupported:
    Attempted to call function marked as skipped; module: _pickle, qualname:
    dumps, skip reason: cannot determine source file for _pickle (likely a C
    extension or builtin)`. Likely source: `torch/_dynamo/trace_rules.py`.
16. `DictTest.test_fromkeys`: `skipped: test hangs`. Likely source:
    `test/dynamo/cpython/3_13/test_dict.py`.

## Failure Categories

Use these categories when closing each gate:

- fixed: sentinel removed with focused regression coverage.
- stdlib/C boundary: Dynamo does not trace an external C or stdlib function.
- CPython implementation detail: GC tracking, refcount, object layout, or
  pickle details that do not represent useful Dynamo coverage.
- side-effect semantic gap: mutation during comparison, lookup, iteration, or
  containment.
- reconstruction gap: value cannot be reconstructed after a graph break.
- unsupported protocol: missing method, slot, descriptor, or object-protocol
  dispatch.
- untriaged: still needs a single-test repro and normalized failure reason.

## Proposed Gate Changes Awaiting Human Approval

None.

## Cycle Log

- G0 baseline and histogram complete: recorded metadata, counts, JUnit XML,
  normalized histogram for all 45 expected failures, and measured Work Queue
  repros under `agent_space/cpython_dynamo/`; no source or sentinel changes.
- G1 partial `dict.update` / merge cycle (2026-05-11):
  - Active gate: G1, not complete. Removed 2 of the 5 target sentinels required
    by the G1 exit criteria.
  - Cluster worked: `dict.update` and `dict.__ior__` merge-from-sequence paths.
  - Failure reason before the change:
    - `DictTest.test_update`, `GeneralMappingTests.test_update`, and
      `SubclassMappingTests.test_update`: `Unsupported method call; call_method
      ConstDictVariable() update [] {}` on no-argument `dict.update()`.
    - `DictTest.test_merge_operator`: `Length mismatch when unpacking object
      for UNPACK_SEQUENCE; expected length: 2, actual: 1` while tracing
      `a.__ior__("BAD")`.
  - Source fix: `ConstDictVariable.update` now handles no-argument updates and
    mirrors CPython `dict_update_arg` order for supported values: dict fast
    path, mapping-like objects with `keys()`, then sequence-of-pairs. Malformed
    sequence elements raise observed `ValueError` instead of graph breaking.
  - Focused regression tests added:
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_update_cpython_merge_paths`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_ior_malformed_sequence_raises`
  - Sentinels removed:
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_update`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_merge_operator`
  - Commands and results:
    - Temporary sentinel-bypass repro before fix for the three update tests and
      merge operator: `4 failed`; update failures were the no-arg
      `ConstDictVariable.update` graph break, merge failure was malformed
      sequence unpack.
    - Temporary sentinel-bypass repro after fix for the same four tests:
      `2 passed, 2 failed`; `DictTest.test_update` and
      `DictTest.test_merge_operator` passed, while the two `mapping_tests`
      variants moved to `Invalid call to __build_class__` at nested
      `SimpleUserDict`.
    - `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_update_cpython_merge_paths test/dynamo/test_dicts.py::DictMethodsTests::test_ior_malformed_sequence_raises`
      -> `2 passed in 1.78s`.
    - `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_update test/dynamo/test_dicts.py::DictMethodsTests::test_binop_ior_iterable test/dynamo/test_dicts.py::DictMethodsTests::test_binop_ior`
      -> `3 passed in 1.80s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update`
      -> `1 passed in 2.36s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_merge_operator`
      -> `1 passed in 2.32s`.
    - `find test/dynamo_expected_failures -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `43`.
    - `find test/dynamo_skips -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `7`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update test/dynamo/cpython/3_13/test_dict.py::GeneralMappingTests::test_update test/dynamo/cpython/3_13/test_dict.py::SubclassMappingTests::test_update test/dynamo/cpython/3_13/test_dict.py::DictTest::test_fromkeys test/dynamo/cpython/3_13/test_dict.py::DictTest::test_fromkeys_operator_modifying_dict_operand test/dynamo/cpython/3_13/test_dict.py::DictTest::test_fromkeys_operator_modifying_set_operand test/dynamo/cpython/3_13/test_dict.py::DictTest::test_merge_operator test/dynamo/cpython/3_13/test_dict.py::DictTest::test_merge_and_mutate`
      -> `2 passed, 6 skipped in 1.64s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> `62 passed, 50 skipped in 5.57s`.
  - Remaining G1 targets and normalized reasons:
    - `GeneralMappingTests.test_update`: unsupported protocol,
      `Invalid call to __build_class__` for nested `SimpleUserDict` closing over
      non-constant `outerself`.
    - `SubclassMappingTests.test_update`: same `__build_class__` closure gap.
    - `DictTest.test_fromkeys`: still skipped by the vendored
      `@unittest.skip` marker for `test hangs`; no sentinel removed.
    - `DictTest.test_fromkeys_operator_modifying_dict_operand`: side-effect /
      closure semantic gap, `Read uninitialized cell` while defining local
      class `X`.
    - `DictTest.test_fromkeys_operator_modifying_set_operand`: same
      uninitialized-cell local class gap.
    - `DictTest.test_merge_and_mutate`: same uninitialized-cell local class
      gap.
  - Known risks: mapping-like `update` coverage is limited to keys iterables
    Dynamo can force-unpack; arbitrary dynamic mappings can still graph break.
  - Recommended next cluster: stay in G1 and fix the local class / closure cell
    support blocking `GeneralMappingTests.test_update`,
    `SubclassMappingTests.test_update`, `DictTest.test_merge_and_mutate`, and
    the two fromkeys mutation tests.
- G1 local class / closure-cell cycle (2026-05-11):
  - Active gate: G1. This cycle completes G1 exit criteria with measured
    evidence: 6 G1 target sentinels removed in total, focused Dynamo regression
    coverage added, and full `test_dict.py` has no unexpected successes or new
    real failures.
  - Cluster worked: local class bodies whose methods close over outer cells,
    including cells holding non-constant Dynamo values and cells assigned after
    the class statement.
  - Failure reason before the change:
    - `GeneralMappingTests.test_update` and
      `SubclassMappingTests.test_update`: `Invalid call to __build_class__`
      for nested `SimpleUserDict` closing over non-constant `outerself`.
    - `DictTest.test_fromkeys_operator_modifying_dict_operand`,
      `DictTest.test_fromkeys_operator_modifying_set_operand`, and
      `DictTest.test_merge_and_mutate`: `Read uninitialized cell` while
      defining a local class whose methods close over a cell assigned later.
  - Source fix: `NestedUserFunctionVariable` can now materialize class body
    functions for `__build_class__` by aliasing closure cells when bytecode only
    forwards freevars into nested methods. Aliased cells are registered with
    side effects so later method tracing reads the current Dynamo cell contents,
    and `STORE_DEREF` mirrors Python constants into aliased CPython cells for
    CPython operations that call Python equality/hash during tracing.
  - Focused regression tests added:
    - `test/dynamo/test_misc.py::MiscTests::test___build_class___non_constant_closure_cell`
    - `test/dynamo/test_misc.py::MiscTests::test___build_class___cell_assigned_after_class_body`
    - `test/dynamo/test_misc.py::MiscTests::test___build_class___constant_cell_reassigned_after_class_body`
    - `test/dynamo/test_misc.py::MiscTests::test___build_class___aliased_cell_visible_to_python_comparison`
  - G1 target sentinels removed:
    - `test/dynamo_expected_failures/CPython313-test_dict-GeneralMappingTests.test_update`
    - `test/dynamo_expected_failures/CPython313-test_dict-SubclassMappingTests.test_update`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_fromkeys_operator_modifying_dict_operand`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_fromkeys_operator_modifying_set_operand`
  - Additional sentinels removed after full-file unexpected-success evidence
    from the same closure-cell fix:
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_dict_contain_use_after_free`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_dictitems_contains_use_after_free`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_getitem`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_init_use_after_free`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_resize2`
  - Commands and results:
    - Temporary sentinel-bypass repro before fix for
      `GeneralMappingTests.test_update`,
      `SubclassMappingTests.test_update`,
      `DictTest.test_fromkeys_operator_modifying_dict_operand`,
      `DictTest.test_fromkeys_operator_modifying_set_operand`, and
      `DictTest.test_merge_and_mutate` -> `5 failed`; the two mapping tests
      failed at `Invalid call to __build_class__`, and the three dict tests
      failed at `Read uninitialized cell`.
    - `pytest -q --tb=short test/dynamo/test_misc.py::MiscTests::test___build_class___non_constant_closure_cell test/dynamo/test_misc.py::MiscTests::test___build_class___cell_assigned_after_class_body test/dynamo/test_misc.py::MiscTests::test___build_class___constant_cell_reassigned_after_class_body test/dynamo/test_misc.py::MiscTests::test___build_class___aliased_cell_visible_to_python_comparison`
      -> final result `4 passed in 3.35s`.
    - Temporary sentinel-bypass repro after the first class-body alias fix for
      the same five CPython targets -> `2 passed, 3 failed`; the remaining
      failures had moved to actual CPython cell aliasing / `assertRaises`
      behavior.
    - Temporary sentinel-bypass repro after CPython cell alias synchronization
      for the same five CPython targets -> `4 passed, 1 failed`; only
      `DictTest.test_merge_and_mutate` remained, now at
      `Data-dependent branching` in `unittest.case._AssertRaisesContext`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update test/dynamo/cpython/3_13/test_dict.py::GeneralMappingTests::test_update test/dynamo/cpython/3_13/test_dict.py::SubclassMappingTests::test_update test/dynamo/cpython/3_13/test_dict.py::DictTest::test_fromkeys test/dynamo/cpython/3_13/test_dict.py::DictTest::test_fromkeys_operator_modifying_dict_operand test/dynamo/cpython/3_13/test_dict.py::DictTest::test_fromkeys_operator_modifying_set_operand test/dynamo/cpython/3_13/test_dict.py::DictTest::test_merge_operator test/dynamo/cpython/3_13/test_dict.py::DictTest::test_merge_and_mutate`
      -> `6 passed, 2 skipped in 1.97s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> initially `5 failed, 66 passed, 41 skipped` due to unexpected
      successes in non-G1 sentinels fixed by the same closure-cell change.
    - After removing those proven-fixed non-G1 sentinels,
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> `71 passed, 41 skipped in 6.51s`; rerun after review fix for
      constant cells reassigned after class creation -> `71 passed, 41 skipped
      in 6.73s`.
    - `find test/dynamo_expected_failures -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `34`.
    - `find test/dynamo_skips -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `7`.
  - Remaining G1 targets and normalized reasons:
    - `DictTest.test_fromkeys`: still skipped by the vendored
      `@unittest.skip("test hangs")` marker; no sentinel removed.
    - `DictTest.test_merge_and_mutate`: side-effect semantic gap. The local
      class closure now traces, but `d.update(other)` does not surface the
      CPython `RuntimeError` expected by `assertRaises`; tracing continues into
      `unittest.case._AssertRaisesContext.__exit__` and graph breaks on
      data-dependent branching over `self.obj_name`.
  - Known risks: class-body closure aliasing is intentionally limited to class
    bodies whose bytecode only forwards freevars into nested methods. Class
    bodies that read freevar contents for class attributes still use the
    existing constant-closure path or graph break.
  - Recommended next cluster: move to G2. `DictTest.test_getitem` and
    `DictTest.test_resize2` were incidentally fixed, so the next coherent G2
    cluster should target `DictTest.test_bad_key`,
    `DictTest.test_invalid_keyword_arguments`, `DictTest.test_setdefault_atomic`,
    and `DictTest.test_setitem_atomic_at_resize`.
- G2 item access / mutating-method cycle (2026-05-11):
  - Active gate: G2. This cycle completes G2 exit criteria with measured
    evidence: 4 G2 target sentinels removed, focused Dynamo regression coverage
    added, and full `test_dict.py` has no unexpected successes or new real
    failures.
  - Cluster worked: `DictTest.test_bad_key`,
    `DictTest.test_invalid_keyword_arguments`, `DictTest.test_setdefault_atomic`,
    and `DictTest.test_setitem_atomic_at_resize`.
  - Failure reason before the change:
    - `DictTest.test_bad_key`: `exec(stmt, locals())` graph broke at
      `Failed to trace builtin operator exec`, so the dict operations under
      test never reached Dynamo's dict protocol.
    - `DictTest.test_invalid_keyword_arguments`: `dict(**invalid)` raised a raw
      `TypeError: keywords must be strings`, surfaced as
      `InternalTorchDynamoError` instead of an observed Python exception.
    - `DictTest.test_setdefault_atomic`: `setdefault` hashed the missing custom
      key twice, producing `AssertionError('2 != 1')`.
    - `DictTest.test_setitem_atomic_at_resize`: dict `__setitem__` did not run
      custom key equality during collision handling, producing
      `AssertionError('0 != 1')`.
  - Source fix: dict lookup/update paths now use explicit lookup helpers that
    compute a lookup hash once, trace CPython-style rich equality on hash
    collisions, restart lookup after equality-triggered key mutation, and reuse
    the lookup key for insertion. `setdefault` now follows the same known-hash
    lookup/insert shape. The lookup scan snapshots items so equality-triggered
    dict mutation does not raise an internal `dictionary changed size during
    iteration`. `CALL_FUNCTION_EX` now converts non-string keyword names into
    an observed `TypeError`. A narrow two-argument constant string `exec()`
    handler supports the CPython bad-key pattern only for `locals()` snapshots,
    including observable `__builtins__` insertion and builtin-name fallback, by
    evaluating simple name, dict, subscript, method call, containment,
    expression, and assignment forms. Dict literals inside the supported
    `exec()` subset and `dict.__init__`, `dict.__delitem__`, and keyword
    `dict.update` now route through the collision-aware helpers.
    Name and builtins lookup inside the supported `exec()` subset also use the
    same collision-aware lookup when the supplied scope or `__builtins__` is a
    Dynamo dict.
  - Focused regression tests added:
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_uses_reflected_subtype_eq`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_rechecks_after_eq_mutation`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_update_kwargs_uses_collision_lookup`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_delitem_uses_collision_lookup`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_init_uses_collision_lookup`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_explicit_globals_locals_unsupported`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_explicit_globals_unsupported`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_injects_builtins`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_resolves_builtin_names`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_dict`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_module`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_builtins_module_ignores_getattr`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_dict_literal_uses_collision_lookup`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_scope_lookup_uses_collision_lookup`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_builtins_lookup_uses_collision_lookup`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_dict_literal_uses_collision_lookup`
    - `test/dynamo/test_dicts.py::DictMethodsTests::test_dict_unpack_uses_collision_lookup`
  - G2 target sentinels removed:
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_bad_key`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_invalid_keyword_arguments`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_setdefault_atomic`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_setitem_atomic_at_resize`
  - Additional G3 sentinel removed after review follow-up evidence from the
    same rich-comparison fix:
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_str_nonstr`
  - Commands and results:
    - Temporary sentinel-bypass repro before fix for the four selected G2
      targets -> `4 failed`; failures matched the reasons above.
    - Temporary sentinel-bypass repro for
      `CAPITest.test_getitem_knownhash` -> `1 failed` at
      `Attempted to call function marked as skipped` for
      `import_helper.import_module('_testinternalcapi')`.
    - `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize`
      -> final result `4 passed in 1.92s`.
    - Review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_uses_reflected_subtype_eq test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_rechecks_after_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_exec_explicit_globals_locals_unsupported test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
      -> final result `7 passed in 2.30s`.
    - Second review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_exec_explicit_globals_unsupported test/dynamo/test_dicts.py::DictMethodsTests::test_exec_explicit_globals_locals_unsupported test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_uses_reflected_subtype_eq test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_rechecks_after_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
      -> final result `9 passed in 2.16s`.
    - Third review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_update_kwargs_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_uses_reflected_subtype_eq test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_rechecks_after_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_exec_explicit_globals_unsupported test/dynamo/test_dicts.py::DictMethodsTests::test_exec_explicit_globals_locals_unsupported`
      -> final result `10 passed in 2.28s`.
    - Fourth review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_injects_builtins test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_resolves_builtin_names test/dynamo/test_dicts.py::DictMethodsTests::test_delitem_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_init_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_update_kwargs_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
      -> final result `10 passed in 2.41s`.
    - Fifth review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_dict test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_injects_builtins test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_resolves_builtin_names test/dynamo/test_dicts.py::DictMethodsTests::test_delitem_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_init_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_update_kwargs_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
      -> final result `11 passed in 2.64s`.
    - Sixth review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_exec_dict_literal_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_module test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_dict test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_injects_builtins test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_resolves_builtin_names test/dynamo/test_dicts.py::DictMethodsTests::test_delitem_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_init_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_update_kwargs_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
      -> final result `13 passed in 2.59s`.
    - Seventh review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_scope_lookup_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_builtins_lookup_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_dict_literal_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_module test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_dict test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_injects_builtins test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_resolves_builtin_names test/dynamo/test_dicts.py::DictMethodsTests::test_delitem_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_init_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_update_kwargs_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
      -> final result `15 passed in 2.49s`.
    - Eighth review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_dict_literal_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_dict_unpack_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_builtins_module_ignores_getattr test/dynamo/test_dicts.py::DictMethodsTests::test_exec_dict_literal_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises`
      -> final result `5 passed in 2.47s`.
    - Ninth review-focused regression rerun:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_scope_lookup_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_builtins_lookup_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_dict_literal_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_module test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_honors_existing_builtins_dict test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_injects_builtins test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_resolves_builtin_names test/dynamo/test_dicts.py::DictMethodsTests::test_delitem_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_init_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_update_kwargs_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_dict_literal_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_dict_unpack_uses_collision_lookup test/dynamo/test_dicts.py::DictMethodsTests::test_exec_locals_builtins_module_ignores_getattr test/dynamo/test_dicts.py::DictMethodsTests::test_bad_key_exec_raises test/dynamo/test_dicts.py::DictMethodsTests::test_dict_key_lookup_restarts_after_false_eq_mutation test/dynamo/test_dicts.py::DictMethodsTests::test_setdefault_custom_key_hash_eq_once test/dynamo/test_dicts.py::DictMethodsTests::test_setitem_custom_key_hash_eq_once_at_resize test/dynamo/test_dicts.py::DictMethodsTests::test_invalid_non_string_keyword_arguments`
      -> final result `18 passed in 2.49s`.
    - Temporary sentinel-bypass repro after fix for the four selected G2
      targets -> `4 passed in 1.27s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_bad_key test/dynamo/cpython/3_13/test_dict.py::DictTest::test_invalid_keyword_arguments test/dynamo/cpython/3_13/test_dict.py::DictTest::test_setdefault_atomic test/dynamo/cpython/3_13/test_dict.py::DictTest::test_setitem_atomic_at_resize`
      -> final result `4 passed in 1.35s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::CAPITest::test_getitem_knownhash test/dynamo/cpython/3_13/test_dict.py::DictTest::test_bad_key test/dynamo/cpython/3_13/test_dict.py::DictTest::test_invalid_keyword_arguments test/dynamo/cpython/3_13/test_dict.py::DictTest::test_setdefault_atomic test/dynamo/cpython/3_13/test_dict.py::DictTest::test_setitem_atomic_at_resize`
      -> final result `4 passed, 1 skipped in 1.41s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_dict_contain_use_after_free`
      -> `1 passed in 1.10s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_pop_pending`
      -> `1 passed in 1.27s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_str_nonstr`
      -> `1 passed in 1.55s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `75 passed, 37 skipped in 8.18s`.
    - Review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 7.41s`.
    - Second review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 10.55s`.
    - Third review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 7.92s`.
    - Fourth review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 8.26s`.
    - Fifth review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 8.25s`.
    - Sixth review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 8.14s`.
    - Seventh review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 8.01s`.
    - Review follow-up G2 subset rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::CAPITest::test_getitem_knownhash test/dynamo/cpython/3_13/test_dict.py::DictTest::test_bad_key test/dynamo/cpython/3_13/test_dict.py::DictTest::test_invalid_keyword_arguments test/dynamo/cpython/3_13/test_dict.py::DictTest::test_setdefault_atomic test/dynamo/cpython/3_13/test_dict.py::DictTest::test_setitem_atomic_at_resize`
      -> final result `4 passed, 1 skipped in 1.31s`.
    - Eighth review follow-up full-file rerun:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> final result `76 passed, 36 skipped in 8.03s`.
    - `find test/dynamo_expected_failures -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `29`.
    - `find test/dynamo_skips -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `7`.
  - Remaining G2 targets and normalized reasons:
    - `CAPITest.test_getitem_knownhash`: stdlib/C boundary. The test imports
      `_testinternalcapi` via `test.support.import_helper.import_module`, which
      graph breaks because `importlib.import_module` is marked skipped. The
      rest of the test targets a CPython internal C API rather than normal
      Python dict item access.
  - Known risks: the `exec()` support is intentionally narrow and only handles
    the two-argument constant-string/`locals()` form needed for CPython dict
    coverage. It models `__builtins__` insertion and builtin-name lookup from
    the supplied scope's `__builtins__` dict/module for the supported syntax,
    but does not attempt general `exec` globals / locals semantics. Dict key
    equality tracing now mirrors more CPython collision behavior for Python
    keys, but dynamic non-constant equality results still graph break.
  - Recommended next cluster: move to G3 split-dict and non-string key edges,
    starting with `DictTest.test_object_set_item_single_instance_non_str_key`,
    `DictTest.test_splittable_to_generic_combinedtable`, and
    `DictTest.test_splittable_setdefault`.
- G3 non-string instance `__dict__` / split-dict order cycle (2026-05-11):
  - Active gate: G3, not complete. This cycle removed 3 current G3 target
    sentinels; G3 still needs at least 4 current-cycle removals for completion.
    `DictTest.test_str_nonstr` remains prior G2 progress and was not
    double-counted.
  - Cluster worked: instance `__dict__` as a mutable mapping, including
    non-string keys that promote CPython split dicts to generic combined tables
    and split-dict delete/reinsert ordering.
  - CPython reference: `Objects/dictobject.c::insertdict` allows non-Unicode
    keys to fall through to `insert_combined_dict`, converting split/unicode
    tables to generic combined tables; `dictresize` documents split-to-combined
    conversion.
  - Failure reason before the change:
    - `DictTest.test_object_set_item_single_instance_non_str_key` and
      `DictTest.test_splittable_to_generic_combinedtable`: internal
      `AssertionError: Expected str key, got <class 'int'>` from
      `SideEffectsProxyDict.__setitem__`.
    - `DictTest.test_splittable_pop`: after `a.pop('y'); a['y'] = 42`,
      Dynamo iterated the pending side-effect table before original keys, so
      `list(a)` did not match CPython's `['x', 'z', 'y']` order.
  - Source fix: `DunderDictVariable` now tracks object `__dict__` as a mapping
    keyed by any hashable Python key instead of only string attribute names.
    Its proxy preserves current dict order across pre-materialization attribute
    writes, non-string item insertion, deletion, and reinsertion. Non-string
    `__dict__` item mutations are replayed with `STORE_SUBSCR`, and
    delete/reinsert cases reconstruct the materialized `__dict__` so external
    objects keep CPython dict order. Review follow-up fixed `del obj.attr`
    after `obj.__dict__` materialization to route through the same deletion
    path, and gave `DunderDictVariable.items.popitem()` dict-style LIFO
    behavior instead of the `MutableMapping` FIFO default.
  - Focused regression tests added:
    - `test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_non_string_key_promotes_to_generic_mapping`
    - `test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_replays_non_string_key_mutation`
    - `test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_reinsert_existing_key_updates_order`
    - `test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_delattr_reinsert_updates_order`
    - `test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_popitem_is_lifo`
  - G3 target sentinels removed:
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_object_set_item_single_instance_non_str_key`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_splittable_pop`
    - `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_splittable_to_generic_combinedtable`
  - Commands and results:
    - Temporary sentinel-bypass repro before fix for
      `DictTest.test_object_set_item_single_instance_non_str_key`,
      `DictTest.test_splittable_to_generic_combinedtable`, and
      `DictTest.test_splittable_setdefault`:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_object_set_item_single_instance_non_str_key test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_to_generic_combinedtable test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_setdefault`
      -> `3 failed`; first two failed on the non-string key assertion,
      `test_splittable_setdefault` failed at skipped `sys.getsizeof`.
    - Temporary sentinel-bypass repro before fix for
      `DictTest.test_splittable_del`, `DictTest.test_splittable_pop`, and
      `DictTest.test_splittable_popitem`:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_del test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_pop test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_popitem`
      -> `3 failed`; `test_splittable_del` and `test_splittable_popitem`
      failed at skipped `sys.getsizeof`, while `test_splittable_pop` failed
      on the post-reinsert key order assertion.
    - `pytest -q --tb=short test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_non_string_key_promotes_to_generic_mapping test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_replays_non_string_key_mutation test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_reinsert_existing_key_updates_order`
      -> `3 passed in 1.74s`.
    - Review follow-up repro for `obj.__dict__.popitem()` before the LIFO fix:
      local `torch.compile(backend="eager", fullgraph=True)` repro returned
      `(('x', 1), ['y', 'z'], {'y': 2, 'z': 3})`; expected first item is
      `('z', 3)`.
    - Review follow-up focused regression:
      `pytest -q --tb=short test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_non_string_key_promotes_to_generic_mapping test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_replays_non_string_key_mutation test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_reinsert_existing_key_updates_order test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_delattr_reinsert_updates_order test/dynamo/test_dicts.py::DunderDictVariableTests::test_dunder_dict_popitem_is_lifo`
      -> `5 passed in 1.89s`.
    - Temporary sentinel-bypass repro after fix for
      `DictTest.test_object_set_item_single_instance_non_str_key`,
      `DictTest.test_splittable_to_generic_combinedtable`, and
      `DictTest.test_splittable_pop`:
      `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_object_set_item_single_instance_non_str_key test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_to_generic_combinedtable test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_pop`
      -> `3 passed in 1.26s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_object_set_item_single_instance_non_str_key test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_to_generic_combinedtable test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_pop test/dynamo/cpython/3_13/test_dict.py::DictTest::test_str_nonstr`
      -> `4 passed in 1.48s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py::DictTest::test_object_set_item_single_instance_non_str_key test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_del test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_pop test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_popitem test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_setdefault test/dynamo/cpython/3_13/test_dict.py::DictTest::test_splittable_to_generic_combinedtable test/dynamo/cpython/3_13/test_dict.py::DictTest::test_str_nonstr`
      -> `4 passed, 3 skipped in 1.72s`; review follow-up rerun after the
      `delattr` and `popitem` fixes -> `4 passed, 3 skipped in 1.78s`.
    - `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py`
      -> `79 passed, 33 skipped in 9.12s`; review follow-up rerun after the
      `delattr` and `popitem` fixes -> `79 passed, 33 skipped in 8.13s`.
    - `find test/dynamo_expected_failures -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `26`.
    - `find test/dynamo_skips -maxdepth 1 -type f -name 'CPython313-test_dict-*' | wc -l`
      -> `7`.
  - Remaining G3 targets and normalized reasons:
    - `DictTest.test_splittable_del`: stdlib/C boundary at skipped
      `sys.getsizeof(a)` before the dict deletion path is reached.
    - `DictTest.test_splittable_setdefault`: stdlib/C boundary at skipped
      `sys.getsizeof(a)` before the setdefault ordering check.
    - `DictTest.test_splittable_popitem`: stdlib/C boundary at skipped
      `sys.getsizeof(a)`, and the later `assertGreater(sys.getsizeof(a),
      orig_size)` is a CPython split-table memory-layout implementation
      detail. A separate review repro confirmed and fixed the intermediate
      `DunderDictVariable.items.popitem()` LIFO gap; the CPython sentinel is
      still not removable while `sys.getsizeof` remains skipped.
  - Known risks: non-string object `__dict__` mutations are replayed for
    hashable constant keys. Dynamic key hashing/equality can still graph break
    through the existing dict-key protocol limits, and full `__dict__`
    reconstruction is only used when deletion/reinsertion makes attribute-store
    replay order observably different.
  - Recommended next cluster: stay in G3 and decide a principled
    `sys.getsizeof` boundary policy for the remaining split-table tests before
    moving to any later gate.
