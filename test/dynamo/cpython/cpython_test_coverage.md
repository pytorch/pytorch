# CPython Dynamo Coverage Plan

Status: current gate is G0.

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
| `test_dict.py` | 112 | 45 | 7 |

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

Status: current.

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

Status: pending.

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

Status: pending.

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

Status: pending.

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
- Propose the next file and category gates under "Proposed Gate Changes Awaiting
  Human Approval".

## Work Queue

G0 owns this section until its exit criteria are met. Replace these seed items
with measured clusters from the current tree.

1. Build `test_dict.py` JUnit XML and normalized histogram.
2. Reproduce `DictTest.test_update`; likely source is
   `ConstDictVariable.call_method`.
3. Reproduce `GeneralMappingTests.test_update`; likely source is mapping
   protocol update handling.
4. Reproduce `DictTest.test_getitem`; likely source is dict getitem handling or
   object protocol dispatch.
5. Reproduce `DictTest.test_items_symmetric_difference`; likely source is dict
   view set operation handling.
6. Reproduce `DictTest.test_object_set_item_single_instance_non_str_key`;
   likely source is split dict or instance dict key assumptions.
7. Reproduce `DictTest.test_mutating_lookup`; likely source is side-effect
   handling during lookup.
8. Reproduce `DictTest.test_reversed`; likely source is dict iterator or
   reconstruction handling.
9. Reproduce `DictTest.test_repr`; likely source is recursive repr behavior.
10. Triage one skip sentinel, starting with `DictTest.test_track_literals`, and
    decide whether it is useful Dynamo coverage or CPython implementation
    detail.

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

No implementation cycles have been run from this plan yet.
