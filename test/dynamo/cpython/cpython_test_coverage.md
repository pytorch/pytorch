# CPython Dynamo Coverage Plan

Status: current gate is G0.

Goal: improve `PYTORCH_TEST_WITH_DYNAMO=1` coverage for CPython tests under
`test/dynamo/cpython/3_13`, starting with five common data-structure files:

```
test_list.py
test_tuple.py
test_dict.py
test_set.py
test_deque.py
```

The current reported overall passrate is about 45%. This plan keeps the first
campaign small and makes gates per-file so the agent loop can land measurable
progress without repeatedly running the full CPython suite.

Operational loop instructions live in `test/dynamo/cpython/agent_manager.md`.
This file is the source of truth for active gate, file scope, exit criteria,
test commands, and measured progress.

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

Expected-failure sentinels still execute the test. If the test now passes, the
harness reports an unexpected success and asks for the sentinel to be removed.

## Initial Five-File Slice

Collection command:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest --collect-only -q \
  test/dynamo/cpython/3_13/test_list.py \
  test/dynamo/cpython/3_13/test_tuple.py \
  test/dynamo/cpython/3_13/test_dict.py \
  test/dynamo/cpython/3_13/test_set.py \
  test/dynamo/cpython/3_13/test_deque.py
```

Measured collection: 914 tests.

| File | Collected | Xfail sentinels | Skip sentinels | Main source areas |
| --- | ---: | ---: | ---: | --- |
| `test_list.py` | 65 | 25 | 0 | `variables/lists.py`, `variables/builtin.py`, `variables/object_protocol.py` |
| `test_tuple.py` | 36 | 16 | 0 | `variables/lists.py`, `variables/builtin.py`, `variables/object_protocol.py` |
| `test_dict.py` | 112 | 45 | 7 | `variables/dicts.py`, `variables/user_defined.py`, `variables/object_protocol.py` |
| `test_set.py` | 623 | 220 | 0 | `variables/sets.py`, `variables/user_defined.py`, `variables/object_protocol.py` |
| `test_deque.py` | 78 | 58 | 0 | `variables/lists.py`, `variables/user_defined.py`, `variables/object_protocol.py` |

Initial five-file totals: 914 collected tests, 364 expected-failure sentinels,
and 7 skip sentinels.

## Measurement Commands

Run one file:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py
```

Run one test:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
```

Run the five-file slice:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_list.py \
  test/dynamo/cpython/3_13/test_tuple.py \
  test/dynamo/cpython/3_13/test_dict.py \
  test/dynamo/cpython/3_13/test_set.py \
  test/dynamo/cpython/3_13/test_deque.py
```

Write five-file JUnit XML:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  --junitxml=agent_space/cpython_dynamo/five_file_slice.xml \
  test/dynamo/cpython/3_13/test_list.py \
  test/dynamo/cpython/3_13/test_tuple.py \
  test/dynamo/cpython/3_13/test_dict.py \
  test/dynamo/cpython/3_13/test_set.py \
  test/dynamo/cpython/3_13/test_deque.py
```

Count expected failures for the five files:

```
for f in test_list test_tuple test_dict test_set test_deque; do
  printf '%s ' "$f"
  find test/dynamo_expected_failures -maxdepth 1 -type f \
    -name "CPython313-$f-*" | wc -l
done
```

Count skips for the five files:

```
find test/dynamo_skips -maxdepth 1 -type f \
  \( -name 'CPython313-test_list-*' \
  -o -name 'CPython313-test_tuple-*' \
  -o -name 'CPython313-test_dict-*' \
  -o -name 'CPython313-test_set-*' \
  -o -name 'CPython313-test_deque-*' \) | wc -l
```

Use Dynamo logs only for opaque single-test repros:

```
TORCH_LOGS="+dynamo,graph_breaks" TORCHDYNAMO_VERBOSE=1 \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
```

## Gates

### G0: Baseline Five Files

Status: current.

Purpose: capture reliable starting data before source changes.

Exit criteria:

- Record git SHA, Python version, exact five-file command, and JUnit XML under
  `agent_space/cpython_dynamo/`.
- Verify the current five-file counts: 914 collected tests, 364 expected-failure
  sentinels, and 7 skip sentinels.
- Build a normalized failure histogram from the five-file run.
- Record the first 10 actionable single-test repros in "Work Queue" with exact
  test names, normalized failure reason, and likely source file.
- Run at least one single-test repro with its sentinel temporarily bypassed and
  restored.
- Leave source files and sentinels unchanged.

### G1: `test_list.py`

Status: pending.

Initial scope: 65 collected tests, 25 expected-failure sentinels, 0 skip
sentinels.

Exit criteria:

- Remove at least 10 `test_list.py` expected-failure sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short
  test/dynamo/cpython/3_13/test_list.py` has no unexpected successes or new
  real failures.
- Remaining `test_list.py` sentinels are categorized in this file.

Seed clusters:

- Constructors and subclass keyword behavior.
- `contains`, `count`, `index`, and `remove` with side effects.
- Repeat and inplace repeat.
- Repr and recursive repr.
- Iterator pickle/reconstruction and `reversed`.

### G2: `test_tuple.py`

Status: pending.

Initial scope: 36 collected tests, 16 expected-failure sentinels, 0 skip
sentinels.

Exit criteria:

- Remove at least 7 `test_tuple.py` expected-failure sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short
  test/dynamo/cpython/3_13/test_tuple.py` has no unexpected successes or new
  real failures.
- Remaining `test_tuple.py` sentinels are categorized in this file.

Seed clusters:

- Constructors and subclass keyword behavior.
- Repeat and `repr(FakeIdVariable)`-adjacent failures.
- Contains order and index behavior.
- Iterator pickle/reconstruction and `reversed`.
- CPython GC tracking tests that may be implementation-detail boundaries.

### G3: `test_dict.py`

Status: pending.

Initial scope: 112 collected tests, 45 expected-failure sentinels, 7 skip
sentinels.

Exit criteria:

- Remove at least 15 `test_dict.py` expected-failure sentinels.
- Do not remove skip sentinels unless the tests are now passing and cheap.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short
  test/dynamo/cpython/3_13/test_dict.py` has no unexpected successes or new
  real failures.
- Remaining `test_dict.py` xfails and skips are categorized in this file.

Seed clusters:

- `ConstDictVariable.update` and mapping update protocol.
- `getitem`, `setdefault`, `pop`, `popitem`, and reconstruction.
- Merge operators and `fromkeys`.
- Dict view operations and symmetric difference.
- Split-dict behavior and non-string key edge cases.
- Mutation during lookup, equality, and iteration.

### G4: `test_set.py`

Status: pending.

Initial scope: 623 collected tests, 220 expected-failure sentinels, 0 skip
sentinels.

Exit criteria:

- Remove at least 40 `test_set.py` expected-failure sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short
  test/dynamo/cpython/3_13/test_set.py` has no unexpected successes or new real
  failures.
- Remaining `test_set.py` sentinels are categorized in this file.

Seed clusters:

- Binary operations across set/set, set/subclass, subclass/set, and
  subclass/subclass.
- Inplace operations and mutation during operation.
- Frozenset hash, copy, and constructor identity.
- Set subclass constructor and keyword behavior.
- Iterator, pickle, repr, and CPython implementation-detail boundaries.

### G5: `test_deque.py`

Status: pending.

Initial scope: 78 collected tests, 58 expected-failure sentinels, 0 skip
sentinels.

Exit criteria:

- Remove at least 18 `test_deque.py` expected-failure sentinels.
- Add focused Dynamo regression tests for each non-trivial semantic fix.
- `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short
  test/dynamo/cpython/3_13/test_deque.py` has no unexpected successes or new
  real failures.
- Remaining `test_deque.py` sentinels are categorized in this file.

Seed clusters:

- `getitem`, `index`, `count`, `contains`, and sequence protocol behavior.
- `add`, `iadd`, `mul`, `imul`, `extend`, and `extendleft`.
- `rotate`, `reverse`, `reversed`, and iterator behavior.
- Copy, deepcopy, pickle, recursive pickle, and repr.
- Subclass and maxlen behavior.

### G6: Five-File Rebaseline

Status: pending.

Purpose: measure the campaign result and choose the next five files.

Exit criteria:

- Run the five-file slice command and record total pass/skip/fail/unexpected
  success counts.
- Record expected-failure and skip counts for all five files after G1-G5.
- Compute sentinel removals by file and total.
- Identify the next five files by expected-failure count and user value.
- Add proposed per-file gates for the next slice under "Proposed Gate Changes
  Awaiting Human Approval".

## Work Queue

G0 owns this section until its exit criteria are met. Replace these seed items
with measured clusters from the current tree.

1. Build five-file JUnit XML and normalized histogram.
2. Reproduce `ListTest.test_repeat`; likely source is sequence repeat or repr
   behavior in `variables/lists.py`.
3. Reproduce `TupleTest.test_repeat`; likely source is tuple repeat or repr
   behavior in `variables/lists.py`.
4. Reproduce `DictTest.test_update`; likely source is
   `ConstDictVariable.call_method`.
5. Reproduce `DictTest.test_items_symmetric_difference`; likely source is dict
   view set operation handling.
6. Reproduce one `TestBinaryOpsMutating_*` set failure; likely source is set
   binary dispatch, subclass priority, or side-effect handling.
7. Reproduce one `TestMethodsMutating_*` set failure; likely source is inplace
   set operation handling.
8. Reproduce `TestBasic.test_getitem` from `test_deque.py`; likely source is
   `DequeVariable` indexing or object protocol indexing.
9. Reproduce `TestBasic.test_rotate` from `test_deque.py`; likely source is
   `DequeVariable.call_method`.
10. Reproduce one pickle/repr failure and decide whether it is normal Python
    behavior or a CPython implementation-detail boundary.

## Remaining Failure Categories

Use these categories when closing each per-file gate:

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
