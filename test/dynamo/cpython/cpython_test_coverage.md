# CPython Dynamo Coverage Plan

Status: current gate is G0.

This plan is for improving `PYTORCH_TEST_WITH_DYNAMO=1` coverage for the
CPython tests under `test/dynamo/cpython`. The current reported passrate is
about 45%. The first campaign focuses on common Python data structures, where
small semantic fixes should pay off across many tests.

The plan is designed for an agent-driven loop. The loop should make small,
coherent changes, keep the active gate fixed until its exit criteria are met,
and update this file with measured evidence after each cycle.

## Scope

Initial test directory:

```
test/dynamo/cpython/3_13
```

Primary command for the full CPython Dynamo suite:

```
cd test/dynamo/cpython/3_13
PYTORCH_TEST_WITH_DYNAMO=1 pytest -vs .
```

Targeted command shape for one file:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short test/dynamo/cpython/3_13/test_dict.py
```

Targeted command shape for one test:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
```

Use `agent_space/` for scratch files, temporary reports, and sentinel backups.
Do not commit files from `agent_space/`.

## How These Tests Work

`torch._dynamo.test_case.CPythonTestCase` wraps CPython unittest methods with
Dynamo when `PYTORCH_TEST_WITH_DYNAMO=1` is set. CPython tests use keys like:

```
CPython313-test_dict-DictTest.test_update
```

Known failures are represented by empty sentinel files in:

```
test/dynamo_expected_failures/
```

Known skips are represented by empty sentinel files in:

```
test/dynamo_skips/
```

Expected failures still execute. If the test fails, the wrapper reports it as a
skip. If the test unexpectedly passes, the wrapper raises:

```
Unexpected success, please remove `test/dynamo_expected_failures/<key>`
```

For diagnosis, temporarily move the relevant sentinel into
`agent_space/cpython_sentinel_backups/` and restore it before ending the cycle
unless the fix really makes the test pass. For an actual fix, remove the
sentinel with `git rm`.

Do not add new expected-failure sentinels as part of this campaign unless the
human explicitly approves. New failures should be treated as regressions or
scope findings.

## Maintenance Rules

Gate exit criteria are fixed once written. Do not relax or rewrite them during
an implementation cycle. If a gate appears unreachable, run a deep replan and
record a proposal under "Proposed Gate Changes Awaiting Human Approval"; do not
change the active criteria without human approval.

Do not mark a gate complete unless the recorded exit criteria are met by
measured evidence from the current tree.

Most source fixes should be in `torch/_dynamo`, usually under:

```
torch/_dynamo/variables/
torch/_dynamo/polyfills/
torch/_dynamo/symbolic_convert.py
torch/_dynamo/bytecode_transformation.py
```

Avoid editing vendored CPython tests under `test/dynamo/cpython/3_13` unless the
task is explicitly to update the CPython import. For behavior fixes, add focused
Dynamo tests in the normal Dynamo test files and remove the relevant CPython
expected-failure sentinels.

When adding focused tests, follow the local test style:

```
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    ...

if __name__ == "__main__":
    run_tests()
```

Use `assertEqual` for equality checks. Use `torch._dynamo.config.patch` for
temporary Dynamo config changes.

Only use `spin` commands for linting. If asked to commit or amend, run
`lintrunner -a` before creating the commit.

## Initial Ten-File Slice

These ten files cover the highest-value common data structures and protocols:
lists, tuples, dicts, sets, ordered/default/user mappings, deques, and
collections ABC wrappers.

Collection was checked with:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest --collect-only -q \
  test/dynamo/cpython/3_13/test_list.py \
  test/dynamo/cpython/3_13/test_tuple.py \
  test/dynamo/cpython/3_13/test_dict.py \
  test/dynamo/cpython/3_13/test_set.py \
  test/dynamo/cpython/3_13/test_defaultdict.py \
  test/dynamo/cpython/3_13/test_ordered_dict.py \
  test/dynamo/cpython/3_13/test_deque.py \
  test/dynamo/cpython/3_13/test_userlist.py \
  test/dynamo/cpython/3_13/test_userdict.py \
  test/dynamo/cpython/3_13/test_collections.py
```

That command collected 1396 tests. It emitted one collection warning for
`test_collections.py::TestNT`, matching normal pytest behavior for a class with
`__new__`.

| File | Collected tests | Expected-failure sentinels | Skip sentinels | Why this file is in the first slice |
| --- | ---: | ---: | ---: | --- |
| `test_list.py` | 65 | 25 | 0 | Core mutable sequence operations and iterator semantics |
| `test_tuple.py` | 36 | 16 | 0 | Core immutable sequence, hashing, repeat, repr, tuple subclass behavior |
| `test_dict.py` | 112 | 45 | 7 | Core mapping operations, views, mutation during lookup, split dict behavior |
| `test_set.py` | 623 | 220 | 0 | Core set/frozenset operations and many repeated protocol clusters |
| `test_defaultdict.py` | 11 | 6 | 0 | Default factory, copy, repr, union behavior |
| `test_ordered_dict.py` | 295 | 223 | 0 | Ordered mapping behavior, pure Python and C implementation parity |
| `test_deque.py` | 78 | 58 | 0 | Common queue/list hybrid with sequence protocol behavior |
| `test_userlist.py` | 51 | 27 | 0 | UserList wrapper behavior, sequence protocol delegation |
| `test_userdict.py` | 25 | 15 | 0 | UserDict wrapper behavior, mapping protocol delegation |
| `test_collections.py` | 100 | 46 | 0 | ChainMap, Counter, namedtuple, collections ABCs |

Total for first slice: 1396 collected tests, 681 expected-failure sentinels,
and 7 skip sentinels.

## First-Slice Failure Clusters

These clusters should be used to pick coherent work, not to justify broad
refactors.

### Sequence Protocol

Files:

```
test_list.py
test_tuple.py
test_deque.py
test_userlist.py
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/object_protocol.py
torch/_dynamo/variables/builtin.py
```

Representative expected failures:

```
ListTest.test_constructors
ListTest.test_contains_order
ListTest.test_extend
ListTest.test_imul
ListTest.test_index
ListTest.test_repeat
ListTest.test_repr
TupleTest.test_constructors
TupleTest.test_repeat
TestBasic.test_getitem
TestBasic.test_iadd
UserListTest.test_getitem
UserListTest.test_setitem
```

Common themes:

- `list`, `tuple`, `deque`, and `UserList` constructor and subclass handling.
- `__contains__`, count, index, and remove semantics with side effects.
- Repeat, inplace repeat, add, and reverse/reversed behavior.
- Iterator exhaustion, mutation while iterating, and reconstruction.
- Repr and recursive repr behavior.

### Mapping Protocol

Files:

```
test_dict.py
test_defaultdict.py
test_ordered_dict.py
test_userdict.py
test_collections.py
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/object_protocol.py
torch/_dynamo/variables/builtin.py
```

Representative expected failures:

```
DictTest.test_update
DictTest.test_getitem
DictTest.test_merge_operator
DictTest.test_views_mapping
DictTest.test_items_symmetric_difference
DictTest.test_setdefault_atomic
TestDefaultDict.test_union
UserDictTest.test_update
CPythonOrderedDictTests.test_update
TestChainMap.test_union_operators
TestCounter.test_update
```

Common themes:

- `dict.update`, `setdefault`, `pop`, `popitem`, `fromkeys`, and merge operators.
- Dict views as live objects with set-like operations.
- OrderedDict behavior across C and pure-Python implementations.
- DefaultDict factory, copy/deepcopy, repr, and union.
- UserDict and ChainMap mapping protocol delegation.
- Counter as dict-like multiset with arithmetic operations.

### Set Protocol

Files:

```
test_set.py
test_collections.py
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/object_protocol.py
torch/_dynamo/variables/builtin.py
```

Representative expected failures:

```
TestSet.test_badcmp
TestSet.test_container_iterator
TestSet.test_copy
TestSet.test_pickling
TestSetSubclass.test_keywords_in_subclass
TestFrozenSet.test_hash_caching
TestBinaryOpsMutating_Set_Set.test_or_with_mutation
TestMethodsMutating_Set_Set.test_update_with_mutation
TestCollectionABCs.test_Set
TestCollectionABCs.test_Set_interoperability_with_real_sets
```

Common themes:

- Binary and inplace set operations.
- Subclass priority and mutation during comparison or iteration.
- Frozenset hash and copy identity behavior.
- Set ABC interoperability with real sets.
- Pickling and iterator behavior, which may often be a hard CPython boundary.

### Iterator, Pickle, Repr, and Implementation-Detail Edges

These recur across the slice and should be triaged separately from normal
container semantics.

Representative themes:

- `_pickle.dumps` on iterators and containers.
- `gc.collect`, `gc.is_tracked`, `sys.getsizeof`, and CPython refcount or GC
  tracking assertions.
- Repr on recursive containers.
- `free_after_iterating`, use-after-free, and mutation during equality tests.
- `importlib.import_module` and other stdlib functions Dynamo currently skips.

Preferred handling:

- Fix normal Python semantics when the test models ordinary user-visible
  behavior.
- Keep or convert to skip only for CPython implementation details that are not
  meaningful Dynamo coverage targets.
- Record hard-boundary decisions in this plan with exact test names and reasons.

## Agent Manager

Operational loop instructions live in `test/dynamo/cpython/agent_manager.md`.
This file is the project plan and the source of truth for scope, active gate,
exit criteria, test targets, and measured progress.

The manager must update this file after each cycle with exact commands, results,
sentinels removed, focused tests added, remaining risks, and the next best
cluster inside the active gate.

## Measurement Commands

Count CPython expected failures by file:

```
find test/dynamo_expected_failures -maxdepth 1 -type f -name 'CPython313-*' \
  | sed 's|.*/CPython313-||; s|-.*||' \
  | sort | uniq -c | sort -nr
```

Count first-slice expected failures:

```
for f in test_list test_tuple test_dict test_set test_defaultdict \
  test_ordered_dict test_deque test_userlist test_userdict test_collections; do
  printf '%s ' "$f"
  find test/dynamo_expected_failures -maxdepth 1 -type f \
    -name "CPython313-$f-*" | wc -l
done
```

Collect first-slice tests:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest --collect-only -q \
  test/dynamo/cpython/3_13/test_list.py \
  test/dynamo/cpython/3_13/test_tuple.py \
  test/dynamo/cpython/3_13/test_dict.py \
  test/dynamo/cpython/3_13/test_set.py \
  test/dynamo/cpython/3_13/test_defaultdict.py \
  test/dynamo/cpython/3_13/test_ordered_dict.py \
  test/dynamo/cpython/3_13/test_deque.py \
  test/dynamo/cpython/3_13/test_userlist.py \
  test/dynamo/cpython/3_13/test_userdict.py \
  test/dynamo/cpython/3_13/test_collections.py
```

Run first-slice tests and write JUnit XML for later histogramming:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  --junitxml=agent_space/cpython_dynamo/first_slice.xml \
  test/dynamo/cpython/3_13/test_list.py \
  test/dynamo/cpython/3_13/test_tuple.py \
  test/dynamo/cpython/3_13/test_dict.py \
  test/dynamo/cpython/3_13/test_set.py \
  test/dynamo/cpython/3_13/test_defaultdict.py \
  test/dynamo/cpython/3_13/test_ordered_dict.py \
  test/dynamo/cpython/3_13/test_deque.py \
  test/dynamo/cpython/3_13/test_userlist.py \
  test/dynamo/cpython/3_13/test_userdict.py \
  test/dynamo/cpython/3_13/test_collections.py
```

Run one active file with Dynamo logs when the error is opaque:

```
TORCH_LOGS="+dynamo,graph_breaks" TORCHDYNAMO_VERBOSE=1 \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
```

## Gates

### G0: Baseline and Triage Harness

Status: current.

Purpose: make the agent loop reliable before changing behavior.

Exit criteria:

- A baseline first-slice run has been captured under `agent_space/cpython_dynamo/`
  with the exact git SHA, command, Python version, and JUnit XML.
- The first-slice expected-failure and skip counts in this file have been
  checked against the current tree and updated if needed.
- The first 20 actionable clusters have been recorded under "Work Queue" with
  exact test names, normalized failure reason, and likely source files.
- At least one single-test repro has been run with its sentinel temporarily
  bypassed and restored.
- No source files or sentinels are changed by this gate.

Recommended first action:

Run the first-slice XML command and build a normalized failure histogram from
the skipped expected failures. Use `agent_space/` for any parser or report.

### G1: Sequence Core

Status: pending.

Purpose: improve list, tuple, deque, and UserList semantics before moving to
larger mapping and set surfaces.

Files in scope:

```
test_list.py
test_tuple.py
test_deque.py
test_userlist.py
```

Initial expected-failure count: 126.

Exit criteria:

- Remove at least 35 expected-failure sentinels across the sequence files.
- Remove no skip sentinels unless the tests are now both passing and cheap.
- Add focused Dynamo regression tests for every non-trivial semantic fix.
- The four sequence files pass under `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q
  --tb=short` with no unexpected successes or new real failures.
- Any remaining sequence expected failures are categorized as one of:
  unsupported stdlib/C boundary, CPython implementation detail, side-effect
  semantic gap, iterator/reconstruction gap, or untriaged.

High-priority clusters:

- `list` and `tuple` repeat, inplace repeat, repr, and contains behavior.
- `deque` getitem, iadd, imul, rotate, reverse, copy, and maxlen behavior.
- `UserList` delegation for getitem, setitem, contains, repeat, append, extend,
  remove, and sort.
- Iterator reconstruction and mutation during iteration where the behavior is
  user-visible.

### G2: Mapping Core

Status: pending.

Purpose: improve dict, defaultdict, and UserDict behavior before tackling the
larger OrderedDict surface.

Files in scope:

```
test_dict.py
test_defaultdict.py
test_userdict.py
```

Initial expected-failure count: 66. Initial skip count: 7, all in `test_dict.py`.

Exit criteria:

- Remove at least 25 expected-failure sentinels across the mapping-core files.
- Do not add new expected-failure or skip sentinels.
- Add focused Dynamo regression tests for every non-trivial semantic fix.
- The three mapping-core files pass under `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q
  --tb=short` with no unexpected successes or new real failures.
- Each remaining `test_dict.py` skip has a recorded reason: slow, CPython
  implementation detail, or real remaining crash/hang risk.

High-priority clusters:

- `ConstDictVariable.update` and mapping protocol update behavior.
- `dict` merge operators and `fromkeys`.
- Dict view operations on keys/items/values.
- `setdefault`, `pop`, `popitem`, reconstruction, and split-dict behavior.
- `defaultdict` repr, copy/deepcopy, recursive repr, pickling, and union.
- `UserDict` read/write/update delegation.

### G3: Set and Frozenset Core

Status: pending.

Purpose: attack the largest repeated cluster in the first slice.

Files in scope:

```
test_set.py
test_collections.py
```

Initial expected-failure count in `test_set.py`: 220.

Exit criteria:

- Remove at least 60 `test_set.py` expected-failure sentinels.
- Add focused Dynamo regression tests for every non-trivial semantic fix.
- `test_set.py` passes under `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short`
  with no unexpected successes or new real failures.
- Remaining `test_set.py` expected failures are grouped by exact protocol gap:
  binary op, inplace op, mutation during comparison, iterator/pickle,
  subclassing, frozenset hash/copy, or CPython implementation detail.

High-priority clusters:

- Binary operations across set/set, set/subclass, subclass/set, and
  subclass/subclass.
- Inplace operations and mutation during operation.
- Frozenset hash and copy identity.
- Set subclass constructor and keyword behavior.
- Set iterator behavior that is not purely CPython implementation detail.

### G4: OrderedDict and Collections Wrappers

Status: pending.

Purpose: cover heavily used collection wrappers after core dict/set behavior has
improved.

Files in scope:

```
test_ordered_dict.py
test_collections.py
```

Initial expected-failure count: 269.

Exit criteria:

- Remove at least 75 expected-failure sentinels across the two files.
- Add focused Dynamo regression tests for every non-trivial semantic fix.
- Both files pass under `PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short` with
  no unexpected successes or new real failures.
- Remaining failures are separated between OrderedDict-specific behavior and
  generic mapping/set/sequence behavior that belongs to earlier gates.

High-priority clusters:

- OrderedDict `update`, `clear`, `delitem`, `pop`, `popitem`, `move_to_end`,
  equality, merge, repr, and iterator behavior.
- Pure-Python versus C OrderedDict parity.
- ChainMap union behavior.
- Counter init, update, subtract, multiset operations, and inplace operations.
- NamedTuple factory, readonly, pickle, defaults, and descriptor behavior.
- Collections ABC `Set`, `Mapping`, `Sequence`, `MutableSequence`, and
  one-trick pony ABCs.

### G5: Cross-Cutting Hard Edges

Status: pending.

Purpose: revisit the recurring failures that may have been deferred during
semantic gates.

Files in scope: all ten first-slice files.

Exit criteria:

- Every remaining expected failure in the first-slice files is categorized with
  an exact reason and a decision: fix now, keep expected failure, convert to
  skip, or ask for human decision.
- At least 20 additional expected-failure sentinels are removed, unless fewer
  than 20 fixable sentinels remain after G1 through G4.
- No test remains in the "untriaged" category.
- The first-slice run completes and its result summary is recorded in this file.

High-priority clusters:

- Pickle and copy behavior that is ordinary Python semantics.
- Recursive repr behavior.
- Iterator reconstruction after graph breaks.
- `gc`, `sys.getsizeof`, refcount, and tracking tests, which may be legitimate
  CPython implementation-detail skips rather than Dynamo coverage targets.
- Stdlib C functions that Dynamo skips but that are common enough to support.

### G6: Full CPython Dynamo Suite Rebaseline

Status: pending.

Purpose: measure the global impact and choose the next domain after the
data-structure campaign.

Files in scope:

```
test/dynamo/cpython/3_13
test/dynamo_expected_failures
test/dynamo_skips
```

Exit criteria:

- Run the full CPython Dynamo suite with:

  ```
  cd test/dynamo/cpython/3_13
  PYTORCH_TEST_WITH_DYNAMO=1 pytest -vs .
  ```

- Record total collected tests, passes, skips, xfails/unexpected successes, and
  true passrate.
- Record expected-failure counts by CPython file after the first-slice campaign.
- Identify the next 10-file slice by expected-failure count and user value.
- Propose the next active gate, leaving it under "Proposed Gate Changes Awaiting
  Human Approval" until approved.

## Work Queue

G0 owns this section until its exit criteria are met. Replace these seed items
with measured clusters from the current tree.

1. Build first-slice XML and normalized histogram.
2. Reproduce `DictTest.test_update` with its expected-failure sentinel
   temporarily bypassed; likely source is `ConstDictVariable.call_method`.
3. Reproduce `ListTest.test_repeat` and `TupleTest.test_repeat`; likely source
   is sequence repeat and `repr(FakeIdVariable)` behavior in list/tuple VTs.
4. Reproduce one `TestBinaryOpsMutating_*` set failure; likely source is
   set binary dispatch, subclass priority, or side-effect handling.
5. Reproduce `TestDefaultDict.test_union`; likely source is
   `DefaultDictVariable` and mapping merge handling.
6. Reproduce `TestBasic.test_getitem` from `test_deque.py`; likely source is
   `DequeVariable` and object protocol indexing.
7. Reproduce `UserListTest.test_getitem` and `UserDictTest.test_update`; likely
   source is user-defined wrapper delegation.
8. Reproduce `TestChainMap.test_union_operators`; likely source is mapping
   protocol and collections wrapper handling.

## Deep Replan Triggers

Run a deep replan instead of continuing the same loop when any of these happens:

- Three consecutive cycles remove no sentinels from the active gate.
- The same failure reason appears in many tests but the source abstraction is
  unclear after targeted repros.
- A proposed fix requires broad changes to bytecode interpretation, exception
  handling, or side-effect modeling.
- A gate's quantitative exit criteria appear unreachable because most remaining
  tests are CPython implementation details.
- A change fixes CPython tests but risks changing normal Dynamo behavior without
  focused regression tests.

Deep replan output should include:

- Active gate and blocker.
- Exact commands run.
- Failure histogram table.
- Top three hypotheses with supporting evidence.
- Recommended next coherent cluster.
- Proposed plan changes, if any, under "Proposed Gate Changes Awaiting Human
  Approval".

## Proposed Gate Changes Awaiting Human Approval

None.

## Cycle Log

No implementation cycles have been run from this plan yet.
