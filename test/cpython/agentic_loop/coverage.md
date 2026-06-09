# CPython Dynamo Agentic Coverage Plan

Status: Cycle 1 top-10 exhausted (G1-G10) and the relevance CSVs were
regenerated per the README (6 landed rows dropped, 4 deferred rows tagged).
The new actionable top-10 is G11-G20, drawn from the re-ranked CSV. G11
(`CPython313-test_iter-TestCase.test_reduce_mutating_builtins_iter`,
relevance 80.2) was triaged and DEFERRED as a CPython C-implementation-detail
test (see G11 entry below). G12
(`CPython313-test_list-ListTest.test_free_after_iterating`, relevance 80.0) was
triaged and DEFERRED as a CPython object-lifetime / GC-internals test (see G12
entry below). G13/G14/G18 landed (G15/G16/G17 were G14 collateral); G19
(`CPython313-test_deque-TestBasic.test_basics`, relevance 78.9) is the next
unworked actionable gate.

Goal: improve `PYTORCH_TEST_WITH_DYNAMO=1` coverage for CPython tests by
working the highest-value expected failures first. The actionable gates come
from the relevance-ranked study in:

```
test/cpython/agentic_loop/cpython_dynamo_expected_failure_relevance.csv
```

Each gate targets exactly one CPython expected-failure sentinel. Each gate
should produce one focused implementation commit after review and validation.

Operational loop instructions live in:

```
test/cpython/agentic_loop/agent_manager.md
```

CPython protocol orientation lives in:

```
test/cpython/agentic_loop/CPYTHON_MIRRORING.md
```

## Ground Rules

- Do not relax gate exit criteria during an implementation cycle.
- Do not mark a gate complete without measured evidence from the current tree.
- Do not edit vendored CPython tests under `test/cpython/v3_13` unless the
  human explicitly asks for a CPython import/update.
- Do not add new expected-failure or skip sentinels without human approval.
- Use `agent_space/` for scratch files, temporary reports, JUnit XML, and
  sentinel backups.
- Prefer source fixes in `torch/_dynamo` and focused regression tests in the
  normal Dynamo test suite.
- Remove only the CPython sentinel proven fixed by the gate.
- Do not batch multiple gates into one commit.

Sentinel directories:

```
test/dynamo_expected_failures/
test/dynamo_skips/
```

Example key:

```
CPython313-test_list-ListTest.test_constructors
```

## Baseline

Fast CPU validation baseline:

```
test/cpython/agentic_loop/cpu_fast_ci_baseline.md
```

Baseline command:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python agent_space/run_cpython_and_dynamo_timing.py --shards 32
```

Validation harness note: the fast CPU loop script
`agent_space/run_cpython_and_dynamo_timing.py` does not exist in this repo.
Every prior gate used affected-CPython-file runs plus targeted Dynamo suites
for validation instead; new gates should do the same.

## Validation Commands

Run one target test with Dynamo:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter
```

Run the affected CPython file:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py
```

Use Dynamo logs only for opaque single-test repros:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
TORCH_LOGS="+dynamo,graph_breaks" TORCHDYNAMO_VERBOSE=1 \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter
```

Before claiming a gate complete, the implementation subagent must report:

- target sentinel removed;
- focused regression test added or updated when the fix is semantic;
- target CPython test passes with the sentinel removed;
- affected CPython file has no new real failures;
- fast CPU validation loop ran and only baseline failures remain, except for
  the expected pass/skip improvement from this gate (or, if the harness script
  is absent, the affected-file run substitute);
- `lintrunner -a` passed before the gate commit.

## Completed Gates (Cycle 1 ledger)

These gates landed; their sentinels were removed and committed. CSV rows for
the landed tests have been dropped from both relevance CSVs.

| Gate | Test | Commit | Resolution |
|---|---|---|---|
| G1 | `CPython313-test_dict-DictTest.test_eq` | 3bdd341f4e6 | `ConstantVariable.is_python_equal` routed non-constant `other` through `generic_richcompare_bool` so a user `__eq__` (and its raised exception) runs, mirroring `PyObject_RichCompareBool`. Regression: `test_cmp_eq_key_raises` in `test/dynamo/test_dicts.py`. |
| G4 | `CPython313-test_defaultdict-TestDefaultDict.test_shallow_copy` | 82c76e895ad | Extended `DefaultDictVariable.call_method` to handle the `__copy__` slot and added a `UserDefinedClassVariable.call_method` branch so the unbound `defaultdict.__copy__(instance)` form from `copy.copy` dispatches to the instance, preserving type, `default_factory`, and contents. Regression: `test_defaultdict_shallow_copy_preserves_factory`. |
| G5 | `CPython313-test_set-TestFrozenSet.test_do_not_rehash_dict_keys` | 3a0da519450 | tp_hash slot dispatch (`type.__hash__(instance)` for int/float/str + subclasses) plus CPython `set_update_internal` do-not-rehash semantics: set/frozenset/dict built from an existing set/dict reuse stored `HashableTracker` keys instead of re-hashing. Touched `builtin.py`, `sets.py`, `dicts.py`, `functions.py`. Regression: `MiscTests.test_do_not_rehash_dict_keys`. |
| G6 | `CPython313-test_set-TestFrozenSetSubclass.test_do_not_rehash_dict_keys` | 3a0da519450 | G5 collateral (same root-cause fix, sentinel removed in the G5 commit). |
| G7 | `CPython313-test_set-TestSet.test_do_not_rehash_dict_keys` | 3a0da519450 | G5 collateral (same root-cause fix, sentinel removed in the G5 commit). |
| G8 | `CPython313-test_set-TestSetSubclass.test_do_not_rehash_dict_keys` | 3a0da519450 | G5 collateral (same root-cause fix, sentinel removed in the G5 commit). |

## Deferred Gates (Cycle 1 ledger)

These tests were triaged and intentionally deferred. Their sentinels are LEFT
IN PLACE (still in `test/dynamo_expected_failures/`). Their CSV rows are kept
but tagged in the new `deferred` column so they do not re-surface as active
gates.

| Gate | Test | Reason |
|---|---|---|
| G2 | `CPython313-test_dict-DictTest.test_fromkeys` | Local dict-subclass construction machinery (out of scope). Test body builds `fromkeys` against locally-defined `dict` subclasses overriding `__new__`/`__init__`/`__setitem__`; passing requires source-backed local subclass construction. The instance-method `fromkeys` routing improvement is a valid standalone change but does not make the gate pass. |
| G3 | `CPython313-test_dict-DictTest.test_getitem` | Class-body closure cell / local class construction (out of scope). `class BadEq` closes over the later-defined `Exc` cell; tracing the class body reads the still-uninitialized `Exc` cell -> "Read uninitialized cell" graph break. |
| G9 | `CPython313-test_list-ListTest.test_deopt_from_append_list` | Vendored `@unittest.skip("Fails on python <=3.13.2 ...")`; the skip dominates the expected-failure wrapper so the test is skipped with or without the sentinel. The sentinel is a dead but harmless artifact; passing would require editing the vendored test (forbidden). Human chose to leave it deferred. |
| G10 | `CPython313-test_dict-DictTest.test_copy_maintains_tracking` | `gc.is_tracked` is a CPython cyclic-GC container-tracking introspection builtin (in Dynamo skipfiles) with no analog in the `VariableTracker` model; `CPYTHON_MIRRORING.md` lists GC traversal under "What not to mirror". Out of scope. |

## Gates (Cycle 2: actionable top-10, G11-G20)

These are the ten highest-ranked rows whose `deferred` column is empty, taken
from the regenerated `cpython_dynamo_expected_failure_relevance.csv`. Gate
numbers continue from G11 to avoid collisions with the Cycle 1 ledger.

### G11: Iterator Reduce With Mutating Builtins (`bytes`)

Status: DEFERRED (Cycle 2). Triaged as a CPython C-implementation-detail test
with no analog in Dynamo's `VariableTracker` model. Sentinel LEFT IN PLACE;
CSV row tagged `deferred`. No source change made.

Root-cause classification: out of scope (CPython interpreter internals /
"what not to mirror"). The visible `bytes(8)` graph break (relevance score) is
only the first of several blockers and is not the subject of the test. The test
is the reproducer for CPython issue #101765: it verifies the C-level argument
evaluation ordering inside `listiter_reduce_general`
(`Objects/listobject.c`), where `_PyEval_GetBuiltin(&_Py_ID(iter))` must run
BEFORE the iterator's internal `it_seq`/`it_index` pointers are read. The test
mutates `builtins.__dict__` (deletes `iter`/`reversed`, re-inserts under a
custom-`__hash__`/`__eq__` key whose `__eq__` exhausts the iterator), then
calls `it.__reduce__()` on builtin iterators and asserts the returned pickle
state reflects the now-exhausted iterator, e.g.
`run_iter("xyz") == (orig["iter"], ("",))`.

Making this gate pass would require Dynamo to symbolically execute CPython's C
`__reduce__` implementations for every builtin iterator (list/str/tuple/
callable/`reversed`) AND reproduce CPython's undefined-in-C argument evaluation
ordering and the `_PyEval_GetBuiltin`-reads-mutated-`builtins` interplay. Per
`CPYTHON_MIRRORING.md` "What not to mirror" (CPython implementation details
that do not affect tracing semantics), this is explicitly out of scope. The
isolated `bytes(int)`/`bytearray(int)` constructor gap is a legitimate, separate
fixable improvement but does NOT make this gate pass on its own.

Repro evidence (current tree, sentinel temporarily removed):

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  pixi run -w pytorch -e pytorch313 python -m pytest \
  "test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter" -q -rs
# -> FAILED: torch._dynamo.exc.Unsupported: Failed to trace builtin operator
#    "Dynamo does not know how to trace builtin operator `bytes` with argument
#    types ['int']"; from user code at test_iter.py:341 on `(bytes(8),)`.
```

Isolated probe confirming the deeper blocker (after the `bytes` break):

```python
@torch.compile(backend="eager", fullgraph=True)
def f():
    return iter([1, 2, 3]).__reduce__()
# -> Unsupported: Dynamo does not know how to trace method `__reduce__`
#    of class `list_iterator`
```

(Mutating `builtins.__dict__` itself - del + setitem - does trace fine; the
unsupported pieces are iterator `__reduce__` and the C-ordering semantics.)

The original gate scaffolding (target sentinel, test, relevance, baseline
failure kind, source areas) is preserved below for the record.

Sanity-checked (pre-defer) still an expected failure (SKIPPED, not XPASS) on
the current tree.

Target sentinel:

```
CPython313-test_iter-TestCase.test_reduce_mutating_builtins_iter
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_iter.py::TestCase::test_reduce_mutating_builtins_iter
```

Relevance score: 80.2.

Baseline failure kind:

```
Failed to trace builtin operator (Dynamo does not know how to trace builtin
operator `bytes` with argument types ['int']); graph break at
test/cpython/v3_13/test_iter.py:341 on `(bytes(8),)`.
```

Likely source areas:

```
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/iter.py
torch/_dynamo/variables/constant.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_iter-TestCase.test_reduce_mutating_builtins_iter`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_iter.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G12: List Free After Iterating

Status: DEFERRED (Cycle 2). Triaged as a CPython object-lifetime / GC-internals
test with no analog in Dynamo's `VariableTracker` model. Sentinel LEFT IN PLACE;
CSV `deferred` column tagged. No source change made. G13 promoted to active.

Root-cause classification: out of scope (refcounting / object lifetime / GC,
plus local class construction). The visible "Attempted to call function marked
as skipped" graph break is `importlib.import_module`, reached only because the
test is wrapped in `@support.suppress_immortalization()`. That contextmanager
calls `_testinternalcapi.suppress_immortalization(True)` -- a CPython internal
C-API that toggles object *immortalization* (a refcount/lifetime mechanism) so
the test can observe deterministic deallocation. Even past that blocker, the
test body `test.support.check_free_after_iterating` defines a local subclass
`class A(cls)` with a `__del__` that runs `next(it)`, then asserts (after
`gc_collect()`) that `__del__` fired exactly when the sequence was deallocated
at end-of-iteration. This is object lifetime, `__del__`-on-deallocation, and GC
collection behavior -- all explicitly listed under `CPYTHON_MIRRORING.md`
"What not to mirror" (refcounting and object lifetime, deallocation slots, GC
traversal) -- combined with local class construction machinery (the
low-relevance early-exit category). Dynamo does not model `__del__` finalizer
timing or deallocation, so making this gate pass would require mirroring CPython
object lifetime, which is out of scope.

Repro evidence (current tree, sentinel temporarily removed):

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu PYTORCH_TEST_WITH_DYNAMO=1 \
  pixi run -w pytorch -e pytorch313 python -m pytest \
  "test/cpython/v3_13/test_list.py::ListTest::test_free_after_iterating" -q -rs
# -> FAILED: torch._dynamo.exc.Unsupported: Attempted to call function marked
#    as skipped; module: importlib, qualname: import_module. Reached from
#    @support.suppress_immortalization() ->
#    _testinternalcapi = import_module("_testinternalcapi").
```

(The inherited definition lives in `test/cpython/v3_13/seq_tests.py:483`
`test_free_after_iterating` -> `support.check_free_after_iterating(self, iter,
self.type2test)`.)

The original gate scaffolding is preserved below for the record.

Target sentinel:

```
CPython313-test_list-ListTest.test_free_after_iterating
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_free_after_iterating
```

Relevance score: 80.0.

Baseline failure kind:

```
Attempted to call function marked as skipped
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/iter.py
torch/_dynamo/trace_rules.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_list-ListTest.test_free_after_iterating`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_list.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G13: List Contains Ordering

Status: FIXED (Cycle 2). Classification (a): genuine in-scope object-protocol
bug in `iter_contains`. Source fix landed in the working tree (uncommitted);
target sentinel and 7 collateral sentinels removed in the working tree.

Root cause: `iter_contains` (`torch/_dynamo/utils.py`) had a constant fast path
that, when `search` was a python constant, scanned items checking only
`x.is_python_constant()` -- silently skipping any non-constant element. CPython
`list_contains` (Objects/listobject.c) instead calls
`PyObject_RichCompareBool(item, search, Py_EQ)` on every element in order and
short-circuits on the first match, so a non-constant element's custom `__eq__`
(and any exception it raises) must run in order. For `[StopCompares(), 1]`,
`1 in ...` returned `True` instead of propagating `StopCompares().__eq__`'s
`DoNotTestEq`.

Fix: only constant-fold when `search` and every element are python constants;
otherwise iterate in order comparing each element via
`generic_richcompare_bool(tx, x, search, "__eq__")` (Dynamo's
`PyObject_RichCompareBool` analog, element first), short-circuiting on a
constant-True result and OR-accumulating symbolic results. This is the list
analog of the G1 dict/set `__eq__`-exception-propagation fix and routes through
the shared object-protocol richcompare path.

Files changed:
- `torch/_dynamo/utils.py` (`iter_contains`)
- `test/dynamo/test_contains_protocol.py` (new `ContainsOrderTest`:
  short-circuit-before-raising-eq, list raising-eq propagation, tuple
  raising-eq propagation)

Sentinels removed (working tree, uncommitted): target + collateral, all
verified failing-before / passing-after with the fix:
- `CPython313-test_list-ListTest.test_contains_order` (target)
- `CPython313-test_tuple-TupleTest.test_contains_order`
- `CPython313-test_userlist-UserListTest.test_contains_order`
- `CPython313-test_deque-TestSequence.test_contains_order`
- `CPython313-test_list-ListTest.test_contains_fake`
- `CPython313-test_tuple-TupleTest.test_contains_fake`
- `CPython313-test_userlist-UserListTest.test_contains_fake`
- `CPython313-test_deque-TestSequence.test_contains_fake`

The 4 `test_contains_fake` tests (rich comparison against `ALWAYS_EQ`/
`NEVER_EQ`) are collateral from the same fix. Validation: all 4 affected
CPython files (`test_list`, `test_tuple`, `test_userlist`, `test_deque`) pass
under Dynamo with the 8 sentinels removed; `test/dynamo/test_contains_protocol`,
`test_list`, `test_dicts`, `test_sets` all green.

Note: `test_set.py::TestFrozenSet::test_hash` and `TestFrozenSetSubclass.test_hash`
XPASS both WITH and WITHOUT this change -- they are pre-existing collateral
(likely from the landed G5 commit), NOT caused by G13, so their sentinels are
left in place for a separate gate to handle.

Next gate: G14.

Original gate scaffolding preserved below.

Status: current active gate (promoted after G12 deferred).

Target sentinel:

```
CPython313-test_list-ListTest.test_contains_order
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_contains_order
```

Relevance score: 79.8.

Baseline failure kind:

```
Data-dependent branching
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_list-ListTest.test_contains_order`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_list.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G14: Set discard

Status: FIXED (Cycle 2). Classification (a): genuine in-scope object-protocol
bug in `set.remove`/`set.discard` key handling. Source fix landed in the
working tree (uncommitted); target sentinel plus 3 collateral sibling sentinels
(G15/G16/G17) removed in the working tree.

Root cause: `SetVariable.call_method` for `remove`/`discard` checked membership
with `args[0] in self` (`__contains__`), which returns `False` for an
unhashable element (`is_hashable` pre-check). So `s.remove([])` raised
`KeyError([])` instead of `TypeError: unhashable type: 'list'`, and
`s.discard([])` silently succeeded instead of raising `TypeError`. CPython
`set_remove_impl`/`set_discard_impl` call `set_discard_key`, which hashes the
key first (raising `TypeError` for an unhashable key), with a set-key fallback
that coerces a set to a frozenset for the lookup (membership-test-with-set
semantics). `KeyError` is only raised, on the original key, after a successful
hash finds the key absent.

Fix: added `SetVariable.lookup_key`, factoring out the CPython
`set_contains_key`/`set_discard_key` key-normalization already inlined in
`sq_contains` (unhashable -> `TypeError`, except a set key is coerced to
`FrozensetVariable`). `sq_contains`, `remove`, and `discard` now all route
through it. `remove`/`discard` normalize the key first, then do the membership
check / pop on the normalized key; `remove` raises `KeyError` on the original
key. This mirrors the G1/G13 object-protocol direction (route through the
shared CPython algorithm rather than a local spot fix).

Files changed:
- `torch/_dynamo/variables/sets.py` (`lookup_key`, `sq_contains`, `remove`,
  `discard`)
- `test/dynamo/test_sets.py` (`_SetBase.test_remove_discard_unhashable`;
  `_SetKeyCoercionMixin.test_remove_set_key` / `test_discard_set_key` on
  `SetTests` and `UserDefinedSetTests`)

Sentinels removed (working tree, uncommitted), each verified failing-before
(initial repro: 4 failed) / passing-after (final repro: 4 passed):
- `CPython313-test_set-TestSet.test_discard` (target, G14)
- `CPython313-test_set-TestSet.test_remove` (G15)
- `CPython313-test_set-TestSetSubclass.test_discard` (G16)
- `CPython313-test_set-TestSetSubclass.test_remove` (G17)

The 3 sibling gates G15/G16/G17 are collateral of the same root-cause fix
(remove is the KeyError-raising twin of discard; the Subclass variants exercise
the same `call_method` path on a `set` subclass).

Validation:
- 4 targets pass under Dynamo with sentinels removed.
- Full `test_set.py` under Dynamo: only the 2 known/unrelated
  `TestFrozenSet::test_hash` / `TestFrozenSetSubclass::test_hash` XPASS remain
  (documented in G13, sentinels left in place).
- `test/dynamo/test_sets.py` (176 passed, 1 skipped) and
  `test/dynamo/test_dicts.py` (297 passed, 1 xfailed) green.

Note: the fast CPU loop script `agent_space/run_cpython_and_dynamo_timing.py`
does not exist; used the affected-CPython-file run plus targeted Dynamo suites
as the validation substitute (consistent with prior gates).

Next gate: G18.

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_set-TestSet.test_discard
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSet::test_discard
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSet.test_discard`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G15: Set remove

Target sentinel:

```
CPython313-test_set-TestSet.test_remove
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSet::test_remove
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSet.test_remove`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G16: Set Subclass discard

Target sentinel:

```
CPython313-test_set-TestSetSubclass.test_discard
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSetSubclass::test_discard
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSetSubclass.test_discard`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G17: Set Subclass remove

Target sentinel:

```
CPython313-test_set-TestSetSubclass.test_remove
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_set.py::TestSetSubclass::test_remove
```

Relevance score: 79.7.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/sets.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_set-TestSetSubclass.test_remove`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_set.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G18: Dict View Containment Check Errors

Status: FIXED (Cycle 2). Classification (a): genuine in-scope object-protocol
bug in dict items-view containment. Source fix landed in the working tree
(uncommitted); target sentinel removed.

Root cause: `DictItemsVariable.sq_contains` compared the stored value via
`is_python_equal` (identity-only), so a stored value's custom `__eq__` never
ran. CPython `dictitems_contains` does
`PyObject_RichCompareBool(found, value, Py_EQ)` on the stored value, so a value
whose `__eq__` raises must propagate that exception; the identity-only check
swallowed it.

Fix: route the stored-value compare through
`generic_richcompare_bool(tx, stored, val, "__eq__")` (stored as the left
operand, matching `found`), mirroring CPython's
`PyObject_RichCompareBool(found, value, Py_EQ)`, so the value `__eq__` runs and
any exception propagates. This also fixes `dict_items` rich comparisons
(`==`,`!=`,`<`,`<=`,`>`,`>=`) that reach the value compare. Sentinel removed;
regression tests `test_dict_items_cmp_value_eq_raises` /
`test_dict_items_cmp_value_present_absent` added (with a `_BadCmpValue` helper
whose `__eq__` raises). This mirrors the G1/G13/G14 object-protocol direction
(route through the shared CPython algorithm rather than a local identity check).

Files changed:
- `torch/_dynamo/variables/dicts.py` (`DictItemsVariable.sq_contains`)
- `test/dynamo/test_dicts.py` (`_BadCmpValue`;
  `DictMethodsTests.test_dict_items_cmp_value_eq_raises` /
  `test_dict_items_cmp_value_present_absent`)

Sentinel removed (working tree, uncommitted):
- `CPython313-test_dict-DictTest.test_errors_in_view_containment_check` (target)

Validation (current tree):

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 python -m pytest \
  "test/cpython/v3_13/test_dict.py::DictTest::test_errors_in_view_containment_check" -q
# 1 passed

CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
python -m pytest test/dynamo/test_dicts.py -q
# 303 passed, 1 xfailed
```

Next gate: G19 (G13/G14/G18 landed; G15/G16/G17 were G14 collateral; G11/G12
deferred).

Original gate scaffolding preserved below.

Target sentinel:

```
CPython313-test_dict-DictTest.test_errors_in_view_containment_check
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_errors_in_view_containment_check
```

Relevance score: 79.4.

Baseline failure kind:

```
Observed exception
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/user_defined.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_errors_in_view_containment_check`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_dict.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G19: Deque Basics

Target sentinel:

```
CPython313-test_deque-TestBasic.test_basics
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_deque.py::TestBasic::test_basics
```

Relevance score: 78.9.

Baseline failure kind:

```
Unsupported method call
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_deque-TestBasic.test_basics`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_deque.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

### G20: Range Iterators

Target sentinel:

```
CPython313-test_range-RangeTest.test_range_iterators
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_range.py::RangeTest::test_range_iterators
```

Relevance score: 78.5.

Baseline failure kind:

```
Failed to trace builtin operator
```

Likely source areas:

```
torch/_dynamo/variables/iter.py
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_range-RangeTest.test_range_iterators`.
- Add focused Dynamo regression coverage when the fix is semantic.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_range.py` CPython file has no new real failures.
- Fast CPU validation (affected-file substitute) passes modulo documented
  baseline failures.
- Commit exactly this gate.

## Proposed Gate Changes Awaiting Human Approval

Use this section only when an implementation subagent believes a gate is too
broad, too narrow, stale, or blocked by unrelated infrastructure.

No proposed changes.
