# CPython Dynamo Agentic Coverage Plan

Status: current gate is G1.

Goal: improve `PYTORCH_TEST_WITH_DYNAMO=1` coverage for CPython tests by
working the highest-value expected failures first. The first ten gates come
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

Measured baseline:

| Suite | Target | Wall time | Passed | Skipped | Expected failures |
|---|---|---:|---:|---:|---:|
| cpython_dynamo | `test/cpython/v3_13` with `PYTORCH_TEST_WITH_DYNAMO=1` | 72.9s | 2279 | 2461 | 1 |
| dynamo | `test/dynamo` | 217.7s | 9790 | 673 | 10 |

Combined wall time: 217.7s.

## Validation Commands

Run one target test with Dynamo:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_constructors
```

Run the affected CPython file:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py
```

Run the fast CPU validation loop:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python agent_space/run_cpython_and_dynamo_timing.py --shards 32
```

Use Dynamo logs only for opaque single-test repros:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
TORCH_LOGS="+dynamo,graph_breaks" TORCHDYNAMO_VERBOSE=1 \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_constructors
```

Before claiming a gate complete, the implementation subagent must report:

- target sentinel removed;
- focused regression test added or updated when the fix is semantic;
- target CPython test passes with the sentinel removed;
- affected CPython file has no new real failures;
- fast CPU validation loop ran and only baseline failures remain, except for
  the expected pass/skip improvement from this gate;
- `lintrunner -a` passed before the gate commit.

## Gates

### G1: List Constructor Semantics

Status: current.

Target sentinel:

```
CPython313-test_list-ListTest.test_constructors
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_constructors
```

Relevance score: 92.6.

Baseline failure kind:

```
Failed to trace list()
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/object_protocol.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_list-ListTest.test_constructors`.
- Add focused Dynamo regression coverage for the supported list-constructor
  behavior if the fix is non-trivial.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_list.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G2: List Repeat

Status: pending.

Target sentinel:

```
CPython313-test_list-ListTest.test_repeat
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_repeat
```

Relevance score: 91.2.

Baseline failure kind:

```
Failed to trace builtin operator
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/object_protocol.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_list-ListTest.test_repeat`.
- Add focused Dynamo regression coverage for list repetition semantics.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_list.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G3: Tuple Constructor Semantics

Status: pending.

Target sentinel:

```
CPython313-test_tuple-TupleTest.test_constructors
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_tuple.py::TupleTest::test_constructors
```

Relevance score: 91.2.

Baseline failure kind:

```
Failed to trace builtin operator
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/object_protocol.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_tuple-TupleTest.test_constructors`.
- Add focused Dynamo regression coverage for tuple construction semantics.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_tuple.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G4: Dict Update

Status: pending.

Target sentinel:

```
CPython313-test_dict-DictTest.test_update
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_update
```

Relevance score: 89.4.

Baseline failure kind:

```
Unsupported method call
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/object_protocol.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_update`.
- Add focused Dynamo regression coverage for `dict.update`.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_dict.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G5: General Mapping Update

Status: pending.

Target sentinel:

```
CPython313-test_dict-GeneralMappingTests.test_update
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::GeneralMappingTests::test_update
```

Relevance score: 89.4.

Baseline failure kind:

```
Unsupported method call
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/object_protocol.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_dict-GeneralMappingTests.test_update`.
- Add focused Dynamo regression coverage for mapping-style update behavior.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_dict.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G6: Dict Subclass Mapping Update

Status: pending.

Target sentinel:

```
CPython313-test_dict-SubclassMappingTests.test_update
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::SubclassMappingTests::test_update
```

Relevance score: 89.4.

Baseline failure kind:

```
Unsupported method call
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/object_protocol.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_dict-SubclassMappingTests.test_update`.
- Add focused Dynamo regression coverage for dict-subclass update behavior.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_dict.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G7: List Sort

Status: pending.

Target sentinel:

```
CPython313-test_list-ListTest.test_sort
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_list.py::ListTest::test_sort
```

Relevance score: 89.0.

Baseline failure kind:

```
Attempted to call function marked as skipped
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/functions.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_list-ListTest.test_sort`.
- Add focused Dynamo regression coverage for supported `list.sort` behavior.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_list.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G8: Dict Literal Constructor

Status: pending.

Target sentinel:

```
CPython313-test_dict-DictTest.test_literal_constructor
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_dict.py::DictTest::test_literal_constructor
```

Relevance score: 88.2.

Baseline failure kind:

```
Attempted to call function marked as skipped
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/builtin.py
torch/_dynamo/variables/object_protocol.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_literal_constructor`.
- Add focused Dynamo regression coverage for the supported literal-constructor
  behavior.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_dict.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G9: Tuple Repeat

Status: pending.

Target sentinel:

```
CPython313-test_tuple-TupleTest.test_repeat
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_tuple.py::TupleTest::test_repeat
```

Relevance score: 88.2.

Baseline failure kind:

```
Failed to trace builtin operator
```

Likely source areas:

```
torch/_dynamo/variables/lists.py
torch/_dynamo/variables/object_protocol.py
torch/_dynamo/variables/builtin.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_tuple-TupleTest.test_repeat`.
- Add focused Dynamo regression coverage for tuple repetition semantics.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_tuple.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

### G10: DefaultDict Union

Status: pending.

Target sentinel:

```
CPython313-test_defaultdict-TestDefaultDict.test_union
```

Target test:

```bash
CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/cpython/v3_13/test_defaultdict.py::TestDefaultDict::test_union
```

Relevance score: 86.2.

Baseline failure kind:

```
Failed to trace builtin operator
```

Likely source areas:

```
torch/_dynamo/variables/dicts.py
torch/_dynamo/variables/user_defined.py
torch/_dynamo/variables/object_protocol.py
```

Exit criteria:

- Remove `test/dynamo_expected_failures/CPython313-test_defaultdict-TestDefaultDict.test_union`.
- Add focused Dynamo regression coverage for `defaultdict` union behavior.
- The target test passes with `PYTORCH_TEST_WITH_DYNAMO=1`.
- The full `test_defaultdict.py` CPython file has no new real failures.
- Fast CPU validation loop passes modulo documented baseline failures.
- Commit exactly this gate.

## Proposed Gate Changes Awaiting Human Approval

Use this section only when an implementation subagent believes a gate is too
broad, too narrow, stale, or blocked by unrelated infrastructure.

No proposed changes.
