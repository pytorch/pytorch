# CPython Dynamo Agent Manager

You are managing the CPython Dynamo coverage project. The project plan and gate
criteria live in `test/dynamo/cpython/cpython_test_coverage.md`; this file only
describes how to run the agent loop.

Do not treat this file as the source of truth for scope. At the start of every
cycle, read `test/dynamo/cpython/cpython_test_coverage.md`, identify the active
gate, and work only on a coherent slice that advances that gate's exit criteria.

## Core Rules

- Gate exit criteria in `test/dynamo/cpython/cpython_test_coverage.md` are
  immutable during a cycle. Do not relax, rewrite, or reinterpret them to claim
  progress.
- Do not mark a gate complete unless the recorded criteria are satisfied by
  measured evidence from the current tree.
- Do not edit vendored CPython tests under `test/dynamo/cpython/3_13` unless the
  human explicitly asks for an import/update of those tests.
- Do not add new expected-failure or skip sentinels unless the human explicitly
  approves.
- Do not commit unless the human explicitly asks. If commits are authorized,
  follow the repository commit instructions and run `lintrunner -a` first.
- Use `agent_space/` for temporary reports, scripts, sentinel backups, and XML.
  Do not commit files from `agent_space/`.
- If a tool such as `python`, `pip`, `pytest`, or `spin` is missing, follow the
  repository environment instructions before trying alternatives.

## Cycle Shape

1. Read `test/dynamo/cpython/cpython_test_coverage.md` and identify the active gate.
2. Pick one coherent cluster from the active gate or Work Queue.
3. Measure the current behavior with exact targeted commands.
4. Temporarily move only the relevant sentinel files to
   `agent_space/cpython_sentinel_backups/` for diagnosis, then restore them
   before the cycle ends unless the fix proves they can be removed.
5. Add or update focused Dynamo regression tests outside the vendored CPython
   tree for every non-trivial user-visible semantic fix.
6. Implement the smallest coherent Dynamo change.
7. Run the single repro, the focused regression test, the affected CPython file,
   and the active gate subset.
8. Remove only sentinels proven fixed by passing tests.
9. Update `test/dynamo/cpython/cpython_test_coverage.md` with evidence,
   remaining risks, and the next best cluster.
10. Append one audit line to `.logs/cpython_dynamo_agent_manager.txt`.

When using agent teams, reuse one implementation agent for the whole cycle so it
keeps context. Use separate implementation agents only for independent clusters
with disjoint write sets. When agent teams are unavailable, run the same cycle
locally and keep the same evidence requirements.

## Implementation Prompt

Use this prompt shape for an implementation agent:

```
Read `test/dynamo/cpython/cpython_test_coverage.md` and identify the active
gate. Work on one coherent cluster that advances that gate's exit criteria. Do
not edit gate exit criteria, do not mark the gate complete without measured
evidence, and do not skip ahead to a later gate.

Use `agent_space/` for temporary files. If you temporarily bypass expected
failure or skip sentinels, restore them before reporting unless your fix proves
they should be removed. Do not add new sentinels.

Prefer source fixes in `torch/_dynamo` and focused regression tests in the
normal Dynamo test suite. Do not edit vendored CPython tests.

Report:
- Active gate
- Cluster worked
- Failure reason before the change
- Files changed
- Sentinels removed
- Tests added or updated
- Exact commands run and results
- Plan updates made
- Known risks
- Final `git status --short`
- Recommended next cluster
```

## Sentinel Workflow

For a single expected failure key such as:

```
CPython313-test_dict-DictTest.test_update
```

diagnose with:

```
mkdir -p agent_space/cpython_sentinel_backups/test/dynamo_expected_failures
mv test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_update \
  agent_space/cpython_sentinel_backups/test/dynamo_expected_failures/
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
mv agent_space/cpython_sentinel_backups/test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_update \
  test/dynamo_expected_failures/
```

If the implementation fixes the test, remove the sentinel with:

```
git rm test/dynamo_expected_failures/CPython313-test_dict-DictTest.test_update
```

For skip sentinels, use the same discipline with `test/dynamo_skips/`. Only
remove a skip when the test is now passing and cheap enough to keep enabled.

## Validation Levels

Single repro:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py::DictTest::test_update
```

Affected file:

```
PYTORCH_TEST_WITH_DYNAMO=1 pytest -q --tb=short \
  test/dynamo/cpython/3_13/test_dict.py
```

Focused Dynamo regression test:

```
pytest -q --tb=short test/dynamo/test_dicts.py -k <new_test_name>
```

Active gate subset:

Use the file list from the active gate in
`test/dynamo/cpython/cpython_test_coverage.md`.

Full CPython Dynamo suite:

```
cd test/dynamo/cpython/3_13
PYTORCH_TEST_WITH_DYNAMO=1 pytest -vs .
```

The full suite is slow. Run it only when the active gate asks for it, when a
change affects shared interpreter behavior, or when rebaselining.

## Plan Update Rules

After each implementation cycle, update
`test/dynamo/cpython/cpython_test_coverage.md` with:

- Cluster completed.
- Exact sentinels removed.
- Focused tests added.
- Exact commands and results.
- New failure clusters discovered.
- Remaining risks.
- Next best cluster inside the active gate.

Do not edit a gate's exit criteria during this pass. If criteria appear wrong or
unreachable, add a proposal under "Proposed Gate Changes Awaiting Human
Approval" and leave the active criteria unchanged.

## Audit Log

Append exactly one line per cycle to:

```
.logs/cpython_dynamo_agent_manager.txt
```

Format:

```
YYYY-MM-DDTHH:MM | action | gate | sentinels_removed | note
```

Use UTC minute precision. Allowed actions:

```
plan
commit
amend
continue
discard
gate-complete
deep-replan
stop
```

Use `-` for `sentinels_removed` when no sentinels changed.

## Deep Replan

Trigger a deep replan when:

- Three consecutive cycles remove no sentinels from the active gate.
- The same failure reason appears across many tests and the source abstraction
  is still unclear after targeted repros.
- The likely fix requires broad bytecode, exception, or side-effect modeling
  changes.
- A gate's quantitative exit criteria appear unreachable because most remaining
  tests are CPython implementation details.
- A fix improves CPython tests but risks changing normal Dynamo behavior without
  a clear focused test strategy.

Deep replan output should include:

- Active gate and blocker.
- Exact commands run.
- Failure histogram table.
- Top three hypotheses with supporting evidence.
- Recommended next coherent cluster.
- Proposed plan changes, if any, under "Proposed Gate Changes Awaiting Human
  Approval" in `test/dynamo/cpython/cpython_test_coverage.md`.
