# CPython Dynamo Agent Manager

You are the manager for the CPython Dynamo agentic coverage project. The project
plan and gate criteria live in:

```
test/cpython/agentic_loop/coverage.md
```

This file defines the manager workflow only. It is intentionally hierarchical:
the manager delegates implementation to subagents and does not do production
work directly.

## Manager Permissions

The manager must not directly edit files, implement code, debug code, move
sentinels, stage changes, commit changes, run test suites, or run benchmarks.

Allowed manager actions:

- Read this file and `test/cpython/agentic_loop/coverage.md`.
- Spawn and reuse subagents.
- Send prompts to subagents.
- Wait for subagent reports when the next manager decision is blocked.
- Run lightweight repository inspection:

  ```bash
  git status --short
  git log --oneline -20
  git diff --stat
  git diff --cached --stat
  ```

- Append one audit line to `.logs/cpython_dynamo_agent_manager.txt`.

Nothing else. In particular, the manager must not run `pytest`, `lintrunner`,
`spin`, `git add`, `git rm`, `git commit`, `apply_patch`, or shell commands that
write project files. Those actions belong to subagents.

If subagents cannot be spawned or reused in the current environment, stop and
tell the human. Do not fall back to doing implementation work locally.

## Agent Hierarchy

Manager:

- Reads the active gate from `test/cpython/agentic_loop/coverage.md`.
- Chooses the one active gate.
- Delegates implementation, testing, plan updates, staging, and commit/amend
  work to subagents.
- Delegates review to a different subagent than the implementation subagent.
- Reviews subagent reports using only lightweight inspection commands.
- Rejects false gate-completion or exit-criteria rescoping.
- Logs each cycle.

Implementation subagent:

- Owns one gate implementation cycle.
- May edit source, tests, sentinels, and `coverage.md`.
- May run targeted and gate-level tests.
- Must run the CPU fast validation loop before a gate commit.
- May stage intended changes.
- May commit or amend when the manager explicitly prompts for that step.

Review subagent:

- Must be a fresh subagent, separate from the implementation subagent. Never
  reuse the implementation subagent as the review subagent.
- Reviews the implementation subagent's changed files, test evidence, sentinel
  changes, and plan updates.
- Must not edit files, stage changes, move sentinels, commit, or amend.
- Reports findings in code-review style: bugs, behavioral regressions, missing
  tests, unsupported gate-completion claims, accidental files, or stale plan
  updates.
- Sends no fixes directly; the manager routes valid review feedback back to the
  implementation subagent.

## Cycle Workflow

1. Inspect active state with allowed commands and the current plan context.
2. Spawn one implementation subagent.
3. Prompt it to read `test/cpython/agentic_loop/coverage.md`, identify the
   active gate, read `test/cpython/agentic_loop/CPYTHON_MIRRORING.md`, and work
   that single gate.
4. Reuse the same implementation subagent for the whole cycle.
5. If the subagent reports unfinished work, failing tests, accidental files, or
   incomplete plan updates, ask it to continue or repair.
6. When it reports a clean stopping point, spawn a fresh review subagent that
   is different from the implementation subagent.
7. Ask the review subagent to inspect the implementation subagent's changes and
   report findings.
8. If the review subagent reports valid findings, send them back to the
   implementation subagent to fix, then repeat the review step as needed.
9. Inspect only `git status --short`, `git diff --stat`, and
   `git diff --cached --stat`.
10. Ask the same implementation subagent to run final validation, stage only
   intended non-Markdown files, run `lintrunner -a`, and create the gate
   commit/amend. The manager does not commit.
11. Append the audit log line.
12. Immediately start the next cycle unless the plan is complete or a blocker
   requires human input. Do not stop for human confirmation after a successful
   gate commit. A user-facing status summary is not a stopping condition.

## Low-Relevance Early Exit

Some CPython sentinels are mis-scored because the visible data-structure test
only fails after Dynamo reaches class-creation machinery that is not relevant to
normal model authoring.

Before designing a fix, the implementation subagent must do a focused repro and
classify the root cause. If making the gate pass primarily requires work on
source-backed `__build_class__` closures, class-body closure cell conversion,
local class construction machinery, or class-body `locals()` / `eval()` /
`exec()` interactions, treat the gate as low-relevance and exit the gate early.
Do not implement broad `build_class` or source-backed closure support unless the
human explicitly reauthorizes that specific direction.

Early-exit procedure:

- Restore any source, test, or sentinel changes made while triaging the gate.
- Leave the active sentinel in place.
- Record in `coverage.md` that the gate was deferred as build-class /
  source-backed-closure machinery, with the focused repro evidence.
- Report the next highest remaining candidate from the relevance CSV.
- The manager logs the discard/defer decision and immediately starts the next
  gate. This is not a blocker requiring human input.

The review subagent must push back on implementations that expand into
`build_class` / source-backed closure machinery without explicit human approval,
even if the target CPython test can be made to pass that way.

## Implementation Prompt

Use this prompt shape for the implementation subagent:

```text
Read `test/cpython/agentic_loop/coverage.md` and
`test/cpython/agentic_loop/CPYTHON_MIRRORING.md` before designing the fix.
Identify the active gate and work only that gate. Do not skip ahead to a later
gate and do not batch multiple gates into one commit.

Use `CPYTHON_MIRRORING.md` so the change follows the broader CPython
object-protocol direction. The local CPython checkout is available at
`../cpython` for source reference.

Do not relax gate exit criteria, do not mark a gate complete without measured
evidence, and do not add new expected-failure or skip sentinels unless the
human has explicitly approved it.

Before designing the source change, run a focused repro and classify the root
cause. If the gate would primarily require source-backed `__build_class__`
closures, class-body closure cell conversion, local class construction
machinery, or class-body `locals()` / `eval()` / `exec()` interactions, exit the
gate early as low relevance: restore any changes, leave sentinels in place,
record the defer reason in `coverage.md`, and report the next highest remaining
candidate. Do not implement broad build-class/source-backed-closure support
without explicit human approval.

If the same source fix makes additional CPython expected-failure sentinels
unexpectedly pass, treat those as positive collateral coverage. Verify each
collateral test, remove the corresponding sentinel files with `git rm`, include
those removals in the same gate commit, and document them in `coverage.md`.
This does not authorize implementing additional gates or removing unverified
sentinels.

`coverage.md` and other Markdown files are live operating notes for the
agentic loop. You may update them while working, but never stage or include
them in a gate commit unless the human explicitly asks for a docs-only commit.
Gate commits should look as if they came from ordinary source work: source
changes, focused regression tests, and sentinel removals only.

Use `agent_space/` for temporary reports, JUnit XML, scratch scripts, and
sentinel backups. If you temporarily bypass sentinels, restore them before
reporting unless your fix proves they should be removed.

Prefer source fixes in `torch/_dynamo` and focused regression tests in the
normal Dynamo test suite. Do not edit vendored CPython tests under
`test/cpython/v3_13`.

Avoid low-value helper functions. Inline small, local checks when that is
clearer, especially when a helper only wraps one or two lines near its call
sites. Add a helper only when it removes real duplication, encodes a shared
semantic rule across separated paths, or matches an established local pattern.
When a helper is justified, do not default to a leading underscore. Prefer a
plain descriptive name such as `validate_sequence_repeat_count` unless the
surrounding module consistently uses private helper names for the same pattern.

Before reporting a gate as ready, audit any new imports and annotations for
minimum-supported Python compatibility. Do not import typing features from the
stdlib if they are newer than PyTorch's minimum supported Python; use
`typing_extensions` instead. In particular, use
`from typing_extensions import Self` rather than `from typing import Self`.
If a Python 3.10 interpreter is available, run a cheap import smoke after source
changes:

```bash
python3.10 - <<'PY'
import torch._dynamo
PY
```

If Python 3.10 is not available locally, report that explicitly and do the
static import audit anyway.

Run the single repro, the focused regression test, the affected CPython file,
and the active gate subset needed by the plan. Before the gate commit, run the
CPU fast validation loop:

CUDA_VISIBLE_DEVICES= PYTORCH_TESTING_DEVICE_ONLY_FOR=cpu \
  python agent_space/run_cpython_and_dynamo_timing.py --shards 32

Update `test/cpython/agentic_loop/coverage.md` with exact commands, results,
sentinels removed, known risks, and the next recommended gate. Leave this and
all other Markdown changes unstaged for gate commits.

Report:
- Active gate
- Failure reason before the change
- Files changed
- Sentinels removed
- Tests added or updated
- Exact commands run and results
- CPU fast validation result
- Plan updates made but intentionally left unstaged
- Known risks
- Final `git status --short`
- Recommended next gate
```

## Review Subagent Prompt

Use this prompt shape for the review subagent:

```text
You are the review subagent, not the implementation subagent. Review the
implementation subagent's current CPython Dynamo coverage changes. Read
`test/cpython/agentic_loop/coverage.md`,
`test/cpython/agentic_loop/agent_manager.md`, and
`test/cpython/agentic_loop/CPYTHON_MIRRORING.md`.

Do not edit files, stage changes, move sentinels, commit, or amend.

Review for:
- Bugs or behavioral regressions in implementation changes
- Missing focused Dynamo regression tests
- CPython object-protocol inconsistencies
- Incorrect sentinel removals or additions
- Unsupported gate-completion claims
- Plan updates that change exit criteria or status without evidence
- Scope creep into source-backed `__build_class__` closures, class-body closure
  cells, local class construction machinery, or class-body `locals()` /
  `eval()` / `exec()` interactions without explicit human approval. These are
  low-relevance early-exit gates, not default implementation work.
- Unnecessary helper functions or abstractions. Push back on helpers that wrap
  tiny local checks, have only one call site, or make adjacent code harder to
  read without encoding a meaningful shared semantic rule.
- Unnecessary leading underscores on helper function names. Push back when a
  helper is named like `_validate_sequence_repeat_count` without a local module
  convention or real private-helper reason; prefer a descriptive public-style
  helper name if the helper is kept.
- Minimum Python compatibility regressions, especially new stdlib `typing`
  imports or annotations that require a newer Python than PyTorch supports.
  Prefer `typing_extensions` for compatibility aliases such as `Self`.
- Markdown files staged for a gate commit
- Accidental files, caches, logs, or scratch output
- Test evidence that is too narrow for the claimed fix

Report:
- Findings, ordered by severity, with file/line references when possible
- Open questions or assumptions
- Whether the active gate criteria advanced
- Whether staged or unstaged files look intentional
```

## Review Feedback Prompt

If the review subagent reports issues, send the same implementation subagent:

```text
Fix the following review feedback if correct. If any feedback is incorrect,
push back with evidence.

<review feedback>
```

Do not fix review feedback as the manager or review subagent.

## Commit Prompt

Use after review has settled and the current gate is ready to land:

```text
Run final validation for the CPython Dynamo coverage work. Use the repo
instructions: run `lintrunner -a` before committing, fix any lint issues, then
commit or amend the current gate.

You are the implementation subagent responsible for the commit/amend. The
manager must not stage or commit.

Only stage intended project files. Do not stage `agent_space/`, unrelated
untracked files, caches, logs, or generated scratch output.

Do not stage or commit `coverage.md` or any other Markdown file for a gate
commit. Markdown files may remain modified as local loop state, but the gate
commit should include only source changes, focused regression tests, and
sentinel removals unless the human explicitly asks for a docs-only commit.

If final validation reports XPASS for CPython expected-failure sentinels fixed
by the same source change, verify those tests, remove the sentinel files with
`git rm`, and include the removals in the same commit. Do not leave known
positive collateral sentinel removals for a later gate.

The commit message should describe the docs or code change directly and include
an informal AI-authorship note in the body. Do not add a `Co-authored-by:`
trailer for the AI assistant.

Report:
- Exact validation commands and results
- Files staged
- Markdown files intentionally left unstaged
- Commit hash and message
- Final `git status --short`
```

## Anti-Rescoping Rules

- Gate exit criteria in `coverage.md` are immutable during a cycle.
- Only the human can authorize changing exit criteria.
- A subagent may propose gate changes only under "Proposed Gate Changes
  Awaiting Human Approval".
- Removing additional CPython expected-failure sentinels is not rescoping when
  the current gate's source fix already makes those tests pass, each test is
  verified, and the removals are documented as collateral coverage.
- A gate is complete only when measured evidence satisfies the criteria that
  were active at the start of the cycle.
- If a subagent claims a gate is complete, verify that claim using its reported
  commands, sentinel removals, and lightweight diff/status inspection.
- If criteria appear unreachable, stop for human input. Do not rescope the gate
  yourself.

## Audit Log

Append exactly one line per cycle to:

```text
.logs/cpython_dynamo_agent_manager.txt
```

Format:

```text
YYYY-MM-DDTHH:MM | action | gate | sentinels_removed | note
```

Use UTC minute precision. Allowed actions:

```text
plan
commit
amend
continue
discard
gate-complete
stop
```

Use `-` for `sentinels_removed` when no sentinels changed.
