# Coverage-Augmented Target Determination

## Context

PyTorch CI today has two simultaneous failure modes:

1. **Running too much.** Every PR triggers all build+test configs. Build cost
   dominates total CI cost.
2. **Skipping things we should have run.** TD's existing top-25% cutoff is a
   single threshold over a sum of weak signals.

The intent: collect *real coverage data periodically on main* and use it both
as a stronger heuristic and (later, behind hard prerequisites) as the input
to a preflight job that prunes the build matrix BEFORE jobs spawn.

Load-bearing safety property: **stale, missing, or unauthenticated coverage
degrades to "run more," never "skip more."**

## Approach in two stages

**Stage A (sections 1-3, can ship in months):** Replace `Profiling` with a
stronger `Coverage` heuristic that issues *positive scores only*. No tiered
selection, no negative scores, no preflight, no job-level skipping. Pure
upgrade to the existing system: same selection model, same 25% top-K cutoff,
just better signal quality at the top.

**Stage B (sections 4-6, gated by Phase 0 prerequisites):** Add tiered
selection with negative scores, preflight, and job-level skipping. Requires
an authoritative test→config mapping, a signed/provenanced map publication,
and resolved mergebot semantics. Without these, Stage B cannot ship safely;
shipping it without them risks silent regressions to main.

The split is deliberate: Stage A captures the easier and lower-risk wins
(better test prioritization within already-spawned jobs) while Stage B
investments are scoped, sequenced, and justified by Stage A's measured
benefit.

## Phase 0 — Prerequisites

These must complete before *any* implementation work starts. None ship
user-visible changes, but each is load-bearing for some later phase.

1. **Mergebot contract.** Investigate `pytorch/test-infra` mergebot
   (`trymerge.py`) to confirm how Skipped jobs are treated for merge
   gating. Engage a test-infra owner. Document actual semantics. Required
   for Stage B / Phase 4.
2. **Per-test coverage cost benchmark.** Run the chosen coverage path
   against a representative subset (~50 test files) on a 24xlarge to
   extrapolate nightly wall time. PyTorch's test set is dynamically
   discovered (`tools/testing/discover_tests.py:19-100`); use the live
   discovery to size N. If isolated per-test coverage exceeds ~12 hrs,
   switch to weekly cadence and 21-day staleness threshold. Required for
   Stage A / Phase 1.
3. **Test→config mapping.** For each test, determine which CI configs and
   shards exercise it. The current configuration logic is spread across
   `.github/workflows/_linux-test.yml:350-353`,
   `.ci/pytorch/test.sh:206-209,2154-2311`, and `test/run_test.py` filter
   logic — there is no single source of truth. Phase 0 must produce a
   deterministic mapping artifact (likely: replay the matrix declarations
   in `pull.yml` and the conditional logic in `test.sh` to compute, per
   `(build-environment, config, shard)`, the set of test files that
   actually run). Required for Stage B's shard carve-out and preflight
   pruning.
4. **Signed map publication path.** Establish a provenance scheme for
   `td_coverage_map.json` so a compromised or wrong-branch test-infra
   publication cannot mass-skip tests. Existing `import_test_stats.py:42-80,178-185`
   fetches unsigned raw GitHub URLs with 3-hour mtime caching. Options:
   GitHub Actions OIDC + `actions/attest-build-provenance`, signed
   manifests via `cosign`, or a hash-chain anchored in the test-infra
   commit. Required for Stage B / Phase 3 (negative scores).
5. **Build artifact consumers audit.** Audit downstream consumers of
   `_linux-build.yml`'s `test-matrix` output (`.github/workflows/_linux-build.yml:150-153,478-481`)
   to determine whether emitting an empty matrix breaks anything beyond
   the test-job skip path. Required before Stage B / Phase 4.
6. **First successful coverage map.** Land `coverage-collection.yml` and
   run it manually until at least one valid map publishes. Required for
   Stage A / Phase 1.

## Stage A — Positive-only Coverage heuristic

### 1. Coverage data pipeline

**Collection workflow.** `.github/workflows/coverage-collection.yml`, new.
Cron `17 7 * * *` UTC nightly on main HEAD only, plus `workflow_dispatch`.
Idempotent: prelude step exits early if the published manifest's
`commit_sha` matches HEAD.

**Driver.** `tools/code_coverage/collect_for_td.py`, new. **This is not a
thin wrapper.** Existing `oss_coverage.py:11-19` calls `summarize_jsons` once
across all test outputs, and `summarize_jsons.py:34-38,137-159,195-223`
accumulates coverage globally without per-test attribution. Concrete new
work required:

- A per-test runner that, for each test file in the live test set
  (`tools/testing/discover_tests.py`), spawns an isolated subprocess
  running `coverage run` and the test, and collects only that test's
  `.coverage` data.
- For the clang/llvm path, emit one `.profraw` per test, then process each
  separately with `llvm-cov export` — do not merge.
- For the gcc path, reset gcov counters between tests via `__gcov_reset()`
  or run each test in a fresh build dir.
- A new `summarize_per_test.py` adjacent to `summarize_jsons.py` that emits
  per-test JSONs without merging.
- Sharded parallel collection: split the test set across N matrix workers
  (suggest 8) to keep wall time reasonable. Phase 0 benchmark sizes N.

**Granularity: file-level.** Function/line-level explode to ~100MB and have
mangled-name stability problems for native code. File-level matches the
existing `test-file-ratings.json` granularity.

**Path normalization.** All paths use repo-relative form. New utility
`tools/testing/target_determination/path_normalization.py` is shared by
collector and heuristic. The clang/llvm and gcov paths emit different forms
(absolute, build-relative, source-relative); normalize at the collector.

**Schema (`td_coverage_map.json`):**

```
{
  "schema_version": 1,
  "collected_at": "2026-05-09T07:17:00Z",
  "commit_sha": "...",
  "coverage_config": "linux-jammy-py3.10-gcc11-cpu",
  "test_to_files": { "test_ops": ["aten/src/ATen/native/UnaryOps.cpp", ...] },
  "file_to_tests": { "aten/src/ATen/native/UnaryOps.cpp": ["test_ops", ...] }
}
```

`coverage_config` is informational in Stage A (no shard carve-out yet) but
required for Stage B.

**Codegen handling — Stage A.** Do NOT attempt to model codegen
input→output mapping. Instead, list ALL codegen inputs in the
always-relevant manifest (Stage B § below), so any codegen-input edit
triggers full CI. This is the conservative-but-simple v1; precision can
improve later.

**Publishing.** Push to `pytorch/test-infra` `generated-stats` branch.
Stage A reads it via the existing unsigned-fetch mechanism in
`tools/stats/import_test_stats.py:42-80`; this is acceptable in Stage A
because the heuristic is positive-score-only — a malicious map can elevate
some tests but cannot suppress them. **In Stage B this is no longer
acceptable; see Phase 0 prerequisite #4.**

### 2. The `Coverage` heuristic — replaces `Profiling`

**File.** `tools/testing/target_determination/heuristics/coverage.py`, new.
`__init__.py:44` swaps `Profiling()` → `Coverage()`. Keep `Profiling` source
in-tree for one rollout cycle.

**Stage A score model.** Per test, given the changed-file set:

- **Touch** (any changed file ∈ `test_to_files[t]`): score `+0.5`.
- **All other cases**: score `0`.

That's it. No negative scores in Stage A. The selection model is unchanged
(`get_top_per_tests(25)`), and the only effect of Coverage is to push
genuinely-affected tests above the 25% cutoff with high confidence —
which is the failure mode the user described as "skipping things we
should have run."

**`normalize_ratings` (`tools/testing/target_determination/heuristics/utils.py:148-169`)
cannot be reused** even for positive scores if any test maps to score 0,
because `min_rating <= 0` triggers `AssertionError`. The Stage A heuristic
constructs `TestPrioritizations` directly via `set_test_score` for the
touched tests only.

**No R1 fix needed.** `EditedByPR`
(`tools/testing/target_determination/heuristics/edited_by_pr.py:41-45`)
already scores any changed `test/*.py` file via `query_changed_files`,
including newly-added ones. There is no problem to fix.

### 3. Stage A rollout

- **Phase 1A (3-4 weeks).** Land `Coverage` heuristic with `trial_mode=True`.
  Diagnostic comparison of `aggregated` vs `aggregated_trial` to verify the
  signal does what we expect. No user-visible change.
- **Phase 2A (2 weeks).** Flip to production. Remove `Profiling`. Watch
  test selection metrics; require 4 weeks of clean operation before any
  Stage B work begins.

Stage A delivers most of the "skipping things we should have run" fix
without taking on the integrity, mapping, or job-spawn risks of Stage B.

## Stage B — Tiered selection + preflight

Stage B does not start until Phase 0 prerequisites #1, #3, #4, and #5 are
complete AND Stage A has 4 weeks of clean production operation.

### 4. Tiered selection with negative Coverage scores

Adds two new conditions to the `Coverage` heuristic:

- **Proven-untouched** (test has a coverage entry, none of its files match,
  AND test is in a config exercised by `coverage_config` per Phase 0
  mapping): score `-0.25`.
- **Unknown** (no coverage entry, OR `now - collected_at > staleness
  threshold`, OR test is in a config NOT exercised by `coverage_config`,
  OR a changed file is unmapped): score `0`.

Tiered selection model added to `TestPrioritizations`
(`tools/testing/target_determination/heuristics/interface.py`):

```python
def get_tiered_selection(self, fallback_pct: int) -> tuple[list[TestRun], list[TestRun]]:
    """Tier A (positive) + top fallback_pct of Tier B (zero) run; Tier C (negative) skip."""
```

`test/run_test.py:2244` switches to `get_tiered_selection(15)` when
`enable_td`.

**Aggregation truth table.** `Filepath`'s `-1.0` docs-only path
(`tools/testing/target_determination/heuristics/filepath.py:124-129`) interacts
with negative Coverage scores. Tests:

| Scenario | Filepath | EditedByPR | Prev. Failed | Coverage | Sum | Tier |
|---|---|---|---|---|---|---|
| Docs-only PR, every test | -1.0 | 0 | 0 | 0 | -1.0 | C |
| C++ edit, test covers it | 0 to 0.25 | 0 to 1.0 | 0 | +0.5 | 0.5 to 1.75 | A |
| C++ edit, test does not, in CPU shard | 0 to 0.25 | 0 | 0 | -0.25 | -0.25 to 0 | B or C |
| C++ edit, test does not, in CUDA shard | 0 to 0.25 | 0 | 0 | 0 | 0 to 0.25 | B |
| YAML edit (codegen) | full-CI fallback | full-CI | full-CI | full-CI | full | (manifest) |
| New test file added | 0 to 0.25 | +1.0 | 0 | 0 | 1.0 to 1.25 | A |

### 5. Always-relevant manifest

`tools/testing/target_determination/preflight_always_relevant.txt`. Any
match short-circuits preflight to `full`. Includes ALL codegen inputs:

```
^setup\.py$
^pyproject\.toml$
^requirements.*\.txt$
^CMakeLists\.txt$
^.+\.cmake$
^cmake/.*
^\.ci/.*
^\.github/.*
^tools/testing/.*
^tools/code_coverage/.*
^tools/stats/.*
^tools/setup_helpers/.*
^torchgen/.*
^third_party/.*\.(c|cc|cpp|cu|h|hpp)$
^c10/.*
^aten/CMakeLists.*
^aten/src/ATen/native/native_functions\.yaml$
^aten/src/ATen/native/tags\.yaml$
^aten/src/ATen/native/ts_native_functions\.yaml$
^aten/src/ATen/native/cuda/.*
^aten/src/ATen/native/mps/.*
^aten/src/ATen/native/xpu/.*
^aten/src/ATen/native/.*\.cu$
^aten/src/ATen/native/.*\.cuh$
^aten/src/ATen/templates/.*
^tools/autograd/derivatives\.yaml$
^tools/autograd/templates/.*
^torch/csrc/cuda/.*
^torch/csrc/distributed/.*
^torch/csrc/jit/.*
```

The list of codegen inputs comes from
`tools/setup_helpers/generate_code.py:19-20,49-76,214-231` and
`cmake/Codegen.cmake:240-242`. Phase 0 task: re-derive this list
programmatically and add a CI check that fails if a new codegen input is
added without updating the manifest.

### 6. Preflight — job-level skipping

**Realistic savings ceiling.** `selected-test-configs` filters at
config-name granularity (`.github/scripts/filter_test_configs.py:206-225`).
Configs are coarse: `default`, `distributed`, `inductor`, `slow`, etc.
Skipping `default` means no default tests run for that build, regardless
of how few are affected. This caps the savings: most PRs touch enough
that `default` survives. Real wins come from skipping less-common
configs (`distributed` for non-distributed PRs, `inductor` for
non-inductor PRs, etc.).

**New workflow.** `.github/workflows/preflight.yml`, called from `pull.yml`
*before* every build job.

**Implementation.** `tools/testing/preflight.py`, new:

1. `query_changed_files()` —
   `tools/testing/target_determination/heuristics/utils.py:76-92`. Uses
   `git diff --name-only`, so add/delete/rename status is not available.
   This is fine for coverage lookup but rules out diff-filter-based
   refinements.
2. Match changed files against `preflight_always_relevant.txt` (§5). Hit
   ⇒ emit `preflight-mode=full`.
3. Verify map signature/provenance via Phase 0 prerequisite #4. Failure
   ⇒ `preflight-mode=full`.
4. Fetch `td_coverage_map.json`. Missing or stale ⇒ `preflight-mode=full`.
5. For each changed file, look up `file_to_tests`. Union ⇒
   `affected_tests`.
6. Per build, intersect `affected_tests` with the test→config map (Phase 0
   prerequisite #3). For each `(build, config)`, if no affected test runs
   in that config AND `coverage_config` covers that config, drop the
   config from the build's `selected-test-configs`.

**Conservative mode (default).** Skip TEST shards, KEEP BUILD. The
existing `_linux-build.yml:244-245` (EC2) and `_linux-build.yml:533-534`
(ARC/OSDC) skip Build when matrix is empty — too aggressive for
conservative mode. Add a new `inputs.preserve-build: bool` to
`_linux-build.yml`; both Build steps' `if:` guards become
`(... existing ...) || inputs.preserve-build == 'true'`. Preflight emits
`preserve-build: 'true'` in conservative mode.

**Aggressive mode.** Opt-in `ci-skip-build-on-clean-coverage` label.
Skips Build too, only when the diff is fully covered by Coverage AND the
existing `reuse-old-whl` mechanism applies. Spelling out the actual diff
required:

- `.github/actions/reuse-old-whl/reuse_old_whl.py:117-130` `ok_changed_file`
  whitelist is currently `torch/*.py` (non-csrc), `test/*.py`,
  `docs/*.{md,rst}`. Aggressive mode generalizes this: instead of a
  static allowlist, accept any changed file proven untouched by Coverage
  AND not in the always-relevant manifest. The function gains a
  data-driven path; the static path remains for non-aggressive PRs.
- This is a tightly-scoped change to the wheel-reuse logic, not just a
  preflight output flag.

**Fail-safe.** Every uncertainty path emits empty
`selected-test-configs`, which `filter_test_configs.py:206-214,625-633`
treats as "keep all." Wrap the preflight Python script with
`continue-on-error: true` AND emit empty default outputs if the script
fails. Test the fail-safe via fault injection in CI.

**Action-yml plumbing.** Adding new label-gated behaviors requires
changes in TWO places:
- `.github/actions/filter-test-configs/action.yml` — declare the labels.
- `.github/scripts/filter_test_configs.py:546-551` — recognize and act
  on them.

### 7. Stage B rollout

- **Phase 3B (3-4 weeks).** Add negative scores to `Coverage` heuristic.
  Switch `test/run_test.py:2244` to `get_tiered_selection(20)`. Drop to
  18%, then 15% in week-long increments contingent on shadow metrics.
- **Phase 4B (4 weeks).** Land preflight in conservative mode. 5% PR
  sample initially, then 100%.
- **Phase 5B (open).** Aggressive build skipping behind opt-in label
  with modified `reuse_old_whl.py`. Default-on only after sustained 0%
  job-level false-skip across opt-in cohort for ≥4 weeks.

## 8. Critical risks (with defenses)

- **R1 — New-test false-skip.** Resolved as a non-issue; `EditedByPR`
  already covers added test files. No code change required.
- **R2 — Codegen file untouched.** Resolved by listing all codegen inputs
  in always-relevant manifest (§5). Phase 0 audit re-derives the list
  programmatically from `tools/setup_helpers/generate_code.py:19-20,49-76,214-231`
  and `cmake/Codegen.cmake:240-242`.
- **R3 — CUDA test skipped because CPU coverage runner doesn't see it.**
  Defense: §4 shard carve-out — but only valid in Stage B with the
  Phase 0 test→config map. Until that map exists, no negative scores
  ship. Stage A (positive-only) is immune.
- **R4 — Aggregation edge cases.** Defense: §4 truth table and synthetic
  Phase 1 fixtures.
- **R5 — Mergebot accepts skipped-by-preflight as green erroneously.**
  Defense: Phase 0 prerequisite #1.
- **R6 — Config-blind false skip.** A test's behavior under `slow`,
  `distributed`, CUDA, ROCm, XPU, `dynamo_wrapped`, or `inductor` differs
  from CPU. Defense: §4's shard carve-out denies negative scores to
  unrepresented configs. Coverage map's `coverage_config` is the
  authority. Phase 0 #3 (test→config map) is the missing piece.
- **R7 — Build artifact consumers.** Empty matrix changes
  `_linux-build.yml`'s outputs (lines 150-153, 478-481), which downstream
  workflows consume. Defense: Phase 0 prerequisite #5 audit; if any
  consumer breaks on empty matrix, fix before Phase 4B.
- **R8 — Malicious or corrupt map.** A bad `file_to_tests` entry could
  selectively suppress tests; the proposed anomaly detector itself reads
  the same channel. Defense: Phase 0 prerequisite #4 (signed publication
  path); Stage A's positive-only model is naturally immune (a bad map
  cannot suppress, only elevate); Stage B blocked until provenance
  scheme is in place.
- **R9 — Coverage-run flakiness.** Tests that mutate global state, fork
  subprocesses, or use multiprocessing produce unstable per-test
  coverage attribution. Defense: per-test isolation per §1; tests known
  to be coverage-unstable get explicit `0` scores via an opt-out list
  (new file `tools/testing/target_determination/coverage_excluded_tests.txt`)
  populated as instabilities are discovered.
- **R10 — `selected-test-configs` granularity is too coarse.** Skipping
  is binary at config-name level; cannot skip "tests A and B in config
  default" without skipping all of `default`. Defense: acknowledge the
  savings ceiling (§6); pursue config splitting (e.g., `default_python`
  vs `default_cpp`) as a separate workstream if savings are inadequate.

## 9. Open questions

1. **Coverage runner cost (Phase 0 #2).** If >12 hrs nightly, switch to
   weekly + 21-day staleness threshold. Is weekly acceptable given the
   freshness-vs-stability trade-off?
2. **CUDA/ROCm/XPU kernel coverage.** Stage A is CPU-only; tests in
   GPU/distributed shards don't appear in `test_to_files`. Stage A still
   works (positive-only), but Stage B's negative scores cannot apply to
   those shards without GPU-shard coverage runs. Acceptable?
3. **Mergebot owner (Phase 0 #1).** Who in `pytorch/test-infra` to engage?
4. **Provenance scheme (Phase 0 #4).** OIDC-attested artifacts vs. cosign
   vs. hash-chain. Which fits PyTorch's current security posture?
5. **Test→config map maintenance (Phase 0 #3).** Is the mapping derivable
   programmatically from `pull.yml` + `test.sh` + `run_test.py`, or must
   it be authored manually? If derivable, where does the deriver live?
6. **CLI flag rename.** `--target-determination 25` semantics change
   under tiered selection. Rename to `--target-determination-fallback-pct`?
7. **Config splitting (R10).** Do we want to split coarse configs like
   `default` into finer-grained ones, or accept the savings ceiling?

## 10. Critical files

**New:**

- `tools/testing/target_determination/heuristics/coverage.py`
- `tools/testing/target_determination/path_normalization.py`
- `tools/testing/target_determination/preflight_always_relevant.txt` (Stage B)
- `tools/testing/target_determination/coverage_excluded_tests.txt` (Stage A, R9)
- `tools/testing/preflight.py` (Stage B)
- `tools/code_coverage/collect_for_td.py`
- `tools/code_coverage/package/tool/summarize_per_test.py`
- `.github/workflows/coverage-collection.yml`
- `.github/workflows/preflight.yml` (Stage B)

**Modify:**

- `tools/testing/target_determination/heuristics/__init__.py:44` — swap
  `Profiling()` → `Coverage()` (Stage A).
- `tools/testing/target_determination/heuristics/interface.py` — add
  `get_tiered_selection` near line 129 (Stage B).
- `tools/stats/import_test_stats.py:42-80,178-185` — add
  `get_td_coverage_map_json` for Stage A; add provenance verification
  for Stage B.
- `tools/testing/do_target_determination_for_s3.py` — invoke new fetch.
- `test/run_test.py:2238-2244` — switch to `get_tiered_selection(15)`
  (Stage B only).
- `.github/workflows/_linux-build.yml:244-245,533-534` — add
  `preserve-build` input to BOTH EC2 and ARC/OSDC build paths (Stage B).
- `.github/actions/reuse-old-whl/reuse_old_whl.py:117-130` — generalize
  `ok_changed_file` (Stage B / Phase 5B).
- `.github/actions/filter-test-configs/action.yml` AND
  `.github/scripts/filter_test_configs.py:546-551` — add
  `ci-no-coverage-skip` and `ci-skip-build-on-clean-coverage` labels.
- `.github/workflows/pull.yml` — add `preflight` job, wire
  `selected-test-configs` from outputs into each build call (Stage B).

**Verified reusable:**

- `.github/workflows/_linux-build.yml:50` — `selected-test-configs` input.
- `.github/workflows/_linux-test.yml:125-129` — matrix-empty skip.
- `tools/stats/import_test_stats.py:42` — `fetch_and_cache` (Stage A);
  must be hardened for Stage B.
- `tools/testing/target_determination/heuristics/utils.py:76-92` —
  `query_changed_files` (note: name-only, no status info).
- `tools/testing/target_determination/heuristics/edited_by_pr.py:41-45` —
  already handles new test files; do NOT modify.

**Verified NOT reusable:**

- `tools/testing/target_determination/heuristics/utils.py:148-169` —
  `normalize_ratings` rejects `min_rating <= 0`.
- `tools/code_coverage/oss_coverage.py:11-19` and
  `tools/code_coverage/package/tool/summarize_jsons.py:34-38,137-159,195-223`
  — produce aggregate map only; per-test attribution is substantial new
  code.

## 11. Verification

**Phase 0:**

- Mergebot semantics document approved by test-infra owner.
- Coverage cost benchmark on 50-test sample with extrapolation to full
  test set.
- Test→config map produced as a JSON artifact; spot-check against 10
  random `(build, config, shard)` tuples.
- Provenance scheme prototype: produce one signed map; verify signature
  via a separate workflow before consumption.
- Build-output consumers list with confirmation each handles empty
  matrix.

**Stage A (positive-only):**

- Unit-test `Coverage` heuristic with synthetic maps for the touch case
  and the no-touch case.
- Phase 1A trial-mode: `aggregated_trial` should differ from `aggregated`
  primarily by promoting touch-positive tests above the 25% line.
  Spot-check 50 PRs.
- Phase 2A production: monitor existing CI metrics for regressions in
  test selection quality.

**Stage B (tiered + preflight):**

- Unit-test for §4 truth table (every row).
- Unit-test for shard carve-out: with `coverage_config="cpu-only"`, a
  CUDA test gets `0` not `-0.25` even when no diff overlap.
- Synthetic-PR adversarial cases for R3, R6, R8.
- Phase 4B fault injection: deliberately break the preflight Python
  script → empty `selected-test-configs` → all builds run. Repeat with
  broken signature verification.
- Synthetic docs-only PR: `Filepath`'s `-1.0` stacks with Coverage `0`;
  total -1.0; Tier C; correct.
- Synthetic codegen YAML edit: always-relevant manifest hits; preflight
  `full`; all jobs run.
- Phase 5B canary: opt-in label + Python-only docstring fix → BUILD
  skips, `reuse-old-whl` provides wheel, tests pass.
