# M04 Audit — Fix High-Priority CI Silent Failures

**Milestone:** M04  
**Mode:** DELTA AUDIT  
**Range:** b352a3bbbee...5fbb0de6125  
**CI Status:** Skipped (Fork Protection)  
**Refactor Posture:** Behavior-Preserving (CI Configuration Only)  
**Audit Verdict:** 🟢 PASS with Evidence Constraint

---

## 1. Executive Summary

### Wins
- ✅ 5 high-severity silent failure patterns removed (M03-R01 through M03-R05)
- ✅ 31 lines of dead/broken CI configuration removed from trunk.yml
- ✅ INV-060 (CI Critical Path Integrity) established as governance invariant
- ✅ OSSF Scorecards security workflow re-enabled for upstream

### Risks
- ⚠️ Fork CI cannot verify runtime behavior (all checks skipped due to `github.repository_owner` guards)
- ⚠️ Upstream verification deferred until PR to pytorch/pytorch

### Most Important Next Action
- Proceed to M05 (CI Workflow Linting) to continue Phase 1 CI hardening

---

## 2. Evidence Constraint: Fork CI is Guarded

### Why Fork CI is Skipped

All PyTorch CI workflows include guards such as:
```yaml
if: github.repository_owner == 'pytorch'
```

This prevents CI execution on forks (`m-cahill/pytorch`) for:
- Cost control (self-hosted runners)
- Secret protection
- Resource management

### What Cannot Be Proven on Fork

The following cannot be directly observed:
- ❌ Target Determination fails when the TD step fails
- ❌ LLM TD Retrieval job failure is now visible
- ❌ tools-unit-tests fails when pytest fails
- ❌ scorecards workflow runs successfully on upstream

### Verification Strategy

M04 verification is limited to:

1. **Diff-based semantic proof** — The changes are mechanically obvious:
   - Removing `continue-on-error: true` = step/job failures propagate
   - Removing `if: false` = job is no longer permanently skipped
   - Changing `if: false && X` to `if: X` = job becomes conditionally eligible

2. **Deferred upstream verification** — When/if merged upstream:
   - Observe TD failure propagation
   - Observe tools-unit-tests failure on actual test failure
   - Observe scorecards workflow execution

---

## 3. Delta Map & Blast Radius

### What Changed

| Component | Type | Change Description |
|-----------|------|-------------------|
| `target_determination.yml` | Workflow | Removed `continue-on-error: true` from Do TD step |
| `llm_td_retrieval.yml` | Workflow | Removed job-level `continue-on-error`; added clarifying comment |
| `trunk.yml` | Workflow | Removed 31 lines (disabled executorch build+test jobs) |
| `tools-unit-tests.yml` | Workflow | Removed `continue-on-error` from both test steps |
| `scorecards.yml` | Workflow | Removed `false &&` to re-enable for upstream |

### Consumer Surfaces Touched

| Surface | Impact |
|---------|--------|
| CLI | ❌ None |
| API | ❌ None |
| Library | ❌ None |
| Schema | ❌ None |
| File formats | ❌ None |
| CI workflows | ✅ 5 files modified |

### Risky Zones

| Zone | Assessment |
|------|------------|
| Persistence | ❌ Not touched |
| Migrations | ❌ Not touched |
| Concurrency | ❌ Not touched |
| Workflow glue | ✅ Modified (5 workflows) |
| Boundary seams | ❌ Not touched |

### Blast Radius Statement

**Where breakage would show up:** CI workflows only. If these changes cause issues:
- Target Determination step failures would now fail the workflow (visible immediately)
- LLM TD Retrieval job failures would now fail the workflow (visible immediately)
- tools-unit-tests would now fail when pytest fails (visible immediately)
- scorecards would run on upstream (visible on first trigger)

**No production code, tests, or runtime behavior affected.**

---

## 4. Architecture & Modularity Review

### Evaluation

| Question | Assessment |
|----------|------------|
| Boundary violations introduced? | ❌ No — CI config only |
| Coupling added that blocks extraction? | ❌ No — changes are decoupling (removing dead code) |
| Dead abstractions created? | ❌ No — dead code removed |
| Layering leaks? | ❌ No — workflows are top-level automation |
| ADR/doc updates needed? | ✅ Yes — REFACTOR.md updated with M04 entry |

### Disposition

| Category | Items |
|----------|-------|
| **Keep** | All 5 workflow changes |
| **Fix now** | None |
| **Defer** | Upstream verification (M04-V01) |

---

## 5. CI/CD & Workflow Audit

### Determinism Assessment

| Aspect | Status |
|--------|--------|
| Deterministic installs | ✅ Not affected |
| Caching | ✅ Not affected |
| Action pinning | ⚠️ Not addressed (M06 scope) |
| Token permissions | ✅ Not changed |
| Matrix correctness | ✅ Not affected |

### Green-but-Misleading Analysis

**Before M04:**
- 4 workflows could report green while masking real failures
- 1 workflow (scorecards) was completely disabled

**After M04:**
- All 5 identified false-positive patterns corrected
- Failures in TD, LLM TD, and tools-unit-tests will now propagate
- Scorecards will execute on upstream

### CI Root Cause Summary

N/A — No CI failures to diagnose. Changes are preventive hardening.

### Minimal Fix Set

All fixes applied in M04. No additional fixes required.

### Guardrails Added

- **INV-060** — CI Critical Path Integrity: If a correctness-critical step fails, CI must fail visibly
- **Removal comment** — Added to trunk.yml explaining why executorch jobs were removed

---

## 6. Tests, Coverage, and Invariants

### Coverage Delta

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Overall coverage | N/A | N/A | No production code changed |
| Touched packages | N/A | N/A | CI config only |

### New Tests Added

None required. M04 is CI configuration change, not code change.

### Invariant Verification Status

| Invariant | Status | Method |
|-----------|--------|--------|
| **INV-060** (new) | ✅ Introduced | Diff-based semantic proof |
| INV-050 (imports) | ✅ Protected | Not touched |

### Flaky Tests

None introduced or affected.

### End-to-End Verification

Deferred to upstream (M04-V01). Fork CI skipped.

### Snapshot/Golden/Contract Harness

N/A — No code changes requiring behavioral verification.

### Missing Invariants

None identified.

### Missing Tests

None required for CI configuration changes.

### Fast Fixes (≤90 min)

None required.

### New Markers/Tags

None suggested.

---

## 7. Security & Supply Chain

### Dependency Deltas

None. No dependencies added or modified.

### Vulnerability Posture

| Change | Impact |
|--------|--------|
| Scorecards re-enabled | ✅ **Improved** — OSSF security scoring now active on upstream |

### Secrets Exposure Risk

None. No new secrets usage; no secrets removed.

### Workflow Trust Boundary Changes

| Workflow | Change |
|----------|--------|
| scorecards.yml | Now eligible to run on upstream (was disabled) |
| Others | No trust boundary changes |

### SBOM/Provenance Continuity

N/A — Not in M04 scope (planned for M08).

---

## 8. Refactor Guardrail Compliance Check

| Guardrail | Status | Evidence |
|-----------|--------|----------|
| **Invariant declaration** | ✅ PASS | INV-060 introduced |
| **Baseline discipline** | ✅ PASS | Changes from M03 baseline (b352a3bbbee) |
| **Consumer contract protection** | ✅ N/A | No consumer contracts modified |
| **Extraction/split safety** | ✅ N/A | No extraction performed |
| **No silent CI weakening** | ✅ PASS | This milestone *removes* silent CI weakening |

---

## 9. Top Issues (Ranked)

### Issues Found: 0

No issues introduced. All changes are minimal, targeted, and reversible.

### Pre-existing Issue Observed (Out of Scope)

| ID | Severity | Observation | Interpretation | Recommendation | Guardrail | Rollback |
|----|----------|-------------|----------------|----------------|-----------|----------|
| PRE-001 | Low | `lumen-cli-compatible-python39` job in tools-unit-tests.yml is missing the `pytest` command | Job only installs but doesn't test on Python 3.9 | Add pytest command to job | Add test verification | N/A (pre-existing) |

**Note:** This is a pre-existing issue, not introduced by M04. Out of scope for this milestone.

---

## 10. PR-Sized Action Plan

| ID | Task | Category | Acceptance Criteria | Risk | Est |
|----|------|----------|---------------------|------|-----|
| 1 | Merge PR #2 | CI | PR merged; changes on main | Low | 5m |
| 2 | Update REFACTOR.md | Docs | M04 entry marked complete | Low | 10m |
| 3 | Add M04-V01 deferral | Docs | Deferral entry in REFACTOR.md | Low | 5m |
| 4 | Seed M05 folder | Docs | M05 plan and toolcalls templates exist | Low | 5m |

**Status:** All actions completed.

---

## 11. Deferred Issues Registry (Cumulative)

| ID | Issue | Discovered (M#) | Deferred To | Reason | Blocker? | Exit Criteria |
|----|-------|-----------------|-------------|--------|----------|---------------|
| **M04-V01** | Upstream CI verification for M04 changes | M04 | Upstream PR | Fork CI skipped due to `repository_owner` guards | No | TD failure propagates; tools-unit-tests fails on pytest failure; scorecards runs cleanly |
| PRE-001 | tools-unit-tests Python 3.9 job missing pytest | M04 | Future | Pre-existing; out of scope | No | Job includes pytest command |

---

## 12. Score Trend (Cumulative)

| Milestone | Invariants | Compat | Arch | CI | Sec | Tests | DX | Docs | Overall |
|-----------|------------|--------|------|-----|-----|-------|----|------|---------|
| M00 (Baseline) | 3.0 | 4.0 | 4.0 | 4.0 | 3.0 | 3.5 | 3.5 | 3.0 | 3.5 |
| M01 (Import Smoke) | 3.5 | 4.0 | 4.0 | 4.0 | 3.0 | 3.5 | 3.5 | 3.0 | 3.6 |
| M02 (Governance) | 3.5 | 4.0 | 4.0 | 4.0 | 3.0 | 3.5 | 3.5 | 4.0 | 3.7 |
| M03 (CI Audit) | 3.5 | 4.0 | 4.0 | 3.5* | 3.0 | 3.5 | 3.5 | 4.0 | 3.6 |
| **M04 (Silent Failures)** | **4.0** | 4.0 | 4.0 | **3.8** | **3.2** | 3.5 | 3.5 | 4.0 | **3.8** |

**Score Movement:**
- **Invariants:** 3.5 → 4.0 (+0.5) — INV-060 established
- **CI:** 3.5 → 3.8 (+0.3) — 5 silent failure patterns removed
- **Security:** 3.0 → 3.2 (+0.2) — OSSF Scorecards re-enabled
- **Overall:** 3.6 → 3.8 (+0.2) — Incremental improvement from CI hardening

*CI score reduced in M03 when silent failures were discovered; partially recovered in M04.

---

## 13. Flake & Regression Log (Cumulative)

| Item | Type | First Seen (M#) | Current Status | Last Evidence | Fix/Defer |
|------|------|-----------------|----------------|---------------|-----------|
| (none) | — | — | — | — | — |

**No flakes or regressions introduced in M04.**

---

## 14. Diff-Based Semantic Proof

### M03-R01: target_determination.yml

**Change:** Remove `continue-on-error: true` from "Do TD" step (line 60)

```diff
       - name: Do TD
         id: td
-        continue-on-error: true
         env:
```

**Semantic effect:** When the TD script fails (non-zero exit), the step now fails. With `continue-on-error: true` removed, step failure propagates to job failure.

**GitHub Actions behavior:** Step `conclusion` changes from `success` (soft-fail) to `failure`. Workflow reports failure.

---

### M03-R02: llm_td_retrieval.yml

**Change 1:** Remove job-level `continue-on-error: true` (line 26)

```diff
   llm-retrieval:
     # Don't run on forked repos
     if: github.repository_owner == 'pytorch'
     runs-on: "${{ needs.get-label-type.outputs.label-type }}linux.4xlarge"
-    continue-on-error: true
     needs: get-label-type
```

**Semantic effect:** Job failure now propagates to workflow failure. Previously, entire job could fail silently.

**Change 2:** Add clarifying comment to step-level `continue-on-error`

```diff
       - name: Run Retriever
         id: run_retriever
-        continue-on-error: true  # ghstack not currently supported due to problems getting git diff
+        # Best-effort step: ghstack not currently supported due to problems getting git diff.
+        # This step is informational and does not gate correctness; failure should not block CI.
+        continue-on-error: true
```

**Semantic effect:** Step-level soft-fail preserved with explicit documentation that it is intentional and informational.

---

### M03-R03: trunk.yml

**Change:** Remove disabled executorch jobs (lines 417-447)

```diff
-  linux-jammy-py3-clang15-executorch-build:
-#    if: ${{ needs.job-filter.outputs.jobs == '' || contains(needs.job-filter.outputs.jobs, ' linux-jammy-py3-clang15-executorch ') }}
-    name: linux-jammy-py3-clang15-executorch
-    uses: ./.github/workflows/_linux-build.yml
-    needs:
-      - get-label-type
-      - job-filter
-    if: false # Has been broken for a while
-    with:
-      ...
-    secrets: inherit
-
-  linux-jammy-py3-clang15-executorch-test:
-    ...
+
+  # NOTE: linux-jammy-py3-clang15-executorch-build and linux-jammy-py3-clang15-executorch-test
+  # were removed in M04 (2026-02-08). They had been disabled with `if: false` and comment
+  # "Has been broken for a while". If executorch CI is needed, re-add as a working job.
```

**Semantic effect:** No runtime change (jobs were already disabled). Removes dead configuration and false signal. `if: false` is no longer present in Tier 1 workflow.

---

### M03-R04: tools-unit-tests.yml

**Change:** Remove `continue-on-error: true` from both test steps (lines 39, 65)

```diff
       - name: Run tests
-        continue-on-error: true
         run: |
           set -ex
           ...
           pytest -v -s .ci/lumen_cli/tests/*
```

**Semantic effect:** When pytest fails (non-zero exit), the step fails. Job failure propagates to workflow failure. Tests no longer silently pass.

---

### M03-R05: scorecards.yml

**Change:** Remove `false &&` from job condition (line 24)

```diff
-    if: false && github.repository == 'pytorch/pytorch'  # don't run on forks
+    if: github.repository == 'pytorch/pytorch'  # don't run on forks
```

**Semantic effect:** Job condition changes from `false` (never runs) to `github.repository == 'pytorch/pytorch'` (runs on upstream). OSSF security scoring becomes active on upstream repository.

---

## 15. Quality Gates Evaluation

| Gate | Status | Evidence |
|------|--------|----------|
| **Invariants** | ✅ PASS | INV-060 established; no existing invariants violated |
| **CI Stability** | ⚠️ UNKNOWN | Cannot run on fork; semantic proof provided |
| **Tests** | ✅ N/A | No test files modified |
| **Coverage** | ✅ N/A | No production code modified |
| **Compatibility** | ✅ PASS | CI configuration only; no API/behavior changes |
| **Workflows** | ✅ PASS | Minimal changes; no new jobs; no permission changes |
| **Security** | ✅ PASS | No secrets; scorecards re-enabled improves posture |
| **DX/Docs** | ✅ PASS | Removal comment added in trunk.yml |

Any **FAIL** must include a one-step fix or defer entry: **None required.**

---

## 16. Conclusion

**Audit Verdict:** 🟢 PASS

M04 executed to specification:
- 5 risks addressed with minimal, targeted changes
- One commit per risk (granular rollback)
- No scope creep
- No unrelated files touched
- Evidence constraint documented with semantic proof

**Recommendation:** Milestone closed. Proceed to M05.

---

## Machine-Readable Appendix

```json
{
  "milestone": "M04",
  "mode": "delta",
  "posture": "preserve",
  "commit": "5fbb0de6125",
  "range": "b352a3bbbee...5fbb0de6125",
  "verdict": "green",
  "quality_gates": {
    "invariants": "pass",
    "compatibility": "pass",
    "ci": "unknown",
    "tests": "pass",
    "coverage": "pass",
    "security": "pass",
    "dx_docs": "pass",
    "guardrails": "pass"
  },
  "issues": [],
  "deferred_registry_updates": [
    {
      "id": "M04-V01",
      "deferred_to": "Upstream PR",
      "reason": "Fork CI is skipped due to repository_owner guards",
      "exit_criteria": "Observe CI failure propagation and scorecards execution on upstream"
    }
  ],
  "score_trend_update": {
    "invariants": 4.0,
    "compat": 4.0,
    "arch": 4.0,
    "ci": 3.8,
    "sec": 3.2,
    "tests": 3.5,
    "dx": 3.5,
    "docs": 4.0,
    "overall": 3.8
  },
  "evidence_constraint": {
    "type": "fork_ci_guarded",
    "verification_method": "diff_semantic_proof",
    "upstream_verification_required": true
  }
}
```

---

**End of M04 Audit**
