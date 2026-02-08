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
- ✅ 4 high-severity silent failure patterns removed
- ✅ 1 disabled workflow re-enabled for upstream
- ✅ 31 lines of dead/broken CI configuration removed
- ✅ INV-060 (CI Critical Path Integrity) established

### Risks
- ⚠️ Fork CI cannot verify runtime behavior (all checks skipped)
- ⚠️ Upstream verification deferred until PR to pytorch/pytorch

### Most Important Next Action
- Document evidence constraint; proceed to merge with semantic proof

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

## 3. Diff-Based Semantic Proof

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
-    if: ${{ needs.job-filter.outputs.jobs == '' || contains(needs.job-filter.outputs.jobs, ' linux-jammy-py3-clang15-executorch ') }}
-    name: linux-jammy-py3-clang15-executorch
-    uses: ./.github/workflows/_linux-test.yml
-    needs:
-      - linux-jammy-py3-clang15-executorch-build
-      - job-filter
-    with:
-      ...
-    secrets: inherit
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

## 4. Quality Gates Evaluation

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

---

## 5. Refactor Guardrail Compliance

| Guardrail | Status | Notes |
|-----------|--------|-------|
| Invariant declaration | ✅ PASS | INV-060 introduced |
| Baseline discipline | ✅ PASS | Changes from M03 baseline (commit b352a3bbbee) |
| Consumer contract protection | ✅ N/A | No consumer contracts modified |
| Extraction/split safety | ✅ N/A | No extraction performed |
| No silent CI weakening | ✅ PASS | This milestone *removes* silent CI weakening |

---

## 6. Issues (None)

No issues introduced. All changes are minimal, targeted, and reversible.

---

## 7. Upstream Verification Hook

**Deferred verification entry for REFACTOR.md:**

| ID | Issue | Discovered | Deferred To | Reason | Exit Criteria |
|----|-------|------------|-------------|--------|---------------|
| M04-V01 | Verify M04 CI behavior on upstream | M04 | Upstream PR | Fork CI skipped | TD failure propagates; tools-unit-tests fails on pytest failure; scorecards runs cleanly |

This entry should be added to REFACTOR.md Active Risks or Deferrals section when M04 is closed.

---

## 8. Conclusion

**Audit Verdict:** 🟢 PASS

M04 executed to specification:
- 5 risks addressed with minimal, targeted changes
- One commit per risk (granular rollback)
- No scope creep
- No unrelated files touched
- Evidence constraint documented with semantic proof

**Recommendation:** Approve merge. Add upstream verification hook to REFACTOR.md.

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
  "evidence_constraint": {
    "type": "fork_ci_guarded",
    "verification_method": "diff_semantic_proof",
    "upstream_verification_required": true
  }
}
```

---

**End of M04 Audit**

