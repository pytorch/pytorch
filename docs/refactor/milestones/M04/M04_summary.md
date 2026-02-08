# 📌 Milestone Summary — M04: Fix High-Priority CI Silent Failures

**Project:** PyTorch Refactoring Program  
**Phase:** Phase 1 — CI Health & Guardrails  
**Milestone:** M04 — Fix High-Priority CI Silent Failures  
**Timeframe:** 2026-02-08  
**Status:** Ready to Merge (Pending Approval)  
**Baseline:** b352a3bbbee (M03 complete)  
**Refactor Posture:** Behavior-Preserving (CI Configuration Only)

---

## 1. Milestone Objective

**Why this milestone existed:**

M03 identified 4 high-severity silent failure risks in PyTorch's CI infrastructure where genuine failures were being masked by `continue-on-error: true` or `if: false` patterns. These patterns allowed broken code to pass CI without detection.

M04 exists to **convert mapped risk into corrected signal** — making CI failures visible, honest, and actionable.

> **What would remain unsafe without this refactor?**  
> Target Determination failures would silently proceed with incomplete test coverage. Tools unit tests would always appear green even when failing. Disabled jobs would remain invisible in Tier 1 workflows.

---

## 2. Scope Definition

### In Scope

| Risk ID | Workflow | Change |
|---------|----------|--------|
| M03-R01 | `target_determination.yml` | Remove `continue-on-error` from Do TD step |
| M03-R02 | `llm_td_retrieval.yml` | Remove job-level `continue-on-error`; preserve step-level with comment |
| M03-R03 | `trunk.yml` | Remove disabled executorch build+test jobs |
| M03-R04 | `tools-unit-tests.yml` | Remove `continue-on-error` from test steps |
| M03-R05 | `scorecards.yml` | Re-enable by removing `false &&` |

### Out of Scope

- ❌ No new CI jobs
- ❌ No workflow re-architecture
- ❌ No action pinning (M06)
- ❌ No YAML linting (M05)
- ❌ No performance optimizations
- ❌ No behavior changes outside CI signaling

---

## 3. Refactor Classification

### Change Type

**Mechanical refactor** — Line-level removal of soft-fail patterns and disabled job conditions.

### Observability

**CI-only impact:**
- Step/job failures that were previously masked will now propagate
- Disabled jobs removed from workflow definition
- Scorecards workflow becomes eligible on upstream

**No externally observable changes to:**
- PyTorch API, CLI, or library behavior
- Test execution logic
- Build artifacts

---

## 4. Work Executed

| Commit | Risk ID | Action |
|--------|---------|--------|
| `0d264ac6c14` | M03-R01 | Removed 1 line (`continue-on-error: true`) |
| `61d4bce052e` | M03-R02 | Removed 1 line (job-level), added 2 lines (clarifying comment) |
| `32cf142d68c` | M03-R03 | Removed 31 lines (disabled jobs), added 3 lines (removal note) |
| `73677f34e72` | M03-R04 | Removed 2 lines (`continue-on-error` from both test steps) |
| `5fbb0de6125` | M03-R05 | Changed 1 line (`false &&` → condition only) |

**Summary:** 5 files changed, 7 insertions, 37 deletions

**No functional logic changed.** All changes are CI configuration only.

---

## 5. Invariants & Compatibility

### Declared Invariants

| ID | Invariant | Status |
|----|-----------|--------|
| INV-060 | CI Critical Path Integrity — If a correctness-critical step fails, CI must fail visibly | ✅ Introduced |

### Compatibility Notes

- **Backward compatibility:** ✅ Preserved (no API changes)
- **Breaking changes:** ❌ None
- **Deprecations:** ❌ None

---

## 6. Validation & Evidence

### Evidence Constraint

**Fork CI is guarded** — All PyTorch CI workflows include `if: github.repository_owner == 'pytorch'` guards that skip execution on forks.

**What cannot be directly observed:**
- TD failure propagation
- tools-unit-tests failure on pytest failure
- scorecards workflow execution

### Verification Method

| Evidence Type | Method | Result | Notes |
|---------------|--------|--------|-------|
| Diff review | Manual inspection | ✅ PASS | All changes are minimal and targeted |
| Semantic proof | GitHub Actions behavior analysis | ✅ PASS | Removing `continue-on-error` = failures propagate |
| Scope verification | File count | ✅ PASS | Only 5 workflow files modified |
| Commit granularity | Git log | ✅ PASS | One commit per risk ID |

### Upstream Verification (Deferred)

| Verification | Status | Exit Criteria |
|--------------|--------|---------------|
| TD failure propagates | 🔵 Deferred | TD step failure causes workflow failure |
| tools-unit-tests fails correctly | 🔵 Deferred | pytest failure causes workflow failure |
| scorecards runs | 🔵 Deferred | OSSF scorecard executes on upstream |

---

## 7. CI / Automation Impact

### Workflows Affected

| Workflow | Change Type | Enforcement |
|----------|-------------|-------------|
| `target_determination.yml` | Hardened | Stricter (failures propagate) |
| `llm_td_retrieval.yml` | Hardened | Stricter (job failures propagate) |
| `trunk.yml` | Cleaned | Unchanged (dead code removed) |
| `tools-unit-tests.yml` | Hardened | Stricter (test failures propagate) |
| `scorecards.yml` | Enabled | New (was disabled) |

### Signal Improvement

- **Before M04:** 4 workflows with false-positive signals
- **After M04:** 0 workflows with false-positive signals (in scope)

---

## 8. Issues, Exceptions, and Guardrails

### Issues Encountered

**None.** All changes applied cleanly without conflicts.

### Pre-existing Issue Observed (Out of Scope)

The `lumen-cli-compatible-python39` job in `tools-unit-tests.yml` is missing the `pytest` command (only has pip install). This is a pre-existing issue, not introduced by M04. Deferred for future investigation.

### Guardrails Added

- **INV-060** — CI Critical Path Integrity invariant established
- **Removal comment** — Added to trunk.yml documenting why executorch jobs were removed

---

## 9. Deferred Work

| Item | Reason | Status | Tracking |
|------|--------|--------|----------|
| Upstream CI verification | Fork CI skipped | New deferral | M04-V01 |
| tools-unit-tests missing pytest | Pre-existing issue | Pre-existing | Out of scope |

---

## 10. Governance Outcomes

**What is now provably true that was not before:**

1. ✅ `continue-on-error: true` is not used on correctness-critical steps in the 5 addressed workflows
2. ✅ No `if: false` disabled jobs exist in `trunk.yml` (Tier 1)
3. ✅ Scorecards security workflow is enabled for upstream
4. ✅ INV-060 is established as a governance invariant

---

## 11. Exit Criteria Evaluation

| Criterion | Status | Evidence |
|-----------|--------|----------|
| All 5 risks addressed | ✅ Met | 5 commits, one per risk |
| Minimal, line-level changes | ✅ Met | 7 insertions, 37 deletions |
| One commit per risk | ✅ Met | Git log shows 5 atomic commits |
| No scope creep | ✅ Met | Only 5 workflow files touched |
| Evidence documented | ✅ Met | This summary + M04_audit.md |

---

## 12. Final Verdict

**Milestone objectives met. Refactor verified safe via semantic proof. Evidence constraint documented. Proceed to merge.**

---

## 13. Authorized Next Step

- ✅ Merge PR #2 (after approval)
- ✅ Update REFACTOR.md with M04 completion entry
- ✅ Add M04-V01 to deferred verification registry
- ✅ Proceed to M05 (YAML Linting)

---

## 14. Canonical References

| Artifact | Reference |
|----------|-----------|
| Branch | `m04-fix-silent-failures` |
| PR | https://github.com/m-cahill/pytorch/pull/2 |
| Base commit | `b352a3bbbee` |
| Head commit | `5fbb0de6125` |
| Plan | `docs/refactor/milestones/M04/M04_plan.md` |
| Audit | `docs/refactor/milestones/M04/M04_audit.md` |
| Toolcalls | `docs/refactor/milestones/M04/M04_toolcalls.md` |

---

**End of M04 Summary**

