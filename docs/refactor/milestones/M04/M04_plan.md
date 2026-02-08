# M04_plan — Fix Silent Failures (High-Priority)

**Status:** Ready to Start  
**Priority:** P1  
**Estimated Effort:** 6 hours  
**Blockers:** None (M03 complete)

---

## Intent / Target

Address the **4 high-severity silent failure risks** identified in M03's CI audit. This milestone performs surgical, targeted fixes to ensure critical CI workflows fail loudly when they should.

**What remains unsafe without M04:**
- Target Determination can fail silently, leading to incomplete test coverage
- LLM TD Retrieval job can fail completely without visibility
- Disabled jobs (`if: false`) remain invisible governance debt
- Tools unit tests always appear to pass

---

## Scope Boundaries

### In Scope

From M03 Risk Register (High Severity):

1. **M03-R01:** Remove `continue-on-error: true` from TD step in `target_determination.yml`
2. **M03-R02:** Remove job-level `continue-on-error` from `llm_td_retrieval.yml`
3. **M03-R03:** Remove or relocate `if: false` disabled jobs:
   - `trunk.yml` — Executorch build
   - `scorecards.yml` — Security scoring
4. **M03-R04:** Remove `continue-on-error` from test steps in `tools-unit-tests.yml`

### Out of Scope

- ❌ Low-severity `continue-on-error` patterns (monitoring, uploads)
- ❌ `fail-fast: false` (intentional matrix behavior)
- ❌ `if: always()` cleanup patterns
- ❌ Action pinning (M06)
- ❌ Adding new workflows or enforcement

---

## Invariants

This milestone affects CI behavior. The following invariants must be preserved:

- **INV-CI-01:** All changes must pass existing CI checks
- **INV-CI-02:** No reduction in test coverage
- **INV-CI-03:** No breaking changes to workflow interfaces

---

## Verification Plan

1. **PR-level verification:**
   - All existing CI checks must pass
   - No new failures introduced

2. **Behavioral verification:**
   - Manually trigger TD workflow to confirm it fails loudly on error
   - Confirm disabled jobs are either removed or moved to unstable.yml

3. **Documentation verification:**
   - Each change documented in M04_audit.md
   - Rollback plan for each workflow change

---

## Implementation Steps (Ordered)

### Step 1: Fix `target_determination.yml`

- Remove `continue-on-error: true` from "Do TD" step (L58-60)
- Add proper error handling or allow natural failure
- Test: TD failure should fail the workflow

### Step 2: Fix `llm_td_retrieval.yml`

- Remove job-level `continue-on-error: true` (L24-26)
- Keep step-level `continue-on-error` on "Run Retriever" if needed for ghstack
- Test: Job failures should be visible in workflow summary

### Step 3: Handle disabled jobs

Option A (Preferred): Remove dead jobs entirely
- Delete `linux-jammy-py3-clang15-executorch-*` jobs from `trunk.yml`
- Delete or fix `scorecards.yml`

Option B: Move to unstable.yml with visibility
- Relocate jobs to `unstable.yml` with proper documentation

### Step 4: Fix `tools-unit-tests.yml`

- Remove `continue-on-error: true` from test steps (L38-39, L64-65)
- If tests are expected to be flaky, move to unstable workflow instead
- Test: Test failures should fail the workflow

---

## Risk & Rollback Plan

**Risks:**
- TD failure may block PRs unexpectedly if there are underlying issues
- LLM TD failure may increase CI latency (retries needed)
- Removing disabled jobs may lose context about why they were disabled

**Mitigation:**
- Each change in a separate commit for easy rollback
- Test changes in a PR before merging
- Document original state before modification

**Rollback:**
- Revert specific commits if issues arise
- Each workflow change is independently revertable

---

## Deliverables

**Required:**
- `docs/refactor/milestones/M04/M04_plan.md` (this document)
- `docs/refactor/milestones/M04/M04_toolcalls.md`
- `docs/refactor/milestones/M04/M04_audit.md`
- `docs/refactor/milestones/M04/M04_summary.md`
- Modified workflow files (4)

---

## Definition of Done

- [ ] All 4 high-severity risks from M03 addressed
- [ ] CI passes on all changes
- [ ] Each change documented with before/after
- [ ] Rollback plan verified
- [ ] REFACTOR.md updated with M04 CLOSED entry
- [ ] No scope creep (low-severity patterns untouched)

---

## Reference

- M03 Risk Register: [`docs/refactor/milestones/M03/M03_audit.md`](../M03/M03_audit.md)
- CI Gaps Baseline: [`docs/refactor/audit/CI_GAPS_AND_GUARDRAILS.md`](../../audit/CI_GAPS_AND_GUARDRAILS.md)

---

**End of M04 Plan**

