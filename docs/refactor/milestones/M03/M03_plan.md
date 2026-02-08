# M03_plan — CI Workflow Audit for Silent Failures

**Status:** 🔒 Closed and Locked  
**Priority:** P1  
**Estimated Effort:** 8 hours (Actual: 4 hours)  
**Blockers:** None (M02 complete)

---

## Intent / Target

Perform a **systematic audit of PyTorch's CI workflows** to identify:

* **silent failures** (checks that pass but fail to assert meaningful correctness),
* **false confidence signals** (green CI that masks risk),
* **structural gaps** (missing coverage, unenforced contracts, skipped paths),
* **governance risks** (where CI behavior contradicts documented expectations).

**M03 is diagnostic only.**
It produces **evidence, classification, and prioritization**, not fixes.

This milestone establishes the **CI truth map** that all later CI refactors will rely on.

---

## Scope Boundaries

### In Scope

* Read and analyze **all CI workflows** under `.github/workflows/`
* Classify workflows by **purpose and assurance level**
* Identify **silent failure modes** and **weak signals**
* Cross-reference CI behavior with:

  * M00 audit findings
  * M01 verification surfaces
  * Governance rules in `REFACTOR.md`

### Out of Scope

* ❌ Modifying workflows
* ❌ Adding or removing CI jobs
* ❌ Changing required/optional status
* ❌ Tightening or weakening checks
* ❌ Adding new tooling or enforcement

**No CI behavior may change in M03.**

---

## Invariants

M03 must not change:

* CI execution behavior
* Required vs optional checks
* Branch protection semantics
* Test selection logic
* Timeouts, runners, matrices

This milestone is **observational only**.

---

## Verification Plan

Verification is documentary and evidentiary:

* Every claim must reference:

  * a workflow file,
  * a job name,
  * or a documented CI behavior.
* No speculative conclusions.
* Clear separation between:

  * **observed fact**
  * **inferred risk**
  * **recommended follow-up** (deferred to later milestones)

Success = a reviewer can answer:

> "Which CI checks can I trust, and why?"

---

## Implementation Steps (Ordered, Reversible)

### 1. Inventory All CI Workflows

Create a complete inventory of `.github/workflows/*` including:

* workflow name
* trigger (PR, push, schedule, manual)
* scope (docs-only, python, C++, CUDA, platform-specific)
* required vs optional status (where visible)
* approximate runtime cost

**Deliverable:** CI Inventory Table (markdown)

---

### 2. Classify Workflows by Signal Strength

Classify each workflow into one of:

* **Strong Signal**

  * deterministically validates correctness
  * fails when invariants are violated
* **Weak Signal**

  * partial coverage, heuristic checks
* **Cosmetic / Informational**

  * formatting, reporting, dashboards
* **Legacy / Unclear**

  * unclear ownership or purpose

No judgment—classification only.

---

### 3. Identify Silent Failure Patterns

For each workflow, assess:

* Does it skip paths silently?
* Does it rely on environment assumptions?
* Does it report success when meaningful checks are bypassed?
* Are failures masked by retries, soft-fail flags, or conditional skips?

Document **patterns**, not fixes.

---

### 4. Cross-Reference with Known Risks

Cross-reference findings against:

* M00 audit risk list
* Known evidence gaps
* Areas touched by M01 (imports, Python surface)

Call out:

* risks already covered by CI
* risks not covered at all
* risks covered weakly

---

### 5. Produce a Prioritized Risk Register (CI-Only)

Create a **CI Silent Failure Risk Register** with:

* workflow(s) involved
* risk description
* severity (Low / Medium / High)
* confidence level
* recommended follow-up milestone (e.g., M04, M05)

This is **advisory**, not binding.

---

### 6. Update REFACTOR.md (Minimal)

* Add an **M03 milestone entry** (status: In Progress → Complete at closeout)
* Optionally add a short note under Phase 1 describing CI audit intent

No other edits.

---

### 7. Tool Logging

Log all analysis steps in:

```
docs/refactor/milestones/M03/M03_toolcalls.md
```

---

## Risk & Rollback Plan

**Risk:**

* Accidental prescriptive language
* Slipping into "how to fix" instead of "what exists"

**Mitigation:**

* Use "observed", "appears", "currently"
* Defer fixes explicitly

**Rollback:**

* Revert documentation commits only

---

## Deliverables

**Required**

* `docs/refactor/milestones/M03/M03_plan.md` (this document)
* `docs/refactor/milestones/M03/M03_toolcalls.md`
* `docs/refactor/milestones/M03/M03_summary.md`
* `docs/refactor/milestones/M03/M03_audit.md`
* CI Inventory Table (embedded in audit)

**No code or CI files modified.**

---

## Definition of Done

- [x] All 142 workflow files inventoried
- [x] Workflows classified by signal strength
- [x] Silent failure patterns identified and documented
- [x] Cross-referenced with M00 known risks
- [x] Prioritized risk register produced
- [x] Evidence-based language used throughout
- [x] No CI or code modifications made
- [x] Taxonomy explicitly defined upfront
- [x] REFACTOR.md updated with M03 CLOSED entry

---

## Execution Notes

### Silent Failure Taxonomy (Locked)

1. **Soft-Fail Mechanics**
   * `continue-on-error: true`
   * non-failing steps masking failure

2. **Conditional Skips**
   * `if:` conditions that bypass checks silently
   * environment-dependent skips

3. **Retry / Flakiness Masking**
   * automatic retries without surfacing instability

4. **Scope Gaps**
   * workflows that do not run on all relevant change surfaces
   * partial matrices with no explanation

5. **Informational-Only Workflows**
   * green signals that assert no correctness properties

### Tiered Analysis Approach

* **Tier 1: Core CI Workflows** — Deep analysis (`pull.yml`, `trunk.yml`, `lint.yml`)
* **Tier 2: Platform / Accelerator** — Medium depth, pattern-based review
* **Tier 3: Generated / Nightly / Experimental** — Summary-level classification

### Evidence Gap: Required Checks

GitHub branch protection rules are not observable from repository contents and are treated as an evidence gap.

---

## Reference

- Baseline Audit: [`docs/refactor/audit/BASELINE_AUDIT.md`](../../audit/BASELINE_AUDIT.md)
- CI Gaps: [`docs/refactor/audit/CI_GAPS_AND_GUARDRAILS.md`](../../audit/CI_GAPS_AND_GUARDRAILS.md)
- Phase Map: [`docs/refactor/audit/REFACTOR_PHASE_MAP.md`](../../audit/REFACTOR_PHASE_MAP.md)

---

**End of M03 Plan**

