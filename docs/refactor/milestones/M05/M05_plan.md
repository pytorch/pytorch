Excellent — with M04 cleanly closed, the **next milestone is M05** as already queued in your phase map. Below is a **Cursor-ready M05 plan**, written to your governance template and scoped so it can execute safely on a fork while still producing real signal.

---

# **M05_plan — CI Workflow Linting & Structural Guardrails**

## Intent / Target

Introduce **structural validation for GitHub Actions workflows** to detect CI anti-patterns *before* they ship. M05 adds **read-only linting and analysis** (no enforcement yet) to surface:

* invalid or malformed YAML,
* unreachable jobs and steps,
* misuse of `if`, `needs`, and outputs,
* deprecated or dangerous workflow constructs.

**What remains unsafe without M05:**
Even after M04 removed known silent failures, **new silent failures can be reintroduced** at any time because workflow YAML is currently unchecked. M05 establishes an *early warning system* for CI integrity.

---

## Scope Boundaries

### **In Scope**

* Add **actionlint** (or equivalent) as a *non-blocking* CI job
* Run linting against `.github/workflows/**/*.yml`
* Capture findings as **visible CI output**
* Document discovered issues (but do not fix them)
* Update REFACTOR.md with M05 milestone entry

### **Out of Scope**

* ❌ No workflow rewrites
* ❌ No fixing lint findings
* ❌ No required/mandatory CI gates
* ❌ No action pinning (M06)
* ❌ No behavior changes to existing workflows

---

## Invariants

### Existing Invariants

* **INV-060 — CI Critical Path Integrity** (must not be weakened)

### New Invariant (Introduced)

* **INV-070 — CI Structural Validity**
  *All CI workflows must be syntactically valid and analyzable by static tooling.*

This invariant is **observational only** in M05 (no enforcement yet).

---

## Verification Plan

### Primary Verification

* actionlint executes successfully on the fork
* actionlint produces a report (warnings/errors)
* CI job itself passes (even if findings exist)

### Evidence Produced

* Lint output attached to CI logs
* Summary of findings captured in `M05_audit.md`
* No workflow behavior altered

### Fork Constraint Handling

* CI job must **not depend on upstream-only runners**
* Use GitHub-hosted runners only
* Findings are informational

---

## Implementation Steps

1. Create branch: `m05-ci-workflow-linting`
2. Add new workflow:

   * `.github/workflows/refactor-actionlint.yml`
3. Configure actionlint:

   * Scan `.github/workflows/**/*.yml`
   * Allow warnings, do not fail job
4. Run on:

   * `pull_request`
   * `workflow_dispatch`
5. Commit workflow addition
6. Observe CI output
7. Document findings (no fixes)
8. Create closeout artifacts

Each step must be independently reversible.

---

## Risk & Rollback

### Risks

* actionlint flags large number of issues (noise)
* False positives due to PyTorch’s scale and complexity

### Mitigation

* Non-blocking mode only
* Findings documented, not acted upon

### Rollback

* Single commit revert removes the workflow entirely

---

## Deliverables

* `.github/workflows/refactor-actionlint.yml`
* `docs/refactor/milestones/M05/M05_toolcalls.md`
* `docs/refactor/milestones/M05/M05_audit.md`
* `docs/refactor/milestones/M05/M05_summary.md`
* REFACTOR.md updated with M05 entry

---

## Definition of Done

* ✅ actionlint runs successfully on fork
* ✅ CI remains green (no new required checks)
* ✅ Findings documented, not fixed
* ✅ No existing workflows modified
* ✅ REFACTOR.md updated
* ✅ M05 audit & summary completed
* ✅ Toolcalls logged

---

## Notes for Cursor

* Treat this as **verification-only**
* Stop if actionlint requires privileged runners
* Prefer visibility over completeness
* Do **not** normalize or auto-fix YAML

---

## Milestone Classification

* **Change Class:** Verification-Only
* **Risk Level:** Low
* **Expected Effort:** ~3–4 hours
* **Phase:** Phase 1 — CI Health & Guardrails

---

If you want, next I can:

* generate the **exact GitHub Actions YAML** for actionlint (ready to paste), or
* help you pre-decide how findings should be categorized in the audit (P0/P1/P2), or
* prep the **M06 plan** (Action pinning & supply-chain hardening) so you can see the runway.
