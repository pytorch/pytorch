# M06_plan — Action Pinning & Supply-Chain Hardening

## Intent / Target

Harden the GitHub Actions supply chain by **auditing and documenting action pinning practices** across all 144 CI workflows. M06 establishes visibility into which actions use floating tags (`@v1`, `@main`, `@master`) versus commit SHA pinning, quantifying supply-chain risk exposure.

**What remains unsafe without M06:**

Any GitHub Action can push a malicious update to a version tag. Workflows using `uses: actions/checkout@v4` trust the current state of that tag, which can change without notice. SHA pinning (`uses: actions/checkout@b4ffde65f46...`) locks to an immutable commit.

Without M06, we have no inventory of pinning practices and no baseline to measure improvement.

---

## Scope Boundaries

### In Scope

* Audit all `uses:` directives across 144 workflow files
* Classify each action by pinning strategy:
  - **SHA-pinned** (40-char commit hash)
  - **Tag-pinned** (`@v1`, `@v4.1.0`)
  - **Branch-pinned** (`@main`, `@master`)
* Generate inventory with blast radius assessment
* Document supply-chain risk exposure
* **Optionally:** Add non-blocking pinning check workflow

### Out of Scope

* ❌ No automatic SHA pinning (too many actions)
* ❌ No action upgrades
* ❌ No workflow modifications (except new validation workflow)
* ❌ No blocking CI enforcement
* ❌ No Dependabot configuration changes
* ❌ No third-party action removal

---

## Invariants

### Existing Invariants (Protected)

* **INV-060** — CI Critical Path Integrity
* **INV-070** — CI Structural Validity

### New Invariant (Proposed)

* **INV-080** — Action Pinning Visibility
  *All GitHub Actions usage must be inventoried with pinning strategy documented.*

This invariant is **observational only** in M06 (audit, no enforcement).

---

## Verification Plan

### Primary Verification

* Script extracts all `uses:` directives from workflow files
* Each action classified by pinning type
* Inventory generated with counts and percentages
* Risk assessment documented

### Evidence Produced

* Action inventory table (by workflow and action)
* Pinning summary statistics
* Blast radius assessment for unpinned actions
* Recommendations for M07+ prioritization

### Fork Constraint Handling

* Analysis is static (no CI execution required)
* All verification can be performed locally

---

## Implementation Steps

1. Create branch: `m06-action-pinning-audit`
2. Write analysis script:
   * Extract all `uses:` directives
   * Parse action references (owner/repo@ref)
   * Classify pinning type (SHA, tag, branch)
3. Run analysis against `.github/workflows/`
4. Generate inventory table
5. Document findings in `M06_audit.md`
6. Assess blast radius for high-risk actions
7. **(Optional)** Create non-blocking validation workflow
8. Create closeout artifacts
9. Update REFACTOR.md

Each step must be independently reversible.

---

## Risk & Rollback

### Risks

* Large number of unpinned actions (expected based on PyTorch scale)
* Noise from internal/PyTorch-owned actions
* Complexity of action reference parsing

### Mitigation

* Audit-only mode (no enforcement)
* Separate internal vs external actions
* Clear documentation of findings

### Rollback

* Documentation-only milestone; no rollback needed
* Optional validation workflow is single-file revert

---

## Deliverables

* `tools/refactor/action_pinning_audit.py` — Analysis script
* `docs/refactor/milestones/M06/M06_audit.md` — Full inventory and risk assessment
* `docs/refactor/milestones/M06/M06_summary.md` — Executive summary
* `docs/refactor/milestones/M06/M06_toolcalls.md` — Tool invocation log
* **(Optional)** `.github/workflows/refactor-pinning-check.yml` — Non-blocking validation
* REFACTOR.md updated with M06 entry

---

## Definition of Done

* ✅ All 144 workflow files analyzed
* ✅ All `uses:` directives inventoried
* ✅ Pinning classification complete (SHA/tag/branch)
* ✅ Blast radius assessed for high-risk actions
* ✅ Findings documented in M06_audit.md
* ✅ REFACTOR.md updated
* ✅ Toolcalls logged

---

## Notes for Cursor

* This is primarily a **documentation/audit** milestone
* Focus on generating accurate inventory
* Distinguish between PyTorch-owned actions and external actions
* External actions with branch pinning are highest risk
* Don't attempt to fix pinning in M06 (defer to M07)

---

## Milestone Classification

* **Change Class:** Documentation-Only (audit) or Verification-Only (if adding check workflow)
* **Risk Level:** Low
* **Expected Effort:** ~4-6 hours
* **Phase:** Phase 1 — CI Health & Guardrails

---

## Context: Supply Chain Risk

### Why This Matters

In 2024, the `actions/checkout` maintainers could push arbitrary code to the `@v4` tag. Any workflow using `@v4` would immediately execute that code. With 144 workflows, exposure is multiplicative.

### Pinning Hierarchy

| Type | Example | Risk | Immutability |
|------|---------|------|--------------|
| SHA | `@b4ffde65f46...` | 🟢 Low | Immutable |
| Full version tag | `@v4.1.0` | 🟡 Medium | Usually immutable |
| Major version tag | `@v4` | 🟠 Medium-High | Mutable |
| Branch | `@main`, `@master` | 🔴 High | Always mutable |

### Expected Findings

Based on typical large repositories:
* Most actions use major version tags (`@v4`)
* Some PyTorch-internal actions may use branches
* Very few actions are SHA-pinned
* External actions with branch pinning are critical findings

---

## Questions to Lock Before Execution

1. **Should we add a non-blocking validation workflow?**
   - Option (a): Audit only (no new workflow)
   - Option (b): Add `refactor-pinning-check.yml` (non-blocking)

2. **How should we categorize PyTorch-owned actions?**
   - Option (a): Same risk assessment as external
   - Option (b): Lower risk (PyTorch controls the source)

3. **Output format for inventory?**
   - Option (a): Markdown tables in M06_audit.md
   - Option (b): JSON + Markdown summary
   - Option (c): Both

---

**End of M06 Plan**

