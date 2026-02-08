Perfect — M06 is a *meaty* one, and it’s exactly where this program starts paying real security dividends.

Below is a **clean, governance-aligned M06 plan** you can hand directly to Cursor. It assumes **action pinning & supply-chain hardening**, stays tightly scoped, and fits seamlessly after M05.

---

# M06_plan — Action Pinning & Supply-Chain Hardening

## Intent / Target

**M06 hardens the PyTorch CI supply chain** by pinning all GitHub Actions used in `.github/workflows/` to **full commit SHAs**, eliminating mutable references (`@v1`, `@v2`, `@main`, `@latest`) on the CI critical path.

> **What remains unsafe without this milestone?**
> Any third-party action referenced by tag can change behavior without notice, introducing:
>
> * Silent CI behavior changes
> * Supply-chain compromise risk
> * Non-reproducible CI results
> * Undetectable regressions in correctness or security

M06 converts CI from *implicitly trusted* to *explicitly verified*.

---

## Scope Boundaries

### In Scope

* Inventory **all `uses:` statements** in `.github/workflows/**`
* Identify **unpinned or weakly pinned actions**:

  * `@v1`, `@v2`, `@v3`
  * `@main`, `@master`, `@latest`
* Pin each action to a **full 40-char commit SHA**
* Preserve the original version tag **as a comment** for maintainability
* Update **only workflow files**
* Add documentation and audit artifacts

### Out of Scope (Hard)

* ❌ No action upgrades or version changes
* ❌ No YAML restructuring
* ❌ No logic changes
* ❌ No new workflows
* ❌ No removal of actions
* ❌ No dependency policy enforcement
* ❌ No SBOM generation (M08)
* ❌ No signing / provenance (future milestone)

---

## Change Class

**CI Configuration — Behavior-Preserving**

This is a **mechanical refactor**:

* Identical action code
* Identical behavior
* Identical inputs/outputs
* Reduced mutability risk

---

## Invariants

### Existing Invariants (Must Hold)

* **INV-060** — CI Critical Path Integrity
* **INV-070** — CI Structural Validity

### New Invariant Introduced

* **INV-080 — Action Immutability**

  > All GitHub Actions on the CI critical path must be referenced by immutable commit SHA.

---

## Verification Plan

Because fork CI is guarded, verification relies on **static proof + structural checks**.

| Verification         | Method                                |
| -------------------- | ------------------------------------- |
| No behavioral change | Diff-based semantic proof             |
| All actions pinned   | Scripted inventory before/after       |
| YAML validity        | actionlint (already present from M05) |
| No new workflows     | File diff check                       |
| Rollback safety      | One-commit-per-group strategy         |

---

## Implementation Steps

> **Cursor must log all tool calls before execution.**

1. **Inventory Actions**

   * Scan `.github/workflows/**/*.yml`
   * Produce table: `{workflow, job, step, action, ref}`

2. **Classify References**

   * Strong: already pinned to SHA
   * Weak: version tag (`@v2`)
   * Unsafe: floating (`@main`, `@latest`)

3. **Resolve SHAs**

   * For each unpinned action:

     * Resolve tag → exact commit SHA
     * Record mapping in notes

4. **Apply Pins**

   * Replace:

     ```yaml
     uses: org/action@v2
     ```

     with:

     ```yaml
     uses: org/action@<full-sha> # v2
     ```
   * No other line changes

5. **Granular Commits**

   * One commit per **action family** (e.g., checkout, setup-python, upload-artifact)
   * Enables safe rollback

6. **Run actionlint**

   * Ensure structural validity unchanged

7. **Document Evidence Constraints**

   * Fork CI guarded
   * Upstream verification deferred

---

## Risk & Rollback

### Risks

| Risk                | Mitigation                            |
| ------------------- | ------------------------------------- |
| Incorrect SHA       | Cross-check tag → commit mapping      |
| Missed action       | Automated inventory                   |
| Merge conflicts     | Granular commits                      |
| Maintainer pushback | Clear audit trail, no behavior change |

### Rollback

* `git revert <commit>` per action family
* Zero state coupling between commits

---

## Deliverables

### Code / Config

* Modified `.github/workflows/*.yml` (pinning only)

### Documentation

* `docs/refactor/milestones/M06/M06_plan.md`
* `docs/refactor/milestones/M06/M06_toolcalls.md`
* `docs/refactor/milestones/M06/M06_audit.md`
* `docs/refactor/milestones/M06/M06_summary.md`
* `REFACTOR.md` updated with M06 entry

---

## Definition of Done

* [ ] All `uses:` entries pinned to full SHA
* [ ] Original tags preserved as comments
* [ ] No new workflows added
* [ ] No workflow logic changed
* [ ] actionlint passes (no new errors)
* [ ] Fork CI constraints documented
* [ ] Audit & summary complete
* [ ] REFACTOR.md updated
* [ ] Explicit closeout approval obtained

---

## Expected Impact

| Area               | Before    | After           |
| ------------------ | --------- | --------------- |
| CI Reproducibility | ❌ Mutable | ✅ Deterministic |
| Supply-Chain Risk  | 🔴 P1     | 🟡 Reduced      |
| Auditability       | Medium    | High            |
| Maintainer Trust   | Neutral   | Improved        |

---

## Authorized Next Step

Once M06 is complete, the program can safely proceed to:

**M07 — Third-Party Action Risk Classification**
or
**M08 — Dependency & SBOM Baseline**

---

If you want, next we can:

* Pre-compute an **action inventory script** for Cursor
* Decide **commit grouping strategy** (by vendor vs by usage)
* Or sanity-check how many actions we expect to pin (ballpark: ~40–60)

Just say the word.
