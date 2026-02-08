Perfect — with **M07 closed** and the audit + summary locked, the next step is to advance in a **clean, non-overlapping way** that preserves signal quality and keeps Cursor tightly scoped.

Below is a **Cursor-ready milestone plan** you can hand off directly. I’m assuming continuity with your refactoring posture in `REFACTOR.md` and that M07 completed CI/action pinning groundwork.

---

# **M08_plan — CI Truthfulness & Silent-Failure Elimination**

## 1. Intent / Target

**Primary objective:**
Eliminate **silent CI failure modes** and restore *truthful signal guarantees* across workflows.

This milestone exists to answer one question conclusively:

> *“If CI is green, do we know it actually exercised the changed surface and failed loudly if something went wrong?”*

M08 is **pure refactor hardening** — no behavior changes, no feature work.

---

## 2. Scope Boundaries

### In Scope

* GitHub Actions workflows under:

  * `.github/workflows/`
  * reusable workflows (`_*.yml`)
* Detection and remediation of:

  * `continue-on-error: true`
  * `if: always()` masking failures
  * shell-level failure suppression (`|| true`, `set +e`)
* Documentation updates:

  * CI guardrail rules in `REFACTOR.md`
  * New CI truthfulness notes (if needed)

### Out of Scope

* No new workflows unless strictly required
* No changes to:

  * test logic
  * product code
  * performance benchmarks
* No action pinning work (already handled in M07)
* No branch protection changes (audit only)

---

## 3. Invariants (Must Not Change)

| Invariant                          | Verification Method                           |
| ---------------------------------- | --------------------------------------------- |
| Required CI checks remain required | Compare branch protection + workflow metadata |
| CI pass/fail semantics unchanged   | Historical comparison of required jobs        |
| No behavior or API changes         | Diff limited to `.github/` + docs             |
| No CI weakening                    | Audit of required vs optional checks          |

---

## 4. Verification Plan

**Evidence sources:**

* Static analysis of workflows:

  * grep/search for `continue-on-error`, `always()`, `|| true`
* Before/after diff of:

  * required checks
  * workflow job graphs
* CI run on a no-op PR:

  * confirms workflows still trigger correctly
  * confirms failures propagate

**Success criteria:**

* Zero unapproved silent-failure constructs in required jobs
* Any remaining exceptions are:

  * explicitly documented
  * non-blocking
  * justified

---

## 5. Implementation Steps (Cursor-Executable)

1. **Inventory silent-failure patterns**

   * Scan all workflows for:

     * `continue-on-error`
     * `if: always()`
     * shell suppression
2. **Classify each occurrence**

   * Required vs informational job
   * Blocking vs non-blocking
3. **Fix or justify**

   * Remove silent suppression in required jobs
   * Add comments + documentation for allowed exceptions
4. **Add guardrail**

   * Lightweight check or documented rule preventing reintroduction
5. **Run CI**

   * Validate no regression in required checks
6. **Update governance docs**

   * Record new CI truthfulness guarantees in `REFACTOR.md`

---

## 6. Risk & Rollback Plan

### Risks

* Accidentally making an informational job blocking
* Surfacing previously hidden flaky failures

### Mitigation

* Explicit required/optional classification before changes
* Small PR, reversible edits only

### Rollback

* `git revert` of workflow-only changes
* No data or artifact migration required

---

## 7. Deliverables

* ✅ Updated workflows with silent-failure paths removed or justified
* 📄 CI truthfulness section added to `REFACTOR.md`
* 📄 M08 summary + audit artifacts
* 🟢 CI green with **truthful enforcement**

---

## 8. Milestone Classification

* **Type:** Hardening / Governance Refactor
* **Posture:** Behavior-Preserving
* **Expected Size:** Small–Medium
* **Blast Radius:** CI only
* **Audit Mode:** DELTA AUDIT

---

### Cursor Handoff Prompt (optional, copy/paste)

> Analyze the repository under a **behavior-preserving refactor posture**.
> Execute **M08_plan — CI Truthfulness & Silent-Failure Elimination**.
>
> Scope is limited to GitHub Actions workflows and governance documentation.
>
> Identify and remediate silent CI failure modes while preserving required check semantics.
>
> Produce:
>
> * implementation changes
> * CI run evidence
> * M08_summary
> * M08_audit
>
> Do not expand scope. Do not change runtime behavior.

---

If you want, next we can:

* pre-draft the **M08 audit expectations** (so Cursor knows what “done” looks like), or
* map **M09–M11** now while M08 runs, to keep momentum without overlap.
