Below is a **Cursor-ready M02 plan + execution prompt**, written to match the discipline you've established in M00–M01. This is a **governance-first milestone**: no code refactors, minimal blast radius, high portfolio signal.

You can paste this directly into Cursor.

---

# M02_plan — Governance & Living-Log Hardening

## Intent / Target

Harden and formalize the **governance spine** of the refactoring program so that all future milestones (M03+) operate under **explicit, enforceable rules** rather than convention.

M02 turns the current *working* governance into a **fully specified, auditable system**:

* what changes are allowed,
* how milestones are opened/closed,
* how risk, deferrals, and evidence are tracked,
* how AI agents are grounded consistently across sessions.

This is a **documentation-only milestone**.

## Scope Boundaries

**In scope**

* Governance documentation
* Clarification and normalization of `REFACTOR.md`
* Milestone lifecycle rules
* Risk / deferral mechanics
* "How to work in this repo" guidance for AI + humans

**Out of scope**

* No production code changes
* No test changes
* No CI changes
* No dependency updates
* No refactors (mechanical or semantic)

## Invariants

M02 must not change:

* Runtime behavior
* Public APIs
* Build behavior
* CI semantics or workflows
* Any conclusions or findings from M00 or M01

M02 **documents** governance; it does not alter system behavior.

## Verification Plan

Verification is documentary and structural:

* All required governance sections exist
* No contradictions between:

  * `REFACTOR.md`
  * audit pack documents
  * milestone closeouts
* No new requirements introduced without rationale
* No scope creep into execution work

Success = reviewers and AI agents can answer **"how do I safely work here?"** without inference.

---

## Implementation Steps (Ordered, Reversible)

### 1. Normalize `REFACTOR.md` as the Canonical Governance Surface

Update `REFACTOR.md` to explicitly include the following **locked sections** (if already present, refine and normalize language; do not rewrite history):

1. **Governance Model**

   * Immutable baseline vs living log
   * Relationship between audit pack and milestones
   * Authority hierarchy (audit > REFACTOR.md > milestone docs)

2. **Milestone Lifecycle**

   * When a milestone may be opened
   * Required artifacts for:

     * plan
     * execution
     * CI analysis (if applicable)
     * audit
     * summary
     * closeout
   * Rules for stopping, deferring, or aborting a milestone

3. **Change Classes**

   * Documentation-only
   * Verification-only
   * Mechanical refactor
   * Behavioral change (explicitly disallowed unless approved)
   * CI wiring vs CI weakening (clear distinction)

4. **Invariant Handling**

   * How invariants are introduced
   * How they are verified
   * How violations are handled
   * How new invariants may be proposed

5. **Deferral & Risk Registry Rules**

   * What qualifies as a deferred issue
   * How deferrals are tracked
   * Required metadata (reason, risk, revisit milestone)
   * Prohibition on "silent deferral"

6. **AI Agent Operating Rules**

   * Expected posture for Cursor / LLM agents
   * When to stop and ask for confirmation
   * Preference for restraint over speculative fixes
   * Explicit reminder: AI is an assistant under governance

---

### 2. Add a Formal "Milestone Template" Section

Inside `REFACTOR.md` (or a linked doc), include a **canonical milestone template** that future milestones must follow:

* Intent
* Scope boundaries
* Invariants
* Verification plan
* Implementation steps
* Risk & rollback
* Deliverables
* Definition of Done

This should be **descriptive, not prescriptive** — a template, not a checklist.

---

### 3. Clarify Phase Boundaries

Normalize phase definitions so it is unambiguous:

* What Phase 0 meant (audit & baseline only)
* What Phase 1 represents (verification surfaces)
* What later phases are allowed to do

Do **not** renumber milestones or rewrite history; only clarify intent and boundaries.

---

### 4. Update Tooling & Logs (Minimal)

* Update `docs/refactor/toolcalls.md` to reflect M02 actions
* Create:

  ```
  docs/refactor/milestones/M02/M02_plan.md
  ```

  capturing this plan
* Do **not** create M02 summary/audit yet (those come at closeout)

---

## Risk & Rollback Plan

**Risk:** Over-specification or accidental scope creep
**Mitigation:**

* Keep language conservative
* Prefer "documents how" over "mandates that"
* Do not introduce new rules that would invalidate prior work

**Rollback:**

* Revert documentation commits (no code touched)

---

## Deliverables

* `REFACTOR.md` updated with explicit governance sections
* `docs/refactor/milestones/M02/M02_plan.md`
* `docs/refactor/toolcalls.md` updated

No other files should be modified.

---

# Cursor Handoff Prompt — Implement M02

You are Cursor AI acting as a **Governance & Refactoring Program Assistant**.

## Context

M00 (Baseline Audit) and M01 (Import Smoke Test) are complete and locked.
This task is **M02**, a documentation-only milestone to harden governance and make future refactors safer and more repeatable.

## Hard Constraints

* **NO code changes**
* **NO test changes**
* **NO CI changes**
* **NO dependency changes**
* **NO refactors**
* Documentation only

Do not reinterpret audit findings or milestone outcomes.

## Objective

Strengthen and normalize the governance framework so that:

* humans and AI agents can operate safely,
* milestones are repeatable and auditable,
* future refactors have clear guardrails.

## Tasks

1. Update `REFACTOR.md` to explicitly document:

   * governance model
   * milestone lifecycle
   * change classes
   * invariant handling
   * deferral/risk rules
   * AI agent operating rules

2. Add a canonical milestone template section.

3. Clarify phase boundaries without renumbering or rewriting history.

4. Create `docs/refactor/milestones/M02/M02_plan.md` reflecting this plan.

5. Update `docs/refactor/toolcalls.md` with actions taken.

## Verification

* Confirm no non-documentation files were modified
* Confirm no new requirements contradict M00/M01
* Confirm governance sections are explicit and readable

## Completion Signal

At the end, report:

* files modified
* confirmation this is documentation-only
* confirmation M02 is ready for execution tracking

Proceed conservatively. When in doubt, **document intent, not enforcement**.

---

If you want, after Cursor finishes M02, I can:

* review it as an external auditor, or
* help you decide whether **M03 (CI silent failure audits)** or a **portfolio pause** is the stronger next move.

