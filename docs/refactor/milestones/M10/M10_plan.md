Perfect. M09 is cleanly closed, and the audit itself already tells us exactly what M10 must be — and just as importantly, what it must **not** be.

Below is a **Cursor-handoff-ready plan for M10**, written to align tightly with the findings and deferrals explicitly recorded in the M09 audit and summary.

---

# **M10_plan — Third-Party Risk & License Classification (SBOM Analysis)**

## 1. Intent / Target

Convert the **M09 SBOM baseline** into **actionable risk intelligence** by classifying:

* **License posture**
* **Provenance confidence**
* **Maintenance / ownership status**
* **Risk category**

This milestone is **analysis-only**.
It does **not** change code, CI, dependencies, or enforcement.

> **What remains unsafe without this work?**
> After M09, we know *what* third-party components exist — but not **which ones matter**, **which ones are risky**, or **which ones require future governance attention**.

M10 answers:
**“Where should we look next — and why?”**

---

## 2. Scope Boundaries

### In Scope

* **All 42 components** enumerated in:

  * `docs/refactor/sbom/M09_sbom.json`
  * `docs/refactor/sbom/M09_THIRD_PARTY.md`
* Classification along **four axes**:

  1. License category
  2. Provenance confidence
  3. Ownership / maintenance status
  4. Risk tier

### Explicitly Out of Scope

* ❌ License remediation
* ❌ Dependency upgrades
* ❌ CVE scanning or vulnerability databases
* ❌ CI enforcement or blocking rules
* ❌ SBOM regeneration or tooling changes
* ❌ Transitive dependency discovery

---

## 3. Refactor / Change Classification

**Pure Analysis + Documentation**

* No runtime changes
* No CI changes
* No SBOM regeneration
* Zero blast radius

---

## 4. Invariants (Must Hold)

| Invariant             | Requirement                                    |
| --------------------- | ---------------------------------------------- |
| Behavior preservation | No code touched                                |
| CI integrity          | No workflows modified                          |
| SBOM immutability     | M09 SBOM treated as read-only input            |
| Audit continuity      | All classifications trace back to M09 evidence |

If any invariant is threatened → stop and surface immediately.

---

## 5. Classification Framework (Authoritative)

Each component must be classified using **only observable evidence**.

### A. License Classification

Use coarse, defensible buckets:

* **Permissive** (MIT, BSD, Apache-2.0, zlib)
* **Weak Copyleft** (LGPL, MPL)
* **Strong Copyleft** (GPL, AGPL)
* **Custom / Non-standard**
* **Unknown**

> If license text is not present → **Unknown** (do not infer from upstream reputation).

---

### B. Provenance Confidence

| Level       | Meaning                                          |
| ----------- | ------------------------------------------------ |
| **High**    | Submodule with pinned SHA and upstream URL       |
| **Medium**  | Vendored copy with LICENSE + README              |
| **Low**     | Embedded or ported code with partial attribution |
| **Unknown** | No clear upstream or license evidence            |

---

### C. Ownership / Maintenance

* **PyTorch-owned** (explicitly identified in M09)
* **Upstream-maintained**
* **Archived / Low activity**
* **Unknown**

---

### D. Risk Tier (Qualitative)

Assign **one**:

* **Low** — well-licensed, well-understood, low churn
* **Medium** — some ambiguity or maintenance risk
* **High** — license uncertainty, unclear provenance, or strategic importance
* **Informational** — present but unlikely to require action

⚠️ This is **not** a CVE or security severity score.

---

## 6. Implementation Steps (Ordered, Reversible)

### Step 1 — Load M09 Artifacts

Treat the following as immutable inputs:

* `M09_sbom.json`
* `M09_THIRD_PARTY.md`
* Deferred issues registry entries from M09

---

### Step 2 — Component-by-Component Classification

For each of the 42 components:

* Record:

  * License bucket
  * Provenance confidence
  * Ownership category
  * Risk tier
* Cite **exact evidence**:

  * LICENSE file
  * README
  * `.gitmodules`
  * Header attribution comments

No evidence → mark **UNKNOWN**.

---

### Step 3 — Produce Risk Matrix

Create a consolidated table:

```
docs/refactor/sbom/M10_RISK_MATRIX.md
```

Including columns:

| Component | License | Provenance | Ownership | Risk | Notes |
| --------- | ------- | ---------- | --------- | ---- | ----- |

---

### Step 4 — Identify Follow-Up Candidates

Explicitly list:

* Components that likely need:

  * license verification
  * ownership clarification
  * future CI enforcement
* Tie each item to:

  * an **existing deferred issue** (preferred), or
  * a **new, clearly scoped future milestone**

---

## 7. Verification Plan

Verification is **internal consistency + traceability**.

### Required Checks

* Every classification links back to M09 evidence
* No component silently dropped or added
* Unknowns are preserved, not “resolved”
* Risk tiers are explained, not implied

---

## 8. Deliverables

| Artifact                                        | Purpose                             |
| ----------------------------------------------- | ----------------------------------- |
| `docs/refactor/sbom/M10_RISK_MATRIX.md`         | Component-level risk classification |
| `docs/refactor/milestones/M10/M10_plan.md`      | Milestone plan                      |
| `docs/refactor/milestones/M10/M10_toolcalls.md` | Tool log                            |
| `docs/refactor/milestones/M10/M10_audit.md`     | Audit report                        |
| `docs/refactor/milestones/M10/M10_summary.md`   | Milestone summary                   |

---

## 9. Definition of Done

M10 is complete when:

* [ ] All 42 components are classified
* [ ] Every classification cites evidence
* [ ] Risk matrix is reviewable and coherent
* [ ] No runtime / CI / SBOM files modified
* [ ] Audit confirms analysis-only posture
* [ ] Follow-up work is clearly scoped (but not executed)

---

## 10. Notes for Cursor (Explicit)

* Do **not** fix anything
* Do **not** “improve” licenses
* Do **not** fetch submodules
* Prefer **UNKNOWN** to speculation
* This milestone informs decisions — it does not make them

---

## M10’s Role in the Program

After M10, the project will have:

1. **Visibility** (M09)
2. **Understanding** (M10)

Only *after that* does enforcement, remediation, or automation become justified.

---

If you want next:

* I can pre-author the **risk tier definitions** in prose for reuse later, or
* Draft **M11** (likely: SBOM drift detection or CI integration) while M10 runs.

But this plan is **locked, scoped, and ready for Cursor**.
