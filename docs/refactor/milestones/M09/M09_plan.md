Got it — thanks for the correction.
We’ll treat this as **one complete milestone**, not a split. The accidental “A” goes away.

Below is the **regenerated, authoritative plan for the full milestone**, named **M09**, suitable to hand directly to Cursor.

---

# **M09_plan — Third-Party Supply Chain Inventory & SBOM Baseline**

## 1. Intent / Target

Establish a **baseline, auditable understanding of third-party and vendored dependencies** in the repository by producing a **machine-readable SBOM** and a **human-readable third-party inventory**.

This milestone is about **visibility and traceability**, not remediation or enforcement.

> **What remains unsafe without this work?**
> Without a baseline inventory, third-party risk is implicit and unbounded. Security posture, licensing exposure, and provenance cannot be reasoned about, audited, or compared across time.

---

## 2. Scope Boundaries

### In Scope

**Third-party and vendored code discovery**, including but not limited to:

* `third_party/`
* Vendored or embedded libraries under:

  * `aten/`
  * `c10/`
  * `torch/csrc/`
* Generated vendored artifacts **only as inventory items** (not modification targets)

**Artifacts to be produced:**

* A **machine-readable SBOM**:

  * CycloneDX JSON (preferred), or
  * SPDX JSON (acceptable)
* A **human-readable inventory document** mapping:

  * component → location
  * version / commit (if known)
  * license (if detectable)
  * provenance notes
* Explicit documentation of **unknowns and evidence gaps**

### Explicitly Out of Scope

* ❌ Dependency upgrades
* ❌ License remediation or normalization
* ❌ CVE scanning or vulnerability scoring
* ❌ Runtime dependency resolution
* ❌ Python `pip` / Conda environment SBOM
* ❌ CI enforcement or blocking rules
* ❌ Generator refactors for vendored files

---

## 3. Refactor / Change Classification

**Documentation + Verification Artifact**

* No runtime behavior changes
* No CI semantics changes
* No build or test logic changes
* Low blast radius, fully reversible

---

## 4. Invariants (Must Hold)

| Invariant             | Requirement                         |
| --------------------- | ----------------------------------- |
| Behavior preservation | No runtime behavior changes         |
| CI integrity          | Required checks unchanged           |
| Action immutability   | No action pinning changes           |
| Audit integrity       | Prior milestone artifacts untouched |

If any invariant is at risk, stop and surface it immediately.

---

## 5. Verification Plan

Verification is **artifact-based**, not execution-based.

### Required Evidence

* SBOM file exists and validates against its schema
* Inventory document matches observed repository contents
* No inferred data presented as fact
* All ambiguity explicitly labeled

### Spot Checks

* Randomly sample 5–10 vendored components
* Confirm paths and metadata align with repo contents
* Ensure “unknown” is used instead of guessing

---

## 6. Implementation Steps (Ordered, Small, Reversible)

### Step 1 — Identify Third-Party Surfaces

* Enumerate directories containing vendored or embedded code
* Produce a table with:

  * path
  * classification (vendored / mirrored / generated)
  * suspected upstream (if known)

### Step 2 — Tooling Decision (Documented)

* Prefer filesystem-based SBOM generation using:

  * `syft` **if available**, otherwise
  * a **manual structured inventory**
* Do **not** install new tools unless explicitly permitted
* If tooling is unavailable, document the limitation and proceed manually

### Step 3 — Generate SBOM

* Generate an SBOM covering **only in-scope paths**
* Output location:

  ```
  docs/refactor/sbom/M09_sbom.json
  ```

### Step 4 — Human-Readable Inventory

Create:

```
docs/refactor/sbom/M09_THIRD_PARTY.md
```

For each component, include:

* Name
* Repo path(s)
* Version / commit (or `UNKNOWN`)
* License (if detectable)
* Provenance notes
* Confidence level (explicit or implied)

### Step 5 — Evidence Gaps & Ambiguity

Explicitly document:

* Components with no version metadata
* Copied code without upstream reference
* Generated artifacts where the generator is the true dependency
* Anything that cannot be proven from repository state alone

---

## 7. Risk & Rollback Plan

### Risks

| Risk            | Mitigation                         |
| --------------- | ---------------------------------- |
| False precision | Use `UNKNOWN` rather than guessing |
| Scope creep     | Limit strictly to vendored code    |
| Tooling limits  | Manual inventory fallback          |

### Rollback

* Remove `docs/refactor/sbom/`
* No other files affected
* No CI or runtime impact

---

## 8. Deliverables

| Artifact                                        | Purpose                  |
| ----------------------------------------------- | ------------------------ |
| `docs/refactor/sbom/M09_sbom.json`              | Machine-readable SBOM    |
| `docs/refactor/sbom/M09_THIRD_PARTY.md`         | Human-readable inventory |
| `docs/refactor/milestones/M09/M09_plan.md`      | Milestone plan           |
| `docs/refactor/milestones/M09/M09_toolcalls.md` | Tool log                 |
| `docs/refactor/milestones/M09/M09_audit.md`     | Audit report             |
| `docs/refactor/milestones/M09/M09_summary.md`   | Milestone summary        |

---

## 9. Definition of Done

M09 is complete when:

* [ ] SBOM artifact exists and validates
* [ ] Third-party inventory is readable and reviewable
* [ ] Unknowns are explicitly documented
* [ ] No CI, build, or runtime files modified
* [ ] Audit confirms scope discipline
* [ ] REFACTOR.md updated with M09 status

---

## 10. Notes for Cursor (Explicit Guidance)

* Operate **observationally**, not inferentially
* Do not normalize or “clean up” dependencies
* Unknowns are acceptable; undocumented assumptions are not
* Prefer fewer, accurate entries over speculative completeness
* Stop and ask if classification is ambiguous

---

If you want next:

* I can draft the **M09 audit acceptance checklist** (what must be true to close), or
* Map **M10** now that Phase 1 supply-chain visibility is almost complete.

But this plan is now **clean, unsplit, and ready for handoff**.
