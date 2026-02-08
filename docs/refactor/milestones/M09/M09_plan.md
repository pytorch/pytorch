# M09_plan — SBOM Generation for Vendored Dependencies

## 1. Intent / Target

**Primary objective:**
Generate a Software Bill of Materials (SBOM) for PyTorch's vendored dependencies in `third_party/` to establish supply chain visibility.

This milestone addresses **I05 (Third-Party Supply Chain Risk)** from the baseline audit.

> **What would remain unsafe without this work?**  
> Without an SBOM, there is no machine-readable inventory of vendored dependencies. Security scanning, license compliance, and vulnerability tracking require manual inspection of 30+ vendored projects.

---

## 2. Scope Boundaries

### In Scope

* Generate SBOM in CycloneDX or SPDX format
* Document vendored dependencies in `third_party/`
* Create human-readable version summary
* Add CI workflow for SBOM generation (if feasible)

### Out of Scope

* No dependency upgrades
* No removal of vendored code
* No changes to build system
* No pip/conda SBOM (different ecosystem)
* No action pinning work (M06/M07 complete)

---

## 3. Invariants

| Invariant | Verification Method |
|-----------|---------------------|
| No product code changes | Diff limited to docs + CI |
| No build system changes | No CMake/setup.py modifications |
| Existing CI unaffected | SBOM generation is additive |

---

## 4. Verification Plan

**Success criteria:**

* SBOM file validates against schema
* All `third_party/` subdirectories represented
* Version information captured where available
* CI workflow runs successfully (if added)

---

## 5. Implementation Steps

1. **Inventory `third_party/`**
   * List all vendored dependencies
   * Identify version sources (CMakeLists.txt, README, etc.)

2. **Select SBOM tool**
   * Options: `syft`, `cyclonedx-cli`, manual generation
   * Constraint: Must run in CI (per AGENTS.md, no local installs)

3. **Generate SBOM**
   * CycloneDX JSON preferred
   * Include: name, version, license, source URL

4. **Create human-readable summary**
   * `THIRD_PARTY_VERSIONS.md` in docs/refactor/

5. **Add CI workflow (optional)**
   * Generate SBOM on schedule or release tags
   * Upload as artifact

6. **Update governance docs**
   * Record in REFACTOR.md

---

## 6. Risk & Rollback

### Risks

* Incomplete version detection for some vendored deps
* SBOM tool may not recognize custom vendoring structure

### Mitigation

* Manual entries for undetected dependencies
* Document gaps explicitly

### Rollback

* Delete SBOM file and workflow
* No other changes to revert

---

## 7. Deliverables

* `docs/refactor/SBOM.json` (or similar)
* `docs/refactor/THIRD_PARTY_VERSIONS.md`
* CI workflow (optional): `.github/workflows/sbom.yml`
* M09 summary + audit artifacts

---

## 8. Milestone Classification

* **Type:** Supply Chain / Documentation
* **Posture:** Additive (no behavior changes)
* **Expected Size:** Medium
* **Blast Radius:** None (documentation only)
* **Audit Mode:** DELTA AUDIT

---

## 9. Dependencies

* M08 complete (CI truthfulness established)
* No blocking dependencies on M06/M07

---

## 10. Definition of Done

- [ ] SBOM file created and validated
- [ ] Human-readable version summary created
- [ ] CI workflow added (or documented as deferred)
- [ ] REFACTOR.md updated
- [ ] M09_audit.md created
- [ ] M09_summary.md created

---

**End of M09 Plan**

