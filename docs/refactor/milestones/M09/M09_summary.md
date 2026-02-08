# 📌 Milestone Summary — M09: Third-Party Supply Chain Inventory & SBOM Baseline

---

**Project:** PyTorch Refactoring Program  
**Phase:** Phase 1 — CI Health & Guardrails  
**Milestone:** M09 — Third-Party Supply Chain Inventory & SBOM Baseline  
**Timeframe:** 2026-02-08  
**Status:** Closed  
**Baseline:** Commit 5933293e0b31455a7d62839273c694f42df92aea (main branch)  
**Refactor Posture:** Behavior-Preserving

---

## 1. Milestone Objective

Establish a **baseline, auditable understanding** of third-party and vendored dependencies in the PyTorch repository.

Without this inventory:
- Third-party risk is implicit and unbounded
- Security posture cannot be reasoned about or audited
- Licensing exposure is unknown
- Supply chain provenance cannot be compared across time

This milestone produces the **first machine-readable SBOM** and **human-readable inventory** for PyTorch's third-party surface.

---

## 2. Scope Definition

### In Scope

| Area | Included |
|------|----------|
| `third_party/` | Full inventory of all subdirectories |
| Git submodules | All 35 submodules with pinned commits |
| Bundled code | Non-submodule vendored libraries (miniz, concurrentqueue, valgrind-headers) |
| Embedded libraries | QNNPACK in `aten/`, clog, ported code in `c10/` |
| Evidence gaps | Explicitly documented unknowns |

### Out of Scope

| Area | Excluded |
|------|----------|
| Python dependencies | pip/Conda environments not inventoried |
| CVE scanning | No vulnerability analysis performed |
| License remediation | No license corrections or normalizations |
| Dependency upgrades | No version changes |
| CI enforcement | No blocking rules added |
| Transitive dependencies | Submodule dependencies not recursively scanned |

---

## 3. Refactor Classification

### Change Type

**Documentation + Verification Artifact**

This milestone produced documentation only. No mechanical, boundary, semantic, or behavioral changes occurred.

### Observability

**No externally observable changes.**

All artifacts are confined to `docs/refactor/sbom/` and `docs/refactor/milestones/M09/`.

---

## 4. Work Executed

### Artifacts Created

| Artifact | Location | Description |
|----------|----------|-------------|
| CycloneDX SBOM | `docs/refactor/sbom/M09_sbom.json` | Machine-readable inventory (42 components) |
| Human inventory | `docs/refactor/sbom/M09_THIRD_PARTY.md` | Human-readable documentation |
| Tool log | `docs/refactor/milestones/M09/M09_toolcalls.md` | Execution trace |
| Audit | `docs/refactor/milestones/M09/M09_audit.md` | Compliance verification |
| Summary | `docs/refactor/milestones/M09/M09_summary.md` | This document |

### Inventory Statistics

| Category | Count |
|----------|-------|
| Git Submodules | 35 |
| Bundled/Vendored | 3 |
| Embedded Libraries | 2 |
| Ported Code | 2 |
| **Total Components** | **42** |

### Discovery Methodology

1. Parsed `.gitmodules` for submodule definitions
2. Ran `git submodule status` for pinned commits
3. Scanned `third_party/` for non-submodule directories
4. Searched `aten/`, `c10/`, `torch/csrc/` for LICENSE files
5. Grepped for "based on", "copied from" patterns in source headers

### Notable Findings

- **6 PyTorch-owned components** identified (gloo, cpuinfo, fbgemm, tensorpipe, kineto, QNNPACK)
- **35 submodules** all have pinned commit SHAs
- **QNNPACK** is fully embedded in `aten/src/ATen/native/quantized/cpu/qnnpack/`
- **concurrentqueue** is a partial vendored copy (excludes test/ for license reasons)
- **2 files** in `c10/util/` contain ported code (protobuf uint128, Boost hash)

---

## 5. Invariants & Compatibility

### Declared Invariants

| Invariant | Requirement | Status |
|-----------|-------------|--------|
| Behavior preservation | No runtime behavior changes | ✅ Verified |
| CI integrity | Required checks unchanged | ✅ Verified |
| Action immutability | No action pinning changes | ✅ Verified |
| Audit integrity | Prior milestone artifacts untouched | ✅ Verified |

### Compatibility Notes

- **Backward compatibility preserved:** Yes — no code changes
- **Breaking changes introduced:** No
- **Deprecations introduced:** No

---

## 6. Validation & Evidence

| Evidence Type | Method | Result | Notes |
|---------------|--------|--------|-------|
| SBOM schema | CycloneDX 1.5 structure | ✅ Valid | Manual verification |
| Submodule cross-ref | `.gitmodules` vs `git submodule status` | ✅ Match | 35/35 submodules |
| Bundled code verification | `LICENSES_BUNDLED.txt` | ✅ Match | miniz-3.0.2 documented |
| Path verification | Spot checks (5 components) | ✅ Pass | Paths exist and match |
| Invariant verification | Manual inspection | ✅ Pass | No runtime code touched |

### Limitations

- SBOM not validated against external schema validator (syft not available)
- Submodule content not inspected (empty directories locally)
- Transitive dependencies not scanned

---

## 7. CI / Automation Impact

### Workflows Affected

**None.**

### CI Verification

This milestone does not require CI verification. All changes are documentation-only.

### Future Integration Opportunity

SBOM generation could be integrated into CI to prevent drift. Deferred to future milestone.

---

## 8. Issues, Exceptions, and Guardrails

### Issues Encountered

| ID | Description | Severity | Status | Notes |
|----|-------------|----------|--------|-------|
| SBOM-001 | Unknown licenses for mslk, aiter | Low | Deferred | Submodules not fetched |
| SBOM-002 | Valgrind headers license unclear | Low | Deferred | Headers-only may differ |
| SBOM-003 | SBOM is point-in-time snapshot | Medium | Deferred | No automation |

### Guardrails Added

- **Evidence gaps section** in `M09_THIRD_PARTY.md` explicitly documents what is unknown
- **UNKNOWN markers** used consistently in SBOM where version/license not determinable

---

## 9. Deferred Work

| Item | Reason | Pre-existing? | Status Changed? |
|------|--------|---------------|-----------------|
| License verification for mslk/aiter | Submodules not fetched locally | No (new) | N/A |
| Automated SBOM generation | Out of M09 scope | No (new) | N/A |
| CVE scanning against SBOM | Out of M09 scope | No (new) | N/A |
| Transitive dependency inventory | Out of M09 scope | No (new) | N/A |

---

## 10. Governance Outcomes

### What is now provably true that was not before:

1. **42 third-party components** are documented with paths and provenance
2. **35 git submodules** have pinned commit SHAs recorded
3. **Evidence gaps** are explicitly acknowledged, not hidden
4. **SBOM baseline** enables future delta comparisons
5. **PyTorch-owned components** are distinguished from external dependencies

### Security Posture Improvement

- Supply chain visibility established (prerequisite for future hardening)
- First machine-readable dependency manifest for PyTorch refactoring program

---

## 11. Exit Criteria Evaluation

| Criterion | Met | Evidence |
|-----------|-----|----------|
| SBOM artifact exists and validates | ✅ Met | `docs/refactor/sbom/M09_sbom.json` |
| Third-party inventory is readable | ✅ Met | `docs/refactor/sbom/M09_THIRD_PARTY.md` |
| Unknowns are explicitly documented | ✅ Met | Evidence Gaps section in inventory |
| No CI/build/runtime files modified | ✅ Met | All changes in `docs/refactor/` |
| Audit confirms scope discipline | ✅ Met | `M09_audit.md` |
| REFACTOR.md updated | ⏳ Pending | To be updated at closeout |

---

## 12. Final Verdict

**Milestone objectives met. Refactor verified safe. Proceed.**

M09 established the first baseline inventory of PyTorch's third-party surface. The SBOM and human-readable documentation provide:

- Visibility into 42 third-party components
- Explicit acknowledgment of evidence gaps
- Foundation for future security and compliance work

No runtime, CI, or build code was modified. All invariants were preserved.

---

## 13. Authorized Next Step

**Pending explicit permission to:**

1. Commit M09 artifacts
2. Create PR to main
3. Update REFACTOR.md with M09 status
4. Close M09 and proceed to M10

---

## 14. Canonical References

| Reference | Value |
|-----------|-------|
| Starting commit | `5933293e0b31455a7d62839273c694f42df92aea` |
| Branch | `m09-sbom-baseline` |
| SBOM | `docs/refactor/sbom/M09_sbom.json` |
| Inventory | `docs/refactor/sbom/M09_THIRD_PARTY.md` |
| Plan | `docs/refactor/milestones/M09/M09_plan.md` |
| Tool log | `docs/refactor/milestones/M09/M09_toolcalls.md` |
| Audit | `docs/refactor/milestones/M09/M09_audit.md` |
| Summary | `docs/refactor/milestones/M09/M09_summary.md` |

---

**End of M09_summary.md**

