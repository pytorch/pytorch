# M09 Milestone Audit — Third-Party Supply Chain Inventory & SBOM Baseline

---

## Header

| Field | Value |
|-------|-------|
| **Milestone** | M09 |
| **Mode** | DELTA AUDIT |
| **Range** | 5933293e0b3...TBD (single branch, docs only) |
| **CI Status** | N/A (documentation-only milestone) |
| **Refactor Posture** | Behavior-Preserving |
| **Audit Verdict** | 🟢 PASS — Documentation milestone completed; no runtime changes |

---

## Executive Summary (Delta-First)

### Wins

1. **SBOM baseline established** — 42 third-party components inventoried in machine-readable CycloneDX JSON format
2. **Human-readable inventory created** — Comprehensive documentation of all vendored dependencies with provenance
3. **Evidence gaps explicitly documented** — 4 categories of unknowns recorded (versions, licenses, provenance, scope limits)
4. **PyTorch-owned components identified** — 6 components (gloo, cpuinfo, fbgemm, tensorpipe, kineto, QNNPACK) flagged for future governance

### Risks

1. **License gaps** — 2 submodules (mslk, aiter) have unknown licenses requiring follow-up
2. **No automated SBOM generation** — Manual inventory is a point-in-time snapshot, not continuously updated
3. **Transitive dependencies not scanned** — Submodule dependencies not inventoried (depth-1 only)
4. **No CVE correlation** — SBOM exists but no vulnerability scanning performed (out of scope)

### Most Important Next Action

Integrate SBOM generation into CI to prevent drift (candidate for M10 or future milestone).

---

## Delta Map & Blast Radius

### What Changed

| Category | Files Added | Files Modified | Files Deleted |
|----------|-------------|----------------|---------------|
| Documentation | 2 | 0 | 0 |
| Milestone artifacts | 3 | 1 | 0 |
| **Total** | **5** | **1** | **0** |

### New Files

- `docs/refactor/sbom/M09_sbom.json` — CycloneDX 1.5 SBOM (42 components)
- `docs/refactor/sbom/M09_THIRD_PARTY.md` — Human-readable inventory
- `docs/refactor/milestones/M09/M09_audit.md` — This file
- `docs/refactor/milestones/M09/M09_summary.md` — Milestone summary

### Modified Files

- `docs/refactor/milestones/M09/M09_toolcalls.md` — Tool invocation log

### Consumer Surfaces Touched

**None.** This milestone produced documentation artifacts only.

### Blast Radius

**Zero runtime impact.** All changes are confined to `docs/refactor/` directory. No CI, build, test, or production code modified.

---

## Architecture & Modularity Review

### Assessment

| Question | Answer |
|----------|--------|
| Boundary violations introduced? | No |
| Coupling added that blocks extraction? | No |
| Dead abstractions created? | No |
| Layering leaks? | No |
| ADR/doc updates needed? | No (this IS the documentation update) |

### Classification

- **Keep:** All changes
- **Fix now:** None
- **Defer:** None

---

## CI/CD & Workflow Audit

### CI Impact

**None.** No workflows added, modified, or affected.

### Verification

This is a documentation-only milestone. CI verification is not applicable, but:

- SBOM JSON validates against CycloneDX 1.5 schema (structure verified manually)
- Inventory cross-references verified against `.gitmodules` and `git submodule status`

---

## Tests, Coverage, and Invariants

### Invariant Verification

| Invariant | Verification Method | Status |
|-----------|---------------------|--------|
| Behavior preservation | No runtime code touched | ✅ PASS |
| CI integrity | No workflows touched | ✅ PASS |
| Action immutability | No action references touched | ✅ PASS |
| Audit integrity | Prior milestone artifacts unchanged | ✅ PASS |

### Coverage Impact

**None.** No code changes.

### New Tests

**None required.** Documentation milestone.

---

## Security & Supply Chain

### SBOM Status

| Metric | Value |
|--------|-------|
| Components inventoried | 42 |
| Submodules with pinned commits | 35/35 |
| Bundled components with version | 1/3 (miniz only) |
| Components with license identified | 38/42 |
| Components with unknown license | 2 (mslk, aiter) + 2 partial (valgrind headers, embedded code) |

### Provenance Improvements

- All submodule commit SHAs recorded
- All upstream URLs documented
- PyTorch-owned components explicitly flagged

### Secrets Exposure

**None.** No secrets added or referenced.

### Workflow Trust Boundary

**Unchanged.** No CI modifications.

---

## Refactor Guardrail Compliance Check

| Guardrail | Status | Notes |
|-----------|--------|-------|
| Invariant declaration | ✅ PASS | 4 invariants declared in M09_plan, all verified |
| Baseline discipline | ✅ PASS | Starting commit documented (5933293e0b3) |
| Consumer contract protection | ✅ N/A | No consumer surfaces touched |
| Extraction/split safety | ✅ N/A | No extraction performed |
| No silent CI weakening | ✅ PASS | No CI changes |

---

## Top Issues (Ranked)

### SBOM-001: Missing Licenses for Two Submodules (Low)

**Observation:** `third_party/mslk` and `third_party/aiter` have no LICENSE file discoverable from repository state.

**Evidence:** `git submodule status` shows commits, but submodules are not checked out (empty directories).

**Interpretation:** License compliance cannot be verified for these components without fetching submodule content.

**Recommendation:** Verify licenses when submodules are fetched in build environment (future milestone).

**Guardrail:** Add license verification to future SBOM automation.

**Rollback:** N/A (documentation)

---

### SBOM-002: Valgrind Headers License Ambiguity (Low)

**Observation:** `third_party/valgrind-headers/` contains GPL-licensed headers from Valgrind.

**Evidence:** `third_party/valgrind-headers/README.md` documents download source but not license terms.

**Interpretation:** Header-only includes may have different terms than full Valgrind (common pattern), but this is not verified.

**Recommendation:** Document license clarification in follow-up milestone.

**Guardrail:** N/A (informational)

**Rollback:** N/A (documentation)

---

### SBOM-003: SBOM is Point-in-Time Snapshot (Medium)

**Observation:** SBOM is manually generated; no automation ensures it stays current.

**Evidence:** `docs/refactor/sbom/M09_sbom.json` created via manual inventory.

**Interpretation:** SBOM may drift from actual dependencies as submodules are updated.

**Recommendation:** Consider adding SBOM generation to CI (future milestone, potentially M10).

**Guardrail:** Future: Add periodic SBOM regeneration to CI.

**Rollback:** N/A

---

## PR-Sized Action Plan

| ID | Task | Category | Acceptance Criteria | Risk | Est |
|----|------|----------|---------------------|------|-----|
| 1 | Commit M09 artifacts | Docs | `git status` shows clean after commit | Low | 5m |
| 2 | Create PR to main | Process | PR created, CI green (if any) | Low | 5m |
| 3 | Update REFACTOR.md | Docs | M09 entry added with status | Low | 10m |

---

## Deferred Issues Registry (Cumulative Update)

| ID | Issue | Discovered | Deferred To | Reason | Blocker? | Exit Criteria |
|----|-------|------------|-------------|--------|----------|---------------|
| M04-V01 | Upstream CI verification | M04 | Upstream PR | Fork CI guarded | No | TD failure propagates |
| M06-V01 | PyTorch-owned @main actions | M06 | Future | Policy needed | No | Internal tagging established |
| M07-V01 | Dependabot runtime verification | M07 | Post-merge | Unobservable locally | No | Dependabot opens PR |
| **SBOM-001** | mslk/aiter license verification | **M09** | **Future** | **Submodules not fetched** | No | **Licenses verified** |
| **SBOM-003** | Automated SBOM generation | **M09** | **M10+** | **Out of scope** | No | **SBOM in CI** |

---

## Score Trend (Cumulative)

| Milestone | Invariants | Compat | Arch | CI | Sec | Tests | DX | Docs | Overall |
|-----------|------------|--------|------|-----|-----|-------|-----|------|---------|
| M00 | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 | 3.5 | 3.5 | 4.0 | 3.8 |
| M02 | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 | 3.5 | 3.5 | 4.5 | 3.8 |
| M08 | 4.5 | 4.0 | 4.0 | 4.5 | 3.5 | 3.5 | 3.5 | 4.5 | 4.0 |
| **M09** | **4.5** | **4.0** | **4.0** | **4.5** | **4.0** | **3.5** | **3.5** | **4.5** | **4.1** |

**Score Movement:**
- **Security +0.5:** SBOM baseline establishes first machine-readable dependency inventory
- **Overall +0.1:** Supply chain visibility improvement

---

## Flake & Regression Log

No flakes or regressions observed (documentation-only milestone).

---

## Machine-Readable Appendix

```json
{
  "milestone": "M09",
  "mode": "delta",
  "posture": "preserve",
  "commit": "TBD",
  "range": "5933293e0b3...TBD",
  "verdict": "green",
  "quality_gates": {
    "invariants": "pass",
    "compatibility": "pass",
    "ci": "pass",
    "tests": "pass",
    "coverage": "pass",
    "security": "pass",
    "dx_docs": "pass",
    "guardrails": "pass"
  },
  "issues": [
    {
      "id": "SBOM-001",
      "category": "security",
      "severity": "low",
      "evidence": "third_party/mslk, third_party/aiter - empty directories",
      "summary": "Unknown licenses for 2 submodules",
      "fix_hint": "Verify licenses when submodules fetched",
      "deferred": true
    },
    {
      "id": "SBOM-002",
      "category": "security",
      "severity": "low",
      "evidence": "third_party/valgrind-headers/README.md",
      "summary": "Valgrind header license terms unclear",
      "fix_hint": "Document license clarification",
      "deferred": true
    },
    {
      "id": "SBOM-003",
      "category": "security",
      "severity": "med",
      "evidence": "docs/refactor/sbom/M09_sbom.json - manual generation",
      "summary": "SBOM is point-in-time snapshot",
      "fix_hint": "Add SBOM generation to CI",
      "deferred": true
    }
  ],
  "deferred_registry_updates": [
    { "id": "SBOM-001", "deferred_to": "Future", "reason": "Submodules not fetched locally", "exit_criteria": "Licenses verified in build env" },
    { "id": "SBOM-003", "deferred_to": "M10+", "reason": "Out of M09 scope", "exit_criteria": "SBOM generation in CI" }
  ],
  "score_trend_update": {
    "invariants": 4.5,
    "compat": 4.0,
    "arch": 4.0,
    "ci": 4.5,
    "sec": 4.0,
    "tests": 3.5,
    "dx": 3.5,
    "docs": 4.5,
    "overall": 4.1
  }
}
```

---

**End of M09_audit.md**

