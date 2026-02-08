# M10 Summary — Third-Party Risk & License Classification

**Milestone:** M10 — Third-Party Risk & License Classification (SBOM Analysis)  
**Status:** ✅ Complete — Pending Closeout Approval  
**Date:** 2026-02-08  
**Effort:** ~2 hours  
**Change Class:** Pure Analysis + Documentation

---

## Executive Summary

M10 converted the M09 SBOM baseline into actionable risk intelligence by classifying all 42 third-party components across four axes: license, provenance, ownership, and risk tier.

**Key outcome:** PyTorch's third-party supply chain is **fundamentally healthy**. 76% of components are Low risk (permissive license, high provenance). Only 3 components require immediate attention due to unknown or copyleft licensing.

---

## What Was Accomplished

### Primary Deliverable

Created `docs/refactor/sbom/M10_RISK_MATRIX.md` containing:
- Complete classification of all 42 components
- Deterministic risk tier assignments using locked framework
- Evidence citations for every classification
- Follow-up item registry with scoped recommendations

### Risk Distribution

| Risk Tier | Count | Percentage | Action |
|-----------|-------|------------|--------|
| **High** | 3 | 7% | Requires attention |
| **Medium** | 1 | 2% | Monitor |
| **Low** | 32 | 76% | No action |
| **Informational** | 4 | 10% | Dev/test only |
| **Total** | **42** | 100% | — |

### High-Risk Components Identified

| Component | Issue | Follow-Up |
|-----------|-------|-----------|
| **mslk** | Unknown license (Meta library, no LICENSE file found in repo) | SBOM-002 |
| **aiter** | Unknown license (ROCm library, no LICENSE file found in repo) | SBOM-002 |
| **valgrind-headers** | GPL-2.0+ (headers-only, terms unclear) | NEW-001 |

### License Distribution

| Category | Count | Examples |
|----------|-------|----------|
| Permissive | 37 | MIT, BSD-2/3-Clause, Apache-2.0, BSL-1.0 |
| Strong Copyleft | 1 | valgrind-headers (GPL) |
| Unknown | 2 | mslk, aiter |
| Custom/Dual | 1 | concurrentqueue (BSD-2/BSL-1.0) |

---

## What Was NOT Done (Scope Honored)

- ❌ No license remediation
- ❌ No dependency upgrades
- ❌ No CVE/vulnerability scanning
- ❌ No CI enforcement changes
- ❌ No SBOM regeneration
- ❌ No external repository fetching
- ❌ No legal interpretation of GPL exceptions

---

## Key Insights

### 1. Supply Chain Maturity

PyTorch's third-party dependency posture is mature:
- 35 of 42 components are git submodules with pinned SHAs
- 6 components are PyTorch-owned (gloo, cpuinfo, fbgemm, tensorpipe, kineto, QNNPACK)
- Well-known, permissively-licensed libraries dominate

### 2. Unknown License Concentration

Both unknown-license components are:
- Recent additions (ROCm/AMD ecosystem: aiter; Meta ecosystem: mslk)
- From trusted organizations but lacking in-repo license evidence
- Easily remediable by fetching LICENSE from upstream

### 3. Copyleft Risk is Minimal

Only one component (valgrind-headers) has copyleft licensing:
- Headers-only (2 files: callgrind.h, valgrind.h)
- GPL header exception likely applies but not verified
- Low practical risk given headers-only usage

### 4. PyTorch-Owned Components

Six components are under `pytorch/` organization:
- All have permissive (BSD-3-Clause or BSD-2-Clause) licenses
- Treated as Low risk but noted for governance alignment with M06-B

---

## Follow-Up Items Created

### Tied to Existing Deferrals

| ID | Components | Action |
|----|------------|--------|
| **SBOM-002** | mslk, aiter | Verify licenses from upstream repositories |

### New Follow-Ups

| ID | Components | Priority | Suggested Milestone |
|----|------------|----------|---------------------|
| **NEW-001** | valgrind-headers | Medium | M11+ — Verify GPL header exception |
| **NEW-002** | concurrentqueue | Low | M11+ — Review dual-license implications |
| **NEW-003** | PyTorch-owned (6) | Informational | M11+ — Document release/governance practices |

---

## Invariants Verified

| Invariant | Status |
|-----------|--------|
| Behavior preservation (no code touched) | ✅ Pass |
| CI integrity (no workflows modified) | ✅ Pass |
| SBOM immutability (M09 artifacts unchanged) | ✅ Pass |
| Audit continuity (all classifications trace to M09) | ✅ Pass |

---

## Artifacts Produced

| Artifact | Purpose |
|----------|---------|
| `docs/refactor/sbom/M10_RISK_MATRIX.md` | Component-level risk classification |
| `docs/refactor/milestones/M10/M10_toolcalls.md` | Tool invocation log |
| `docs/refactor/milestones/M10/M10_audit.md` | Compliance verification |
| `docs/refactor/milestones/M10/M10_summary.md` | This document |

---

## Relationship to Program Goals

### M10's Role in Phase 1

M10 completes the supply chain visibility arc started in M09:

| Milestone | Contribution |
|-----------|--------------|
| M09 | Established **visibility** (what exists) |
| M10 | Established **understanding** (what matters) |
| M11+ | Will establish **enforcement** (what to do) |

### Security Score Impact

| Metric | Before M10 | After M10 |
|--------|------------|-----------|
| Security Score | 7/10 | 7/10 (maintained) |
| Supply Chain Visibility | Inventory only | Risk-classified |
| Actionable Intelligence | Limited | High-risk items identified |

M10 does not change the security score directly but provides the foundation for future improvements.

---

## Next Steps (For Future Milestones)

1. **SBOM-002 resolution:** Fetch LICENSE files from mslk and aiter upstream repositories
2. **NEW-001 investigation:** Verify valgrind GPL header exception posture
3. **Consider M11:** SBOM drift detection or CI integration for supply chain monitoring

---

## Conclusion

M10 successfully converted the M09 SBOM baseline into actionable risk intelligence. The analysis confirms PyTorch's third-party supply chain is healthy, with only 3 high-risk components requiring future attention.

**Milestone Status:** ✅ Complete — Awaiting closeout permission

---

**End of M10_summary.md**

