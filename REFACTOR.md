# PyTorch Refactoring Program — Living Log

**Repository:** pytorch/pytorch (fork: m-cahill/pytorch)  
**Program Start:** 2026-02-08  
**Baseline Commit:** c5f1d40892292ef79cb583a8df00ceb1c8812a12  
**Current Phase:** Phase 0 Complete → Phase 1 Ready

---

## Project Context — AI-Native Refactoring Program

This project applies an **AI-native, audit-first refactoring methodology** to an existing, large-scale production codebase.

The program is **not greenfield**. The default posture is **behavior preservation**, with all changes evaluated against explicitly declared **invariants**, **blast-radius awareness**, and **verifiable evidence** (tests, CI signals, documented gaps).

Work proceeds in **small, non-overlapping milestones**, each with:

* a clearly stated intent and scope boundary,
* explicit non-goals,
* defined verification methods,
* and a safe rollback plan.

A completed **Phase 0 (M00) baseline audit** establishes the authoritative starting point:

* system surfaces,
* invariants,
* risks,
* CI posture,
* security and supply-chain considerations,
* and accepted evidence gaps.

All subsequent milestones (M01+) must:

* reference the Phase 0 audit,
* avoid implicit behavior changes,
* treat CI as a truth signal (not a suggestion),
* and produce objective proof that correctness is preserved.

Cursor is expected to operate as a **refactoring assistant under governance**, not as a code generator or optimizer. When in doubt, prefer **restraint, documentation, and explicit deferral** over speculative fixes.

---

## Repository Facts (Baseline)

| Metric | Value | Source |
|--------|-------|--------|
| **Total Tracked Files** | 20,440 | `git ls-files` (2026-02-08) |
| **Python Files** | 4,216 | File count |
| **C/C++ Files** | 4,403 | File count |
| **CUDA Files** | 345 | File count |
| **Python Test Files** | 1,353+ | `test/` directory |
| **C++ Test Files** | 279+ | `test/cpp/`, `c10/test/`, etc. |
| **CI Workflows** | 130+ | `.github/workflows/` |
| **Estimated Test Coverage** | 70-75% | CI inference |
| **Python Version** | 3.12.10 | Development environment |

---

## Governance Structure

**This document (`REFACTOR.md`)** = Living refactor log (updated after each milestone)

**Audit Pack (`docs/refactor/audit/`)** = Immutable baseline (Phase 0 snapshot, commit c5f1d40)

### Key References

- **Baseline Audit:** [`docs/refactor/audit/BASELINE_AUDIT.md`](docs/refactor/audit/BASELINE_AUDIT.md)
- **System Surfaces:** [`docs/refactor/audit/SYSTEM_SURFACES.md`](docs/refactor/audit/SYSTEM_SURFACES.md)
- **Invariants Catalog:** [`docs/refactor/audit/INVARIANTS_CATALOG.md`](docs/refactor/audit/INVARIANTS_CATALOG.md)
- **Milestone Roadmap:** [`docs/refactor/audit/REFACTOR_PHASE_MAP.md`](docs/refactor/audit/REFACTOR_PHASE_MAP.md)
- **Next Actions:** [`docs/refactor/audit/NEXT_ACTIONS.md`](docs/refactor/audit/NEXT_ACTIONS.md)
- **Audit Pack Index:** [`docs/refactor/audit/README.md`](docs/refactor/audit/README.md)

---

## Milestone History

### Phase 0: Foundation

#### M00 — Baseline Audit ✅ COMPLETE

**Status:** 🔒 Closed and Locked  
**Date:** 2026-02-08  
**Mode:** BASELINE RESET  
**Scope:** Documentation & audit only

**Intent:**
Establish an authoritative, evidence-based baseline for the PyTorch refactoring program through comprehensive audit of codebase, CI infrastructure, testing, security posture, and module boundaries.

**Deliverables:**
- ✅ 11 audit documents (~15,000 lines) in `docs/refactor/audit/`
- ✅ Top 7 issues identified and triaged (P0-P2)
- ✅ 22 milestones planned (M00-M22)
- ✅ 80+ invariants cataloged with verification methods
- ✅ Phase 0 non-goals explicitly stated
- ✅ 7 evidence gaps acknowledged and documented
- ✅ Governance structure established

**Key Findings:**
- **Wins:** Production-grade CI (130+ workflows), comprehensive tests (1,600+ files), modern dependency management, modular architecture
- **Top Risks:** No local build environment (CI-only), 130+ workflows with potential silent failures, mixed action pinning (supply chain risk), large third-party footprint
- **Coverage:** 70-75% estimated (actual metrics deferred to M01+)
- **Security Score:** 5.2/10 baseline (target 7/10 after Phase 1)

**Non-Goals (Honored):**
- ❌ No behavior changes
- ❌ No refactors
- ❌ No formatting sweeps
- ❌ No dependency upgrades
- ❌ No CI enforcement changes
- ❌ No performance work
- ❌ No test rewrites

**Evidence Gaps (Accepted):**
- No local build/test execution (per AGENTS.md constraints)
- No runtime coverage metrics (estimated from structure)
- No fork-level CI execution (observational only)
- No SBOM generated (planned for M08)
- CI behavior inferred from workflow structure
- No dependency version audit (planned for M08-M09)
- No ABI compatibility baseline (planned for M15)

**Outcome:**
- Baseline locked at commit c5f1d40
- Audit pack is now immutable reference
- Ready to proceed to M01 (Import Smoke Test)

**Verification:**
- ✅ All scope boundaries honored (documentation-only)
- ✅ All invariants preserved (zero behavioral impact)
- ✅ All quality gates passed
- ✅ Third-party reviewable

**Closeout Artifacts:**
- [`docs/refactor/milestones/M00/M00_summary.md`](docs/refactor/milestones/M00/M00_summary.md)
- [`docs/refactor/milestones/M00/M00_audit.md`](docs/refactor/milestones/M00/M00_audit.md)
- [`docs/refactor/milestones/M00/M00_closeout.md`](docs/refactor/milestones/M00/M00_closeout.md)

**Effort:** ~40 hours (audit pack creation + closeout)

---

### Phase 1: CI Health & Guardrails (Planned)

#### M01 — Import Smoke Test 🟡 IN PROGRESS

**Status:** Implementation Complete (pending PR)  
**Priority:** P0 (Critical)  
**Effort:** 4 hours  
**Branch:** `m01-import-smoke-test`

**Intent:**
Create static import-graph checker that validates Python import integrity without requiring a C++ build. Establishes minimal "truth surface" for all future refactor validation.

**Scope:**
- ✅ Created `tools/refactor/import_smoke_static.py` — Static AST-based import analyzer
- ✅ Created `test/test_import_smoke_static.py` — Test suite (8 tests)
- ✅ Created `.github/workflows/refactor-smoke.yml` — Isolated CI workflow
- ✅ Allowlist for compiled/generated/optional modules

**Locked Targets (M01):**
- `torch`
- `torch.nn`
- `torch.optim`
- `torch.utils`
- `torch.autograd`

**Invariants Protected:**
- INV-050: Import Path Stability

**Verification Results:**
```
Files scanned:       2,372
Total imports found: 21,254
Unresolved imports:  0
Tests:               8/8 passed
```

**See:** [`docs/refactor/milestones/M01/M01_plan.md`](docs/refactor/milestones/M01/M01_plan.md)

---

#### M02 — Populate REFACTOR.md 🔵 PLANNED

**Status:** Blocked by M01  
**Priority:** P0 (Critical)  
**Effort:** 2 hours  

**Intent:**
Activate governance by populating this document with architectural principles, deprecation policy, and milestone tracking.

**Note:** Initial population complete with M00 entry. M02 will add architectural principles and expand governance sections.

**See:** [`docs/refactor/audit/NEXT_ACTIONS.md`](docs/refactor/audit/NEXT_ACTIONS.md) — Action 2

---

#### M03-M10 — CI Health (Planned)

See [`docs/refactor/audit/REFACTOR_PHASE_MAP.md`](docs/refactor/audit/REFACTOR_PHASE_MAP.md) for full Phase 1 plan.

---

## Architectural Principles

*(To be expanded in M02)*

1. **Behavior Preservation First** — All changes must prove correctness
2. **Evidence-Based Decisions** — No changes without measurable justification
3. **Small, Verifiable Milestones** — PR-sized work with clear rollback
4. **Invariant Protection** — 80+ cataloged invariants must be verified
5. **CI as Truth Signal** — Green CI is necessary, not sufficient

---

## Deprecation Policy

*(To be expanded in M02)*

**Standard Cycle:** 2 releases
- Release N: Add deprecation warning
- Release N+1: Warning remains
- Release N+2: Remove deprecated feature

See [`docs/refactor/audit/INVARIANTS_CATALOG.md`](docs/refactor/audit/INVARIANTS_CATALOG.md) — INV-060 for details.

---

## Active Risks (P0-P1)

From baseline audit, top risks requiring mitigation:

| ID | Risk | Priority | Status | Milestone |
|----|------|---------|--------|-----------|
| **I01** | No Working Build Environment | P0 | 🔵 Active | M01 (mitigation) |
| **I02** | Empty REFACTOR.md | P0 | ✅ Resolved | M00-M02 |
| **I03** | 130+ CI Workflows (Maintenance) | P1 | 🔵 Active | M03-M05 |
| **I04** | Mixed Action Pinning | P1 | 🔵 Active | M06-M07 |
| **I05** | Third-Party Supply Chain Risk | P1 | 🔵 Active | M08-M10 |
| **I06** | Implicit Distributed Protocol | P2 | 🔵 Active | M11-M12 |
| **I07** | No Pre-Commit Hooks | P2 | 🔵 Active | M13-M14 |

---

## Milestone Progress

| Phase | Milestones | Complete | In Progress | Planned |
|-------|-----------|----------|-------------|---------|
| **Phase 0** | M00-M02 | 1 (M00) | 0 | 1 (M01) |
| **Phase 1** | M03-M10 | 0 | 0 | 8 |
| **Phase 2** | M11-M14 | 0 | 0 | 4 |
| **Phase 3** | M15-M19 | 0 | 0 | 5 |
| **Phase 4** | M23-M30 | 0 | 0 | TBD |

**Program Progress:** 1/22 milestones complete (5%)  
**Estimated Remaining:** ~204 hours (M01-M19)

---

## Score Trend

| Date | Phase | Architecture | Tests | CI | Security | Velocity |
|------|-------|-------------|-------|----|---------| ---------|
| 2026-02-08 | M00 (Baseline) | 8/10 | 7/10 | 8/10 | 6/10 | N/A |

**Targets (Post-Phase 3):**
- Architecture: Maintain 8/10
- Tests: Improve to 8/10
- CI: Maintain 8/10
- Security: Improve to 7/10
- Velocity: Establish baseline after M01

---

## Tool Logging

All tool invocations are logged in: [`docs/refactor/toolcalls.md`](docs/refactor/toolcalls.md)

Milestone-specific logs are in: `docs/refactor/milestones/MNN/MNN_toolcalls.md`

---

## Recovery Protocol

If session is interrupted:

1. Read [`docs/refactor/toolcalls.md`](docs/refactor/toolcalls.md) for last recorded action
2. Report: last action, next planned action, completion status
3. Wait for user confirmation before resuming

For milestone-specific recovery, consult: `docs/refactor/milestones/MNN/MNN_toolcalls.md`

---

## Document Version

**Last Updated:** 2026-02-08 (M00 closeout)  
**Next Update:** M01 completion  
**Baseline Locked:** Commit c5f1d40

---

**End of REFACTOR.md**

