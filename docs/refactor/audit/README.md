# PyTorch Refactoring Audit Pack

**Audit Date:** 2026-02-08  
**Baseline Commit:** c5f1d40  
**Repository:** pytorch/pytorch (fork: m-cahill/pytorch)  
**Auditor:** Cursor AI (Claude Sonnet 4.5)

---

## Purpose

This audit pack establishes the baseline for AI-Native refactoring of PyTorch. It documents the current state of the codebase, identifies risks, and provides a phased roadmap for safe refactoring.

**Status:** ✅ **BASELINE ESTABLISHED**

---

## Quick Navigation

| Document | Purpose | Read First? |
|----------|---------|-------------|
| [**BASELINE_AUDIT.md**](./BASELINE_AUDIT.md) | Comprehensive codebase audit (wins, risks, top issues) | ✅ Start here |
| [**NEXT_ACTIONS.md**](./NEXT_ACTIONS.md) | Top 10 next actions (PR-sized tasks) | ✅ Then read this |
| [**SYSTEM_SURFACES.md**](./SYSTEM_SURFACES.md) | External interfaces that must not break | 🔍 When planning refactors |
| [**INVARIANTS_CATALOG.md**](./INVARIANTS_CATALOG.md) | Behavior/API promises to preserve | 🔍 Before every refactor |
| [**REFACTOR_PHASE_MAP.md**](./REFACTOR_PHASE_MAP.md) | 22 milestones (M01-M22) with dependencies | 📅 For planning |
| [**MODULE_BOUNDARY_MAP.md**](./MODULE_BOUNDARY_MAP.md) | Module structure, coupling, refactor risks | 🗺️ When restructuring |
| [**TESTING_GAPS_AND_PLAN.md**](./TESTING_GAPS_AND_PLAN.md) | Test coverage gaps + fixes | 🧪 When adding tests |
| [**CI_GAPS_AND_GUARDRAILS.md**](./CI_GAPS_AND_GUARDRAILS.md) | CI reliability improvements | ⚙️ When changing workflows |
| [**SECURITY_AND_SUPPLY_CHAIN_BASELINE.md**](./SECURITY_AND_SUPPLY_CHAIN_BASELINE.md) | Supply chain risks + mitigations | 🔒 For security review |
| [**DEFINITION_OF_DONE.md**](./DEFINITION_OF_DONE.md) | Success criteria for refactor program | ✅ For governance |

---

## Executive Summary (from BASELINE_AUDIT.md)

### Wins ✅

1. **Production-Grade CI**: 130+ workflows, extensive hardware coverage
2. **Comprehensive Tests**: 1,353+ Python tests, 279+ C++ tests, sharded execution
3. **Modern Dependency Management**: Migrating to `pyproject.toml`
4. **Modular Architecture**: Clear layering (c10 → aten → torch)
5. **Active Linting**: `ruff`, `lintrunner` configured
6. **Extensive Documentation**: 170+ markdown files, 73+ RST files

### Top Risks 🔴

1. **No Working Build Environment** (per AGENTS.md) - CI-only validation
2. **Empty REFACTOR.md** - No governance baseline until this audit
3. **130+ CI Workflows** - High maintenance burden, potential silent failures
4. **Mixed Action Pinning** - Supply chain risk (`@main` references)
5. **Large Third-Party Footprint** - Vendored deps, manual updates
6. **Implicit Distributed Protocol** - No version checks, cross-version failures possible
7. **No Pre-Commit Hooks** - Lint failures waste CI cycles

### One Next Action 🎯

**M01: Create minimal smoke test harness** - Verify Python imports without C++ build. Foundation for all future refactor validation.

---

## How to Use This Audit Pack

### Before Starting Any Refactor

1. **Read** [BASELINE_AUDIT.md](./BASELINE_AUDIT.md) - Understand the codebase
2. **Check** [SYSTEM_SURFACES.md](./SYSTEM_SURFACES.md) - Identify affected surfaces
3. **Review** [INVARIANTS_CATALOG.md](./INVARIANTS_CATALOG.md) - Ensure invariants are verified
4. **Consult** [MODULE_BOUNDARY_MAP.md](./MODULE_BOUNDARY_MAP.md) - Check coupling risks

### During Refactor

1. **Verify Invariants** - Run tests from invariants catalog
2. **Log Tool Calls** - Update `docs/refactor/milestones/MNN/MNN_toolcalls.md`
3. **Run Tests** - Use tiered testing (smoke → unit → integration → CI)

### After Refactor

1. **Update REFACTOR.md** - Document milestone completion
2. **Update Audit Pack** - If new risks discovered, update relevant audit doc
3. **Verify CI** - Ensure no silent failures introduced

---

## Key Statistics

| Metric | Value | Source |
|--------|-------|--------|
| **Total Tracked Files** | 20,440 | `git ls-files` |
| **Python Files** | 4,216 | File count |
| **C/C++ Files** | 4,403 | File count |
| **CUDA Files** | 345 | File count |
| **Python Test Files** | 1,353+ | `test/` directory |
| **C++ Test Files** | 279+ | `test/cpp/`, `c10/test/`, etc. |
| **CI Workflows** | 130+ | `.github/workflows/` |
| **Estimated LOC** | 1M+ | (Full count deferred) |
| **Test Coverage (Est.)** | 70-75% | CI inference |
| **CI Execution Time** | ~1 hour (sharded) | Workflow observation |

---

## Top Issues (Summary)

| ID | Issue | Priority | Milestone | Effort |
|----|-------|---------|-----------|--------|
| I01 | No Working Build Environment | P0 | M01 | 4h |
| I02 | Empty REFACTOR.md | P0 | M02 | 2h |
| I03 | 130+ CI Workflows (Maintenance Burden) | P1 | M03-M05 | 18h |
| I04 | Mixed Action Pinning | P1 | M06-M07 | 14h |
| I05 | Third-Party Supply Chain Risk | P1 | M08-M10 | 26h |
| I06 | Implicit Distributed Protocol Version | P2 | M11-M12 | 28h |
| I07 | No Pre-Commit Hooks | P2 | M13-M14 | 6h |

**Total Effort (P0-P1)**: ~64 hours (~1.5 weeks)

---

## Milestone Overview (22 Milestones)

| Phase | Milestones | Effort | Focus |
|-------|-----------|--------|-------|
| **Phase 0: Foundation** | M00-M02 | 46h | Audit pack, smoke test, governance |
| **Phase 1: CI Health** | M03-M10 | 60h | Silent failures, action pinning, SBOM |
| **Phase 2: Testing** | M11-M14 | 34h | Protocol version, pre-commit |
| **Phase 3: Verification** | M15-M19 | 64h | ABI check, determinism, state dict keys |
| **Phase 4: Structural** | M23-M30 | TBD | (Future refactors, scope TBD) |

**Total Estimated Effort (M01-M19)**: ~204 hours (~5 weeks for one person)

---

## Critical Invariants (P0)

These invariants MUST NOT break during refactoring (from INVARIANTS_CATALOG.md):

1. **INV-001**: Python Public API Signatures
2. **INV-002**: C++ API (ATen) Signatures
3. **INV-003**: `torch.nn.Module` Subclass Contract
4. **INV-004**: Autograd Custom Function Contract
5. **INV-020**: Checkpoint Backward Compatibility
6. **INV-021**: TorchScript Model Compatibility
7. **INV-030**: Distributed Wire Protocol Compatibility
8. **INV-040**: Device String Parsing
9. **INV-050**: Import Path Stability
10. **INV-060**: Deprecation Cycle Enforcement
11. **INV-061**: `torch.nn.functional` API Stability

**Verification**: Each invariant has a verification method (see INVARIANTS_CATALOG.md)

---

## Recommended Reading Order

### For Project Manager / Lead

1. [BASELINE_AUDIT.md](./BASELINE_AUDIT.md) - High-level overview
2. [NEXT_ACTIONS.md](./NEXT_ACTIONS.md) - Immediate priorities
3. [REFACTOR_PHASE_MAP.md](./REFACTOR_PHASE_MAP.md) - Timeline & dependencies
4. [DEFINITION_OF_DONE.md](./DEFINITION_OF_DONE.md) - Success criteria

### For Developer (Implementing Refactor)

1. [SYSTEM_SURFACES.md](./SYSTEM_SURFACES.md) - What not to break
2. [INVARIANTS_CATALOG.md](./INVARIANTS_CATALOG.md) - How to verify
3. [MODULE_BOUNDARY_MAP.md](./MODULE_BOUNDARY_MAP.md) - Coupling risks
4. [TESTING_GAPS_AND_PLAN.md](./TESTING_GAPS_AND_PLAN.md) - Test strategy

### For Security / DevOps

1. [SECURITY_AND_SUPPLY_CHAIN_BASELINE.md](./SECURITY_AND_SUPPLY_CHAIN_BASELINE.md) - Supply chain risks
2. [CI_GAPS_AND_GUARDRAILS.md](./CI_GAPS_AND_GUARDRAILS.md) - CI improvements

---

## Related Documents

- **Refactoring Workflow**: See `.cursorrules` (milestone workflow process)
- **Governance**: `../../REFACTOR.md` (to be populated in M02)
- **Tool Logging**: `../toolcalls.md` (active logging)

---

## Updates & Maintenance

**This audit pack is a living document.** Updates required:

- **After each milestone**: Update `REFACTOR_PHASE_MAP.md` with completion status
- **If new risks discovered**: Update relevant audit document (e.g., new invariant → update `INVARIANTS_CATALOG.md`)
- **Quarterly**: Re-run security posture assessment (SECURITY_AND_SUPPLY_CHAIN_BASELINE.md)

**Last Updated**: 2026-02-08 (Baseline)  
**Next Review**: After M10 (end of Phase 1)

---

## Contact / Questions

For questions about this audit or the refactoring program:
- See `.cursorrules` for workflow details
- Refer to `docs/refactor/toolcalls.md` for recovery protocol
- Check `REFACTOR.md` for milestone history (after M02)

---

**End of Audit Pack README**

