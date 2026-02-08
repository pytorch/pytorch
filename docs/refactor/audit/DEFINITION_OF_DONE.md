# PyTorch Refactoring Program: Definition of Done

**Purpose**: Define measurable outcomes that must be achieved to consider the refactoring program complete.

**Audit Date**: 2026-02-08  
**Baseline Commit**: c5f1d40

---

## Scope

This document defines "done" for the **AI-Native Refactoring Demonstration** using PyTorch as a case study. The program has two levels:

1. **Phase-Level Done**: Completion criteria for Foundation, CI Health, Testing, and Verification phases
2. **Program-Level Done**: Overall refactoring program completion

---

## Phase 0: Foundation (M00-M02)

### Done Criteria

✅ **Baseline Audit Complete**:
- [ ] All audit pack documents created (9 files)
- [ ] `REFACTOR.md` populated with governance baseline
- [ ] Tool logging infrastructure (`docs/refactor/toolcalls.md`) active

✅ **Import Smoke Test Exists**:
- [ ] `test/test_import_smoke.py` created
- [ ] Test runs in <1 second without C++ build
- [ ] Added to CI as fast pre-check

✅ **Governance Established**:
- [ ] `REFACTOR.md` contains milestone table
- [ ] Audit pack linked from `REFACTOR.md`
- [ ] Recovery protocol documented

**Evidence**:
- All Phase 0 files committed to `main`
- Smoke test passing in CI
- `REFACTOR.md` contains M00-M02 entries

---

## Phase 1: CI Health & Guardrails (M03-M10)

### Done Criteria

✅ **No Silent CI Failures**:
- [ ] All critical workflows (`pull.yml`, `trunk.yml`, `lint.yml`) audited
- [ ] `continue-on-error: true` removed or justified
- [ ] `if: always()` patterns reviewed
- [ ] CI fails loudly when tests fail

**Evidence**: CI run shows explicit failure (not silent skip)

---

✅ **Workflow Linting Active**:
- [ ] `actionlint.yml` workflow added to CI
- [ ] Actionlint is a required check (branch protection)
- [ ] All workflows pass actionlint

**Evidence**: PR with YAML error is blocked by actionlint

---

✅ **All Actions Pinned to SHA**:
- [ ] No `@main` references in workflows
- [ ] No mutable tags (e.g., `@v4` without SHA)
- [ ] All actions pinned to commit SHA with version comment

**Evidence**: `grep -r "@main" .github/workflows/` returns 0 results (excluding internal reusable workflows)

---

✅ **Dependabot Active for Actions**:
- [ ] `.github/dependabot.yml` created
- [ ] Dependabot opens PRs for action updates

**Evidence**: Dependabot PR for action update (or scheduled check confirms config)

---

✅ **SBOM Generated**:
- [ ] `docs/refactor/SBOM.json` exists (machine-readable)
- [ ] `docs/refactor/THIRD_PARTY_VERSIONS.md` exists (human-readable)
- [ ] SBOM validates with schema

**Evidence**: SBOM file committed, contains 50+ vendored libraries

---

✅ **Third-Party Audit Automated**:
- [ ] `scripts/audit_third_party_versions.py` created
- [ ] Script runs monthly in CI (cron job)
- [ ] Reports outdated vendored dependencies

**Evidence**: Script execution output (or CI artifact)

---

### Phase 1 Summary

**Must be True**:
- CI is reliable (no silent failures)
- CI is secure (actions pinned, SBOM generated)
- CI is maintainable (linting, Dependabot)

---

## Phase 2: Test Infrastructure (M11-M14)

### Done Criteria

✅ **Distributed Protocol Version Enforced**:
- [ ] `torch.distributed._protocol_version` constant added
- [ ] `init_process_group()` checks version
- [ ] Test exists: mixed-version processes fail gracefully

**Evidence**: `test/distributed/test_protocol_version.py` passes

---

✅ **Cross-Version Integration Test Exists**:
- [ ] Test runs two PyTorch versions (e.g., 2.1 + 2.2)
- [ ] `all_reduce()` across versions fails with clear error
- [ ] Same-version test passes

**Evidence**: `test/distributed/test_cross_version.py` passes

---

✅ **Pre-Commit Hooks Available**:
- [ ] `.pre-commit-config.yaml` created
- [ ] Includes `lintrunner`, `ruff format --check`
- [ ] Documented in `CONTRIBUTING.md`

**Evidence**: Developer can run `pre-commit install` + `pre-commit run --all-files`

---

### Phase 2 Summary

**Must be True**:
- Distributed protocol versioned (prevents cross-version failures)
- Pre-commit hooks available (improves dev experience)

---

## Phase 3: Verification Infrastructure (M15-M22)

### Done Criteria

✅ **All P0 Invariants Have Strong Verification**:

From `INVARIANTS_CATALOG.md`, P0 invariants:

| Invariant ID | Description | Verification Strength Required | Milestone |
|--------------|-------------|-------------------------------|-----------|
| INV-001 | Python API Signatures | 🟢 Strong (already exists) | N/A |
| INV-002 | C++ API Signatures | 🟢 Strong (ABI checker added) | M15 |
| INV-003 | `nn.Module` Contract | 🟢 Strong (already exists) | N/A |
| INV-004 | Autograd Function Contract | 🟢 Strong (already exists) | N/A |
| INV-020 | Checkpoint Backward Compat | 🟢 Strong (already exists) | N/A |
| INV-021 | TorchScript Compat | 🟢 Strong (BC tests improved) | M17 |
| INV-030 | Distributed Wire Protocol | 🟢 Strong (tests added) | M11, M12 |
| INV-040 | Device String Parsing | 🟢 Strong (already exists) | N/A |
| INV-050 | Import Path Stability | 🟢 Strong (smoke test added) | M01 |
| INV-060 | Deprecation Cycle | 🟡 Partial (automated check added) | M19 |
| INV-061 | `torch.nn.functional` API | 🟢 Strong (already exists) | N/A |

**Evidence**:
- ABI checker CI job exists and passes
- TorchScript BC tests expanded (10+ new models)
- Distributed protocol version tests pass
- Deprecation policy checker runs in CI

---

✅ **All P1 Invariants Have Partial+ Verification**:

| Invariant ID | Description | Verification Strength Required | Milestone |
|--------------|-------------|-------------------------------|-----------|
| INV-010 | Operator Correctness | 🟢 Strong (already exists) | N/A |
| INV-011 | Gradient Correctness | 🟢 Strong (already exists) | N/A |
| INV-022 | State Dict Key Stability | 🟡 Partial (regression test added) | M18 |
| INV-031 | RPC Serialization | 🟡 Partial (tests added) | M11 |
| INV-041 | CUDA Stream Semantics | 🟢 Strong (already exists) | N/A |

**Evidence**:
- State dict key regression test exists (`test/test_state_dict_keys.py`)
- RPC cross-version test exists

---

### Phase 3 Summary

**Must be True**:
- All critical invariants (P0) have automated verification
- High-priority invariants (P1) have at least partial verification
- No refactor can break these invariants without failing tests

---

## Program-Level Done: Refactoring Program Complete

**The refactoring program is "done" when all phases are complete AND the following outcomes are achieved:**

### 1. Behavioral Safety ✅

- [ ] **Invariants Declared**: All invariants documented in `INVARIANTS_CATALOG.md`
- [ ] **Invariants Verified**: All P0 invariants have strong verification (automated tests)
- [ ] **Compatibility Preserved**: No breaking changes without deprecation cycle (INV-060)
- [ ] **No Regressions**: All tests pass (Python, C++, distributed, JIT, ONNX)

**Measurement**: 
- `pytest test/` passes 100%
- Invariant verification checklist (from Phase 3) all green

---

### 2. Quality Gates ✅

- [ ] **Tests Passing**: All CI checks green (`pull.yml`, `trunk.yml`)
- [ ] **Coverage Maintained**: Overall coverage ≥75% (baseline 70-75%)
- [ ] **Lint/Type Checks Stable**: `lintrunner -a` passes, `mypy` passes

**Measurement**:
- CI dashboard shows green checks
- Coverage report (if available) shows ≥75%

---

### 3. CI Truthfulness ✅

- [ ] **No Silent Bypasses**: All `continue-on-error` removed or justified
- [ ] **Required Checks Aligned**: Branch protection rules match critical workflows
- [ ] **Deterministic Workflows**: No flaky tests (or flakes tracked + fixed)

**Measurement**:
- Audit log shows 0 instances of `continue-on-error: true` in critical workflows
- Flake dashboard shows <1% flake rate (or no dashboard = flakes addressed ad-hoc)

---

### 4. Security Posture ✅

- [ ] **No Hardcoded Secrets**: `git grep` for secrets returns 0 results
- [ ] **Dependency Audits**: Dependabot active, SBOM generated
- [ ] **SBOM Exists**: `docs/refactor/SBOM.json` + `THIRD_PARTY_VERSIONS.md`
- [ ] **Actions Pinned**: All GitHub Actions pinned to SHA

**Measurement**:
- Security posture score ≥7/10 (from `SECURITY_AND_SUPPLY_CHAIN_BASELINE.md`)

---

### 5. Architectural Outcome ✅

- [ ] **Boundaries Achieved**: Module dependencies remain acyclic (from `MODULE_BOUNDARY_MAP.md`)
- [ ] **No New Coupling**: Coupling scores (from Module Boundary Map) do not increase
- [ ] **Documentation Updated**: `REFACTOR.md` reflects all milestone changes

**Measurement**:
- Module coupling matrix (from Module Boundary Map) shows no regressions
- `REFACTOR.md` is up-to-date

---

### 6. Governance & Documentation ✅

- [ ] **REFACTOR.md Complete**: All milestones documented in milestone table
- [ ] **Audit Pack Maintained**: Audit documents updated (if program goals change)
- [ ] **Recovery Protocol Works**: Toolcalls log is accurate, recovery is possible

**Measurement**:
- `REFACTOR.md` contains M00-M22 (or final milestone)
- Toolcalls log has entries for each milestone

---

## Done = Evidence Required

For each milestone and phase, "done" requires:

1. **Code Artifact**: Files created/modified (committed to `main`)
2. **CI Proof**: CI run showing tests pass
3. **Documentation**: Milestone summary in `REFACTOR.md`
4. **Rollback Plan**: Documented in milestone plan

**Example (M01: Import Smoke Test)**:
- ✅ Code: `test/test_import_smoke.py` exists
- ✅ CI: Workflow run shows smoke test passed
- ✅ Docs: `REFACTOR.md` has M01 entry
- ✅ Rollback: "Delete `test/test_import_smoke.py`" (documented in `REFACTOR_PHASE_MAP.md`)

---

## Success Metrics (Quantitative)

| Metric | Baseline | Target | Measurement Method |
|--------|----------|--------|-------------------|
| **CI Reliability** | ??? (unknown) | 95%+ green | CI dashboard (30-day window) |
| **Test Coverage** | 70-75% | 75-80% | Coverage report (if available) |
| **Security Score** | 5.2/10 | 7/10 | Security posture scorecard |
| **Coupling Score (torch/)** | 9/10 (high) | 8/10 (improved) | Module boundary analysis |
| **Invariant Coverage** | P0: 70% strong | P0: 100% strong | Invariants catalog checklist |

---

## Not Done Until...

**The following are BLOCKERS for "done" status:**

- ❌ Any P0 invariant lacks strong verification
- ❌ Any critical CI workflow has silent failures
- ❌ Any GitHub Action is unpinned (`@main` or mutable tag)
- ❌ `REFACTOR.md` is empty or outdated
- ❌ Tests are failing (unless explicitly deferred with justification)

---

## Long-Term Success (Post-Program)

**The refactoring program is "successful" in the long term if:**

1. **Maintainability Improved**: Future refactors are faster (less risk, better tooling)
2. **Team Confidence**: Developers trust tests to catch regressions
3. **No Major Incidents**: No production outages caused by refactoring
4. **Community Adoption**: Refactoring practices documented and reusable

**Measurement** (6 months post-completion):
- Survey PyTorch team: "Do you feel more confident refactoring?"
- Track refactor-related incidents (should be 0)
- Check if other projects adopted the refactoring workflow

---

## Appendix: Completion Checklist (Top-Level)

Use this checklist to track program-level completion:

### Phase 0: Foundation
- [ ] M00: Baseline audit complete
- [ ] M01: Import smoke test exists
- [ ] M02: REFACTOR.md populated

### Phase 1: CI Health
- [ ] M03: Workflows audited for silent failures
- [ ] M04: Silent failures fixed
- [ ] M05: Actionlint active
- [ ] M06: Actions pinned to SHA
- [ ] M07: Dependabot active
- [ ] M08: SBOM generated
- [ ] M09: Third-party audit automated
- [ ] M10: C++ package manager researched

### Phase 2: Testing
- [ ] M11: Protocol version enforced
- [ ] M12: Cross-version test exists
- [ ] M13: Pre-commit config exists
- [ ] M14: Pre-commit documented

### Phase 3: Verification
- [ ] M15: ABI checker active
- [ ] M16: Determinism test harness exists
- [ ] M17: TorchScript BC tests improved
- [ ] M18: State dict key regression test exists
- [ ] M19: Deprecation policy check automated
- [ ] M20: (Deferred) Doctest coverage
- [ ] M21: (Deferred) Perf regression gate
- [ ] M22: (Deferred) Memory regression test

### Program-Level
- [ ] All P0 invariants have strong verification
- [ ] All P1 invariants have partial+ verification
- [ ] CI reliability ≥95%
- [ ] Security posture ≥7/10
- [ ] No regressions (all tests pass)
- [ ] `REFACTOR.md` up-to-date

**When all checkboxes are green: The refactoring program is DONE.** 🎉

---

**End of Definition of Done**

