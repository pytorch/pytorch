# PyTorch Refactoring: Top 10 Next Actions

**Audit Date:** 2026-02-08  
**Baseline Commit:** c5f1d40  
**Purpose:** Prioritized, PR-sized tasks to begin the refactoring program

---

## How to Use This List

1. **Start with M01** (Import Smoke Test) - Establishes baseline verification
2. **Then M02** (Populate REFACTOR.md) - Establishes governance
3. **Parallelize where possible** - M03-M10 are mostly independent

**Each action includes:**
- **ID**: Milestone identifier
- **Title**: Short description
- **Effort**: Estimated hours
- **Priority**: P0 (critical), P1 (high), P2 (medium)
- **Blockers**: Dependencies (if any)
- **Deliverables**: Specific files/changes

---

## Priority 0 (Critical) - Start Here

### Action 1: M01 - Create Import Smoke Test

**Title**: Import Smoke Test (No C++ Build Required)

**Priority**: P0

**Effort**: 4 hours

**Blockers**: None (can start immediately)

**Problem**: 
- No test verifies Python import paths work without C++ build
- Refactors can break imports, only discovered after full CI build (60+ min)

**Solution**:
Create `test/test_import_smoke.py` that imports all public APIs:

```python
# test/test_import_smoke.py
def test_torch_import():
    import torch
    assert torch.__version__

def test_nn_import():
    from torch import nn
    assert nn.Module

def test_optim_import():
    from torch import optim
    assert optim.SGD

def test_distributed_import():
    from torch import distributed
    # Lazy-loaded, just check import doesn't fail

# ... repeat for torch.fx, torch.export, torch.autograd, etc.
```

**Deliverables**:
- [ ] `test/test_import_smoke.py` (~100 lines)
- [ ] Update CI: Add smoke test job (runs before build, <1 second)
- [ ] Update `REFACTOR.md`: Add M01 entry

**Verification**: `python test/test_import_smoke.py` runs in <1 second, no C++ build

**Rollback**: Delete `test/test_import_smoke.py`

**Next**: After M01 → M02

---

### Action 2: M02 - Populate REFACTOR.md

**Title**: Establish Governance Baseline

**Priority**: P0

**Effort**: 2 hours

**Blockers**: M01 (smoke test should exist first)

**Problem**:
- `REFACTOR.md` is empty (1 line)
- No milestone history, no architectural decisions documented

**Solution**:
Populate `REFACTOR.md` with:
- Link to audit pack (`docs/refactor/audit/README.md`)
- Milestone table (M00, M01, M02 completed)
- Architectural principles (from baseline audit)
- Deprecation policy (2-release cycle)

**Deliverables**:
- [ ] `REFACTOR.md` (~200 lines)
- [ ] Milestone table with M00-M02

**Verification**: `REFACTOR.md` is readable, contains audit pack link

**Rollback**: N/A (documentation only)

**Next**: After M02 → Start Phase 1 (M03-M10 in parallel)

---

## Priority 1 (High) - CI Health & Supply Chain

### Action 3: M03 - Audit Workflows for Silent Failures

**Title**: Identify Silent Failure Patterns in CI

**Priority**: P1

**Effort**: 8 hours

**Blockers**: None (can start immediately)

**Problem**:
- 130+ workflows; some may have `continue-on-error: true` or `if: always()` causing tests to fail silently
- CI reports success when tests actually failed

**Solution**:
- Audit all `.github/workflows/*.yml` files
- Search for: `continue-on-error`, `if: always()`, `|| true` (shell-level suppression)
- Document findings in `docs/refactor/milestones/M03/M03_findings.md`

**Deliverables**:
- [ ] `docs/refactor/milestones/M03/M03_findings.md` (list of workflows with silent failures)
- [ ] Recommendations for M04 (which patterns to fix)

**Verification**: Document contains ≥5 findings (or states "no issues found" if clean)

**Rollback**: N/A (audit only, no code changes)

**Next**: After M03 → M04 (fix identified issues)

---

### Action 4: M04 - Fix Silent Failures (High-Priority Workflows)

**Title**: Remove Silent Failures from Critical Workflows

**Priority**: P1

**Effort**: 6 hours

**Blockers**: M03 (needs findings first)

**Problem**:
- Silent failures in `pull.yml`, `trunk.yml`, `lint.yml` mask real test failures

**Solution**:
- Edit 3-5 workflows identified in M03
- Remove `continue-on-error: true` or replace with explicit failure handling
- Add comments explaining why certain steps are allowed to fail (if any)

**Deliverables**:
- [ ] Updated workflows (3-5 files in `.github/workflows/`)
- [ ] `docs/refactor/milestones/M04/M04_summary.md` (what was fixed)

**Verification**: Introduce intentional test failure; verify CI fails loudly (not silent)

**Rollback**: `git revert` workflow changes

**Next**: After M04 → M05 (add linting to prevent regression)

---

### Action 5: M05 - Add `actionlint` to CI

**Title**: Prevent Workflow YAML Errors

**Priority**: P1

**Effort**: 4 hours

**Blockers**: None (can start immediately, or after M04)

**Problem**:
- YAML syntax errors in workflows are not caught before merge
- Broken workflow = silent CI failure (workflow doesn't run)

**Solution**:
- Add `.github/workflows/actionlint.yml` workflow
- Run `actionlint` on all workflow files
- Make it a required check (branch protection)

**Deliverables**:
- [ ] `.github/workflows/actionlint.yml` (~30 lines)
- [ ] Update branch protection: Add `actionlint` to required checks

**Verification**: Create PR with intentional YAML error; verify actionlint blocks merge

**Rollback**: Remove `.github/workflows/actionlint.yml`

**Next**: After M05 → M06 (pin actions for security)

---

### Action 6: M06 - Pin All Workflow Actions to SHA

**Title**: Eliminate Supply Chain Risk from Mutable Action References

**Priority**: P1

**Effort**: 12 hours

**Blockers**: None (can start immediately)

**Problem**:
- Some workflows use `@main` (mutable, can break)
- Some use `@v4` (mutable tag, supply chain attack vector)

**Solution**:
- Audit all `uses:` statements in workflows
- Replace `@main`, `@v4` with commit SHAs
- Add comments: `# v4.0.0 - sha: abc123`
- Script this (don't do 130 files manually)

**Deliverables**:
- [ ] All workflows updated (130+ files, scripted)
- [ ] `docs/refactor/milestones/M06/M06_action_pins.json` (mapping: action → SHA → version)

**Verification**: `grep -r "@main" .github/workflows/` returns 0 results (excluding internal reusable workflows)

**Rollback**: `git revert` workflow changes (if incorrect SHA breaks CI)

**Next**: After M06 → M07 (automate action updates)

---

### Action 7: M08 - Generate SBOM for Vendored Dependencies

**Title**: Document Supply Chain (Third-Party Code)

**Priority**: P1

**Effort**: 6 hours

**Blockers**: None (can start immediately)

**Problem**:
- `third_party/` contains 50+ vendored libraries
- No documentation of versions (security risk, license compliance risk)

**Solution**:
- Use `syft` or `cyclonedx-cli` to generate SBOM (Software Bill of Materials)
- Create human-readable version list

**Deliverables**:
- [ ] `docs/refactor/SBOM.json` (machine-readable, CycloneDX or SPDX format)
- [ ] `docs/refactor/THIRD_PARTY_VERSIONS.md` (human-readable table)

**Verification**: SBOM validates with SBOM schema, contains 50+ entries

**Rollback**: Delete SBOM files

**Next**: After M08 → M09 (automate version checks)

---

### Action 8: M09 - Add Periodic Third-Party Version Audit Script

**Title**: Alert When Vendored Dependencies Are Outdated

**Priority**: P1

**Effort**: 8 hours

**Blockers**: M08 (SBOM provides input)

**Problem**:
- Vendored dependencies are updated manually, infrequently
- Security vulnerabilities may go unnoticed

**Solution**:
- Create `scripts/audit_third_party_versions.py`
- Compare vendored versions vs GitHub releases (or CVE databases)
- Run monthly in CI (cron job)

**Deliverables**:
- [ ] `scripts/audit_third_party_versions.py` (~200 lines)
- [ ] `.github/workflows/audit-third-party.yml` (cron job, monthly)

**Verification**: Script runs, reports known outdated dependency (e.g., `pybind11`)

**Rollback**: Delete script + workflow

**Next**: After M09 → M10 (research long-term solution)

---

## Priority 2 (Medium) - Testing Infrastructure

### Action 9: M11 - Add Distributed Protocol Version Check

**Title**: Prevent Cross-Version Distributed Failures

**Priority**: P2 (but high impact)

**Effort**: 16 hours

**Blockers**: None (can start immediately)

**Problem**:
- No protocol version in `torch.distributed`
- Old worker + new parameter server = silent failure or crash

**Solution**:
- Add `torch/distributed/_protocol_version.py` with version constant
- Check version in `init_process_group()`
- Add test: Two processes with mismatched versions fail gracefully

**Deliverables**:
- [ ] `torch/distributed/_protocol_version.py` (~20 lines)
- [ ] Updated `init_process_group()` with version check
- [ ] `test/distributed/test_protocol_version.py` (~100 lines)

**Verification**: Run `test/distributed/test_protocol_version.py`; passes

**Rollback**: Remove version check (revert to implicit compatibility)

**Next**: After M11 → M12 (add cross-version integration test)

---

### Action 10: M13 - Add Pre-Commit Configuration

**Title**: Enable Local Linting Before Push

**Priority**: P2

**Effort**: 4 hours

**Blockers**: None (can start immediately)

**Problem**:
- No pre-commit hooks; developers may push code that fails lint
- Wastes CI cycles

**Solution**:
- Add `.pre-commit-config.yaml`
- Include: `lintrunner`, `ruff format --check`, trailing whitespace check
- Document in `CONTRIBUTING.md`

**Deliverables**:
- [ ] `.pre-commit-config.yaml` (~30 lines)
- [ ] Update `CONTRIBUTING.md` (setup instructions)

**Verification**: Run `pre-commit install` + `pre-commit run --all-files`; passes

**Rollback**: Delete `.pre-commit-config.yaml`

**Next**: After M13 → M14 (document usage)

---

## Summary Table

| ID | Action | Priority | Effort | Blockers | Deliverables |
|----|--------|---------|--------|----------|--------------|
| **M01** | Import Smoke Test | P0 | 4h | None | `test/test_import_smoke.py` |
| **M02** | Populate REFACTOR.md | P0 | 2h | M01 | `REFACTOR.md` |
| **M03** | Audit Workflows | P1 | 8h | None | `M03_findings.md` |
| **M04** | Fix Silent Failures | P1 | 6h | M03 | Updated workflows |
| **M05** | Add actionlint | P1 | 4h | None | `actionlint.yml` |
| **M06** | Pin Actions | P1 | 12h | None | 130+ workflows updated |
| **M08** | Generate SBOM | P1 | 6h | None | `SBOM.json`, `THIRD_PARTY_VERSIONS.md` |
| **M09** | Third-Party Audit Script | P1 | 8h | M08 | `audit_third_party_versions.py` |
| **M11** | Protocol Version Check | P2 | 16h | None | `_protocol_version.py`, test |
| **M13** | Pre-Commit Config | P2 | 4h | None | `.pre-commit-config.yaml` |

**Total Effort**: 70 hours (~2 weeks for one person)

**Parallelizable**: M03, M05, M06, M08, M11, M13 can run in parallel (non-overlapping files)

---

## Recommended Start Sequence

### Week 1: Foundation + Quick Wins
1. **Day 1-2**: M01 (Import Smoke Test) + M02 (REFACTOR.md)
2. **Day 3-4**: M03 (Audit Workflows) + M05 (actionlint)
3. **Day 5**: M04 (Fix Silent Failures)

### Week 2: Security + Testing
4. **Day 6-7**: M06 (Pin Actions) - scripted, but requires validation
5. **Day 8**: M08 (Generate SBOM)
6. **Day 9**: M09 (Third-Party Audit Script)
7. **Day 10**: M13 (Pre-Commit Config)

### Week 3+ (Optional, Higher Effort)
8. **Days 11-13**: M11 (Protocol Version Check) - 16 hours, high impact

---

## Success Criteria

**After completing these 10 actions:**

✅ **Foundation Established**:
- Smoke test catches import breakage
- `REFACTOR.md` is populated (governance)

✅ **CI is Reliable**:
- No silent failures
- Workflows are linted (actionlint)

✅ **CI is Secure**:
- Actions pinned to SHA
- SBOM generated, third-party audit automated

✅ **Testing Improved**:
- Protocol version enforced (distributed)
- Pre-commit hooks available

**Outcome**: The refactoring program has a solid foundation for Phase 2-3 (verification infrastructure).

---

## What's Next After Top 10?

See [REFACTOR_PHASE_MAP.md](./REFACTOR_PHASE_MAP.md) for full milestone list:
- **M07**: Add Dependabot (automation)
- **M10**: Research C++ package manager (long-term)
- **M12**: Cross-version integration test (distributed)
- **M14**: Pre-commit docs (finalize)
- **M15-M19**: Verification infrastructure (ABI check, determinism, etc.)
- **M23-M30**: Structural refactors (TBD based on Phase 1-3 results)

---

**End of Next Actions**

