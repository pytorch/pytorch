# M03 Audit — CI Workflow Audit for Silent Failures

**Milestone:** M03  
**Status:** In Progress  
**Date:** 2026-02-08  
**Mode:** Audit-Only (No CI modifications)  
**Baseline Commit:** c5f1d40

---

## 1. Executive Summary

This audit systematically analyzed **142 CI workflow files** in `.github/workflows/` to identify silent failure patterns, false confidence signals, and structural gaps in PyTorch's CI infrastructure.

### Key Findings

| Category | Count | Risk Level |
|----------|-------|------------|
| **Workflows with `continue-on-error: true`** | 50+ instances across 25+ files | 🔴 High (some legitimate) |
| **Workflows with `if: always()`** | 35+ files | 🟡 Medium (mostly cleanup) |
| **Workflows with `fail-fast: false`** | 18+ files | 🟢 Low (intentional matrix behavior) |
| **Disabled jobs (`if: false`)** | 2 jobs | 🔴 High (silent skip) |
| **Fork-blocked workflows** | 90+ jobs | 🟡 Medium (expected for cost control) |
| **Target Determination with `continue-on-error`** | 1 critical workflow | 🔴 High (can fail silently) |

### Top 5 CI Silent Failure Risks

1. **Target Determination fails silently** — If TD fails, tests still run but may miss critical coverage
2. **`tools-unit-tests.yml` always passes** — Tests run with `continue-on-error: true`
3. **Disabled jobs remain in trunk.yml** — `linux-jammy-py3-clang15-executorch` marked `if: false`
4. **LLM TD Retrieval job-level continue-on-error** — Entire job can fail silently
5. **Monitoring and utilization steps mask errors** — Multiple `continue-on-error` on auxiliary steps

---

## 2. Silent Failure Taxonomy

This audit uses the following classification system for silent failure patterns. Categories were defined upfront; taxonomy extensions are documented inline.

### 2.1 Soft-Fail Mechanics

**Definition:** Mechanisms that allow a step or job to fail without failing the workflow.

| Pattern | Description | Risk |
|---------|-------------|------|
| `continue-on-error: true` (step-level) | Step failure does not fail the job | Medium to High |
| `continue-on-error: true` (job-level) | Entire job failure does not fail the workflow | High |
| `|| true` (shell) | Command failure masked in script | High (not systematically audited) |

### 2.2 Conditional Skips

**Definition:** Conditions that bypass checks without clear visibility.

| Pattern | Description | Risk |
|---------|-------------|------|
| `if: github.repository_owner == 'pytorch'` | Blocks execution on forks | Low (expected) |
| `if: false` | Permanently disabled job | High (should be removed) |
| Complex `if:` with multiple conditions | May skip unexpectedly | Medium |
| Job-filter exclusion | Dynamic job skipping | Medium (opaque) |

### 2.3 Retry / Flakiness Masking

**Definition:** Automatic retries that hide instability.

| Pattern | Description | Risk |
|---------|-------------|------|
| `pytest-rerunfailures` | Retries failed tests | Medium (hides flakes) |
| S390x pip install retries | Network resilience (legitimate) | Low |
| `llm_td_retrieval.yml` retry logic | May mask underlying issues | Medium |

### 2.4 Scope Gaps

**Definition:** Workflows that do not run on all relevant change surfaces.

| Pattern | Description | Risk |
|---------|-------------|------|
| Schedule-only workflows | Never run on PRs | Medium (delayed feedback) |
| `paths:` trigger filtering | May miss relevant changes | Medium |
| Matrix exclusions | Reduced coverage | Low to Medium |

### 2.5 Informational-Only Workflows

**Definition:** Workflows that report metrics or status but assert no correctness properties.

| Pattern | Description | Risk |
|---------|-------------|------|
| Upload stats workflows | Reporting only | Low |
| Benchmark workflows | Performance, not correctness | Low |
| Stale issue management | Housekeeping | None |

---

## 3. CI Inventory Table

### 3.1 Total Workflow Count

| Metric | Value |
|--------|-------|
| **Total `.yml` files** | 142 |
| **Reusable workflows** (`_*.yml`) | 21 |
| **Generated workflows** (`generated-*.yml`) | 12 |
| **Scheduled workflows** | 60+ |
| **Workflow-dispatch enabled** | 95+ |

### 3.2 Workflow Classification by Tier

#### Tier 1: Core CI Workflows (Deep Analysis)

| Workflow | Trigger | Purpose | Signal Strength |
|----------|---------|---------|-----------------|
| `pull.yml` | PR, push to main | Primary PR gate | **Strong** |
| `trunk.yml` | Push to main | Post-merge full matrix | **Strong** |
| `lint.yml` | PR, push | Code quality checks | **Strong** |
| `inductor.yml` | Push to main | Inductor benchmarks | **Strong** |
| `inductor-unittest.yml` | Schedule | Inductor unit tests | **Strong** |
| `periodic.yml` | Schedule | Extended test coverage | **Strong** |

#### Tier 2: Platform / Accelerator Workflows (Medium Analysis)

| Workflow | Platform | Purpose | Signal Strength |
|----------|----------|---------|-----------------|
| `rocm-*.yml` (6) | AMD ROCm | GPU testing | Strong |
| `xpu.yml` | Intel XPU | XPU testing | Strong |
| `mac-mps.yml` | macOS MPS | Apple GPU testing | Strong |
| `linux-aarch64.yml` | ARM64 | Architecture testing | Strong |
| `s390.yml`, `s390x-periodic.yml` | IBM Z | Mainframe testing | Strong |
| `riscv64.yml` | RISC-V | Emerging architecture | Medium |
| `h100-*.yml` (3) | H100 GPU | High-end GPU testing | Strong |
| `b200-*.yml` (2) | B200 GPU | Next-gen GPU testing | Strong |

#### Tier 3: Generated / Nightly / Experimental (Summary Analysis)

| Workflow Pattern | Count | Purpose | Signal Strength |
|------------------|-------|---------|-----------------|
| `generated-*.yml` | 12 | Binary builds (nightly) | Strong (release pipeline) |
| `inductor-perf-*.yml` | 11 | Performance benchmarks | Weak (informational) |
| `nightly*.yml` | 2 | Nightly CI | Strong (stability) |
| `unstable*.yml` | 2 | Experimental jobs | **Weak** (explicitly allowed to fail) |
| `weekly.yml` | 1 | Weekly tasks | Medium |

---

## 4. Silent Failure Pattern Analysis

### 4.1 `continue-on-error: true` — Full Inventory

#### Critical Findings (High Risk)

| Workflow | Location | Pattern | Risk Assessment |
|----------|----------|---------|-----------------|
| `target_determination.yml` | "Do TD" step (L58-60) | `continue-on-error: true` | **🔴 HIGH** — TD failure means test selection may be incomplete, but workflow continues as if successful |
| `llm_td_retrieval.yml` | Job-level (L24-26) | `continue-on-error: true` | **🔴 HIGH** — Entire LLM retrieval can fail silently |
| `llm_td_retrieval.yml` | "Run Retriever" step (L85-87) | `continue-on-error: true` | **🔴 HIGH** — Ghstack issues masked |
| `tools-unit-tests.yml` | "Run tests" steps (L38-39, L64-65) | `continue-on-error: true` | **🔴 HIGH** — Tests always appear to pass |
| `unstable.yml` | Job-level (L22-24) | `continue-on-error: true` | 🟡 MEDIUM — Explicitly for unstable jobs (documented) |
| `unstable-periodic.yml` | Job-level (L21-23) | `continue-on-error: true` | 🟡 MEDIUM — Explicitly for unstable jobs (documented) |

#### Legitimate Uses (Lower Risk)

| Workflow | Location | Pattern | Justification |
|----------|----------|---------|---------------|
| `_linux-test.yml` | "Start monitoring script" (L200-204) | `continue-on-error: true` | Non-critical monitoring |
| `_linux-test.yml` | "Download TD artifacts" (L223-225) | `continue-on-error: true` | Fallback behavior acceptable |
| `_linux-test.yml` | "Upload pytest cache" (L478-481) | `continue-on-error: true` | Cache miss is not fatal |
| `_linux-test.yml` | "Stop monitoring script" (L513-516) | `continue-on-error: true` | Cleanup step |
| `_linux-test.yml` | "Upload utilization stats" (L545-548) | `continue-on-error: true` | Non-critical metrics |
| `_linux-build.yml` | "Start monitoring script" (L223-227) | `continue-on-error: true` | Non-critical monitoring |
| `_linux-build.yml` | "Download pytest cache" (L244-247) | `continue-on-error: true` | Cache miss is not fatal |
| `_linux-build.yml` | Various upload steps | `continue-on-error: true` | Non-critical metrics |
| `trymerge.yml` | "Comment on Canceled" (L77-79) | `continue-on-error: true` | Nice-to-have notification |
| `trymerge.yml` | "Upload merge record" (L94-97) | `continue-on-error: true` | Audit trail, not blocking |
| `tryrebase.yml` | "Comment on Canceled" (L47-49) | `continue-on-error: true` | Nice-to-have notification |
| `revert.yml` | "Comment on Canceled" (L56-58) | `continue-on-error: true` | Nice-to-have notification |
| `lint-autoformat.yml` | Multiple steps | `continue-on-error: true` | Suggestion workflow, not blocking |
| Windows binary workflows | SSH setup, Defender config | `continue-on-error: true` | Optional developer convenience |

### 4.2 `if: always()` — Usage Analysis

**Total occurrences:** 35+ files

**Primary use cases (all legitimate):**

1. **Cleanup steps** — `teardown-linux`, `docker cleanup`
2. **Artifact upload** — Upload test results even on failure
3. **Notification** — Comment on canceled workflows
4. **Job ID retrieval** — `get-workflow-job-id` for logging

**Risk Assessment:** 🟢 **LOW** — `if: always()` is used correctly for cleanup and artifact preservation.

### 4.3 `fail-fast: false` — Usage Analysis

**Total occurrences:** 18+ files

**Observed in:** All reusable test workflows (`_linux-test.yml`, `_win-test.yml`, `_mac-test.yml`, etc.)

**Pattern:**
```yaml
strategy:
  matrix: ${{ fromJSON(inputs.test-matrix) }}
  fail-fast: false
```

**Risk Assessment:** 🟢 **LOW** — This is intentional. When running a sharded test matrix, you want all shards to complete so you see all failures, not just the first one.

### 4.4 `if: false` — Disabled Jobs

| Workflow | Job | Status | Risk |
|----------|-----|--------|------|
| `trunk.yml` | `linux-jammy-py3-clang15-executorch-build` (L417-424) | `if: false # Has been broken for a while` | **🔴 HIGH** — Silently skipped, no visibility |
| `scorecards.yml` | Main job (L22-24) | `if: false && github.repository == 'pytorch/pytorch'` | 🟡 MEDIUM — Entire workflow disabled |

**Observation:** Disabled jobs should either be:
1. Removed entirely, or
2. Moved to `unstable.yml` for visibility

### 4.5 Repository Owner Check — Fork Blocking

**Pattern:** `if: github.repository_owner == 'pytorch'`

**Total occurrences:** 90+ jobs across most workflows

**Risk Assessment:** 🟢 **LOW** — This is expected behavior to:
1. Reduce CI cost on forks
2. Prevent secret exposure
3. Control self-hosted runner usage

**Evidence Gap:** GitHub branch protection "required checks" cannot be observed from workflow files. We cannot confirm which checks are enforced on the main repository.

---

## 5. Cross-Reference with M00 Known Risks

### 5.1 Risks Already Covered by CI

| M00 Risk | CI Coverage | Status |
|----------|-------------|--------|
| Import path stability (INV-050) | `refactor-smoke.yml` | ✅ Covered (M01) |
| Python version compatibility | Multi-version matrix in `pull.yml` | ✅ Covered |
| CUDA compatibility | CUDA 12.4, 12.8, 13.0 in various workflows | ✅ Covered |
| ROCm compatibility | `rocm-*.yml` workflows | ✅ Covered |
| Windows compatibility | `_win-build.yml`, `_win-test.yml` | ✅ Covered |
| macOS compatibility | `_mac-build.yml`, `_mac-test.yml` | ✅ Covered |

### 5.2 Risks Not Covered or Covered Weakly

| M00 Risk | CI Coverage | Status |
|----------|-------------|--------|
| **Action pinning (supply chain)** | No enforcement | 🔴 Not Covered (M06) |
| **Workflow YAML validation** | No `actionlint` | 🔴 Not Covered (M05) |
| **ABI compatibility** | No ABI checker | 🔴 Not Covered (M15) |
| **Dependency version audit** | No SBOM generation | 🔴 Not Covered (M08) |
| **Target Determination reliability** | TD can fail silently | 🟡 Weakly Covered |
| **Tools unit tests** | Tests always pass due to `continue-on-error` | 🔴 Not Covered |

### 5.3 New Risks Discovered in M03

| Risk ID | Description | Severity | Discovered In |
|---------|-------------|----------|---------------|
| **M03-R01** | Target Determination fails silently | High | `target_determination.yml` |
| **M03-R02** | LLM TD Retrieval job-level continue-on-error | High | `llm_td_retrieval.yml` |
| **M03-R03** | Executorch build disabled with `if: false` | High | `trunk.yml` |
| **M03-R04** | Tools unit tests always pass | High | `tools-unit-tests.yml` |
| **M03-R05** | Scorecards workflow entirely disabled | Medium | `scorecards.yml` |

---

## 6. CI Signal Strength Classification

### 6.1 Strong Signal Workflows

These workflows **deterministically validate correctness** and fail when invariants are violated.

| Workflow | Reason |
|----------|--------|
| `pull.yml` | Primary PR gate, runs build + test matrix |
| `trunk.yml` | Full post-merge matrix, GPU tests |
| `lint.yml` | Linter checks fail on violations |
| `inductor.yml` | Benchmark + unit test combination |
| `periodic.yml` | Extended test coverage |
| `_linux-build.yml` | Build failures are hard failures |
| `_linux-test.yml` | Test failures are hard failures (minus monitoring steps) |

### 6.2 Weak Signal Workflows

These workflows have **partial coverage, heuristic checks, or can fail silently**.

| Workflow | Reason |
|----------|--------|
| `target_determination.yml` | Can fail with `continue-on-error: true` |
| `llm_td_retrieval.yml` | Job-level `continue-on-error: true` |
| `unstable.yml` | Explicitly designed to allow failures |
| `unstable-periodic.yml` | Explicitly designed to allow failures |
| `tools-unit-tests.yml` | `continue-on-error: true` on test steps |

### 6.3 Cosmetic / Informational Workflows

These workflows **report metrics or status** but assert no correctness properties.

| Workflow | Purpose |
|----------|---------|
| `upload-test-stats.yml` | Metrics collection |
| `upload-torch-dynamo-perf-stats.yml` | Performance tracking |
| `inductor-perf-*.yml` (11 workflows) | Benchmark results |
| `stale.yml` | Issue housekeeping |
| `docathon-sync-label.yml` | Label management |
| `assigntome-docathon.yml` | Issue assignment |

### 6.4 Legacy / Unclear Workflows

| Workflow | Observation |
|----------|-------------|
| `scorecards.yml` | Entirely disabled (`if: false`) |
| `trunk.yml` executorch job | Disabled (`if: false`) with comment "Has been broken for a while" |

---

## 7. CI Silent Failure Risk Register

### Prioritized Risk Register

| ID | Workflow(s) | Risk Description | Severity | Confidence | Recommended Follow-up |
|----|-------------|------------------|----------|------------|----------------------|
| **M03-R01** | `target_determination.yml` | TD step uses `continue-on-error: true`; if TD fails, tests run with potentially incomplete coverage but workflow reports success | **High** | High | M04: Evaluate if TD failure should fail the workflow or trigger fallback behavior |
| **M03-R02** | `llm_td_retrieval.yml` | Entire job uses `continue-on-error: true`; complete failure is silently ignored | **High** | High | M04: Remove job-level continue-on-error; allow step-level only |
| **M03-R03** | `trunk.yml` | Executorch build job disabled with `if: false`; silently skipped with no visibility | **High** | High | M04: Remove job or move to unstable.yml |
| **M03-R04** | `tools-unit-tests.yml` | Test steps use `continue-on-error: true`; tests always appear to pass | **High** | High | M04: Remove continue-on-error from test steps |
| **M03-R05** | `scorecards.yml` | Entire workflow disabled with `if: false`; security scoring never runs | **Medium** | High | M04: Either enable or remove workflow |
| **M03-R06** | All reusable workflows | Monitoring/upload steps use `continue-on-error: true`; failures in metrics collection are hidden | **Low** | High | Acceptable: Non-critical steps |
| **M03-R07** | `lint-autoformat.yml` | Suggestion workflow uses `continue-on-error: true`; but this is intentional (suggestions, not blocking) | **Low** | High | Acceptable: By design |
| **M03-R08** | Windows binary workflows | SSH setup, Defender config use `continue-on-error: true`; developer convenience, not correctness | **Low** | High | Acceptable: Non-critical |

### Risk Summary

| Severity | Count | Action Required |
|----------|-------|-----------------|
| **High** | 4 | Fix in M04 |
| **Medium** | 1 | Evaluate in M04 |
| **Low (Acceptable)** | 3 | Document as acceptable |

---

## 8. Evidence Gaps

### 8.1 Cannot Observe from Workflow Files

| Gap | Description | Impact |
|-----|-------------|--------|
| **Branch protection required checks** | GitHub repo settings, not in workflow files | Cannot confirm which checks block merge |
| **Actual CI pass rates** | Requires HUD/metrics access | Cannot quantify flakiness |
| **Runtime test coverage** | Requires coverage artifacts | Estimates only |
| **Action version security** | Requires vulnerability DB lookup | Not audited |

### 8.2 Not Systematically Audited

| Gap | Reason | Recommendation |
|-----|--------|----------------|
| `|| true` in shell scripts | Requires script-level analysis | Future milestone (lower priority) |
| External action behavior | Third-party code | M06/M07 (action pinning) |
| Composite action internals | `.github/actions/*` | Future audit if needed |

---

## 9. Recommendations for M04+

### M04: Fix High-Priority Silent Failures

1. **Remove `continue-on-error: true` from `target_determination.yml`** — Add proper error handling instead
2. **Remove job-level `continue-on-error` from `llm_td_retrieval.yml`** — Allow step-level only
3. **Remove or relocate disabled jobs** — `trunk.yml` executorch, `scorecards.yml`
4. **Fix `tools-unit-tests.yml`** — Remove `continue-on-error` from test steps

### M05: Add Workflow Linting

1. **Add `actionlint` to CI** — Catch YAML errors and anti-patterns
2. **Lint for `continue-on-error` usage** — Require justification comments

### M06: Action Pinning

1. **Pin all actions to SHA** — Eliminate `@main` and mutable tags
2. **Document pinning policy** — Exceptions require explicit approval

### M07: Dependabot for Actions

1. **Enable Dependabot** — Automated action update PRs

---

## 10. Verification Checklist

This audit was conducted according to M03_plan.md requirements:

- [x] All 142 workflow files inventoried
- [x] Workflows classified by signal strength
- [x] Silent failure patterns identified and documented
- [x] Cross-referenced with M00 known risks
- [x] Prioritized risk register produced
- [x] Evidence-based language used throughout
- [x] No CI or code modifications made
- [x] Taxonomy explicitly defined upfront
- [x] Taxonomy extensions documented inline
- [x] Evidence gaps acknowledged

---

## 11. Audit Methodology

### Scope

- All `.yml` files in `.github/workflows/` (142 files)
- Tiered analysis: Tier 1 (deep), Tier 2 (medium), Tier 3 (summary)

### Tools Used

- `grep` for pattern matching
- `read_file` for manual inspection
- Terminal commands for file counting

### Patterns Searched

| Pattern | Purpose |
|---------|---------|
| `continue-on-error:\s*true` | Soft-fail mechanics |
| `if:\s*always\(\)` | Always-run conditions |
| `fail-fast:\s*false` | Matrix behavior |
| `if:\s*github\.repository_owner` | Fork blocking |
| `if:\s*false` | Disabled jobs |
| `schedule:` | Scheduled workflows |
| `workflow_dispatch:` | Manual trigger support |

### Analysis Date

2026-02-08

---

**End of M03 Audit**

