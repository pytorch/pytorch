# PyTorch CI/CD Gaps & Guardrails

**Purpose**: Audit CI infrastructure, identify silent failures, and recommend guardrails.

**Audit Date**: 2026-02-08  
**Baseline Commit**: c5f1d40

---

## 1. CI Overview

**Total Workflows**: 130+ files in `.github/workflows/`

**Key Workflows**:
- `pull.yml` - PR gate (Linux CPU, minimal GPU, docs)
- `trunk.yml` - Post-merge (full matrix: CUDA, ROCm, XPU, distributed)
- `lint.yml` / `lint-autoformat.yml` - Code quality
- `inductor*.yml` - Compiler tests (10+ workflows)
- `nightly.yml` - Nightly builds & benchmarks
- `periodic*.yml` - Weekly deep tests

**CI Platform**: GitHub Actions (self-hosted runners + GitHub-hosted)

**Estimated Cost**: $$$$ (large self-hosted fleet, GPU runners)

---

## 2. CI Architecture

### 2.1 Workflow Composition

**Reusable Workflows** (composable building blocks):
- `_linux-build.yml`, `_linux-test.yml`
- `_win-build.yml`, `_win-test.yml`
- `_mac-build.yml`, `_mac-test.yml`

**Job Filtering**:
- `.github/workflows/job-filter.yml` - Dynamically selects jobs based on changed files

**Target Determination**:
- `.github/workflows/target_determination.yml` - ML-based test selection
- `.github/workflows/llm_td_retrieval.yml` - LLM assists in test prioritization

**Test Sharding**:
- Default tests: 5 shards
- Distributed tests: 3 shards
- Parallelism = ~5-10x speedup

---

### 2.2 CI Health Metrics (Observational)

| Metric | Status | Evidence |
|--------|--------|----------|
| **Workflow Organization** | 🟢 Excellent | Reusable workflows, clear naming |
| **Action Pinning** | 🟡 Partial | Mix of `@main` (risky) and `@v4` (safer) |
| **Permissions** | 🟢 Good | `id-token: write` (OIDC), `contents: read` default |
| **Caching** | 🟢 Good | Docker layer cache, pip cache, ccache (inferred) |
| **Secret Handling** | 🟢 Good | `secrets: inherit`, no hardcoded secrets |
| **Error Handling** | ⚠️ Unknown | Requires line-by-line audit (M03) |

---

## 3. Identified CI Gaps

### Gap 1: Mixed Action Pinning (`@main` vs SHA) [P1]

**Observation**: Some workflows use `@main` (unstable), others use `@v4` (mutable tag).

**Examples**:
```yaml
# Risky: @main can introduce breaking changes
uses: pytorch/pytorch/.github/workflows/_runner-determinator.yml@main

# Better: version tag (but still mutable)
uses: actions/checkout@v4

# Best: SHA pin (immutable)
uses: actions/checkout@8e5e7e5abc...
```

**Risk**: Supply chain attack (tag retargeting) or accidental breakage

**Recommendation**: **M06** - Pin all actions to commit SHA

---

### Gap 2: No Workflow Schema Validation [P1]

**Observation**: 130+ YAML files; no pre-merge validation

**Risk**: YAML syntax errors break CI silently (workflow doesn't run)

**Recommendation**: **M05** - Add `actionlint` to CI

---

### Gap 3: Silent Failure Patterns (TBD) [P1]

**Observation**: Need systematic audit for:
- `continue-on-error: true`
- `if: always()`
- `|| true` (shell-level failure suppression)

**Risk**: Tests fail but CI reports success

**Recommendation**: **M03, M04** - Audit and remove silent failures

---

### Gap 4: No Centralized Flake Tracking [P2]

**Observation**: Flaky tests are reported ad-hoc (GitHub issues)

**Risk**: Flakes mask real failures

**Recommendation**: Add flake dashboard (track retry counts, failure rates)

---

### Gap 5: No Required Check Alignment Audit [P2]

**Observation**: Branch protection "required checks" may not match actual critical workflows

**Risk**: Non-critical workflow failure blocks merge, or critical workflow not required

**Recommendation**: Audit branch protection rules vs workflow structure

---

## 4. CI Guardrails (Recommendations)

### Guardrail 1: Action Pinning Policy

**Rule**: All external actions MUST be pinned to immutable commit SHA or version tag.

**Enforcement**:
- Automated: `actionlint` (checks for `@main`)
- Manual: PR review checklist

**Exception**: Internal reusable workflows (e.g., `pytorch/pytorch/.github/workflows/_linux-build.yml`) may use `@main` if tightly controlled

**Implementation**: M06

---

### Guardrail 2: No Silent Failures

**Rule**: `continue-on-error: true` is banned unless explicitly justified in PR description.

**Enforcement**:
- Automated: Grep for `continue-on-error` in PR diffs, require approval
- Manual: PR review checklist

**Exception**: Non-critical jobs (e.g., benchmarks, nightly uploads) may use `continue-on-error` if explicitly documented

**Implementation**: M04

---

### Guardrail 3: Workflow YAML Linting

**Rule**: All workflow YAMLs must pass `actionlint` before merge.

**Enforcement**:
- CI job: `.github/workflows/actionlint.yml` (required check)

**Implementation**: M05

---

### Guardrail 4: Required Check Governance

**Rule**: Branch protection "required checks" must be documented and reviewed quarterly.

**Enforcement**:
- Manual: Document in `docs/refactor/CI_REQUIRED_CHECKS.md`
- Review: Q1 audit of required checks vs actual workflow needs

**Implementation**: Post-M05

---

### Guardrail 5: Workflow Change Review

**Rule**: Changes to reusable workflows (`_*.yml`) require two approvals (higher risk).

**Enforcement**:
- GitHub CODEOWNERS: `/.github/workflows/_*.yml @pytorch/core`

**Implementation**: Add CODEOWNERS rule

---

## 5. CI Workflow Audit Findings (Preliminary)

**Note**: Full audit deferred to M03. Preliminary findings:

### 5.1 Reusable Workflows (Good Practice)

**Count**: ~20 reusable workflows (`_*.yml`)

**Examples**:
- `_linux-build.yml` - Used by `pull.yml`, `trunk.yml`, `periodic.yml`
- `_binary-build-linux.yml` - Used by nightly/release workflows

**Assessment**: 🟢 **Excellent** - Reduces duplication, improves maintainability

---

### 5.2 Job Filtering (Good Practice)

**Workflow**: `.github/workflows/job-filter.yml`

**Purpose**: Dynamically determine which jobs to run based on changed files

**Assessment**: 🟢 **Good** - Reduces CI time for small PRs

**Concern**: Complex logic; requires testing to avoid false negatives (skipping critical tests)

---

### 5.3 Target Determination (ML-Based)

**Workflows**:
- `.github/workflows/target_determination.yml`
- `.github/workflows/llm_td_retrieval.yml`

**Purpose**: Use ML to predict which tests are affected by changes

**Assessment**: 🟡 **Experimental** - Innovative, but risk of false negatives

**Recommendation**: Keep as optimization, but ensure "full test suite" option always available

---

### 5.4 Action Pinning (Mixed)

**Observation**: Random sample of 10 workflows

| Workflow | Actions Used | Pinning Status |
|----------|--------------|----------------|
| `pull.yml` | `actions/checkout@v4` | 🟡 Mutable tag |
| `trunk.yml` | `pytorch/pytorch/.github/workflows/_runner-determinator.yml@main` | 🔴 @main (risky) |
| `lint.yml` | `actions/setup-python@v5` | 🟡 Mutable tag |

**Recommendation**: M06 - Pin all to SHA

---

## 6. CI Maintenance Burden

### 6.1 Workflow Count Growth

**Current**: 130+ workflows

**Concern**: High maintenance burden (updates, security patches)

**Mitigation**:
- Maximize reuse of `_*.yml` reusable workflows
- Consider workflow generation (e.g., YAML from templates)

---

### 6.2 Workflow Duplication

**Example**: Many workflows have similar build steps (checkout, setup Python, install deps)

**Mitigation**:
- Extract common steps into reusable workflows
- Already partially done; continue in M04

---

## 7. CI Speed & Efficiency

### 7.1 Critical Path Analysis (Estimated)

| Phase | Time | Bottleneck |
|-------|------|------------|
| **Checkout & Setup** | ~2 min | Network (shallow clone helps) |
| **Build (C++)** | ~30-60 min | CPU (ccache helps) |
| **Test (Python, sharded)** | ~10-20 min/shard | GPU availability (for GPU tests) |
| **Upload Artifacts** | ~5 min | Network |

**Total (wall time, sharded)**: ~1 hour (5 shards in parallel)

**Recommendation**: Current setup is well-optimized; no urgent improvements needed

---

### 7.2 Caching Strategy

**Observed Caching**:
- Docker layer cache (for build images)
- `ccache` (C++ compilation cache)
- `pip` cache (Python deps)

**Assessment**: 🟢 **Good** - Standard practices in use

---

## 8. CI Security Posture

### 8.1 Action Supply Chain Risk

**Risk**: Third-party actions can be compromised (tag retargeting, account takeover)

**Current Mitigation**: None (actions not pinned to SHA)

**Recommendation**: M06 - Pin to SHA + M07 - Add Dependabot for action updates

---

### 8.2 Secrets Management

**Observation**: `secrets: inherit` used in reusable workflows

**Assessment**: 🟢 **Good** - Secrets not hardcoded, scope limited

**Recommendation**: Periodic audit of secret usage (ensure least privilege)

---

### 8.3 PR from Forks (Untrusted Code)

**Risk**: PR from fork can execute malicious code in CI

**Current Mitigation**: `pull_request` trigger (not `pull_request_target`); secrets not exposed to forks

**Assessment**: 🟢 **Good** - Standard GitHub Actions security model

---

## 9. Recommended CI Improvements (Prioritized)

| ID | Improvement | Priority | Effort | Milestone | Impact |
|----|-------------|---------|--------|-----------|--------|
| **CI-1** | Audit workflows for silent failures | P1 | 8h | M03 | Prevent false successes |
| **CI-2** | Fix identified silent failures | P1 | 6h | M04 | Improve CI reliability |
| **CI-3** | Add actionlint to CI | P1 | 4h | M05 | Catch YAML errors |
| **CI-4** | Pin all actions to SHA | P1 | 12h | M06 | Security hardening |
| **CI-5** | Add Dependabot for actions | P2 | 2h | M07 | Maintenance automation |
| **CI-6** | Audit required checks vs workflows | P2 | 4h | Post-M05 | Governance |
| **CI-7** | Add flake tracking dashboard | P2 | 16h | Future | Observability |
| **CI-8** | Workflow change CODEOWNERS rule | P3 | 1h | Post-M05 | Safety |

**Total Effort (P1)**: 30 hours  
**Total Effort (P1-P2)**: 36 hours

---

## 10. CI Guardrail Summary Table

| Guardrail | Status | Enforcement | Milestone |
|-----------|--------|-------------|-----------|
| **Action Pinning** | 🔴 Not Enforced | Automated (actionlint) + Manual | M06 |
| **No Silent Failures** | 🔴 Not Enforced | Manual (PR review) | M04 |
| **Workflow Linting** | 🔴 Not Enforced | Automated (actionlint CI job) | M05 |
| **Required Check Governance** | 🔴 Not Documented | Manual (quarterly audit) | Post-M05 |
| **Workflow Change Review** | 🔴 Not Enforced | Automated (CODEOWNERS) | Post-M05 |

---

## 11. Verification Commands

| Check | Command | When |
|-------|---------|------|
| **Lint Workflows** | `actionlint .github/workflows/*.yml` | Before committing workflow changes |
| **Check Action Pins** | `grep -r "@main" .github/workflows/` | Before M06 completion |
| **Find Silent Failures** | `grep -r "continue-on-error" .github/workflows/` | M03 audit |
| **List Required Checks** | GitHub UI → Repo Settings → Branches → main → Edit | Quarterly review |

---

**End of CI/CD Gaps & Guardrails**

