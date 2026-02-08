# PyTorch Security & Supply Chain Baseline

**Purpose**: Establish security posture baseline for supply chain, dependencies, and secrets management.

**Audit Date**: 2026-02-08  
**Baseline Commit**: c5f1d40

---

## 1. Dependency Posture

### 1.1 Dependency Management Files

| File | Purpose | Status |
|------|---------|--------|
| `pyproject.toml` | Modern Python packaging (PEP 621) | 🟢 Migrating to this |
| `requirements.txt` | Runtime Python deps | 🟡 Min versions, not pinned |
| `requirements-build.txt` | Build-time Python deps | 🟡 Min versions, not pinned |
| `setup.py` | Legacy build script (1,764 lines) | 🟡 Still primary build mechanism |
| `third_party/` | Vendored C++ dependencies | 🔴 High risk (manual updates) |

**Assessment**: 🟡 **Mixed** - Modern tooling in progress, but lacks lockfiles and vendored code is high-risk

---

### 1.2 Dependency Pinning

**Python Dependencies** (from `requirements.txt`):
```
sympy>=1.13.3
filelock
fs spec>=0.8.5
jinja2
networkx>=2.5.1
optree>=0.13.0
psutil
```

**Assessment**: 🟡 **Partial** - Min versions specified, but no max versions or exact pins

**Risk**: 
- Unexpected breaking changes in transitive deps
- Reproducibility issues (different versions in dev vs prod)

**Recommendation**: **M06** (related) - Generate lockfile (`requirements.lock` or use `uv`, `pip-tools`)

---

### 1.3 Vendored Dependencies (`third_party/`)

**Examples** (from directory listing):
- `pybind11` - Python/C++ bindings
- `NNPACK` - Neural network primitives
- `FP16` - Half-precision math
- `googletest` - C++ testing framework
- `protobuf` - Serialization

**Count**: 50+ vendored libraries

**Assessment**: 🔴 **High Risk**
- Manual updates required (no automation)
- Security vulnerabilities may go unnoticed
- Version skew (vendored version vs upstream release)

**Recommendation**: **M08, M09** - Generate SBOM + periodic version audit

---

## 2. SBOM (Software Bill of Materials)

**Current State**: ❌ **Not Generated**

**Recommendation**: **M08** - Generate SBOM using `syft` or `cyclonedx-cli`

**Target Output**: `docs/refactor/SBOM.json` (machine-readable) + `docs/refactor/THIRD_PARTY_VERSIONS.md` (human-readable)

**SBOM Use Cases**:
- Vulnerability scanning (CVE lookup)
- License compliance
- Supply chain transparency

---

## 3. Vulnerability Scanning

### 3.1 Python Dependencies

**Tool**: GitHub Dependabot Security Alerts (enabled by default for public repos)

**Status**: 🟢 **Likely Active** (no visible `dependabot.yml`, but security alerts are default)

**Recommendation**: Verify Dependabot is active; add `dependabot.yml` for explicit config

---

### 3.2 Vendored C++ Dependencies

**Tool**: ❌ **None**

**Risk**: Vulnerabilities in vendored code (e.g., `protobuf`, `pybind11`) may not be detected

**Recommendation**: 
- M08 - Generate SBOM
- M09 - Periodic audit (compare vendored versions vs CVE databases)

---

### 3.3 GitHub Actions

**Tool**: Dependabot (for actions, if configured)

**Status**: 🔴 **Not Configured**

**Recommendation**: **M07** - Add `.github/dependabot.yml` with `package-ecosystem: "github-actions"`

---

## 4. Secret Scanning

### 4.1 GitHub Secret Scanning

**Status**: 🟢 **Enabled** (default for public repos)

**Coverage**:
- Detects hardcoded API keys, tokens, passwords in commits
- Sends alerts to maintainers

**Assessment**: 🟢 **Good**

---

### 4.2 Secrets in CI

**Observation**: No hardcoded secrets in `.github/workflows/*.yml`

**Pattern Used**: `${{ secrets.SECRET_NAME }}` (correct)

**Assessment**: 🟢 **Good**

**Recommendation**: Periodic audit (ensure no accidental secret leaks in logs)

---

## 5. SBOM / Provenance

### 5.1 SBOM Generation

**Current State**: ❌ **Not Generated**

**Recommendation**: **M08** - Generate SBOM for:
- Python dependencies (`requirements.txt`)
- Vendored C++ dependencies (`third_party/`)

**Tools**:
- `syft` (SPDX, CycloneDX formats)
- `cyclonedx-cli` (CycloneDX format)

---

### 5.2 SLSA Provenance

**Current State**: ❌ **Not Generated**

**SLSA Level**: L0 (no provenance)

**Recommendation**: (Low priority) Add SLSA provenance to release builds

**Benefit**: Users can verify binary authenticity (defense against supply chain tampering)

---

## 6. CI Trust Boundaries

### 6.1 Workflow Permissions

**Observation**: Workflows use `id-token: write`, `contents: read` (OIDC for AWS/Azure auth)

**Assessment**: 🟢 **Good** - Least privilege (not `write-all`)

---

### 6.2 PR from Forks (Untrusted Code)

**Trigger**: `pull_request` (NOT `pull_request_target`)

**Assessment**: 🟢 **Good** - Secrets not exposed to forks; fork PRs run in isolated context

**Risk**: Fork can still consume CI resources (DoS), but cannot steal secrets

---

### 6.3 Unpinned Actions (Supply Chain Risk)

**Observation**: Some workflows use `@main` or `@v4` (mutable references)

**Risk**: 
- Compromised action can execute malicious code
- Tag retargeting attack (rare, but possible)

**Recommendation**: **M06** - Pin all actions to commit SHA

---

### 6.4 Self-Hosted Runners

**Observation**: PyTorch uses self-hosted runners (for GPU, specialized hardware)

**Risk**: Self-hosted runners can be compromised (persistent state, network access)

**Mitigation** (assumed, not verified):
- Ephemeral runners (created/destroyed per job)
- Network isolation (no internet access from runner)

**Recommendation**: Audit self-hosted runner security (out of scope for this audit; defer to PyTorch infra team)

---

## 7. Top 3 Supply Chain Risks

### Risk 1: Vendored Dependencies (`third_party/`) [P1]

**Description**: 50+ vendored libraries; manual updates; no automated vulnerability scanning

**Impact**: Security vulnerabilities may go undetected for months

**Likelihood**: High (vendored code is rarely updated)

**Mitigation**:
- M08 - Generate SBOM
- M09 - Periodic version audit
- M10 - Investigate package manager (conan, vcpkg) to automate updates

---

### Risk 2: Unpinned GitHub Actions [P1]

**Description**: Actions using `@main` or mutable tags

**Impact**: Compromised action can execute arbitrary code in CI (steal secrets, inject backdoors)

**Likelihood**: Low (GitHub has security measures), but **impact is severe**

**Mitigation**:
- M06 - Pin all actions to commit SHA
- M07 - Add Dependabot for automated action updates

---

### Risk 3: No Python Dependency Lockfile [P2]

**Description**: `requirements.txt` specifies min versions, not exact pins

**Impact**: 
- Non-reproducible builds
- Unexpected breaking changes from transitive deps

**Likelihood**: Medium (Python ecosystem moves fast)

**Mitigation**:
- Generate lockfile (`requirements.lock` or use `uv`, `pip-tools`)
- Pin transitive dependencies

---

## 8. Security Checklist for Refactors

Before merging refactor PRs:

- [ ] **No new hardcoded secrets** (check with `git grep -E "(password|token|key|secret)" <changed_files>`)
- [ ] **No new vendored dependencies** (if unavoidable, document in `third_party/` + SBOM)
- [ ] **Workflow changes reviewed** (if touching `.github/workflows/`, require 2 approvals)
- [ ] **Actions pinned** (no `@main`, no mutable tags)
- [ ] **Dependency updates justified** (if updating `requirements.txt`, explain why)

---

## 9. Security Posture Score (Baseline)

| Category | Score | Notes |
|----------|-------|-------|
| **Dependency Management** | 🟡 5/10 | Modern tooling in progress, but no lockfiles |
| **Vulnerability Scanning** | 🟡 6/10 | Python deps scanned (Dependabot), but vendored C++ not covered |
| **Secret Management** | 🟢 8/10 | No hardcoded secrets, GitHub secret scanning active |
| **SBOM / Provenance** | 🔴 2/10 | No SBOM, no SLSA provenance |
| **CI Supply Chain** | 🟡 5/10 | Good permissions, but unpinned actions |

**Overall Score**: 🟡 **5.2/10** (Baseline, room for improvement)

**Target Post-Phase 1**: 🟢 **7/10** (After M06-M09 completion)

---

## 10. Recommended Actions (Prioritized)

| ID | Action | Priority | Effort | Milestone | Impact |
|----|--------|---------|--------|-----------|--------|
| **SEC-1** | Generate SBOM for vendored deps | P1 | 6h | M08 | Visibility |
| **SEC-2** | Periodic third-party version audit | P1 | 8h | M09 | Maintenance |
| **SEC-3** | Pin GitHub Actions to SHA | P1 | 12h | M06 | Security hardening |
| **SEC-4** | Add Dependabot for actions | P2 | 2h | M07 | Automation |
| **SEC-5** | Generate Python lockfile | P2 | 4h | Future | Reproducibility |
| **SEC-6** | Investigate C++ package manager | P2 | 12h | M10 | Long-term security |
| **SEC-7** | Add SLSA provenance (release builds) | P3 | 20h | Future | Supply chain trust |

**Total Effort (P1)**: 26 hours  
**Total Effort (P1-P2)**: 44 hours

---

## 11. Compliance Considerations

**Note**: This audit does not cover legal/compliance requirements (e.g., GDPR, SOC 2). Defer to PyTorch legal/security team.

**Relevant Standards**:
- **SLSA** (Supply Chain Levels for Software Artifacts) - Currently L0, target L2
- **CycloneDX / SPDX** (SBOM formats) - Not yet adopted
- **OpenSSF Scorecard** - Not run (could add as CI check)

**Recommendation**: Run OpenSSF Scorecard to get automated security assessment

---

**End of Security & Supply Chain Baseline**

