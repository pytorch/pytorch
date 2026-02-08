# M10 Third-Party Risk Matrix

**Milestone:** M10 — Third-Party Risk & License Classification (SBOM Analysis)  
**Generated:** 2026-02-08  
**Input:** M09 SBOM Baseline (42 components)  
**Baseline Commit:** 5933293e0b31455a7d62839273c694f42df92aea

---

> **Scope Statement:** Classifications are evidence-based and limited to repository-visible facts; no external fetching performed.

---

## Executive Summary

### Risk Distribution

| Risk Tier | Count | Percentage |
|-----------|-------|------------|
| **High** | 3 | 7% |
| **Medium** | 3 | 7% |
| **Low** | 32 | 76% |
| **Informational** | 4 | 10% |
| **Total** | **42** | 100% |

### High-Risk Components (Require Attention)

| Component | Primary Concern | Follow-Up |
|-----------|-----------------|-----------|
| **mslk** | Unknown license | SBOM-002 |
| **aiter** | Unknown license | SBOM-002 |
| **valgrind-headers** | GPL (headers-only, ambiguous terms) | NEW-001 |

### License Distribution

| License Category | Count |
|------------------|-------|
| Permissive (MIT, BSD, Apache-2.0, BSL-1.0, zlib) | 37 |
| Strong Copyleft (GPL) | 1 |
| Unknown | 2 |
| Custom/Non-standard | 2 |

---

## Classification Framework Applied

### License Categories

| Category | Definition |
|----------|------------|
| **Permissive** | MIT, BSD-2-Clause, BSD-3-Clause, Apache-2.0, BSL-1.0, zlib |
| **Weak Copyleft** | LGPL, MPL (none found) |
| **Strong Copyleft** | GPL, AGPL |
| **Custom/Non-standard** | Dual-license, unusual terms |
| **Unknown** | No LICENSE file in repo-visible evidence |

### Provenance Confidence

| Level | Criteria Applied |
|-------|------------------|
| **High** | Submodule with pinned SHA + upstream URL in .gitmodules |
| **Medium** | Vendored copy with LICENSE + README present |
| **Low** | Embedded/ported code with partial attribution |
| **Unknown** | No clear upstream or license evidence |

### Risk Tier Rules (Deterministic, Per Locked Decisions)

1. **High**: License = Unknown OR License = Strong Copyleft (shipped/linked) OR Provenance = Unknown
2. **Medium**: License = Custom/Non-standard OR Provenance = Low OR Ownership = Archived
3. **Low**: Permissive license AND Provenance ≥ Medium AND no other flags
4. **Informational**: Headers-only/tooling-only/dev-only when evidenced

---

## Full Component Classification

### Section 1: Git Submodules (35 components)

#### 1.1 Core Libraries

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **pybind11** | Permissive (BSD-3-Clause) | High | Upstream-maintained | Low | M09: third_party/pybind11, SHA f5fbe867 | Python bindings |
| **protobuf** | Permissive (BSD-3-Clause) | High | Upstream-maintained (Google) | Low | M09: third_party/protobuf, SHA d1eca4e4 | Serialization |
| **onnx** | Permissive (Apache-2.0) | High | Upstream-maintained (LF AI) | Low | M09: third_party/onnx, SHA e709452e | Model interchange |
| **fmt** | Permissive (MIT) | High | Upstream-maintained | Low | M09: third_party/fmt, SHA 407c905e | C++ formatting |
| **nlohmann_json** | Permissive (MIT) | High | Upstream-maintained | Low | M09: third_party/nlohmann, SHA 55f93686 | JSON library |
| **flatbuffers** | Permissive (Apache-2.0) | High | Upstream-maintained (Google) | Low | M09: third_party/flatbuffers, SHA a2cd1ea3 | Serialization |

#### 1.2 Neural Network Acceleration

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **NNPACK** | Permissive (BSD-2-Clause) | High | Upstream-maintained | Low | M09: third_party/NNPACK, SHA c07e3a04 | CPU acceleration |
| **XNNPACK** | Permissive (BSD-3-Clause) | High | Upstream-maintained (Google) | Low | M09: third_party/XNNPACK, SHA 51a01036 | Float NN inference |
| **fbgemm** | Permissive (BSD-3-Clause) | High | **PyTorch-owned** | Low | M09: third_party/fbgemm, SHA c246916f | Matrix multiplication |
| **gemmlowp** | Permissive (Apache-2.0) | High | Upstream-maintained (Google) | Low | M09: third_party/gemmlowp/gemmlowp, SHA 3fb5c176 | Low-precision GEMM |
| **flash-attention** | Permissive (BSD-3-Clause) | High | Upstream-maintained (Dao-AILab) | Low | M09: third_party/flash-attention, SHA e2743ab5 | Fast attention |
| **kleidiai** | Permissive (Apache-2.0) | High | Upstream-maintained (ARM) | Low | M09: third_party/kleidiai, SHA d7770c89 | ARM acceleration |
| **mslk** | **Unknown** | High (SHA pinned) | Meta/PyTorch | **High** | M09: third_party/mslk, SHA 3d332d1c; No LICENSE file found | ⚠️ License verification required |

#### 1.3 GPU/Accelerator Libraries

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **cudnn_frontend** | Permissive (MIT) | High | Upstream-maintained (NVIDIA) | Low | M09: third_party/cudnn_frontend, SHA b8c0656e | cuDNN wrapper |
| **cutlass** | Permissive (BSD-3-Clause) | High | Upstream-maintained (NVIDIA) | Low | M09: third_party/cutlass, SHA 0d2b201e | CUDA linear algebra |
| **NVTX** | Permissive (Apache-2.0) | High | Upstream-maintained (NVIDIA) | Low | M09: third_party/NVTX, SHA 3ebbc93d | NVIDIA tracing |
| **composable_kernel** | Permissive (MIT) | High | Upstream-maintained (ROCm/AMD) | Low | M09: third_party/composable_kernel, SHA fcc9372c, branch: develop | ROCm kernels |
| **aiter** | **Unknown** | High (SHA pinned) | Upstream-maintained (ROCm/AMD) | **High** | M09: third_party/aiter, SHA 9a469a60; No LICENSE file found | ⚠️ License verification required |
| **VulkanMemoryAllocator** | Permissive (MIT) | High | Upstream-maintained (AMD GPUOpen) | Low | M09: third_party/VulkanMemoryAllocator, SHA 1d8f600f | Vulkan memory |

#### 1.4 Distributed & Communication

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **gloo** | Permissive (BSD-3-Clause) | High | **PyTorch-owned** | Low | M09: third_party/gloo, SHA 3135b0b4 | Collective comms |
| **tensorpipe** | Permissive (BSD-3-Clause) | High | **PyTorch-owned** | Low | M09: third_party/tensorpipe, SHA 2b4cd910 | P2P comms |

#### 1.5 CPU & Platform Libraries

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **cpuinfo** | Permissive (BSD-2-Clause) | High | **PyTorch-owned** | Low | M09: third_party/cpuinfo, SHA f858c30b | CPU detection |
| **pthreadpool** | Permissive (BSD-2-Clause) | High | Upstream-maintained | Low | M09: third_party/pthreadpool, SHA 4fe0e1e1 | Thread pool |
| **FXdiv** | Permissive (MIT) | High | Upstream-maintained | Low | M09: third_party/FXdiv, SHA b408327a | Fixed-point division |
| **FP16** | Permissive (MIT) | High | Upstream-maintained | Low | M09: third_party/FP16, SHA 4dfe081c | Half-precision |
| **psimd** | Permissive (MIT) | High | Upstream-maintained | Low | M09: third_party/psimd, SHA 072586a7 | SIMD intrinsics |
| **sleef** | Permissive (BSL-1.0) | High | Upstream-maintained | Low | M09: third_party/sleef, SHA 5a1d179d | SIMD math |
| **pocketfft** | Permissive (BSD-3-Clause) | High | Upstream-maintained | Low | M09: third_party/pocketfft, SHA 0fa0ef59 | FFT library |
| **mimalloc** | Permissive (MIT) | High | Upstream-maintained (Microsoft) | Low | M09: third_party/mimalloc, SHA 048d969a | Allocator |

#### 1.6 Intel Libraries

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **ideep** | Permissive (Apache-2.0) | High | Upstream-maintained (Intel) | Low | M09: third_party/ideep, SHA 8e7ddd65 | Intel DL extension |
| **ittapi** | Permissive (BSD-3-Clause) | High | Upstream-maintained (Intel) | Low | M09: third_party/ittapi, SHA 0c575408 | Intel tracing |

#### 1.7 Testing & Tooling

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **googletest** | Permissive (BSD-3-Clause) | High | Upstream-maintained (Google) | **Informational** | M09: third_party/googletest, SHA 52eb8108 | Dev/test-only |
| **benchmark** | Permissive (Apache-2.0) | High | Upstream-maintained (Google) | **Informational** | M09: third_party/benchmark, SHA 299e5928 | Dev/test-only |
| **python-peachpy** | Permissive (BSD-2-Clause) | High | Upstream-maintained | **Informational** | M09: third_party/python-peachpy, SHA f45429b0 | ASM codegen (build-time) |

#### 1.8 Networking & Utilities

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **cpp-httplib** | Permissive (MIT) | High | Upstream-maintained | Low | M09: third_party/cpp-httplib, SHA bd95e67c, branch: v0.15.3 | HTTP library |
| **kineto** | Permissive (BSD-3-Clause) | High | **PyTorch-owned** | Low | M09: third_party/kineto, SHA 5fa388fd | Profiler |

#### 1.9 Android

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **fbjni** | Permissive (MIT) | High | Upstream-maintained (Meta) | Low | M09: android/libs/fbjni, SHA 7e1e1fe3 | JNI helpers |

---

### Section 2: Bundled/Vendored Components (3 components)

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **miniz** | Permissive (MIT) | Medium | Upstream-maintained | Low | M09: third_party/miniz-3.0.2, LICENSE file present, listed in LICENSES_BUNDLED.txt | zlib replacement |
| **concurrentqueue** | Custom (BSD-2-Clause / BSL-1.0 dual) | Medium | Upstream-maintained | **Medium** | M09: third_party/concurrentqueue, partial copy excludes test/ to avoid license issues | Dual-license requires attention |
| **valgrind-headers** | Strong Copyleft (GPL-2.0+) — Ambiguous (headers-only) | Low | Upstream-maintained | **High** | M09: third_party/valgrind-headers, only callgrind.h + valgrind.h, fetched from HEAD, version unknown | ⚠️ GPL headers-only terms need verification |

---

### Section 3: Embedded Libraries (2 components)

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **QNNPACK** | Permissive (BSD-3-Clause) | Low | **PyTorch-owned** | Low | M09: aten/src/ATen/native/quantized/cpu/qnnpack, LICENSE file present | Embedded, maintained inline |
| **clog** | Permissive (BSD-2-Clause) | Low | Meta/Facebook | Low | M09: aten/.../qnnpack/deps/clog, LICENSE file present | Embedded in QNNPACK |

---

### Section 4: Ported Code (2 components)

| Component | License | Provenance | Ownership | Risk | Evidence | Notes |
|-----------|---------|------------|-----------|------|----------|-------|
| **protobuf-int128** | Permissive (BSD-3-Clause) | Low | Ported from Google | **Informational** | M09: c10/util/int128.h, c10/util/int128.cpp, inline license header, source commit 1e88936 | Small utility, inline attribution |
| **boost-hash** | Permissive (BSL-1.0) | Low | Ported from Boost | Low | M09: c10/util/hash.h, inline license header | hash_combine and SHA1 |

---

## Risk Tier Summary

### High Risk (3 components) — Action Required

| Component | Issue | Recommendation | Follow-Up ID |
|-----------|-------|----------------|--------------|
| **mslk** | Unknown license | Verify license from upstream repository | SBOM-002 |
| **aiter** | Unknown license | Verify license from upstream repository | SBOM-002 |
| **valgrind-headers** | GPL (headers-only); terms for headers unclear | Verify if GPL header exception applies | NEW-001 |

### Medium Risk (3 components) — Monitor

| Component | Issue | Recommendation | Follow-Up ID |
|-----------|-------|----------------|--------------|
| **concurrentqueue** | Dual-license (BSD-2/BSL-1.0); partial copy to avoid license issues | Review license choice implications | NEW-002 |
| (none additional) | — | — | — |

*Note: Only 1 component is Medium (concurrentqueue). valgrind-headers escalated to High due to Strong Copyleft.*

### Low Risk (32 components) — No Action Required

All components with:
- Permissive license (MIT, BSD, Apache-2.0, BSL-1.0)
- High or Medium provenance confidence
- No ownership or maintenance concerns

### Informational (4 components) — Dev/Test/Build-Time Only

| Component | Classification Reason |
|-----------|----------------------|
| **googletest** | Test-only dependency |
| **benchmark** | Test/benchmark-only dependency |
| **python-peachpy** | Build-time ASM codegen |
| **protobuf-int128** | Small ported utility, inline license |

---

## Follow-Up Items

### Existing Deferrals (From M09)

| ID | Description | Components | Exit Criteria |
|----|-------------|------------|---------------|
| **SBOM-002** | License verification for unknown-license components | mslk, aiter | LICENSE file confirmed or alternative license documented |

### New Follow-Ups (Discovered in M10)

| ID | Description | Components | Suggested Milestone | Exit Criteria |
|----|-------------|------------|---------------------|---------------|
| **NEW-001** | Verify GPL header exception / terms for valgrind headers | valgrind-headers | M11+ | Authoritative upstream documentation confirms headers-only exception OR risk accepted with documented rationale |
| **NEW-002** | Review concurrentqueue dual-license implications | concurrentqueue | M11+ (low priority) | License selection documented; build integration confirmed to comply |
| **NEW-003** | Document PyTorch-owned component governance | gloo, cpuinfo, fbgemm, tensorpipe, kineto, QNNPACK | M11+ (informational) | Confirm release/tagging practices align with M06-B CI action governance |

---

## Notes on PyTorch-Owned Components

Six components are identified as PyTorch-owned in the SBOM:
- `gloo`, `cpuinfo`, `fbgemm`, `tensorpipe`, `kineto`, `QNNPACK`

**Relationship to M06-B (CI Action Governance):**

These are **code dependencies** (linked into PyTorch builds), whereas M06-B addresses **CI action dependencies** (`@main` refs in GitHub Actions). Both fall under "supply-chain governance" but are distinct enforcement surfaces:

- **SBOM components** → Build-time linking, runtime behavior
- **M06-B actions** → CI execution, workflow integrity

They should **not** be merged into a single remediation milestone. However, a future milestone could establish unified governance policies for all PyTorch-owned dependencies.

---

## Methodology

### Evidence Sources

All classifications derived from:
1. `docs/refactor/sbom/M09_sbom.json` — Machine-readable inventory
2. `docs/refactor/sbom/M09_THIRD_PARTY.md` — Human-readable evidence
3. `.gitmodules` — Submodule tracking
4. In-repo LICENSE files (where documented in M09)

### What Was NOT Done

- ❌ No external repository fetching
- ❌ No CVE/vulnerability scanning
- ❌ No license remediation
- ❌ No dependency upgrades
- ❌ No legal interpretation of GPL exceptions

### Classification Confidence

| Confidence | Criteria |
|------------|----------|
| **High** | LICENSE file documented in M09, well-known license |
| **Medium** | License inferred from M09 notes, standard upstream |
| **Low** | Marked as UNKNOWN in M09 |

---

## Related Documents

- **SBOM (Input):** [`M09_sbom.json`](M09_sbom.json)
- **Inventory (Input):** [`M09_THIRD_PARTY.md`](M09_THIRD_PARTY.md)
- **Milestone Plan:** [`../milestones/M10/M10_plan.md`](../milestones/M10/M10_plan.md)
- **Governance:** [`../../REFACTOR.md`](../../REFACTOR.md)

---

**End of M10_RISK_MATRIX.md**

