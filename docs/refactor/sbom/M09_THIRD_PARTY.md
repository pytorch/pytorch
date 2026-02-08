# PyTorch Third-Party Component Inventory

**Milestone:** M09 — Third-Party Supply Chain Inventory & SBOM Baseline  
**Generated:** 2026-02-08  
**Baseline Commit:** 5933293e0b31455a7d62839273c694f42df92aea  
**Scope:** `third_party/` + embedded libraries in `aten/`, `c10/`

---

## Executive Summary

This document provides a **human-readable inventory** of all third-party and vendored dependencies discovered in the PyTorch repository. It accompanies the machine-readable SBOM at `docs/refactor/sbom/M09_sbom.json`.

### Inventory Statistics

| Category | Count |
|----------|-------|
| Git Submodules | 35 |
| Bundled/Vendored (non-submodule) | 3 |
| Embedded Libraries (in aten/c10) | 2 |
| Ported Code (inline) | 2 |
| **Total Components** | **42** |

### Ownership Summary

| Owner | Count | Components |
|-------|-------|------------|
| **PyTorch Organization** | 6 | gloo, cpuinfo, fbgemm, tensorpipe, kineto, QNNPACK (embedded) |
| **Google** | 6 | googletest, benchmark, gemmlowp, XNNPACK, flatbuffers |
| **NVIDIA** | 4 | cudnn_frontend, cutlass, NVTX |
| **Meta/Facebook** | 2 | fbjni, clog |
| **Intel** | 3 | ideep, ittapi |
| **AMD/ROCm** | 2 | composable_kernel, aiter |
| **Other/Independent** | 19 | Various |

---

## Component Inventory

### Section 1: Git Submodules (35 components)

These components are managed as git submodules and tracked in `.gitmodules`. Each has a pinned commit SHA.

#### 1.1 Core Libraries

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **pybind11** | `third_party/pybind11` | [pybind/pybind11](https://github.com/pybind/pybind11.git) | `f5fbe867...` | BSD-3-Clause | Python bindings for C++ |
| **protobuf** | `third_party/protobuf` | [protocolbuffers/protobuf](https://github.com/protocolbuffers/protobuf.git) | `d1eca4e4...` | BSD-3-Clause | Serialization |
| **onnx** | `third_party/onnx` | [onnx/onnx](https://github.com/onnx/onnx.git) | `e709452e...` | Apache-2.0 | Model interchange |
| **fmt** | `third_party/fmt` | [fmtlib/fmt](https://github.com/fmtlib/fmt.git) | `407c905e...` | MIT | C++ formatting |
| **nlohmann_json** | `third_party/nlohmann` | [nlohmann/json](https://github.com/nlohmann/json.git) | `55f93686...` | MIT | JSON for C++ |
| **flatbuffers** | `third_party/flatbuffers` | [google/flatbuffers](https://github.com/google/flatbuffers.git) | `a2cd1ea3...` | Apache-2.0 | Serialization |

#### 1.2 Neural Network Acceleration

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **NNPACK** | `third_party/NNPACK` | [Maratyszcza/NNPACK](https://github.com/Maratyszcza/NNPACK.git) | `c07e3a04...` | BSD-2-Clause | CPU acceleration |
| **XNNPACK** | `third_party/XNNPACK` | [google/XNNPACK](https://github.com/google/XNNPACK.git) | `51a01036...` | BSD-3-Clause | Float NN inference |
| **fbgemm** | `third_party/fbgemm` | [pytorch/fbgemm](https://github.com/pytorch/fbgemm) | `c246916f...` | BSD-3-Clause | Matrix multiplication (PyTorch-owned) |
| **gemmlowp** | `third_party/gemmlowp/gemmlowp` | [google/gemmlowp](https://github.com/google/gemmlowp.git) | `3fb5c176...` | Apache-2.0 | Low-precision GEMM |
| **flash-attention** | `third_party/flash-attention` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention.git) | `e2743ab5...` | BSD-3-Clause | Fast attention |
| **kleidiai** | `third_party/kleidiai` | [ARM-software/kleidiai](https://github.com/ARM-software/kleidiai.git) | `d7770c89...` | Apache-2.0 | ARM acceleration |
| **mslk** | `third_party/mslk` | [meta-pytorch/MSLK](https://github.com/meta-pytorch/MSLK.git) | `3d332d1c...` | UNKNOWN | Meta library |

#### 1.3 GPU/Accelerator Libraries

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **cudnn_frontend** | `third_party/cudnn_frontend` | [NVIDIA/cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend.git) | `b8c0656e...` | MIT | cuDNN wrapper |
| **cutlass** | `third_party/cutlass` | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass.git) | `0d2b201e...` | BSD-3-Clause | CUDA linear algebra |
| **NVTX** | `third_party/NVTX` | [NVIDIA/NVTX](https://github.com/NVIDIA/NVTX.git) | `3ebbc93d...` | Apache-2.0 | NVIDIA tracing |
| **composable_kernel** | `third_party/composable_kernel` | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel.git) | `fcc9372c...` | MIT | ROCm kernels (branch: develop) |
| **aiter** | `third_party/aiter` | [ROCm/aiter](https://github.com/ROCm/aiter.git) | `9a469a60...` | UNKNOWN | ROCm library |
| **VulkanMemoryAllocator** | `third_party/VulkanMemoryAllocator` | [GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git) | `1d8f600f...` | MIT | Vulkan memory |

#### 1.4 Distributed & Communication

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **gloo** | `third_party/gloo` | [pytorch/gloo](https://github.com/pytorch/gloo) | `3135b0b4...` | BSD-3-Clause | Collective comms (PyTorch-owned) |
| **tensorpipe** | `third_party/tensorpipe` | [pytorch/tensorpipe](https://github.com/pytorch/tensorpipe.git) | `2b4cd910...` | BSD-3-Clause | P2P comms (PyTorch-owned) |

#### 1.5 CPU & Platform Libraries

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **cpuinfo** | `third_party/cpuinfo` | [pytorch/cpuinfo](https://github.com/pytorch/cpuinfo.git) | `f858c30b...` | BSD-2-Clause | CPU detection (PyTorch-owned) |
| **pthreadpool** | `third_party/pthreadpool` | [Maratyszcza/pthreadpool](https://github.com/Maratyszcza/pthreadpool.git) | `4fe0e1e1...` | BSD-2-Clause | Thread pool |
| **FXdiv** | `third_party/FXdiv` | [Maratyszcza/FXdiv](https://github.com/Maratyszcza/FXdiv.git) | `b408327a...` | MIT | Fixed-point division |
| **FP16** | `third_party/FP16` | [Maratyszcza/FP16](https://github.com/Maratyszcza/FP16.git) | `4dfe081c...` | MIT | Half-precision |
| **psimd** | `third_party/psimd` | [Maratyszcza/psimd](https://github.com/Maratyszcza/psimd.git) | `072586a7...` | MIT | SIMD intrinsics |
| **sleef** | `third_party/sleef` | [shibatch/sleef](https://github.com/shibatch/sleef) | `5a1d179d...` | BSL-1.0 | SIMD math |
| **pocketfft** | `third_party/pocketfft` | [mreineck/pocketfft](https://github.com/mreineck/pocketfft) | `0fa0ef59...` | BSD-3-Clause | FFT library |
| **mimalloc** | `third_party/mimalloc` | [microsoft/mimalloc](https://github.com/microsoft/mimalloc.git) | `048d969a...` | MIT | Allocator |

#### 1.6 Intel Libraries

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **ideep** | `third_party/ideep` | [intel/ideep](https://github.com/intel/ideep) | `8e7ddd65...` | Apache-2.0 | Intel DL extension |
| **ittapi** | `third_party/ittapi` | [intel/ittapi](https://github.com/intel/ittapi.git) | `0c575408...` | BSD-3-Clause | Intel tracing |

#### 1.7 Testing & Tooling

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **googletest** | `third_party/googletest` | [google/googletest](https://github.com/google/googletest.git) | `52eb8108...` | BSD-3-Clause | Unit testing |
| **benchmark** | `third_party/benchmark` | [google/benchmark](https://github.com/google/benchmark.git) | `299e5928...` | Apache-2.0 | Microbenchmarking |
| **python-peachpy** | `third_party/python-peachpy` | [malfet/PeachPy](https://github.com/malfet/PeachPy.git) | `f45429b0...` | BSD-2-Clause | ASM codegen |

#### 1.8 Networking & Utilities

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **cpp-httplib** | `third_party/cpp-httplib` | [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib.git) | `bd95e67c...` | MIT | HTTP library (branch: v0.15.3) |
| **kineto** | `third_party/kineto` | [pytorch/kineto](https://github.com/pytorch/kineto) | `5fa388fd...` | BSD-3-Clause | Profiler (PyTorch-owned) |

#### 1.9 Android

| Component | Path | Upstream | Commit | License | Notes |
|-----------|------|----------|--------|---------|-------|
| **fbjni** | `android/libs/fbjni` | [facebookincubator/fbjni](https://github.com/facebookincubator/fbjni.git) | `7e1e1fe3...` | MIT | JNI helpers |

---

### Section 2: Bundled/Vendored (Non-Submodule) Components (3 components)

These components are directly included in the repository (not as submodules).

| Component | Path | Upstream | Version/Commit | License | Notes |
|-----------|------|----------|----------------|---------|-------|
| **miniz** | `third_party/miniz-3.0.2` | [richgel999/miniz](https://github.com/richgel999/miniz) | 3.0.2 | MIT | zlib replacement. Listed in `LICENSES_BUNDLED.txt` |
| **concurrentqueue** | `third_party/concurrentqueue` | [cameron314/concurrentqueue](https://github.com/cameron314/concurrentqueue) | `24b78782...` | BSD-2-Clause / BSL-1.0 | Partial copy. Excludes `test/` to avoid license issues. Has `update.sh` script. |
| **valgrind-headers** | `third_party/valgrind-headers` | [valgrind.git](https://sourceware.org/git/?p=valgrind.git) | UNKNOWN (HEAD) | GPL-2.0+ (headers only) | Only `callgrind.h` and `valgrind.h` |

---

### Section 3: Embedded Libraries in Source Tree (2 components)

These are full third-party libraries embedded within PyTorch source directories.

| Component | Path | Upstream | Version | License | Notes |
|-----------|------|----------|---------|---------|-------|
| **QNNPACK** | `aten/src/ATen/native/quantized/cpu/qnnpack` | [pytorch/QNNPACK](https://github.com/pytorch/QNNPACK) | UNKNOWN | BSD-3-Clause | Full embedded copy. Originally Facebook, now maintained inline. |
| **clog** | `aten/src/ATen/native/quantized/cpu/qnnpack/deps/clog` | N/A (part of QNNPACK) | UNKNOWN | BSD-2-Clause | Logging library, dependency of QNNPACK |

---

### Section 4: Ported/Adapted Code (2 components)

These are code files ported from other projects with inline license headers.

| Component | Path | Original Source | License | Notes |
|-----------|------|-----------------|---------|-------|
| **protobuf uint128** | `c10/util/int128.h`, `c10/util/int128.cpp` | [protobuf @1e88936](https://github.com/protocolbuffers/protobuf/blob/1e88936fce10cf773cb72b44c6a7f48b38c7578b/src/google/protobuf/stubs/int128.h) | BSD-3-Clause | Inline license header. Modified for PyTorch. |
| **Boost hash/SHA1** | `c10/util/hash.h` | Boost Libraries | BSL-1.0 | `hash_combine` and `sha1` implementation. Inline license header. |

---

### Section 5: Reference Files (Not Libraries)

These files document external dependencies but don't contain vendored code.

| File | Purpose | Content |
|------|---------|---------|
| `third_party/eigen_pin.txt` | Eigen version pin | `5.0.0` |
| `third_party/xpu.txt` | XPU commit reference | `7f6bf0aa8b10c22023722d7fe02b05d730a5a6a9` |
| `third_party/LICENSES_BUNDLED.txt` | License manifest | Lists miniz-3.0.2 |

---

## Evidence Gaps & Unknowns

This section documents components where complete information could not be determined from repository state alone.

### 1. Version Information Gaps

| Component | Gap | Notes |
|-----------|-----|-------|
| All submodules | Semantic version unknown | Only commit SHAs available; release tags not inspected |
| valgrind-headers | Exact version unknown | Fetched from HEAD at unknown time |
| QNNPACK | Version unknown | No version file; maintained inline |
| clog | Version unknown | No version metadata |

### 2. License Verification Gaps

| Component | Gap | Notes |
|-----------|-----|-------|
| mslk | License unknown | No LICENSE file found in submodule reference |
| aiter | License unknown | No LICENSE file found in submodule reference |
| valgrind-headers | GPL implications unclear | Headers may have different terms than full Valgrind |

### 3. Provenance Gaps

| Component | Gap | Notes |
|-----------|-----|-------|
| Generated wrappers | Generator is true dependency | Files in `third_party/*_wrappers/` are generated |
| QNNPACK | Divergence from upstream unknown | May have PyTorch-specific modifications |

### 4. Scope Limitations

| Area | What Was Not Scanned | Reason |
|------|---------------------|--------|
| Python dependencies | `requirements.txt`, `setup.py` | Out of scope per M09_plan |
| Transitive submodule deps | Submodules of submodules | Depth-1 scan only |
| Build-time dependencies | CMake `FetchContent`, Bazel | Runtime focus |

---

## Methodology

### Discovery Process

1. **Submodules:** Parsed `.gitmodules` and `git submodule status`
2. **Bundled:** Scanned `third_party/` for non-empty directories without `.git`
3. **Embedded:** Searched for LICENSE files in `aten/`, `c10/`, `torch/csrc/`
4. **Ported:** Grepped for "based on", "copied from", "ported from" patterns

### Classification Rules

| Type | Definition |
|------|------------|
| **Submodule** | Git submodule tracked in `.gitmodules` |
| **Bundled** | Full library copy in repository (not submodule) |
| **Vendored** | Partial library copy (subset of files) |
| **Embedded** | Third-party code within PyTorch source tree |
| **Ported** | Code adapted from another project with modifications |

### Confidence Levels

- **HIGH:** Commit SHA verified, LICENSE file present
- **MEDIUM:** Upstream URL known, version/license inferred
- **LOW:** Provenance unclear, marked UNKNOWN

---

## Related Documents

- **SBOM (Machine-Readable):** [`docs/refactor/sbom/M09_sbom.json`](M09_sbom.json)
- **Milestone Plan:** [`docs/refactor/milestones/M09/M09_plan.md`](../milestones/M09/M09_plan.md)
- **Baseline Audit:** [`docs/refactor/audit/BASELINE_AUDIT.md`](../audit/BASELINE_AUDIT.md)

---

**End of M09_THIRD_PARTY.md**

