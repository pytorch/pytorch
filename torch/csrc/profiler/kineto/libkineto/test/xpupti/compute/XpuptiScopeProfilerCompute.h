//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#pragma once

// Runs a small GEMM workload on the XPU. Implemented with SYCL device code in
// XpuptiScopeProfilerCompute.cpp, which must be compiled by a SYCL compiler
// (icpx -fsycl) and linked as the kernel-only shared library xpupti_compute.
// The signature is a pure host ABI (POD args, no STL on the boundary) so the
// GCC/MSVC-compiled test code can call it across the shared-library boundary.
//
// The symbol must be exported from the shared library: on Windows nothing is
// exported by default (dllexport when building the library, dllimport when the
// tests consume it); on ELF the default visibility already exports it.
#if defined(_WIN32)
#if defined(XPUPTI_COMPUTE_BUILDING)
#define XPUPTI_COMPUTE_API __declspec(dllexport)
#else
#define XPUPTI_COMPUTE_API __declspec(dllimport)
#endif
#else
#define XPUPTI_COMPUTE_API
#endif

XPUPTI_COMPUTE_API void ComputeOnXpu(unsigned size, unsigned repeatCount);
