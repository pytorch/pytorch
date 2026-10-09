/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

// ============================================================================
// CUDA Compatibility Shims
//
// Provides compatibility functions for features introduced in newer CUDA versions.
// These shims enable code written for CUDA 13+ to compile and run on CUDA 12.x
// by implementing missing functions using PTX assembly.
//
// Functions provided:
// - elect_sync: Warp leader election primitive (CUDA 13.0+)
// - cp_async_bulk overload: Async bulk copy shared→global (CUDA 13.0+)
//
// These shims automatically disable when the native implementations are available.
// ============================================================================

#include <cuda/ptx>

// Only provide shims for CUDA versions that need them
#if !defined(__CUDACC_VER_MAJOR__) || (__CUDACC_VER_MAJOR__ < 13) || \
  ((__CUDACC_VER_MAJOR__ == 13) &&                                   \
   (!defined(__CUDACC_VER_MINOR__) || (__CUDACC_VER_MINOR__ < 1)))

namespace cuda { namespace ptx {

  // ============================================================================
  // elect_sync - Warp Leader Election
  // ============================================================================
  // Elects a single thread from active threads in a warp.
  // This is a hardware primitive available since SM_70, but the C++ API
  // was added in CUDA 13.0 / CCCL 2.3.0.
  //
  // Usage: if (cuda::ptx::elect_sync(mask)) { /* leader code */ }
  //
  // PTX instruction: elect.sync (PTX ISA 7.1+)
  // ============================================================================
  __device__ __forceinline__ bool elect_sync(unsigned mask) {
    int result;
    asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  elect.sync _|p, %1;\n\t"  // Sets predicate p=1 for elected thread
        "  selp.b32 %0, 1, 0, p;\n\t" // Convert predicate to integer
        "}\n\t"
        : "=r"(result)
        : "r"(mask)
    );
    return result != 0;
  }

  // ============================================================================
  // cp_async_bulk - Async Bulk Copy (shared→global overload)
  // ============================================================================
  // Performs asynchronous bulk copy from shared memory to global memory
  // with mbarrier-based completion tracking.
  //
  // This specific overload (space_shared, space_global) is available in
  // hardware since SM_90 but the C++ API was added in CUDA 13.0.
  //
  // Parameters:
  // - First two params: Memory space tags (shared source, global destination)
  // - __dst: Destination pointer in shared memory
  // - __src: Source pointer in global memory
  // - __size: Number of bytes to copy
  // - __mbar: Memory barrier for tracking completion
  //
  // PTX instruction: cp.async.bulk (PTX ISA 8.1+, SM_90+)
  // ============================================================================
  __device__ __forceinline__ void cp_async_bulk(
      space_shared_t, space_global_t,
      void* __dst, const void* __src,
      const uint32_t& __size, uint64_t* __mbar)
  {
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :
        : "r"(static_cast<unsigned>(__cvta_generic_to_shared(__dst))),
          "l"(__src),
          "r"(__size),
          "r"(static_cast<unsigned>(__cvta_generic_to_shared(__mbar)))
        : "memory"
    );
  }

}} // namespace cuda::ptx

#endif // CUDA < 13
