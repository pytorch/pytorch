// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef CK_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#endif

#define CK_TIME_KERNEL 1

// constant address space for kernel parameter
// https://llvm.org/docs/AMDGPUUsage.html#address-spaces
#define CK_CONSTANT_ADDRESS_SPACE __attribute__((address_space(4)))

// launch bounds
#define CK_USE_LAUNCH_BOUNDS 1

#ifdef CK_USE_LAUNCH_BOUNDS
#define CK_MAX_THREAD_PER_BLOCK 256
#define CK_MIN_BLOCK_PER_CU 2
#endif

// check GPU target
#ifdef __HIP_DEVICE_COMPILE__
#if !(defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) || \
      defined(__gfx90a__) || defined(__gfx1030__) || defined(__gfx1100__))
#error Not supported target
#endif
#endif

// buffer resource
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_BUFFER_RESOURCE_3RD_DWORD -1
#elif defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) || \
    defined(__gfx90a__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx1030__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31014000
#elif defined(__gfx1100__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x10020000
#endif

// FMA instruction
#ifndef __HIP_DEVICE_COMPILE__                   // for host code, define nothing
#elif defined(__gfx803__) || defined(__gfx900__) // for GPU code
#define CK_USE_AMD_V_MAC_F32
#elif defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx1030__) // for GPU code
#define CK_USE_AMD_V_FMAC_F32
#define CK_USE_AMD_V_DOT2_F32_F16
#define CK_USE_AMD_V_DOT4_I32_I8
#endif

// MFMA instruction
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_USE_AMD_MFMA
#elif defined(__gfx908__) || defined(__gfx90a__) // for GPU code
#define CK_USE_AMD_MFMA
#endif

#if defined(__gfx90a__)
#define CK_USE_AMD_MFMA_BF16_1K_OP
#endif

// WMMA instruction
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_USE_AMD_WMMA
#elif defined(__gfx1100__) // for GPU code
#define CK_USE_AMD_WMMA
#endif

// buffer load
#define CK_USE_AMD_BUFFER_LOAD 1

// buffer store
#define CK_USE_AMD_BUFFER_STORE 1

// buffer atomic add: integer
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_INTEGER 1

// buffer atomic add: floating point
#ifndef __HIP_DEVICE_COMPILE__ // for host code
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#elif defined(__gfx908__) || defined(__gfx90a__) // for GPU code
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 1
#else // for GPU code
#define CK_USE_AMD_BUFFER_ATOMIC_ADD_FLOAT 0
#endif

#if defined(__gfx90a__) // for GPU code
#define CK_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 1
#else
#define CK_USE_AMD_BUFFER_ATOMIC_MAX_FLOAT64 0
#endif

// inline asm
#define CK_USE_AMD_INLINE_ASM 1

// inner product (DLOP)
#define CK_USE_AMD_INNER_PRODUCT_INLINE_ASM 1

// block synchronization only s_wait lgkmcnt(0), not vmcnt(0)
#define CK_EXPERIMENTAL_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM 1

// experimental feature: multi index implemented as array
#define CK_EXPERIMENTAL_USE_DYNAMICALLY_INDEXED_MULTI_INDEX 0

// experimental feature: static tensor descriptor
#define CK_EXPERIMENTAL_STATIC_TENSOR_DESCRIPTOR 0

// experimental feature: buffer load/store/atomic-add/ OOB trick
// This (ifndef) is a hack to use customized behavior for buffer load rather than using default
// setting. Don't use this hack unless absolutely necessary!
// FIXME: make the behavior of buffer load a configurable (template) parameter for each usage
#ifndef CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
#define CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK 0
#endif
#define CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK 1
#define CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK 1
#define CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK 1

// experimental feature: in-regsiter sub-dword transpose
#define CK_EXPERIMENTAL_USE_IN_REGISTER_SUB_DWORD_TRANSPOSE 1

// experimental feature: merge transformation use magic number division
#define CK_EXPERIMENTAL_MERGE_USE_MAGIC_DIVISION 1

// experimental feature: use __builtin_memcpy instead of pointer cast to access a vector from
// pointer of scalar
#define CK_EXPERIMENTAL_USE_MEMCPY_FOR_VECTOR_ACCESS 0

// experimental feature: use __builtin_memcpy instead of union to do bit_cast
#define CK_EXPERIMENTAL_USE_MEMCPY_FOR_BIT_CAST 1

// experimental feature: optimize for inter-wave scheduling policy
#define CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING 1
#define CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS 1
// this will let make_default_loop_scheduler() return interwave scheduling flag by default
#define CK_EXPERIMENTAL_DEFAULT_TO_INTER_WAVE_SCHEDULING 0
// experimental feature: add instances using interwave scheduling
#define CK_EXPERIMENTAL_INTER_WAVE_INSTANCES 1
// experimental feature: add instances using pipeline v2
#define CK_EXPERIMENTAL_PIPELINE_V2_INSTANCES 1

// hack: have underlying assumption that need to be satsified, otherwise it's a bug
// hack for forcing register to keep idx_diff_low_const in SGPR. idx_diff_low_const must be
// thread-invariant, otherwise it's a bug
// TODO: separate index calculation into "compile-time", "global", "block", "wave", "thread"
#define CK_HACK_MERGE_CALCULATE_IDX_DIFF_LOW_CONST_USE_AMD_GCN_READ_FIRST_LANE 0

// workaround: compiler crash when compiling recursive lambda
#define CK_WORKAROUND_SWDEV_275126 1

// workaround: compiler crash when using buffer load/store for i8
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_BUFFER_LOAD_STORE_ISSUE 1

// workaround: compiler gnerating inefficient ds_write instructions
#define CK_WORKAROUND_SWDEV_XXXXXX_INT8_DS_WRITE_ISSUE 1

// workaround: verifaction failure, due to compiler regression, for conv bwd-data fp16 using some
// tuning parameter
#define CK_WORKAROUND_SWDEV_325164 0

// workaround: a BF16 attention kernel for gfx908 is likely affected by a compiler issue
#ifdef __gfx908__
#define CK_WORKAROUND_SWDEV_XXXXXX_BF16_ATTEN_FWD_GFX908_ISSUE 1
#else // __gfx90a__, ...
#define CK_WORKAROUND_SWDEV_XXXXXX_BF16_ATTEN_FWD_GFX908_ISSUE 0
#endif // __gfx908__

// flag to enable (1) or disable (0) the debugging output in some kernels
#define DEBUG_LOG 0

namespace ck {

enum struct InMemoryDataOperationEnum
{
    Set,
    AtomicAdd,
    AtomicMax,
    Add
};

// FIXME: use regular Sequence and remove this
template <InMemoryDataOperationEnum... Is>
struct InMemoryDataOperationEnumSequence
{
    static constexpr int mSize = sizeof...(Is);

    __host__ __device__ static constexpr InMemoryDataOperationEnum At(int I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const InMemoryDataOperationEnum mData[mSize + 1] = {Is..., InMemoryDataOperationEnum::Set};
        return mData[I];
    }
};

// index type
using index_t      = int32_t;
using long_index_t = int64_t;

} // namespace ck
