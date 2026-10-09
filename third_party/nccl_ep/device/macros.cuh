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

#include <exception>
#include <string>
#include <cuda_runtime.h>

//==============================================================================
// Static Assert
//==============================================================================

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

//==============================================================================
// Exception Handling
//==============================================================================

class EPException: public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char *what() const noexcept override { return message.c_str(); }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd) \
do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
} while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond) \
do { \
    if (not (cond)) { \
        throw EPException("Assertion", __FILE__, __LINE__, #cond); \
    } \
} while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
#endif

//==============================================================================
// Kernel Launch Configuration
//==============================================================================

#ifndef SETUP_LAUNCH_CONFIG
#ifndef DISABLE_SM90_FEATURES
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[2]; \
    attr[0].id = cudaLaunchAttributeCooperative; \
    attr[0].val.cooperative = 1; \
    attr[1].id = cudaLaunchAttributeClusterDimension; \
    attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1); \
    attr[1].val.clusterDim.y = 1; \
    attr[1].val.clusterDim.z = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 2
#else
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
    int __num_sms = (sms); \
    int __num_threads = (threads); \
    auto __stream = (stream)
#endif
#endif

#ifndef LAUNCH_KERNEL
#ifndef DISABLE_SM90_FEATURES
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#else
#define LAUNCH_KERNEL(config, kernel, ...) \
do { \
    kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__); \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
        EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
        fprintf(stderr, "%s\n", cuda_exception.what()); \
        throw cuda_exception; \
    } \
} while (0)
#endif
#endif

#ifndef SET_SHARED_MEMORY_FOR_TMA
#ifndef DISABLE_SM90_FEATURES
#define SET_SHARED_MEMORY_FOR_TMA(kernel) \
EP_HOST_ASSERT(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) == cudaSuccess); \
cfg.dynamicSmemBytes = smem_size;
#else
#define SET_SHARED_MEMORY_FOR_TMA(kernel) void()
#endif
#endif

//==============================================================================
// Switch Macros for Template Dispatch
//==============================================================================

#define SWITCH_RANKS(case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(2); \
        case 4: case_macro(4); \
        case 8: case_macro(8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while(0)

#define SWITCH_RDMA_RANKS(rdma_ranks_val, case_macro) \
    do { switch (rdma_ranks_val) { \
        case 1:  case_macro(1);  break; \
        case 2:  case_macro(2);  break; \
        case 4:  case_macro(4);  break; \
        case 8:  case_macro(8);  break; \
        case 16: case_macro(16); break; \
        case 18: case_macro(18); break; \
        case 20: case_macro(20); break; \
        default: EP_HOST_ASSERT(false and "Unsupported RDMA ranks"); \
    } } while(0)

#define SWITCH_RANKS_WITH_DTYPE(dtype, case_macro) \
    switch (num_ranks) { \
        case 2: case_macro(dtype, 2); \
        case 4: case_macro(dtype, 4); \
        case 8: case_macro(dtype, 8); \
        default: EP_HOST_ASSERT(false and "Unsupported ranks"); \
    } while(0)

#define SWITCH_TYPES(case_macro) \
    switch (type) { \
        case CUDA_R_16BF: case_macro(nv_bfloat16); \
        default: EP_HOST_ASSERT(false and "Unsupported type"); \
    } while(0)

#define SWITCH_HIDDEN(case_macro) \
    switch (hidden) { \
        case 2048: case_macro(2048); \
        case 2560: case_macro(2560); \
        case 4096: case_macro(4096); \
        case 5120: case_macro(5120); \
        case 6144: case_macro(6144); /* For qwen3 coder */ \
        case 7168: case_macro(7168); \
        case 8192: case_macro(8192); \
        default: EP_HOST_ASSERT(false and "Unsupported hidden"); \
    } while(0)

