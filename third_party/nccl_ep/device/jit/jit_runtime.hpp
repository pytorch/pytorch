/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace nccl_ep {
namespace jit {

enum class JitKernelStatus {
    kLaunched,
    kDisabled,
    kUnsupportedDevice,
    kCompileFailed,
    kLoadFailed,
    kAttributeFailed,
    kLaunchFailed,
};

struct JitKernelVariant {
    std::string_view kernel_family;
    std::string_view variant_name;
    std::string_view source;
    std::string_view entry_name;
    const void* identity = nullptr;
    std::uint64_t runtime_key = 0;
    int num_blocks = 0;
    int block_dim = 0;
    int dynamic_smem_bytes = 0;
    int min_sm = 90;
};

JitKernelStatus launch_jit_kernel(
    const JitKernelVariant& variant,
    void* kernel_param,
    std::size_t kernel_param_size,
    cudaStream_t stream,
    std::string* error);

JitKernelStatus launch_jit_kernel(
    const JitKernelVariant& variant,
    void* kernel_param,
    cudaStream_t stream,
    std::string* error);

const char* jit_kernel_status_name(JitKernelStatus status);

} // namespace jit
} // namespace nccl_ep
