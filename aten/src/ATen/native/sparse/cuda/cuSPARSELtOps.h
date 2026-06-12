#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Half.h>
#include <cusparse.h>
#include <cstdint>

#if AT_CUSPARSELT_ENABLED()
// ROCm 7.0.2's amd_hip_bf16.h (pulled in via hipsparselt.h -> hip_fp8.h ->
// amd_hip_fp8.h -> amd_hip_bf16.h) contains device-only builtins (warpSize,
// __shfl_*_sync) that fail during host compilation. Pre-define the hip_fp8.h
// include guard to prevent this chain. The hipsparselt C API uses opaque
// handles and enum types, not C++ fp8 types directly.
#if defined(USE_ROCM) && !defined(__HIP_DEVICE_COMPILE__)
#ifndef HIP_INCLUDE_HIP_HIP_FP8_H
#define HIP_INCLUDE_HIP_HIP_FP8_H
#endif
#endif
#include <cusparseLt.h>
#endif

namespace at::native {

at::Tensor _cslt_compress(const Tensor& sparse_input);

TORCH_CUDA_CPP_API std::tuple<at::Tensor, int64_t, int64_t, int64_t, int64_t> _cslt_sparse_mm_impl(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int alg_id,
    int split_k,
    int split_k_mode,
    bool search_alg_id
);

at::Tensor _cslt_sparse_mm(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int64_t alg_id,
    int64_t split_k,
    int64_t split_k_mode
);

int64_t _cslt_sparse_mm_search(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result
);

} // namespace at::native
