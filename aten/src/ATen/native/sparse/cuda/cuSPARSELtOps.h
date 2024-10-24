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
#include <cusparseLt.h>
#endif

namespace at::native {

// Ideally we would use the same DeviceThreadHandlePool mechanism as used in aten/src/ATen/cuda/CuSparseHandlePool.cpp
// which would handle this for us. However, the cuSPARSELt handle signature is different from that of cuSPARSE/cuBLAS,
// so it's not possible to reuse the existing pooling mechanism. Instead we have to handle our handles ourselves, which
// is why these variables are thread local. Once cuSPARSELt updates their handle signature to be consistent with the rest
// of CUDA, we can switch to using DeviceThreadHandlePool.
thread_local cusparseLtHandle_t handle;
thread_local bool handle_initialized = false;

at::Tensor _cslt_compress(const Tensor& sparse_input);

TORCH_CUDA_CPP_API std::tuple<at::Tensor, int64_t, int64_t, bool, int64_t> _cslt_sparse_mm_impl(
    const Tensor& compressed_A,
    const Tensor& dense_B,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& alpha_opt,
    const std::optional<c10::ScalarType> out_dtype_opt,
    bool transpose_result,
    int alg_id,
    int split_k,
    bool split_k_one_kernel,
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
    bool split_k_one_kernel
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
