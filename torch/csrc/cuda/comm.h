#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>
#include <torch/csrc/Export.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace cuda {

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

TORCH_CUDA_CU_API std::vector<at::Tensor>& broadcast_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors);
TORCH_CUDA_CU_API std::vector<at::Tensor> broadcast(
    const at::Tensor& tensor,
    at::IntArrayRef devices);
TORCH_CUDA_CU_API tensor_list2d broadcast_coalesced(
    at::TensorList tensors,
    at::IntArrayRef devices,
    size_t buffer_size);

TORCH_CUDA_CU_API std::vector<at::Tensor>& scatter_out(
    const at::Tensor& tensor,
    std::vector<at::Tensor>& out_tensors,
    int64_t dim = 0,
    const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>>&
        streams = c10::nullopt);

TORCH_CUDA_CU_API std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const c10::optional<std::vector<int64_t>>& chunk_sizes = c10::nullopt,
    int64_t dim = 0,
    const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>>&
        streams = c10::nullopt);

TORCH_CUDA_CU_API at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim);

TORCH_CUDA_CU_API at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    c10::optional<int32_t> destination_index);
} // namespace cuda
} // namespace torch
