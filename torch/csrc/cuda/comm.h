#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/Export.h>
#include <optional>

#include <cstddef>
#include <vector>

namespace torch::cuda {

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
    const std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>&
        streams = std::nullopt);

TORCH_CUDA_CU_API std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const std::optional<std::vector<int64_t>>& chunk_sizes = std::nullopt,
    int64_t dim = 0,
    const std::optional<std::vector<std::optional<at::cuda::CUDAStream>>>&
        streams = std::nullopt);

TORCH_CUDA_CU_API at::Tensor& gather_out(
    at::TensorList tensors,
    at::Tensor& out_tensor,
    int64_t dim);

TORCH_CUDA_CU_API at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    std::optional<int32_t> destination_index);

} // namespace torch::cuda
