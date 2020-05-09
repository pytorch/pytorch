#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/cuda/ATenCUDAGeneral.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <vector>

namespace torch { namespace cuda {

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

TORCH_CUDA_API std::vector<at::Tensor> broadcast(const at::Tensor& tensor, at::IntArrayRef devices);
TORCH_CUDA_API tensor_list2d broadcast_coalesced(at::TensorList tensors, at::IntArrayRef devices,
                                  size_t buffer_size);

TORCH_CUDA_API std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntArrayRef devices,
    const c10::optional<std::vector<int64_t>>& chunk_sizes = c10::nullopt,
    int64_t dim = 0,
    const c10::optional<std::vector<c10::optional<at::cuda::CUDAStream>>>& streams =
        c10::nullopt);

TORCH_CUDA_API at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    c10::optional<int32_t> destination_index);
}}
