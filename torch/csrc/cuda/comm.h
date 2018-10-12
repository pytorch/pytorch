#pragma once

#ifdef USE_CUDA

#include <ATen/ATen.h>
#include <ATen/core/optional.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <vector>

namespace torch { namespace cuda {

using tensor_list2d = std::vector<std::vector<at::Tensor>>;

std::vector<at::Tensor> broadcast(const at::Tensor& tensor, at::IntList devices);
tensor_list2d broadcast_coalesced(at::TensorList tensors, at::IntList devices,
                                  size_t buffer_size);

std::vector<at::Tensor> scatter(
    const at::Tensor& tensor,
    at::IntList devices,
    const at::optional<std::vector<int64_t>>& chunk_sizes = at::nullopt,
    int64_t dim = 0,
    const at::optional<std::vector<at::cuda::CUDAStream>>& streams = at::nullopt);

at::Tensor gather(
    at::TensorList tensors,
    int64_t dim,
    at::optional<int32_t> destination_index);
}}

#endif
