#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Optional.h>

#include <nccl.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace cuda {
namespace nccl {

// NOTE: this is exposed only so that python_nccl.cpp can some of these helpers.
// Don't use them outside of these files.
namespace detail {

TORCH_CUDA_API void throw_nccl_error(ncclResult_t status);

static inline void NCCL_CHECK(ncclResult_t status) {
  if (status != ncclSuccess) {
    throw_nccl_error(status);
  }
}

struct AutoNcclGroup {
  AutoNcclGroup() {
    (c10::cuda::CUDACachingAllocator::getFreeMutex())->lock();
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    NCCL_CHECK(ncclGroupStart());
#endif
  }
  ~AutoNcclGroup() {
#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
    NCCL_CHECK(ncclGroupEnd());
#endif
    (c10::cuda::CUDACachingAllocator::getFreeMutex())->unlock();
  }
};

TORCH_CUDA_API at::ArrayRef<ncclComm_t> get_communicators(at::TensorList inputs);
TORCH_CUDA_API void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    int input_multiplier,
    int output_multiplier);
TORCH_CUDA_API ncclDataType_t get_data_type(const at::Tensor& t);

} // namespace detail

using comm_list = std::vector<ncclComm_t>;
using stream_list = std::vector<c10::optional<at::cuda::CUDAStream>>;

TORCH_CUDA_API std::uint64_t version();

bool is_available(at::TensorList tensors);

TORCH_CUDA_API void get_unique_id(ncclUniqueId& id);
TORCH_CUDA_API ncclComm_t comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank);
TORCH_CUDA_API void comm_destroy(ncclComm_t comm);

TORCH_CUDA_API void broadcast(
    at::TensorList tensors,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

size_t get_max_count();

TORCH_CUDA_API void reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t root = 0,
    int32_t op = ncclSum,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = ncclSum,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = ncclSum,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = ncclSum,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

} // namespace nccl
} // namespace cuda
} // namespace torch
