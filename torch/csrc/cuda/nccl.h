#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <vector>

typedef struct ncclComm* ncclComm_t;

namespace torch {
namespace cuda {
namespace nccl {

// NOTE: this is exposed only so that python_nccl.cpp can some of these helpers.
// Don't use them outside of these files.
namespace detail {
/*
The below enums are from https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in 
however they're declare anyonomously there, so we can't forward declare them here.*/
/* Error type */
typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclNumResults = 6
} torchNcclResult_t;

/* Data types */
typedef enum {
  ncclInt8 = 0,
  ncclChar = 0,
  ncclUint8 = 1,
  ncclInt32 = 2,
  ncclInt = 2,
  ncclUint32 = 3,
  ncclInt64 = 4,
  ncclUint64 = 5,
  ncclFloat16 = 6,
  ncclHalf = 6,
  ncclFloat32 = 7,
  ncclFloat = 7,
  ncclFloat64 = 8,
  ncclDouble = 8,
  ncclNumTypes = 9
} torchNcclDataType_t;

#define TORCH_CUDA_NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  char internal[TORCH_CUDA_NCCL_UNIQUE_ID_BYTES];
} torchNcclUniqueId;

TORCH_CUDA_API void throw_nccl_error(torchNcclResult_t status);

static void NCCL_CHECK(torchNcclResult_t status);

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
TORCH_CUDA_API torchNcclDataType_t get_data_type(const at::Tensor& t);

} // namespace detail

using comm_list = std::vector<ncclComm_t>;
using stream_list = std::vector<c10::optional<at::cuda::CUDAStream>>;

TORCH_CUDA_API std::uint64_t version();

bool is_available(at::TensorList tensors);

TORCH_CUDA_API void get_unique_id(torchNcclUniqueId& id);
TORCH_CUDA_API ncclComm_t
comm_init_rank(int nranks, const torchNcclUniqueId& comm_id, int rank);
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
    int32_t op = 0,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = 0,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = 0,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = 0,
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
