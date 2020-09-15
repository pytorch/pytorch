#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Optional.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace cuda {
namespace nccl {

/* The following are copied from <nccl.h> and redefined in torch::cuda::nccl namespace */
/* pytorch should only use the following definition within pytorch scope */

/* Opaque handle to communicator to ncclComm*, this will reinterpret as ncclComm in nccl.cpp */
typedef void* ncclComm_t;

/** redefine nccl unique ID in torch scope. this should be identical to native nccl impp. */
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

/* Error type */
enum class ncclResult {
    Success                 =  0,
    UnhandledCudaError      =  1,
    SystemError             =  2,
    InternalError           =  3,
    InvalidArgument         =  4,
    InvalidUsage            =  5,
    NumResults              =  6 };

/* Reduction operation selector */
enum class ncclRedOp {
    Sum        = 0,
    Prod       = 1,
    Max        = 2,
    Min        = 3,
    NumOps     = 4 };

/* Data types */
enum class ncclDataType {
    Int8       = 0, Char       = 0,
    Uint8      = 1,
    Int32      = 2, Int        = 2,
    Uint32     = 3,
    Int64      = 4,
    Uint64     = 5,
    Float16    = 6, Half       = 6,
    Float32    = 7, Float      = 7,
    Float64    = 8, Double     = 8,
    numTypes   = 9 };



// NOTE: this is exposed only so that python_nccl.cpp can some of these helpers.
// Don't use them outside of these files.
namespace detail {

TORCH_CUDA_API void throw_nccl_error(ncclResult status);

static inline void NCCL_CHECK(ncclResult status) {
  if (status != ncclResult::Success) {
    throw_nccl_error(status);
  }
}

TORCH_CUDA_API at::ArrayRef<ncclComm_t> get_communicators(at::TensorList inputs);
TORCH_CUDA_API void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    int input_multiplier,
    int output_multiplier);
TORCH_CUDA_API void check_inputs(
    at::TensorList inputs,
    const at::Tensor& output,
    int root,
    int input_multiplier,
    int output_multiplier);

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
    at::Tensor& output,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
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
