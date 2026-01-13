// nccl_symm_comm.cpp
// Host-side implementation for NCCL symmetric memory communication
//
// This file provides a "pure" example of a host-side class NCCLSymmComm that
// initializes NCCL, allocates symmetric memory buffer, registers it with a
// window, and creates the device communicator directly (without using
// NCCLDevCommManager).
//
// Usage from Python:
//   from torch._C._distributed_c10d import NCCLSymmComm
//   comm = NCCLSymmComm(group_name, buffer_size, device_idx)
//   ctx_ptr = comm.get_context_ptr()  # Pass to Triton kernel

#include <torch/csrc/_extern_triton/nccl_symm_comm.hpp>

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace c10d {
namespace symmetric_memory {

// Maximum number of memory barriers for NCCL device communicator.
// Each CTA will need a separate memory barrier.
constexpr int kLsaBarrierCount = 32;

NCCLSymmComm::NCCLSymmComm(
    const std::string& group_name,
    size_t buffer_size,
    int device_idx)
    : group_name_(group_name),
      buffer_size_(buffer_size),
      device_idx_(device_idx),
      rank_(0),
      world_size_(0),
      buffer_ptr_(nullptr),
      signal_pad_ptr_(nullptr),
      buffer_window_(nullptr),
      signal_window_(nullptr),
      comm_(nullptr),
      dev_comm_storage_(nullptr),
      owns_dev_comm_(false),
      context_dev_(nullptr) {
  // Allocate storage for ncclDevComm
  dev_comm_storage_ = new ncclDevComm();
  initialize();
}

NCCLSymmComm::~NCCLSymmComm() {
  cleanup();
  if (dev_comm_storage_ != nullptr) {
    delete static_cast<ncclDevComm*>(dev_comm_storage_);
    dev_comm_storage_ = nullptr;
  }
}

int64_t NCCLSymmComm::get_context_ptr() const {
  return reinterpret_cast<int64_t>(context_dev_);
}

int64_t NCCLSymmComm::get_buffer_ptr() const {
  return reinterpret_cast<int64_t>(buffer_ptr_);
}

size_t NCCLSymmComm::get_buffer_size() const {
  return buffer_size_;
}

int NCCLSymmComm::get_rank() const {
  return rank_;
}

int NCCLSymmComm::get_world_size() const {
  return world_size_;
}

int NCCLSymmComm::get_device_idx() const {
  return device_idx_;
}

void NCCLSymmComm::initialize() {
  c10::cuda::CUDAGuard guard(device_idx_);

  // Resolve process group
  auto group = resolve_process_group(group_name_);
  rank_ = group->getRank();
  world_size_ = group->getSize();

  // Get NCCL communicator from process group
  auto* ncclPg = dynamic_cast<c10d::ProcessGroupNCCL*>(
      group->getBackend(c10::DeviceType::CUDA).get());
  TORCH_CHECK(ncclPg != nullptr, "Backend must be an NCCL process group");

  // The NCCL communicator is lazily created when the first collective is
  // performed. We need to ensure the communicator exists before using
  // symmetric memory APIs. Perform a barrier to initialize the communicator.
  auto commPtr = ncclPg->getCommPtr();
  if (commPtr == 0) {
    // Communicator not initialized yet - perform a barrier to initialize it
    // We use a small tensor for the barrier operation
    auto dummy = at::empty(
        {1},
        at::TensorOptions().dtype(at::kByte).device(at::kCUDA, device_idx_));
    std::vector<at::Tensor> tensors = {dummy};
    auto work = ncclPg->barrier();
    work->wait();

    // Now get the communicator pointer again
    commPtr = ncclPg->getCommPtr();
    TORCH_CHECK(
        commPtr != 0,
        "NCCLSymmComm: Failed to initialize NCCL communicator. "
        "The NCCL backend requires at least one collective operation "
        "to be performed before symmetric memory APIs can be used.");
  }

  comm_ = reinterpret_cast<ncclComm_t>(commPtr);

  // Step 1: Allocate symmetric memory using ncclMemAlloc
  C10D_NCCL_CHECK(
      ncclMemAlloc(&buffer_ptr_, buffer_size_),
      "ncclMemAlloc failed for buffer");

  // Allocate signal pad (for synchronization)
  const size_t signal_pad_size = 4096; // Minimum page size
  C10D_NCCL_CHECK(
      ncclMemAlloc(&signal_pad_ptr_, signal_pad_size),
      "ncclMemAlloc failed for signal pad");

  // Step 2: Register buffer with window
  ncclWindow_t buffer_win;
  C10D_NCCL_CHECK(
      ncclCommWindowRegister(
          comm_,
          buffer_ptr_,
          buffer_size_,
          &buffer_win,
          NCCL_WIN_COLL_SYMMETRIC),
      c10::str(
          "Failed to window register buffer with ptr ",
          buffer_ptr_,
          ", size ",
          buffer_size_,
          " on rank ",
          rank_));
  buffer_window_ = reinterpret_cast<void*>(buffer_win);

  // Register signal pad with window
  ncclWindow_t signal_win;
  C10D_NCCL_CHECK(
      ncclCommWindowRegister(
          comm_,
          signal_pad_ptr_,
          signal_pad_size,
          &signal_win,
          NCCL_WIN_COLL_SYMMETRIC),
      c10::str(
          "Failed to window register signal pad with ptr ",
          signal_pad_ptr_,
          ", size ",
          signal_pad_size,
          " on rank ",
          rank_));
  signal_window_ = reinterpret_cast<void*>(signal_win);

  // Step 3: Create device communicator directly
  // See example in NCCL docs:
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#simple-lsa-kernel
  ncclDevComm* dev_comm = static_cast<ncclDevComm*>(dev_comm_storage_);
  ncclDevCommRequirements reqs;
  memset(&reqs, 0, sizeof(ncclDevCommRequirements));
  reqs.lsaBarrierCount = kLsaBarrierCount;
  C10D_NCCL_CHECK(
      ncclDevCommCreate(comm_, &reqs, dev_comm), "ncclDevCommCreate failed");
  owns_dev_comm_ = true;

  // Step 4: Create host-side context
  context_host_ = NCCLSymmContext(
      rank_,
      world_size_,
      buffer_win,
      signal_win,
      dev_comm,
      buffer_ptr_,
      buffer_size_,
      device_idx_);

  // Step 5: Copy context to device
  C10_CUDA_CHECK(cudaMalloc(&context_dev_, sizeof(NCCLSymmContext)));
  C10_CUDA_CHECK(cudaMemcpy(
      context_dev_,
      &context_host_,
      sizeof(NCCLSymmContext),
      cudaMemcpyHostToDevice));
}

void NCCLSymmComm::cleanup() {
  // Skip cleanup if CUDA context has exited
  if (is_finalizing()) {
    return;
  }

  try {
    c10::cuda::CUDAGuard guard(device_idx_);

    // Synchronize before destroying resources
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    // Free device context
    if (context_dev_ != nullptr) {
      cudaFree(context_dev_);
      context_dev_ = nullptr;
    }

    // Destroy device communicator (must be done before freeing memory)
    if (owns_dev_comm_ && comm_ != nullptr && dev_comm_storage_ != nullptr) {
      ncclDevComm* dev_comm = static_cast<ncclDevComm*>(dev_comm_storage_);
      ncclDevCommDestroy(comm_, dev_comm);
      owns_dev_comm_ = false;
    }

    // Free signal pad
    if (signal_pad_ptr_ != nullptr) {
      ncclMemFree(signal_pad_ptr_);
      signal_pad_ptr_ = nullptr;
    }

    // Free buffer
    if (buffer_ptr_ != nullptr) {
      ncclMemFree(buffer_ptr_);
      buffer_ptr_ = nullptr;
    }
  } catch (...) {
    // Ignore cleanup errors
    std::cerr << "Failed to cleanup NCCLSymmComm, skipping\n";
  }
}

} // namespace symmetric_memory
} // namespace c10d

#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
