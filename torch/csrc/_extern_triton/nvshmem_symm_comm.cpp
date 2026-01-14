// nvshmem_symm_comm.cpp
// Host-side implementation for NVSHMEM symmetric memory communication
//
// This file provides a host-side class NVSHMEMSymmComm that initializes
// NVSHMEM, allocates symmetric memory buffer, and creates the device context
// that can be passed to Triton kernels.
//
// Usage from Python:
//   from torch._C._distributed_c10d import NVSHMEMSymmComm
//   comm = NVSHMEMSymmComm(group_name, buffer_size, device_idx)
//   ctx_ptr = comm.get_context_ptr()  # Pass to Triton kernel

#include <torch/csrc/_extern_triton/nvshmem_symm_comm.hpp>

#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/cuda/utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemoryUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <c10/util/error.h>

#include <cuda_runtime.h>

// Include NVSHMEM host API headers
#include <host/nvshmem_api.h>
#include <host/nvshmemx_api.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace c10d {
namespace symmetric_memory {

// Helper macro for NVSHMEM error checking
#define NVSHMEM_CHECK(stmt, msg)                                             \
  do {                                                                       \
    int result = (stmt);                                                     \
    TORCH_CHECK(                                                             \
        result == 0,                                                         \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) + " " + msg + \
            ". Error code: " + std::to_string(result));                      \
  } while (0)

// Static store exchange helper for NVSHMEM initialization
static StoreExchange storeExchange = StoreExchange("NVSHMEMSymmComm");

// Static flag to track NVSHMEM initialization state
static bool nvshmem_initialized = false;

// Initialize environment variables from NCCL settings for convenience
static void maybe_initialize_env_vars() {
  auto nccl_socket_if_name = c10::utils::get_env("NCCL_SOCKET_IFNAME");
  auto nccl_hca_list = c10::utils::get_env("NCCL_IB_HCA");
  auto nccl_ib_gid_index = c10::utils::get_env("NCCL_IB_GID_INDEX");
  auto nvshmem_socket_if_name =
      c10::utils::get_env("NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME");
  auto nvshmem_hca_list = c10::utils::get_env("NVSHMEM_HCA_LIST");
  auto nvshmem_ib_gid_index = c10::utils::get_env("NVSHMEM_IB_GID_INDEX");

  if (!nvshmem_socket_if_name.has_value() && nccl_socket_if_name.has_value()) {
    c10::utils::set_env(
        "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME", nccl_socket_if_name->c_str());
  }
  if (!nvshmem_hca_list.has_value() && nccl_hca_list.has_value()) {
    c10::utils::set_env("NVSHMEM_ENABLE_NIC_PE_MAPPING", "1");
    c10::utils::set_env("NVSHMEM_HCA_LIST", nccl_hca_list->c_str());
  }
  if (!nvshmem_ib_gid_index.has_value() && nccl_ib_gid_index.has_value()) {
    c10::utils::set_env("NVSHMEM_IB_GID_INDEX", nccl_ib_gid_index->c_str());
  }
}

// Initialize NVSHMEM with the process group's store
static void initialize_nvshmem_with_store(
    c10::intrusive_ptr<c10d::Store> store,
    int rank,
    int world_size,
    int device_idx) {
  if (nvshmem_initialized) {
    return;
  }

  // Check if NVSHMEM is already initialized by another code path
  // (e.g., NVSHMEMSymmetricMemoryAllocator)
  int init_status = nvshmemx_init_status();
  if (init_status == NVSHMEM_STATUS_IS_INITIALIZED) {
    nvshmem_initialized = true;
    LOG(INFO) << "NVSHMEMSymmComm: NVSHMEM already initialized by another path";
    return;
  }

  c10::cuda::CUDAGuard guard(device_idx);
  maybe_initialize_env_vars();

  // Make sure the CUDA runtime is initialized
  cudaFree(nullptr);

  // Get unique ID for NVSHMEM bootstrap
  nvshmemx_uniqueid_t unique_id;
  NVSHMEM_CHECK(
      nvshmemx_get_uniqueid(&unique_id), "nvshmemx_get_uniqueid failed");

  // Exchange unique IDs using the store
  auto unique_ids =
      storeExchange.all_gather(store, rank, world_size, unique_id);

  // Initialize NVSHMEM with the unique ID
  nvshmemx_init_attr_t attr;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, &unique_ids[0], &attr);

  NVSHMEM_CHECK(
      nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr),
      "nvshmemx_init_attr failed");

  nvshmem_initialized = true;

  // Print version for logging
  int major, minor;
  nvshmem_info_get_version(&major, &minor);
  LOG(INFO) << "NVSHMEMSymmComm: NVSHMEM initialized, version: " << major << '.'
            << minor;
}

NVSHMEMSymmComm::NVSHMEMSymmComm(
    const std::string& group_name,
    size_t buffer_size,
    int device_idx)
    : group_name_(group_name),
      buffer_size_(buffer_size),
      device_idx_(device_idx),
      rank_(0),
      world_size_(0),
      buffer_ptr_(nullptr),
      context_dev_(nullptr) {
  initialize();
}

NVSHMEMSymmComm::~NVSHMEMSymmComm() {
  cleanup();
}

int64_t NVSHMEMSymmComm::get_context_ptr() const {
  return reinterpret_cast<int64_t>(context_dev_);
}

int64_t NVSHMEMSymmComm::get_buffer_ptr() const {
  return reinterpret_cast<int64_t>(buffer_ptr_);
}

size_t NVSHMEMSymmComm::get_buffer_size() const {
  return buffer_size_;
}

int NVSHMEMSymmComm::get_rank() const {
  return rank_;
}

int NVSHMEMSymmComm::get_world_size() const {
  return world_size_;
}

int NVSHMEMSymmComm::get_device_idx() const {
  return device_idx_;
}

void NVSHMEMSymmComm::initialize() {
  c10::cuda::CUDAGuard guard(device_idx_);

  // Resolve process group for group-local rank
  auto group = resolve_process_group(group_name_);
  rank_ = group->getRank();
  world_size_ = group->getSize();

  // Get global rank from the global process group (group "0")
  // NVSHMEM uses global PE numbers, not group-local ranks
  auto global_group = resolve_process_group("0");
  int global_rank = global_group->getRank();
  int global_world_size = global_group->getSize();

  // Initialize NVSHMEM using the global process group's store
  // This follows the pattern from NVSHMEMSymmetricMemoryAllocator
  initialize_nvshmem_with_store(
      global_group->getStore(), global_rank, global_world_size, device_idx_);

  // Allocate symmetric memory using nvshmem_malloc directly
  // This is the NVSHMEM-prescribed way per its documentation
  buffer_ptr_ = nvshmem_malloc(buffer_size_);
  TORCH_CHECK(
      buffer_ptr_ != nullptr || buffer_size_ == 0,
      "nvshmem_malloc failed to allocate ",
      buffer_size_,
      " bytes");

  // Zero-initialize the buffer
  if (buffer_ptr_ != nullptr && buffer_size_ > 0) {
    C10_CUDA_CHECK(cudaMemset(buffer_ptr_, 0, buffer_size_));
  }

  // Synchronize to ensure all PEs have completed allocation
  C10_CUDA_CHECK(cudaDeviceSynchronize());

  // Create host-side context with GLOBAL PE numbers for NVSHMEM
  // The kernel needs global_rank and global_world_size for nvshmem_ptr()
  context_host_ = NVSHMEMSymmContext(
      rank_,
      world_size_,
      buffer_ptr_,
      buffer_size_,
      device_idx_,
      0, // offset = 0
      global_rank,
      global_world_size);

  // Copy context to device
  C10_CUDA_CHECK(cudaMalloc(&context_dev_, sizeof(NVSHMEMSymmContext)));
  C10_CUDA_CHECK(cudaMemcpy(
      context_dev_,
      &context_host_,
      sizeof(NVSHMEMSymmContext),
      cudaMemcpyHostToDevice));
}

void NVSHMEMSymmComm::cleanup() {
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

    // Free symmetric memory using nvshmem_free
    // nvshmem_free has no return value
    if (buffer_ptr_ != nullptr) {
      nvshmem_free(buffer_ptr_);
      buffer_ptr_ = nullptr;
    }
  } catch (...) {
    // Ignore cleanup errors
    std::cerr << "Failed to cleanup NVSHMEMSymmComm, skipping\n";
  }
}

} // namespace symmetric_memory
} // namespace c10d
