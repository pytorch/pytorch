#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <string>
#include <unordered_map>

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

namespace c10d::symmetric_memory {
// Maximum number of memory barriers for NCCL device communicator.
// Each CTA will need a separate memory barrier.
constexpr int NCCL_LSA_BARRIER_COUNT = 32;

// Manage all the NCCL device communicator business. Singleton.
class NCCLDevCommManager {
 public:
  // Constructor
  explicit NCCLDevCommManager(const c10::Device device) : device_(device) {}

  // Get single, global manager.
  static NCCLDevCommManager& get(const c10::Device device) {
    static NCCLDevCommManager manager(device);
    TORCH_CHECK_VALUE(
        manager.device_ == device,
        "Detected use of NCCLDevCommManager on multiple devices. This is not supported.");
    return manager;
  }

  // Get an NCCL device communicator for a group.
  ncclDevComm& get_devcomm(const std::string& group_name) {
    auto it = group_to_comms_.find(group_name);
    if (it == group_to_comms_.end()) {
      TORCH_CHECK(
          false,
          "NCCL device communicator for group ",
          group_name,
          " not found. Have you rendezvoused any tensor with this group?");
    }
    return it->second.second;
  }

  // Get a host-side communicator for a group.
  ncclComm_t get_comm(const std::string& group_name) {
    auto it = group_to_comms_.find(group_name);
    if (it == group_to_comms_.end()) {
      TORCH_CHECK(
          false,
          "NCCL host communicator for group ",
          group_name,
          " not found. Have you rendezvoused any tensor with this group?");
    }
    return it->second.first;
  }

  // Create device communicator if it doesn't exist. Skip if it already exists.
  void try_emplace_devcomm(const std::string& group_name, ncclComm_t comm) {
    auto it = group_to_comms_.find(group_name);
    if (it != group_to_comms_.end()) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_);
    ncclDevComm devComm;

    // Initializer available from NCCL 2.29
#ifdef NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
    // In 2.28, we can set it to zero
    ncclDevCommRequirements reqs;
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
#endif

    // Specifies the number of memory barriers to allocate.
    reqs.lsaBarrierCount = NCCL_LSA_BARRIER_COUNT;
    // TODO (kwen2501): Add network barrier count.
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devComm), "ncclDevCommCreate failed");
    // Cache the device communicator for future reuse
    // TODO (kwen2501):
    // Cache devComm not just based on group name, but also on requirements.
    group_to_comms_.emplace(group_name, std::make_pair(comm, devComm));
  }

  ~NCCLDevCommManager() noexcept {
    // Best effort to destroy the device communicators. Skip if CUDA context has
    // exited.
    try {
      c10::cuda::CUDAGuard guard(device_);
      // Make sure all kernels have completed before destroying the device
      // communicator.
      C10_CUDA_CHECK(cudaDeviceSynchronize());
      for (auto& [_, comm_pair] : group_to_comms_) {
        auto& [comm, devcomm] = comm_pair;
        ncclDevCommDestroy(comm, &devcomm);
      }
    } catch (...) {
      // Ignore the error
      std::cerr << "Failed to destroy the NCCL device communicator, skipping\n";
    }
  }

 private:
  // Device where the NCCL device communicator manager is created
  const c10::Device device_;
  // A map from group name to NCCL device communicator for that group.
  std::unordered_map<std::string, std::pair<ncclComm_t, ncclDevComm>>
      group_to_comms_;
};

} // namespace c10d::symmetric_memory
#endif // NCCL_HAS_SYMMEM_DEVICE_SUPPORT
