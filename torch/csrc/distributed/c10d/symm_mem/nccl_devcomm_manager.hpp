#pragma once

#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <nccl_device.h>
#include <string>
#include <unordered_map>

namespace c10d::symmetric_memory {

// Manage all the NCCL device communicator business. Singleton.
class NCCLDevCommManager {
 public:
  // Constructor
  explicit NCCLDevCommManager(const c10::Device device) : device_(device) {}

  // Get single, global manager.
  static NCCLDevCommManager& get(const c10::Device device) {
    static NCCLDevCommManager manager(device);
    TORCH_CHECK(
        manager.device_ == device,
        "Detected use of NCCLDevCommManager on multiple devices. This is not supported.");
    return manager;
  }

  // Get an NCCL device communicator for a group.
  ncclDevComm get_devcomm(const std::string& group_name) {
    auto it = group_to_comms_.find(group_name);
    if (it == group_to_comms_.end()) {
      TORCH_CHECK(
          false,
          "NCCL device communicator for group ",
          group_name,
          " not found");
    }
    return it->second.second;
  }

  // Create device communicator if it doesn't exist. Skip if it already exists.
  void try_emplace_devcomm(const std::string& group_name, ncclComm_t comm) {
    auto it = group_to_comms_.find(group_name);
    if (it != group_to_comms_.end()) {
      return;
    }
    c10::cuda::CUDAGuard guard(device_);
    ncclDevComm devComm;
    ncclDevCommRequirements reqs;
    // See example in
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#simple-lsa-kernel
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
    // TODO: we need to figure out how to set the number of CTA and
    // requirements. See
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html#nccldevcommrequirements
    int nCTAs = 16;
    reqs.lsaBarrierCount = nCTAs;
    C10D_NCCL_CHECK(
        ncclDevCommCreate(comm, &reqs, &devComm), "ncclDevCommCreate failed");
    // Cache the device communicator for future reuse
    group_to_comms_[group_name] = std::make_pair(comm, devComm);
  }

  ~NCCLDevCommManager() noexcept {
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