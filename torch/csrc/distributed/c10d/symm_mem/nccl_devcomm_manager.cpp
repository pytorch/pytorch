// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

#ifdef NCCL_HAS_SYMMEM_SUPPORT

#include <memory>

namespace c10d::symmetric_memory {

NCCLDevCommManager& NCCLDevCommManager::get(const c10::Device device) {
  static std::mutex mu;
  static std::
      unordered_map<c10::DeviceIndex, std::unique_ptr<NCCLDevCommManager>>
          managers;
  std::lock_guard<std::mutex> lock(mu);
  auto& slot = managers[device.index()];
  if (!slot) {
    slot = std::unique_ptr<NCCLDevCommManager>(new NCCLDevCommManager(device));
    LOG(INFO) << "[NCCLDevCommManager] created manager for device=" << device;
  }
  return *slot;
}

} // namespace c10d::symmetric_memory

#endif // NCCL_HAS_SYMMEM_SUPPORT
