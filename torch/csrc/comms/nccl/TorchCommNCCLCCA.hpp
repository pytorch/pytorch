// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/CallOnce.h>
#include <torch/csrc/comms/nccl/TorchCommNCCL.hpp>
#include <memory>

namespace torch::comms {

class NcclCachingAllocatorHookImpl {
 public:
  virtual ~NcclCachingAllocatorHookImpl() = default;
  virtual void regDeregMem(
      const c10::cuda::CUDACachingAllocator::TraceEntry& te);
  virtual void registerComm(TorchCommNCCL* comm);
  virtual void deregisterComm(TorchCommNCCL* comm);
  virtual void registerMemPreHook();
  virtual void clear();

  virtual bool isCommRegistered(TorchCommNCCL* comm);

 private:
  std::mutex mutex_;

  struct MemInfo {
    size_t len;
    int32_t device;

    MemInfo(size_t l, int32_t d) : len(l), device(d) {}
  };

  // Map of registered memory addresses to their sizes and device
  std::unordered_map<void*, MemInfo> registeredMemMap_;
  // Set of registered communicators. TorchComms, manages it's membership inside
  // this set.
  std::set<TorchCommNCCL*> registeredComms_;
};

class DefaultNcclCachingAllocatorHookImpl
    : public NcclCachingAllocatorHookImpl {
 public:
  DefaultNcclCachingAllocatorHookImpl();
  virtual ~DefaultNcclCachingAllocatorHookImpl() = default;

  // Delete copy constructor and assignment operator
  DefaultNcclCachingAllocatorHookImpl(
      const DefaultNcclCachingAllocatorHookImpl&) = delete;
  DefaultNcclCachingAllocatorHookImpl& operator=(
      const DefaultNcclCachingAllocatorHookImpl&) = delete;
  // Delete move constructor and assignment operator
  DefaultNcclCachingAllocatorHookImpl(DefaultNcclCachingAllocatorHookImpl&&) =
      delete;
  DefaultNcclCachingAllocatorHookImpl& operator=(
      DefaultNcclCachingAllocatorHookImpl&&) = delete;
};

class NcclCachingAllocatorHook {
 public:
  // Get the singleton instance
  static NcclCachingAllocatorHookImpl& getInstance();

  // only for use by tests
  static void setInstance(
      std::unique_ptr<NcclCachingAllocatorHookImpl> instance) {
    instance_ = std::move(instance);
  }

 protected:
  static void createInstance() {
    if (!instance_) {
      instance_ = std::make_unique<DefaultNcclCachingAllocatorHookImpl>();
    }
  }

  inline static std::unique_ptr<NcclCachingAllocatorHookImpl> instance_ =
      nullptr;
  inline static c10::once_flag init_flag_;
};

// Global function to be registered as a hook
void ncclCachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te);

} // namespace torch::comms
