// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifdef USE_C10D_NCCL

#include <torch/csrc/distributed/c10d/nccl2/NCCLMemAllocator.hpp>

#include <algorithm>
#include <iterator>

#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/distributed/c10d/nccl2/NcclApi.hpp>
#include <torch/csrc/distributed/c10d/nccl2/ProcessGroupNCCL.hpp>

namespace c10d::nccl2 {

namespace {

// Window calls are collective, so issue them in group-name order, which is the
// same on every rank.
std::vector<ProcessGroupNCCL*> pgsForDevice(
    const std::vector<ProcessGroupNCCL*>& pgs,
    int device) {
  std::vector<ProcessGroupNCCL*> out;
  std::ranges::copy_if(pgs, std::back_inserter(out), [&](auto* pg) {
    return pg->getDevice().index() == device;
  });
  std::ranges::stable_sort(out, {}, [](auto* pg) { return pg->getCommName(); });
  return out;
}

} // namespace

const std::shared_ptr<NCCLMemAllocator>& NCCLMemAllocator::global() {
  static const auto allocator =
      std::make_shared<NCCLMemAllocator>(/*symm=*/false);
  return allocator;
}

NCCLMemAllocator::NCCLMemAllocator(bool symm)
    : CUDAPluggableAllocator(
          [this](size_t size, int device, cudaStream_t /*stream*/) {
            return allocSegment(size, device);
          },
          [this](void* ptr, size_t /*size*/, int device, cudaStream_t) {
            freeSegment(ptr, device);
          }),
      nccl_api_(std::make_shared<DefaultNcclApi>()),
      symm_(symm) {}

void* NCCLMemAllocator::allocSegment(size_t size, int device) {
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  void* ptr = nullptr;
  auto rc = nccl_api_->memAlloc(&ptr, size);
  TORCH_CHECK(
      rc == ncclSuccess,
      "ncclMemAlloc failed: ",
      nccl_api_->getErrorString(rc));
  const auto deviceIndex = static_cast<c10::DeviceIndex>(device);
  std::lock_guard<std::mutex> lock(mutex_);
  if (symm_) {
    try {
      for (auto* pg : pgs_) {
        pg->registerWindow(ptr, size, deviceIndex);
      }
    } catch (...) {
      for (auto* pg : pgs_) {
        pg->deregisterWindow(ptr);
      }
      NCCL_CHECK_IGNORE(
          nccl_api_, nccl_api_->memFree(ptr), "ncclMemFree failed");
      throw;
    }
  }
  std::lock_guard<std::mutex> segmentsLock(segments_mutex_);
  segments_.push_back({ptr, size, deviceIndex});
  return ptr;
}

void NCCLMemAllocator::freeSegment(void* ptr, int device) {
  at::cuda::OptionalCUDAGuard gpuGuard(device);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    {
      std::lock_guard<std::mutex> segmentsLock(segments_mutex_);
      auto it = std::ranges::find(segments_, ptr, &Segment::ptr);
      TORCH_INTERNAL_ASSERT(
          it != segments_.end(), "freeing an unknown segment");
      segments_.erase(it);
    }
    for (auto* pg : pgsForDevice(pgs_, device)) {
      pg->deregisterWindow(ptr);
    }
  }
  auto rc = nccl_api_->memFree(ptr);
  TORCH_CHECK(
      rc == ncclSuccess, "ncclMemFree failed: ", nccl_api_->getErrorString(rc));
}

bool NCCLMemAllocator::ownsSegment(const void* ptr) const {
  std::lock_guard<std::mutex> lock(segments_mutex_);
  return std::ranges::find(segments_, ptr, &Segment::ptr) != segments_.end();
}

void NCCLMemAllocator::attach(ProcessGroupNCCL& pg) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (std::ranges::find(pgs_, &pg) == pgs_.end()) {
    pgs_.push_back(&pg);
  }
  if (!symm_) {
    return;
  }
  // segments_ only changes with mutex_ held.
  for (const auto& segment : segments_) {
    pg.registerWindow(segment.ptr, segment.size, segment.device);
  }
}

void NCCLMemAllocator::detach(ProcessGroupNCCL& pg) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = std::ranges::find(pgs_, &pg);
  if (it == pgs_.end()) {
    return;
  }
  pgs_.erase(it);
  for (const auto& segment : segments_) {
    pg.deregisterWindow(segment.ptr);
  }
}

} // namespace c10d::nccl2

#endif // USE_C10D_NCCL
