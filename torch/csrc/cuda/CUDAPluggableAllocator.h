#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <mutex>

namespace torch::cuda::CUDAPluggableAllocator {

#if defined(USE_ROCM)
using streamType = c10::hip::HIPStream;
#else
using streamType = c10::cuda::CUDAStream;
#endif

TORCH_CUDA_CPP_API std::shared_ptr<
    c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator();
TORCH_CUDA_CPP_API std::shared_ptr<
    c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn);
TORCH_CUDA_CPP_API void changeCurrentAllocator(
    const std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>&
        allocator);

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(
      size_t size,
      c10::DeviceIndex device_idx,
      cudaStream_t stream);
  size_t size;
  c10::DeviceIndex device_idx;
  cudaStream_t stream{};
};

struct TORCH_CUDA_CPP_API CUDAPluggableAllocator
    : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
  CUDAPluggableAllocator(
      std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
      std::function<void(void*, size_t, int, cudaStream_t)> free_fn);

  CUDAPluggableAllocator(CUDAPluggableAllocator& other);
  CUDAPluggableAllocator(CUDAPluggableAllocator&& other) = delete;
  CUDAPluggableAllocator& operator=(const CUDAPluggableAllocator& other) =
      delete;
  CUDAPluggableAllocator& operator=(CUDAPluggableAllocator&& other) = delete;
  ~CUDAPluggableAllocator() override = default;

  void set_init_fn(std::function<void(int)> init_fn);

  void set_reset_fn(std::function<void()> reset_fn);

  void set_memory_fraction_fn(
      std::function<void(double, int)> memory_fraction_fn);

  void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);

  void set_record_stream_fn(
      std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn);

  void set_begin_allocate_to_pool(
      std::function<
          void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
          capture_begin_fn);

  void set_end_allocate_to_pool_fn(
      std::function<void(int, c10::cuda::MempoolId_t)> capture_about_to_end_fn);

  void set_release_pool(
      std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn);

  void* malloc(size_t size, c10::DeviceIndex device, cudaStream_t stream);

  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  void* raw_alloc(size_t nbytes) override;
  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream) override;
  void raw_delete(void* ptr) override;
  void init(int device_count) override;
  bool initialized() override;
  double getMemoryFraction(c10::DeviceIndex device) override;
  void setMemoryFraction(double fraction, c10::DeviceIndex device) override;
  std::vector<c10::cuda::CUDACachingAllocator::StreamSegmentSize>
  getExpandableSegmentSizes(c10::DeviceIndex device) override;
  void emptyCache(c10::cuda::MempoolId_t mempool_id = {0, 0}) override;
  void enable(bool /*value*/) override {}
  bool isEnabled() const override {
    return true;
  }
  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override;
  void* getBaseAllocation(void* ptr, size_t* size) override;

  void recordStream(const c10::DataPtr& /*ptr*/, streamType stream) override;

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot(
      c10::cuda::MempoolId_t mempool) override;
  void beginAllocateToPool(
      c10::DeviceIndex device,
      c10::cuda::MempoolId_t mempool_id,
      std::function<bool(cudaStream_t)> /*filter*/) override;
  void endAllocateToPool(
      c10::DeviceIndex device,
      c10::cuda::MempoolId_t mempool_id) override;
  void releasePool(c10::DeviceIndex device, c10::cuda::MempoolId_t mempool_id)
      override;
  std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(
      void* /*ptr*/) override;
  void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      c10::cuda::CUDACachingAllocator::RecordContext when,
      bool clearHistory,
      const std::vector<std::string>& skip_actions) override;
  void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  void attachAllocatorTraceTracker(
      c10::cuda::CUDACachingAllocator::AllocatorTraceTracker tracker) override;
  std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex device, at::cuda::MempoolId_t id)
      override;
  c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps)
      override;
  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access)
      override;
  cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override;
  std::string name() override;
  void copy_data(void* dest, const void* src, std::size_t count) const final;

 protected:
  std::function<void*(size_t, int, cudaStream_t)> alloc_fn_;
  std::function<void(void*, size_t, int, cudaStream_t)> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn_;
  std::function<
      void(int, c10::cuda::MempoolId_t, std::function<bool(cudaStream_t)>)>
      begin_allocate_to_pool_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> end_allocate_to_pool_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookkeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace torch::cuda::CUDAPluggableAllocator
