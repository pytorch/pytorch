#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <array>
#include <mutex>

namespace torch::cuda::CUDAPluggableAllocator {

#if defined(TORCH_HIP_VERSION)
using streamType = c10::hip::HIPStream;
#else
using streamType = c10::cuda::CUDAStream;
#endif

std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
getCurrentAllocator();
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
createCustomAllocator(
    std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
    std::function<void(void*, size_t, int, cudaStream_t)> free_fn);
void changeCurrentAllocator(
    std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator> allocator);

struct _AllocationMetadata {
  _AllocationMetadata();
  _AllocationMetadata(size_t size, int device_idx, cudaStream_t stream);
  size_t size;
  int device_idx;
  cudaStream_t stream;
};

struct CUDAPluggableAllocator
    : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
  CUDAPluggableAllocator(
      std::function<void*(size_t, int, cudaStream_t)> alloc_fn,
      std::function<void(void*, size_t, int, cudaStream_t)> free_fn);

  CUDAPluggableAllocator(CUDAPluggableAllocator& other);

  void set_init_fn(std::function<void(int)> init_fn);

  void set_reset_fn(std::function<void()> reset_fn);

  void set_memory_fraction_fn(
      std::function<void(double, int)> memory_fraction_fn);

  void set_base_alloc_fn(std::function<void*(void*, size_t*)> base_alloc_fn);

  void set_record_stream_fn(
      std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn);

  void set_begin_allocate_stream_to_pool(
      std::function<void(int, cudaStream_t, c10::cuda::MempoolId_t)>
          capture_begin_fn);

  void set_end_allocate_stream_to_pool_fn(
      std::function<void(int, cudaStream_t)> capture_about_to_end_fn);

  void set_release_pool(
      std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn);

  void* malloc(size_t size, int device, cudaStream_t stream);

  c10::DataPtr allocate(size_t size) const override;
  c10::DeleterFnPtr raw_deleter() const override;

  virtual void* raw_alloc(size_t nbytes) override;
  virtual void* raw_alloc_with_stream(size_t nbytes, cudaStream_t stream)
      override;
  virtual void raw_delete(void* ptr) override;
  virtual void init(int device_count) override;
  virtual bool initialized() override;
  virtual void setMemoryFraction(double fraction, int device) override;
  virtual void emptyCache() override;
  virtual void cacheInfo(int dev_id, size_t* largestBlock) override;
  virtual void* getBaseAllocation(void* ptr, size_t* size) override;

  virtual void recordStream(const c10::DataPtr&, streamType stream) override;

  virtual c10::cuda::CUDACachingAllocator::DeviceStats getDeviceStats(
      int device) override;
  virtual void resetAccumulatedStats(int device) override;
  virtual void resetPeakStats(int device) override;
  virtual c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override;
  virtual void beginAllocateStreamToPool(
      int device,
      cudaStream_t stream,
      c10::cuda::MempoolId_t mempool_id) override;
  virtual void endAllocateStreamToPool(int device, cudaStream_t stream)
      override;
  virtual void releasePool(int device, c10::cuda::MempoolId_t mempool_id)
      override;
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  virtual void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) override;
  virtual void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  virtual std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
  getCheckpointState(int device, at::cuda::MempoolId_t id) override;
  virtual c10::cuda::CUDACachingAllocator::CheckpointDelta
  setCheckpointPoolState(
      int device,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> pps)
      override;
  virtual void enablePeerAccess(int dev, int dev_to_access) override;
  virtual cudaError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream,
      bool p2p_enabled) override;
  virtual std::string name() override;

 protected:
  std::function<void*(size_t, int, cudaStream_t)> alloc_fn_;
  std::function<void(void*, size_t, int, cudaStream_t)> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn_;
  std::function<void(int, cudaStream_t, c10::cuda::MempoolId_t)>
      begin_allocate_stream_to_pool_fn_;
  std::function<void(int, cudaStream_t)> end_allocate_stream_to_pool_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> relase_pool_fn_;
  std::mutex allocator_mutex_;
  // We do the bookeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace torch::cuda::CUDAPluggableAllocator
