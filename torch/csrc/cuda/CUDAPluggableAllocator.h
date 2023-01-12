#pragma once

#include <c10/core/Allocator.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/cuda/CUDAStream.h>

#include <c10/cuda/CUDACachingAllocator.h>

#include <array>
#include <mutex>

namespace torch {

namespace cuda {

namespace CUDAPluggableAllocator {

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

  void set_capture_begin_fn(
      std::function<void(int, c10::cuda::CaptureId_t, c10::cuda::MempoolId_t)>
          capture_begin_fn);

  void set_capture_about_to_end_fn(
      std::function<void(int, c10::cuda::CaptureId_t)> capture_about_to_end_fn);

  void set_capture_ended_fn(
      std::function<void(int, c10::cuda::CaptureId_t)> capture_ended_fn);

  void set_capture_destroy_fn(
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
  virtual void notifyCaptureBegin(
      int device,
      c10::cuda::CaptureId_t graph_id,
      c10::cuda::MempoolId_t mempool_id) override;
  virtual void notifyCaptureAboutToEnd(
      int device,
      c10::cuda::CaptureId_t graph_id) override;
  virtual void notifyCaptureEnded(int device, c10::cuda::CaptureId_t graph_id)
      override;
  virtual void notifyCaptureDestroy(
      int device,
      c10::cuda::MempoolId_t mempool_id) override;
  virtual std::shared_ptr<void> getIpcDevPtr(std::string handle) override;
  virtual void recordHistory(
      bool enabled,
      c10::cuda::CUDACachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      bool alloc_trace_record_context) override;
  virtual void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver observer) override;
  virtual bool needsPoolSpecificPeerAccess() override;
  virtual std::string name() override;

 protected:
  std::function<void*(size_t, int, cudaStream_t)> alloc_fn_;
  std::function<void(void*, size_t, int, cudaStream_t)> free_fn_;
  std::function<void(int)> init_fn_;
  std::function<void()> reset_fn_;
  std::function<void(double, int)> memory_fraction_fn_;
  std::function<void*(void*, size_t*)> base_alloc_fn_;
  std::function<void(void* ptr, cudaStream_t stream)> record_stream_fn_;
  std::function<void(int, c10::cuda::CaptureId_t, c10::cuda::MempoolId_t)>
      capture_begin_fn_;
  std::function<void(int, c10::cuda::CaptureId_t)> capture_about_to_end_fn_;
  std::function<void(int, c10::cuda::CaptureId_t)> capture_ended_fn_;
  std::function<void(int, c10::cuda::MempoolId_t)> capture_destroy_fn_;
  std::mutex allocator_mutex_;
  // We do the bookeeping here in order to simplify custom allocators
  std::unordered_map<void*, _AllocationMetadata> allocation_metadata_;

  bool initialized_ = false;
};
} // namespace CUDAPluggableAllocator
} // namespace cuda
} // namespace torch
