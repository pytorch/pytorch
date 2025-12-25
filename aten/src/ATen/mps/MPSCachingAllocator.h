#pragma once

#include <ATen/mps/MPSAllocatorInterface.h>
#include <c10/core/CachingDeviceAllocator.h>
#include <memory>

namespace at::mps {

// Forward declaration for friend
struct MPSCachingAllocatorBuilder;

class MPSCachingAllocator final : public IMPSAllocator {
 public:
  static MPSCachingAllocator* get();
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  // c10::Allocator interface
  c10::DataPtr allocate(size_t size) override;
  c10::DeleterFnPtr raw_deleter() const override;

  // c10::DeviceAllocator interface
  bool initialized() override;
  void emptyCache(c10::MempoolId_t mempool_id = {0, 0}) override;
  void recordStream(const c10::DataPtr& ptr, c10::Stream stream) override;
  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override;
  void resetAccumulatedStats(c10::DeviceIndex device) override;
  void resetPeakStats(c10::DeviceIndex device) override;
  std::pair<size_t, size_t> getMemoryInfo(c10::DeviceIndex device) override;

  // IMPSAllocator MPS-specific interface
  void emptyCache() const override;
  void freeInactiveBuffers() const override;
  ssize_t getUnalignedBufferSize(const void* ptr) const override;
  IntArrayRef getBufferShape(const void* ptr) const override;
  id_t getBufferId(const void* ptr) const override;
  void setBufferShape(const void* ptr, const IntArrayRef& shape) const override;
  bool isSharedBuffer(const void* ptr) const override;
  bool isSharedStorageSupported() const override;
  c10::DataPtr allocScalarBufferWithValue(void* value, size_t size)
      const override;
  std::string formatSize(size_t size) const override;
  void setLowWatermarkRatio(double ratio) const override;
  void setHighWatermarkRatio(double ratio) const override;
  ssize_t getLowWatermarkValue() const override;
  size_t getLowWatermarkLimit() const override;
  size_t getHighWatermarkLimit() const override;
  size_t getTotalAllocatedMemory() const override;
  size_t getCurrentAllocatedMemory() const override;
  size_t getDriverAllocatedMemory() const override;
  size_t getRecommendedMaxMemory() const override;
  std::pair<const void*, uint32_t> getSharedBufferPtr(
      const void* ptr) const override;
  bool recordEvents(c10::ArrayRef<const void*> buffers) const override;
  bool waitForEvents(c10::ArrayRef<const void*> buffers) const override;

 private:
  MPSCachingAllocator() = default;
  friend struct MPSCachingAllocatorBuilder;
};

} // namespace at::mps