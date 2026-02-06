#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSCachingAllocator.h>
#include <c10/util/ArrayRef.h>

namespace at::mps {

// Helper struct implementation (friend of MPSCachingAllocator)
struct MPSCachingAllocatorBuilder {
  static std::unique_ptr<MPSCachingAllocator> build() {
    return std::unique_ptr<MPSCachingAllocator>(new MPSCachingAllocator());
  }
};

MPSCachingAllocator* MPSCachingAllocator::get() {
  static std::unique_ptr<MPSCachingAllocator> allocator = MPSCachingAllocatorBuilder::build();
  return allocator.get();
}

// c10::Allocator interface
c10::DataPtr MPSCachingAllocator::allocate(size_t size) {
  return getIMPSAllocator()->allocate(size);
}

c10::DeleterFnPtr MPSCachingAllocator::raw_deleter() const {
  return getIMPSAllocator()->raw_deleter();
}

void MPSCachingAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  default_copy_data(dest, src, count);
}

// c10::DeviceAllocator interface
bool MPSCachingAllocator::initialized() {
  return getIMPSAllocator()->initialized();
}

void MPSCachingAllocator::emptyCache(c10::MempoolId_t mempool_id) {
  getIMPSAllocator()->emptyCache(mempool_id);
}

void MPSCachingAllocator::recordStream(const c10::DataPtr& ptr, c10::Stream stream) {
  getIMPSAllocator()->recordStream(ptr, stream);
}

c10::CachingDeviceAllocator::DeviceStats MPSCachingAllocator::getDeviceStats(c10::DeviceIndex device) {
  return getIMPSAllocator()->getDeviceStats(device);
}

void MPSCachingAllocator::resetAccumulatedStats(c10::DeviceIndex device) {
  getIMPSAllocator()->resetAccumulatedStats(device);
}

void MPSCachingAllocator::resetPeakStats(c10::DeviceIndex device) {
  getIMPSAllocator()->resetPeakStats(device);
}

std::pair<size_t, size_t> MPSCachingAllocator::getMemoryInfo(c10::DeviceIndex device) {
  return getIMPSAllocator()->getMemoryInfo(device);
}

// IMPSAllocator MPS-specific interface
void MPSCachingAllocator::emptyCache() const {
  getIMPSAllocator()->emptyCache();
}

void MPSCachingAllocator::freeInactiveBuffers() const {
  getIMPSAllocator()->freeInactiveBuffers();
}

ssize_t MPSCachingAllocator::getUnalignedBufferSize(const void* ptr) const {
  return getIMPSAllocator()->getUnalignedBufferSize(ptr);
}

IntArrayRef MPSCachingAllocator::getBufferShape(const void* ptr) const {
  return getIMPSAllocator()->getBufferShape(ptr);
}

id_t MPSCachingAllocator::getBufferId(const void* ptr) const {
  return getIMPSAllocator()->getBufferId(ptr);
}

void MPSCachingAllocator::setBufferShape(const void* ptr, const IntArrayRef& shape) const {
  getIMPSAllocator()->setBufferShape(ptr, shape);
}

bool MPSCachingAllocator::isSharedBuffer(const void* ptr) const {
  return getIMPSAllocator()->isSharedBuffer(ptr);
}

bool MPSCachingAllocator::isSharedStorageSupported() const {
  return getIMPSAllocator()->isSharedStorageSupported();
}

c10::DataPtr MPSCachingAllocator::allocScalarBufferWithValue(void* value, size_t size) const {
  return getIMPSAllocator()->allocScalarBufferWithValue(value, size);
}

std::string MPSCachingAllocator::formatSize(size_t size) const {
  return getIMPSAllocator()->formatSize(size);
}

void MPSCachingAllocator::setLowWatermarkRatio(double ratio) const {
  getIMPSAllocator()->setLowWatermarkRatio(ratio);
}

void MPSCachingAllocator::setHighWatermarkRatio(double ratio) const {
  getIMPSAllocator()->setHighWatermarkRatio(ratio);
}

ssize_t MPSCachingAllocator::getLowWatermarkValue() const {
  return getIMPSAllocator()->getLowWatermarkValue();
}

size_t MPSCachingAllocator::getLowWatermarkLimit() const {
  return getIMPSAllocator()->getLowWatermarkLimit();
}

size_t MPSCachingAllocator::getHighWatermarkLimit() const {
  return getIMPSAllocator()->getHighWatermarkLimit();
}

size_t MPSCachingAllocator::getTotalAllocatedMemory() const {
  return getIMPSAllocator()->getTotalAllocatedMemory();
}

size_t MPSCachingAllocator::getCurrentAllocatedMemory() const {
  return getIMPSAllocator()->getCurrentAllocatedMemory();
}

size_t MPSCachingAllocator::getDriverAllocatedMemory() const {
  return getIMPSAllocator()->getDriverAllocatedMemory();
}

size_t MPSCachingAllocator::getRecommendedMaxMemory() const {
  return getIMPSAllocator()->getRecommendedMaxMemory();
}

std::pair<const void*, uint32_t> MPSCachingAllocator::getSharedBufferPtr(const void* ptr) const {
  return getIMPSAllocator()->getSharedBufferPtr(ptr);
}

bool MPSCachingAllocator::recordEvents(c10::ArrayRef<const void*> buffers) const {
  return getIMPSAllocator()->recordEvents(buffers);
}

bool MPSCachingAllocator::waitForEvents(c10::ArrayRef<const void*> buffers) const {
  return getIMPSAllocator()->waitForEvents(buffers);
}

} // namespace at::mps