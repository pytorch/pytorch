#pragma once

#include <c10/core/CachingDeviceAllocator.h>
#include <c10/core/DeviceType.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10::hip {

// Takes a valid HIPAllocator (of any sort) and turns it into
// an allocator pretending to be a CUDA allocator.  See
// Note [Masquerading as CUDA]
class HIPAllocatorMasqueradingAsCUDA final : public DeviceAllocator {
  DeviceAllocator* allocator_;
public:
  explicit HIPAllocatorMasqueradingAsCUDA(DeviceAllocator* allocator)
    : allocator_(allocator) {}
  DataPtr allocate(size_t size) override {
    DataPtr r = allocator_->allocate(size);
    r.unsafe_set_device(Device(c10::DeviceType::CUDA, r.device().index()));
    return r;
  }
  DeleterFnPtr raw_deleter() const override {
    return allocator_->raw_deleter();
  }
  void copy_data(void* dest, const void* src, std::size_t count) const final {
    allocator_->copy_data(dest, src, count);
  }
  bool initialized() override {
    return allocator_->initialized();
  }
  void emptyCache(MempoolId_t mempool_id = {0, 0}) {
    allocator_->emptyCache(mempool_id);
  }
  void recordStream(const DataPtr& ptr, c10::Stream stream) {
    allocator_->recordStream(ptr, stream);
  }
  CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) {
    return allocator_->getDeviceStats(device);
  }
  void resetAccumulatedStats(c10::DeviceIndex device) {
    allocator_->resetAccumulatedStats(device);
  }
  void resetPeakStats(c10::DeviceIndex device) {
    allocator_->resetPeakStats(device);
  }
};

} // namespace c10::hip
