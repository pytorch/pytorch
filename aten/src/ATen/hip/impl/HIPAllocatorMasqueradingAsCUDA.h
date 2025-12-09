#pragma once

#include <c10/hip/HIPCachingAllocator.h>

// Use of c10::hip namespace here makes hipification easier, because
// I don't have to also fix namespaces.  Sorry!
namespace c10::hip {

// Takes a valid HIPAllocator (of any sort) and turns it into
// an allocator pretending to be a CUDA allocator.  See
// Note [Masquerading as CUDA]
class HIPAllocatorMasqueradingAsCUDA final : public HIPCachingAllocator::HIPAllocator {
  HIPCachingAllocator::HIPAllocator* allocator_;
public:
  explicit HIPAllocatorMasqueradingAsCUDA(HIPCachingAllocator::HIPAllocator* allocator)
    : allocator_(allocator) {}

  virtual ~HIPAllocatorMasqueradingAsCUDA() = default;

  // From c10::Allocator

  DataPtr allocate(size_t size) override {
    DataPtr r = allocator_->allocate(size);
    r.unsafe_set_device(Device(c10::DeviceType::CUDA, r.device().index()));
    return r;
  }

  bool is_simple_data_ptr(const DataPtr& data_ptr) const override {
    return allocator_->is_simple_data_ptr(data_ptr);
  }

  DeleterFnPtr raw_deleter() const override {
    return allocator_->raw_deleter();
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    allocator_->copy_data(dest, src, count);
  }

  // From DeviceAllocator

  bool initialized() override {
    return allocator_->initialized();
  }

  void emptyCache(MempoolId_t mempool_id = {0, 0}) override {
    allocator_->emptyCache(mempool_id);
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    HIPStream hip_stream = HIPStream(stream);
    recordStream(ptr, hip_stream);
  }

  CachingDeviceAllocator::DeviceStats getDeviceStats(c10::DeviceIndex device) override {
    return allocator_->getDeviceStats(device);
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    allocator_->resetAccumulatedStats(device);
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    allocator_->resetPeakStats(device);
  }

  // From CUDAAllocator

  void* raw_alloc(size_t nbytes) override {
    return allocator_->raw_alloc(nbytes);
  }

  void* raw_alloc_with_stream(size_t nbytes, hipStream_t stream) override {
    return allocator_->raw_alloc_with_stream(nbytes, stream);
  }

  void raw_delete(void* ptr) override {
    allocator_->raw_delete(ptr);
  }

  void init(int device_count) override {
    allocator_->init(device_count);
  }

  double getMemoryFraction(c10::DeviceIndex device) override {
    return allocator_->getMemoryFraction(device);
  }

  void setMemoryFraction(double fraction, c10::DeviceIndex device) override {
    allocator_->setMemoryFraction(fraction, device);
  }

  std::vector<HIPCachingAllocator::StreamSegmentSize> getExpandableSegmentSizes(c10::DeviceIndex device) override {
    return allocator_->getExpandableSegmentSizes(device);
  }

  void enable(bool value) override {
    allocator_->enable(value);
  }

  bool isEnabled() const override {
    return allocator_->isEnabled();
  }

  void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) override {
    allocator_->cacheInfo(device, largestBlock);
  }

  void* getBaseAllocation(void* ptr, size_t* size) override {
    return allocator_->getBaseAllocation(ptr, size);
  }

  void recordStream(const DataPtr& ptr, HIPStream stream) override {
    allocator_->recordStream(ptr, stream);
  }

  HIPCachingAllocator::SnapshotInfo snapshot(MempoolId_t mempool_id = {0, 0}) override {
    return allocator_->snapshot(mempool_id);
  }

  void beginAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      std::function<bool(hipStream_t)> filter) override {
    allocator_->beginAllocateToPool(device, mempool_id, filter);
  }

  void endAllocateToPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id) override {
    allocator_->endAllocateToPool(device, mempool_id);
  }

  void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    allocator_->releasePool(device, mempool_id);
  }

  int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    return allocator_->getPoolUseCount(device, mempool_id);
  }

  void createOrIncrefPool(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      HIPAllocator* allocator = nullptr) override {
    allocator_->createOrIncrefPool(device, mempool_id, allocator);
  }

  void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    allocator_->setUseOnOOM(device, mempool_id);
  }

  void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) override {
    allocator_->setNoSplit(device, mempool_id);
  }

  bool checkPoolLiveAllocations(
      c10::DeviceIndex device,
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) override {
    return allocator_->checkPoolLiveAllocations(device, mempool_id, expected_live_allocations);
  }

  HIPCachingAllocator::ShareableHandle shareIpcHandle(void* ptr) override {
    return allocator_->shareIpcHandle(ptr);
  }

  std::shared_ptr<void> getIpcDevPtr(std::string handle) override {
    return allocator_->getIpcDevPtr(handle);
  }

  bool isHistoryEnabled() override {
    return allocator_->isHistoryEnabled();
  }

  void recordHistory(
      bool enabled,
      HIPCachingAllocator::CreateContextFn context_recorder,
      size_t alloc_trace_max_entries,
      HIPCachingAllocator::RecordContext when,
      bool clearHistory) override {
    allocator_->recordHistory(enabled, context_recorder, alloc_trace_max_entries, when, clearHistory);
  }

  void recordAnnotation(
      const std::vector<std::pair<std::string, std::string>>& md) override {
    allocator_->recordAnnotation(md);
  }

  void pushCompileContext(std::string& md) override {
    allocator_->pushCompileContext(md);
  }

  void popCompileContext() override {
    allocator_->popCompileContext();
  }

  void attachOutOfMemoryObserver(HIPCachingAllocator::OutOfMemoryObserver observer) override {
    allocator_->attachOutOfMemoryObserver(observer);
  }

  void attachAllocatorTraceTracker(HIPCachingAllocator::AllocatorTraceTracker tracker) override {
    allocator_->attachAllocatorTraceTracker(tracker);
  }

  void enablePeerAccess(c10::DeviceIndex dev, c10::DeviceIndex dev_to_access) override {
    allocator_->enablePeerAccess(dev, dev_to_access);
  }

  hipError_t memcpyAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      hipStream_t stream,
      bool p2p_enabled) override {
    return allocator_->memcpyAsync(dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
  }

  std::shared_ptr<HIPCachingAllocator::AllocatorState> getCheckpointState(
      c10::DeviceIndex device,
      MempoolId_t id) override {
    return allocator_->getCheckpointState(device, id);
  }

  HIPCachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex device,
      std::shared_ptr<HIPCachingAllocator::AllocatorState> pps) override {
    auto cpd = allocator_->setCheckpointPoolState(device, pps);
    for (auto& ptr : cpd.dataptrs_allocd) {
      ptr.unsafe_set_device(Device(c10::DeviceType::CUDA, ptr.device().index()));
    }
    return cpd;
  }

  std::string name() override {
    return allocator_->name();
  }

};

} // namespace c10::hip
