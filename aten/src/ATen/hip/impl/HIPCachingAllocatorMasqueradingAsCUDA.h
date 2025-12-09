#pragma once

#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/impl/HIPAllocatorMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace c10 {
// forward declaration
class DataPtr;
namespace hip {
namespace HIPCachingAllocatorMasqueradingAsCUDA {

C10_HIP_API HIPCachingAllocator::HIPAllocator* get();
C10_HIP_API void recordStreamMasqueradingAsCUDA(const DataPtr& ptr, HIPStreamMasqueradingAsCUDA stream);

inline void* raw_alloc(size_t nbytes) {
  return get()->raw_alloc(nbytes);
}

inline void* raw_alloc_with_stream(size_t nbytes, hipStream_t stream) {
  return get()->raw_alloc_with_stream(nbytes, stream);
}

inline void raw_delete(void* ptr) {
  return get()->raw_delete(ptr);
}

inline void init(int device_count) {
  return get()->init(device_count);
}

inline double getMemoryFraction(c10::DeviceIndex device) {
  return get()->getMemoryFraction(device);
}

inline void setMemoryFraction(double fraction, c10::DeviceIndex device) {
  return get()->setMemoryFraction(fraction, device);
}

inline void emptyCache(MempoolId_t mempool_id = {0, 0}) {
  return get()->emptyCache(mempool_id);
}

inline void enable(bool value) {
  return get()->enable(value);
}

inline bool isEnabled() {
  return get()->isEnabled();
}

inline void cacheInfo(c10::DeviceIndex device, size_t* largestBlock) {
  return get()->cacheInfo(device, largestBlock);
}

inline void* getBaseAllocation(void* ptr, size_t* size) {
  return get()->getBaseAllocation(ptr, size);
}

inline c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
    c10::DeviceIndex device) {
  return get()->getDeviceStats(device);
}

inline void resetAccumulatedStats(c10::DeviceIndex device) {
  return get()->resetAccumulatedStats(device);
}

inline void resetPeakStats(c10::DeviceIndex device) {
  return get()->resetPeakStats(device);
}

inline HIPCachingAllocator::SnapshotInfo snapshot(MempoolId_t mempool_id = {0, 0}) {
  return get()->snapshot(mempool_id);
}

inline std::shared_ptr<HIPCachingAllocator::AllocatorState> getCheckpointState(
    c10::DeviceIndex device,
    MempoolId_t id) {
  return get()->getCheckpointState(device, id);
}

inline HIPCachingAllocator::CheckpointDelta setCheckpointPoolState(
    c10::DeviceIndex device,
    std::shared_ptr<HIPCachingAllocator::AllocatorState> pps) {
  return get()->setCheckpointPoolState(device, std::move(pps));
}

inline void beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(hipStream_t)> filter) {
  get()->beginAllocateToPool(device, mempool_id, std::move(filter));
}

inline void endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->endAllocateToPool(device, mempool_id);
}

inline void recordHistory(
    bool enabled,
    HIPCachingAllocator::CreateContextFn context_recorder,
    size_t alloc_trace_max_entries,
    HIPCachingAllocator::RecordContext when,
    bool clearHistory) {
  return get()->recordHistory(
      enabled, context_recorder, alloc_trace_max_entries, when, clearHistory);
}

inline void recordAnnotation(
    const std::vector<std::pair<std::string, std::string>>& md) {
  return get()->recordAnnotation(md);
}

inline void pushCompileContext(std::string& md) {
  return get()->pushCompileContext(md);
}

inline void popCompileContext() {
  return get()->popCompileContext();
}

inline bool isHistoryEnabled() {
  return get()->isHistoryEnabled();
}

inline bool checkPoolLiveAllocations(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    const std::unordered_set<void*>& expected_live_allocations) {
  return get()->checkPoolLiveAllocations(
      device, mempool_id, expected_live_allocations);
}

inline void attachOutOfMemoryObserver(HIPCachingAllocator::OutOfMemoryObserver observer) {
  return get()->attachOutOfMemoryObserver(std::move(observer));
}

inline void attachAllocatorTraceTracker(HIPCachingAllocator::AllocatorTraceTracker tracker) {
  return get()->attachAllocatorTraceTracker(std::move(tracker));
}

inline void releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->releasePool(device, mempool_id);
}

inline void createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    HIPCachingAllocator::HIPAllocator* allocator_ptr = nullptr) {
  get()->createOrIncrefPool(device, mempool_id, allocator_ptr);
}

inline void setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->setUseOnOOM(device, mempool_id);
}

inline void setNoSplit(c10::DeviceIndex device, MempoolId_t mempool_id) {
  get()->setNoSplit(device, mempool_id);
}

inline int getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return get()->getPoolUseCount(device, mempool_id);
}

inline std::shared_ptr<void> getIpcDevPtr(std::string handle) {
  return get()->getIpcDevPtr(std::move(handle));
}

inline HIPCachingAllocator::ShareableHandle shareIpcHandle(void* ptr) {
  return get()->shareIpcHandle(ptr);
}

inline std::string name() {
  return get()->name();
}

inline hipError_t memcpyAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    hipStream_t stream,
    bool p2p_enabled) {
  return get()->memcpyAsync(
      dst, dstDevice, src, srcDevice, count, stream, p2p_enabled);
}

inline void enablePeerAccess(
    c10::DeviceIndex dev,
    c10::DeviceIndex dev_to_access) {
  return get()->enablePeerAccess(dev, dev_to_access);
}

} // namespace HIPCachingAllocatorMasqueradingAsCUDA
} // namespace hip
} // namespace c10
