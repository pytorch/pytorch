#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/xccl2/XCCLCachingAllocatorHook.hpp>

#include <algorithm>
#include <vector>

#include <ATen/Context.h>
#include <c10/core/AllocatorConfig.h>
#include <torch/csrc/distributed/c10d/xccl2/Logging.hpp>
#include <torch/csrc/distributed/c10d/xccl2/ProcessGroupXCCL.hpp>

namespace c10d::xccl2 {

namespace {
void xcclCachingAllocatorHookFn(
    const c10::CachingDeviceAllocator::TraceEntry& te) {
  XCCLCachingAllocatorHook::getInstance().regDeregMem(te);
}

// registeredComms_ is ordered by pointer, which differs between ranks. Order
// by group name instead so every rank walks the comms in the same order.
std::vector<ProcessGroupXCCL*> commsForDevice(
    const std::set<ProcessGroupXCCL*>& comms,
    c10::DeviceIndex device) {
  std::vector<ProcessGroupXCCL*> out;
  for (auto* comm : comms) {
    if (device == comm->getDevice().index()) {
      out.push_back(comm);
    }
  }
  std::stable_sort(out.begin(), out.end(), [](auto* a, auto* b) {
    return a->getCommName() < b->getCommName();
  });
  return out;
}
} // namespace

XCCLCachingAllocatorHook& XCCLCachingAllocatorHook::getInstance() {
  // Leaked singleton: allocator trace trackers cannot be detached, so the
  // hook must outlive every allocator event.
  static auto* instance = new XCCLCachingAllocatorHook();
  return *instance;
}

XCCLCachingAllocatorHook::XCCLCachingAllocatorHook()
    : register_default_pool_segments_(
          !c10::CachingAllocator::AcceleratorAllocatorConfig::
              use_expandable_segments()) {
  at::globalContext().lazyInitDevice(c10::DeviceType::XPU);
  if (!register_default_pool_segments_) {
    TC_LOG(INFO)
        << "Disabling default-pool XCCL memory registration because it is "
           "incompatible with XPU allocator expandable segments mode.";
  }
  registerMemPreHook();
  c10::xpu::XPUCachingAllocator::attachAllocatorTraceTracker(
      &xcclCachingAllocatorHookFn);
}

bool XCCLCachingAllocatorHook::shouldTrackSegment(
    const c10::MempoolId_t& mempool_id) const {
  return register_default_pool_segments_ ||
      mempool_id != c10::MempoolId_t{0, 0};
}

void XCCLCachingAllocatorHook::registerMemPreHook() {
  auto snapshot = c10::xpu::XPUCachingAllocator::snapshot();
  for (const auto& segmentInfo : snapshot.segments) {
    if (!shouldTrackSegment(segmentInfo.owner_private_pool_id)) {
      continue;
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    registeredMemMap_.emplace(
        addr, MemInfo{segmentInfo.total_size, segmentInfo.device});
  }
}

void XCCLCachingAllocatorHook::regDeregMem(
    const c10::CachingDeviceAllocator::TraceEntry& te) {
  if (!shouldTrackSegment(te.mempool_)) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (te.action_ ==
      c10::CachingDeviceAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;
    TORCH_CHECK(
        !registeredMemMap_.count(addr), "Memory already registered with XCCL");
    registeredMemMap_.emplace(addr, MemInfo{len, te.device_});
    for (auto* comm : commsForDevice(registeredComms_, te.device_)) {
      comm->register_address(addr, len);
    }
  } else if (
      te.action_ ==
      c10::CachingDeviceAllocator::TraceEntry::Action::SEGMENT_FREE) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    TORCH_CHECK(
        registeredMemMap_.count(addr), "Memory not registered with XCCL");
    registeredMemMap_.erase(addr);
    for (auto* comm : commsForDevice(registeredComms_, te.device_)) {
      comm->deregister_address(addr);
    }
  }
}

void XCCLCachingAllocatorHook::registerComm(ProcessGroupXCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  TORCH_CHECK(!registeredComms_.count(comm), "Communicator already registered");
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->register_address(addr, mem_info.len);
    }
  }
  registeredComms_.insert(comm);
}

void XCCLCachingAllocatorHook::deregisterComm(ProcessGroupXCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!registeredComms_.count(comm)) {
    return;
  }
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->deregister_address(addr);
    }
  }
  registeredComms_.erase(comm);
}

} // namespace c10d::xccl2

#endif // USE_C10D_XCCL
