#include <torch/csrc/jit/passes/memory_planning.h>

#include <sstream> //for std::stringstream

#include <c10/util/Backtrace.h>
#include <torch/csrc/jit/mobile/interpreter.h>

namespace c10 {

static void DoNothing(void* ptr) {
  return;
}

inline int64_t timeSinceEpoch(
    const std::chrono::time_point<std::chrono::system_clock>& t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             t.time_since_epoch())
      .count();
}

MemoryPlanningAllocator::MemoryPlanningAllocator(at::DeviceType device_type)
    : allocator_priority_(c10::GetAllocatorPriority(device_type)),
      device_type_(device_type),
      orig_allocator_(*c10::GetAllocator(device_type)) {
  c10::SetAllocator(device_type, this, allocator_priority_);
};

at::DataPtr MemoryPlanningAllocator::allocate(size_t nbytes) const {
  auto alloc = allocs_.top();
  allocs_.pop();
  auto size = alloc.first;
  TORCH_CHECK(size == nbytes);
  auto data = alloc.second;
  return {data, data, &DoNothing, at::Device(device_type_)};
}

at::DeleterFnPtr MemoryPlanningAllocator::raw_deleter() const {
  return Allocator::raw_deleter();
}

void MemoryPlanningAllocator::push_allocation(
    c10::Storage buffer,
    size_t size,
    size_t offset,
    at::DeviceType device_type) {
  TORCH_CHECK(device_type == device_type_);
  uint8_t* start = static_cast<uint8_t*>(buffer.data());
  void* src = static_cast<void*>(start + offset);
  allocs_.push(std::make_pair(size, src));
}

at::DeleterFnPtr raw_deleter() {
  return &DoNothing;
}

std::string dataPtrAddrToStr(void* ptr) {
  std::stringstream ss;
  ss << ptr;
  return ss.str();
}

at::DataPtr MemoryTracingAllocator::allocate(size_t nbytes) const {
  auto orig_ptr = orig_allocator_.raw_allocate(nbytes);
  auto bt = c10::get_backtrace(0, 200, true);
  auto frame_node_id = torch::jit::currentFrameId();
  if (frame_node_id.has_value()) {
    allocation_traces_.emplace_back(torch::jit::MemEvent{
        frame_node_id.value().pc,
        bt,
        dataPtrAddrToStr(orig_ptr),
        nbytes,
        torch::jit::MemEvent::EventType::Allocate,
        frame_node_id});
  } else {
    allocation_traces_.emplace_back(torch::jit::MemEvent{
        0,
        bt,
        dataPtrAddrToStr(orig_ptr),
        nbytes,
        torch::jit::MemEvent::EventType::Allocate,
        c10::nullopt});
  }
  allocations_.insert({orig_ptr, nbytes});

  auto deleter = [this, nbytes = nbytes](void* ptr) {
    auto bt = c10::get_backtrace(0, 200, true);
    auto frame_node_id = torch::jit::currentFrameId();
    if (frame_node_id.has_value()) {
      allocation_traces_.emplace_back(torch::jit::MemEvent{
          frame_node_id.value().pc,
          bt,
          dataPtrAddrToStr(ptr),
          nbytes,
          torch::jit::MemEvent::EventType::Free,
          frame_node_id});
    } else {
      allocation_traces_.emplace_back(torch::jit::MemEvent{
          std::numeric_limits<uint64_t>::max(),
          bt,
          dataPtrAddrToStr(ptr),
          nbytes,
          torch::jit::MemEvent::EventType::Free,
          c10::nullopt});
    }
    orig_allocator_.raw_deallocate(ptr);
  };

  return c10::InefficientStdFunctionContext::makeDataPtr(
      orig_ptr, deleter, at::Device(device_type_));
}

WithProfileTracingAllocationsGuard::WithProfileTracingAllocationsGuard(
    at::DeviceType device_type)
    : tracer_(MemoryTracingAllocator{device_type}), device_type_(device_type) {}

std::vector<torch::jit::MemEvent> WithProfileTracingAllocationsGuard::
    getAllocationTraces() {
  return tracer_.allocation_traces_;
}

} // namespace c10