#include <torch/csrc/jit/passes/memory_planning/MemoryPlanningAllocator.h>
#include <torch/csrc/jit/passes/memory_planning/memory_observer.h>

#include <sstream>

#include <c10/util/Backtrace.h>
#include <torch/csrc/jit/mobile/interpreter.h>

namespace c10 {

static void DoNothing(void* ptr) {
  return;
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

using namespace torch::jit;

at::DataPtr MemoryTracingAllocator::allocate(size_t nbytes) const {
  auto orig_ptr = orig_allocator_.raw_allocate(nbytes);
  auto bt = c10::get_backtrace(0, 200, true);
  auto frame_node_id = torch::jit::currentFrameId();
  auto time = torch::jit::timeSinceEpoch();
  allocation_traces_.emplace_back(MemoryEvent{
      time,
      time,
      bt,
      reinterpret_cast<intptr_t>(orig_ptr),
      (int64_t)nbytes,
      MemoryEvent::EventType::ALLOCATE,
      frame_node_id});
  allocations_.insert({orig_ptr, nbytes});

  auto deleter = [this, nbytes = nbytes](void* ptr) {
    auto bt = c10::get_backtrace(0, 200, true);
    auto frame_node_id = torch::jit::currentFrameId();
    auto time = torch::jit::timeSinceEpoch();
    allocation_traces_.emplace_back(MemoryEvent{
      time,
      time,
      bt,
      reinterpret_cast<intptr_t>(ptr),
      (int64_t)nbytes,
      MemoryEvent::EventType::FREE,
      frame_node_id});
    orig_allocator_.raw_deallocate(ptr);
  };

  return c10::InefficientStdFunctionContext::makeDataPtr(
      orig_ptr, deleter, at::Device(device_type_));
}

WithProfileTracingAllocationsGuard::WithProfileTracingAllocationsGuard(
    at::DeviceType device_type)
    : tracer_(MemoryTracingAllocator{device_type}), device_type_(device_type) {}

std::vector<torch::jit::MemoryEvent> WithProfileTracingAllocationsGuard::
    getAllocationTraces() {
  return tracer_.allocation_traces_;
}

} // namespace c10