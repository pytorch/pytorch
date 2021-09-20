#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/Storage.h>

#include <torch/csrc/jit/passes/memory_planning/memory_observer.h>

#include <stack>

namespace c10 {

struct C10_API MemoryPlanningAllocator final : at::Allocator {
  MemoryPlanningAllocator(at::DeviceType device_type);
  at::DataPtr allocate(size_t nbytes) const override;
  at::DeleterFnPtr raw_deleter() const override;
  void push_allocation(
      c10::Storage buffer,
      size_t size,
      size_t offset,
      at::DeviceType device);

 private:
  uint8_t allocator_priority_;
  at::DeviceType device_type_;
  c10::Allocator& orig_allocator_;
  mutable std::stack<std::pair<size_t, void*>> allocs_;
};

class C10_API WithProfileTracingAllocationsGuard;

struct C10_API MemorizingAllocator final : at::Allocator {
  MemorizingAllocator(at::DeviceType device_type)
      : allocator_priority_(c10::GetAllocatorPriority(device_type)),
        device_type_(device_type),
        orig_allocator_(*c10::GetAllocator(device_type)) {
    c10::SetAllocator(device_type, this, allocator_priority_);
  }

  at::DataPtr allocate(size_t nbytes) const override;

  uint8_t allocator_priority_;
  at::DeviceType device_type_;
  c10::Allocator& orig_allocator_;
  mutable std::vector<torch::jit::MemoryEvent> allocation_traces_;
  mutable std::map<void*, size_t> allocations_;
  friend WithProfileTracingAllocationsGuard;

 private:
};

class C10_API WithProfileTracingAllocationsGuard {
 public:
  WithProfileTracingAllocationsGuard(at::DeviceType device_type);
  std::vector<torch::jit::MemoryEvent> getAllocationTraces();
  ~WithProfileTracingAllocationsGuard() {
    c10::SetAllocator(
        device_type_, &tracer_.orig_allocator_, tracer_.allocator_priority_);
  }

 private:
  MemorizingAllocator tracer_;
  at::DeviceType device_type_;
};
} // namespace c10