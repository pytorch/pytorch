#pragma once

#include <c10/core/Allocator.h>
#include <torch/csrc/jit/passes/memory_planning.h>

#include <utility>

struct TORCH_API MemEvent {
  enum class EventType { Allocate = 0, Free };

  uint64_t time;
  std::string allocation_trace;
  std::string ptr_addr;
  uint64_t size;
  EventType type;
  MemEvent(
      uint64_t t,
      std::string alloc_trace,
      std::string address,
      uint64_t s,
      EventType e)
      : time(t),
        allocation_trace(std::move(alloc_trace)),
        ptr_addr(std::move(address)),
        size(s),
        type(e) {}

};

struct C10_API MemoryPlanningAllocator final : at::Allocator {
  MemoryPlanningAllocator(at::DeviceType& device_type);
  at::DataPtr allocate(size_t nbytes) const override;
  at::DeleterFnPtr raw_deleter() const override;
  void push_allocation(
      c10::Storage buffer,
      size_t size,
      size_t offset,
      at::DeviceType device);

 private:
  at::DeviceType& device_type_;
  c10::Allocator& orig_allocator_;
  mutable std::stack<std::pair<size_t, void*>> allocs_;
};

struct MemoryTracingAllocator;

class C10_API WithProfileAllocationsGuard {
 public:
  WithProfileAllocationsGuard(at::DeviceType& device_type);
  std::vector<MemEvent> getAllocationTraces();
  ~WithProfileAllocationsGuard();

 private:
  std::shared_ptr<MemoryTracingAllocator> tracer_;
  at::DeviceType& device_type_;
};
