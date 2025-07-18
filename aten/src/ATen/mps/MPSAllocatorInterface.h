//  Copyright © 2023 Apple Inc.

#pragma once

#include <ATen/core/ATen_fwd.h>
#include <c10/core/Allocator.h>
#include <c10/core/IMPSAllocator.h>
#include <c10/util/Registry.h>

#define MB(x) (x * 1048576UL)

namespace at::mps {

using IMPSAllocator = c10::IMPSAllocator;

class IMpsAllocatorCallback {
 public:
  enum class EventType {
    ALLOCATED, // buffer got allocated to be used immediately
    RECYCLED, // buffer pulled from free list to be reused
    FREED, // buffer put to free list for future recycling
    RELEASED, // buffer memory released
    ALLOCATION_FAILED // buffer allocation failed
  };
  virtual ~IMpsAllocatorCallback() = default;
  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// MPS allocator will execute every registered callback when a block of memory
// is freed.
TORCH_DECLARE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);
#define REGISTER_MPS_ALLOCATOR_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(MPSAllocatorCallbacksRegistry, name, __VA_ARGS__)

IMPSAllocator* getIMPSAllocator(
    bool sharedAllocator = true,
    bool pinnedAllocator = false);

at::Allocator* getPinnedMemoryAllocator();

bool isMPSPinnedPtr(const void* data);

} // namespace at::mps
