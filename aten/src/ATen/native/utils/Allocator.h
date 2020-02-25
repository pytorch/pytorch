#pragma once

#include <c10/core/CPUAllocator.h>

namespace at {
namespace native {

// QNNPACK AND XNNPACK may out-of-bound access the input and / or output tensors.
// This behavior will trigger ASAN, and may result in a segfault if the accessed
// memory just so happens to fall on a page the current process has no read access
// to.  Here we define a custom allocator that allocates the extra storage required
// to keep this behavior safe.
//
// PreGuardBytes: Number of guard bytes to allocate before the allocation.
// PostGuardBytes: Number of guard bytes to allocate after the allocation.

template <uint32_t PreGuardBytes, uint32_t PostGuardBytes>
class GuardingAllocator final : public at::Allocator {
 public:
  GuardingAllocator() = default;
  virtual ~GuardingAllocator() override = default;

  static void deleter(void* pointer) {
    const Cast memory{pointer};
    c10::free_cpu(memory.as_byte_ptr - kPreGuardBytes);
  }

  virtual DataPtr allocate(size_t nbytes) const override {
    Cast memory{c10::alloc_cpu(kPreGuardBytes + nbytes + kPostGuardBytes)};
    memory.as_byte_ptr += kPreGuardBytes;

    return {
      memory.as_void_ptr,
      memory.as_void_ptr,
      &deleter,
      at::Device(DeviceType::CPU),
    };
  }

  virtual DeleterFnPtr raw_deleter() const override {
    return deleter;
  }

 private:
  static constexpr uint32_t kPreGuardBytes = PreGuardBytes;
  static constexpr uint32_t kPostGuardBytes = PostGuardBytes;

  union Cast final {
    void * const as_void_ptr;
    uint8_t * as_byte_ptr;
  };
};

} // namespace native
} // namespace at
