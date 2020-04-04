#include <c10/core/MobileAllocator.h>
#include <c10/core/CPUAllocator.h>

namespace c10 {
namespace {

// QNNPACK AND XNNPACK may out-of-bound access the input and / or output tensors.
// This behavior will trigger ASAN, and may result in a segfault if the accessed
// memory just so happens to fall on a page the current process has no read access
// to.  Here we define a custom allocator that allocates the extra storage required
// to keep this behavior safe.
//
// PreGuardBytes: Number of guard bytes to allocate before the allocation.
// PostGuardBytes: Number of guard bytes to allocate after the allocation.

template <uint32_t PreGuardBytes, uint32_t PostGuardBytes>
class DefaultMobileCPUAllocator final : public at::Allocator {
 public:
  DefaultMobileCPUAllocator() = default;
  virtual ~DefaultMobileCPUAllocator() override = default;

  static void deleter(void* const pointer) {
    const Cast memory{pointer};
    c10::free_cpu(memory.as_byte_ptr - PreGuardBytes);
  }

  virtual DataPtr allocate(const size_t nbytes) const override {
    Cast memory{c10::alloc_cpu(PreGuardBytes + nbytes + PostGuardBytes)};
    memory.as_byte_ptr += PreGuardBytes;

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
  union Cast final {
    void * const as_void_ptr;
    uint8_t * as_byte_ptr;
  };
};

DefaultMobileCPUAllocator<8u, 16u> g_mobile_cpu_allocator;
REGISTER_ALLOCATOR(DeviceType::CPU, &g_mobile_cpu_allocator);

} // namespace

at::Allocator* GetDefaultMobileCPUAllocator() {
  return &g_mobile_cpu_allocator;
}

#ifdef C10_MOBILE

at::Allocator* GetDefaultCPUAllocator() {
  return GetDefaultMobileCPUAllocator();
}

#endif /* C10_Mobile */

} // namespace c10
