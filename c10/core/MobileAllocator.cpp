#include <c10/core/MobileAllocator.h>
#include <c10/core/CPUAllocator.h>

namespace c10 {
namespace {

struct Configuration final {
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
  static constexpr uint32_t kAlignment = 16u;
#else
  static constexpr uint32_t kAlignment = gAlignment;
#endif
};

// QNNPACK AND XNNPACK may out-of-bound access the input and / or output tensors.
// This behavior will trigger ASAN, and may result in a segfault if the accessed
// memory location just so happens to fall on a page the current process has no
// read access to.  Here we define a custom allocator that allocates the extra
// storage required to keep this behavior safe.  This allocator could have been
// restricted to QNNPACK and XNNPACK only, but that would have negative
// performance ramifications, as input tensors must be reallocated, and copied
// over, if not allocated with this allocator to begin with.  Making this
// allocator the default on mobile builds minimizes the probability of unnecessary
// reallocations and copies.
//
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
    Cast memory{
      c10::alloc_cpu(
        PreGuardBytes + nbytes + PostGuardBytes,
        Configuration::kAlignment),
    };

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

DefaultMobileCPUAllocator<std::max(8u, Configuration::kAlignment), 16u> g_mobile_cpu_allocator;

} // namespace

// The CPU Mobile allocator must always be present even on non-mobile builds
// because QNNPACK and XNNPACK are not mobile specific.

at::Allocator* GetDefaultMobileCPUAllocator() {
  return &g_mobile_cpu_allocator;
}

// #ifdef C10_MOBILE

// Having said that, only register the mobile CPU allocator as the default CPU
// memory allocator on mobile builds.

at::Allocator* GetDefaultCPUAllocator() {
  return GetDefaultMobileCPUAllocator();
}

REGISTER_ALLOCATOR(DeviceType::CPU, &g_mobile_cpu_allocator);

// #endif /* C10_Mobile */

} // namespace c10
