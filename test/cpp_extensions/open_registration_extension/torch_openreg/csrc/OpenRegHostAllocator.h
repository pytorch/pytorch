#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include "../backend/include/openreg.h"

struct OpenRegHostAllocator final : at::HostAllocator {
  OpenRegHostAllocator() = default;

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    orFreeHost(ptr);
  }

  at::DataPtr allocate(size_t nbytes) override {
    void* data = nullptr;
    if (nbytes > 0) {
      orMallocHost(&data, nbytes);
      TORCH_CHECK(data, "Failed to allocator ", nbytes, " bytes on host.");
    }
    return {data, data, &ReportAndDelete, at::Device(at::kCPU)};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    orMemcpy(dest, src, count, orMemcpyHostToHost);
  }

  // ignore
  bool record_event(void* ptr, void* ctx, c10::Stream stream) override {
    return true;
  }
  void empty_cache() override {}
  at::HostStats get_stats() override {
    return at::HostStats();
  }
  void reset_accumulated_stats() override {}
  void reset_peak_stats() override {}
};
