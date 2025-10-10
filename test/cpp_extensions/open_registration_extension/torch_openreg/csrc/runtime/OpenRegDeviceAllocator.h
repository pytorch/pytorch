#include <ATen/core/CachingHostAllocator.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <include/openreg.h>

namespace c10::openreg {
struct OpenRegDeviceAllocator final : at::Allocator {
  OpenRegDeviceAllocator() = default;

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    orFreeHost(ptr);
  }

  at::DataPtr allocate(size_t nbytes) override {
    int current_device_index = -1;
    orGetDevice(&current_device_index);

    auto curr_device =
        c10::Device(c10::DeviceType::PrivateUse1, current_device_index);
    void* data = nullptr;
    if (nbytes > 0) {
      orMalloc(&data, nbytes);
      TORCH_CHECK(
          data, "Failed to allocator ", nbytes, " bytes on openreg device.");
    }
    return {data, data, &ReportAndDelete, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    orMemcpy(dest, src, count, orMemcpyDeviceToDevice);
  }
};

} // namespace c10::openreg
