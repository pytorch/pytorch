#include <ATen/core/CachingHostAllocator.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

#include <include/openreg.h>

#include "OpenRegGenerator.h"

namespace c10::openreg {
struct OpenRegHooksInterface : public at::PrivateUse1HooksInterface {
  OpenRegHooksInterface() {};
  ~OpenRegHooksInterface() override = default;

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    return true;
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return at::getHostAllocator(at::kPrivateUse1);
  }

  bool isPinnedPtr(const void* data) const override {
    orPointerAttributes attr{};
    orPointerGetAttributes(&attr, data);

    return attr.type == orMemoryTypeHost;
  }

  const at::Generator& getDefaultGenerator(
      c10::DeviceIndex device_index) const override {
    return getDefaultOpenRegGenerator(device_index);
  }

  at::Generator getNewGenerator(c10::DeviceIndex device_index) const override {
    return at::make_generator<OpenRegGeneratorImpl>(device_index);
  }
};

} // namespace c10::openreg
