#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include <ATen/detail/PrivateUse1HooksInterface.h>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

class OpenRegGeneratorImpl : public at::CPUGeneratorImpl {
 public:
  OpenRegGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~OpenRegGeneratorImpl() override = default;
};

const at::Generator& getDefaultOpenRegGenerator(
    c10::DeviceIndex device_index = -1);
