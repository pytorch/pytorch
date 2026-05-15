#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/GeneratorForPrivateuseone.h>

#include <c10/core/Device.h>

#include "OpenRegFunctions.h"

namespace c10::openreg {
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

} // namespace c10::openreg
