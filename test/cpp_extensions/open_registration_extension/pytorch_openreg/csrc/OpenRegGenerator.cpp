#include "ATen/CPUGeneratorImpl.h"
#include "ATen/core/GeneratorForPrivateuseone.h"
#include "OpenReg.h"

namespace openreg {
class OpenRegGeneratorImpl : public at::CPUGeneratorImpl {
 public:
  // Constructors
  OpenRegGeneratorImpl(c10::DeviceIndex device_index) {
    device_ = c10::Device(c10::DeviceType::PrivateUse1, device_index);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
  ~OpenRegGeneratorImpl() override = default;
};

// this is used to register generator
at::Generator make_openreg_generator(c10::DeviceIndex device_index) {
  return at::make_generator<OpenRegGeneratorImpl>(device_index);
}
REGISTER_GENERATOR_PRIVATEUSE1(make_openreg_generator)

void register_generator() {
  // no-op, just to trigger REGISTER_GENERATOR_PRIVATEUSE1
}
} // namespace openreg