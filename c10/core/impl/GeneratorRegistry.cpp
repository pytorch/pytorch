#include <c10/core/Device.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/impl/GeneratorRegistry.h>

namespace c10::impl {

GeneratorRegistry& GeneratorRegistry::instance() {
  static GeneratorRegistry g;
  return g;
}

void GeneratorRegistry::register_factory(c10::DeviceType device_type, GeneratorFactory factory) {
  std::lock_guard<std::mutex> lock(mu_);
  factories_[device_type] = std::move(factory);
}

c10::intrusive_ptr<c10::GeneratorImpl> GeneratorRegistry::make(c10::Device device) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = factories_.find(device.type());
  TORCH_CHECK(it != factories_.end(), "No registered generator factory for device type ", device.type());
  return it->second(device);
}

bool GeneratorRegistry::has(c10::DeviceType device_type) const {
  std::lock_guard<std::mutex> lock(mu_);
  return factories_.count(device_type) != 0;
}

void registerGenerator(c10::DeviceType device_type, GeneratorFactory factory) {
  GeneratorRegistry::instance().register_factory(device_type, std::move(factory));
}

c10::intrusive_ptr<c10::GeneratorImpl> makeGenerator(c10::Device device) {
  return GeneratorRegistry::instance().make(device);
}

bool hasGenerator(c10::DeviceType device_type) {
  return GeneratorRegistry::instance().has(device_type);
}

} // namespace c10::impl