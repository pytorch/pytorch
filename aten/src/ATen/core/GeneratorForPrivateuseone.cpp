#include <mutex>
#include <ATen/core/GeneratorForPrivateuseone.h>

namespace at {

static std::mutex _generator_mutex_lock;

std::optional<GeneratorFuncType>& GetGeneratorPrivate() {
  static std::optional<GeneratorFuncType> generator_privateuse1 = c10::nullopt;
  return generator_privateuse1;
}

_GeneratorRegister::_GeneratorRegister(const GeneratorFuncType& func) {
  std::lock_guard<std::mutex> lock(_generator_mutex_lock);
  TORCH_CHECK(
      !GetGeneratorPrivate().has_value(),
      "Only can register a generator to the PrivateUse1 dispatch key once!");

  auto& m_generator = GetGeneratorPrivate();
  m_generator = func;
}

at::Generator GetGeneratorForPrivateuse1(c10::DeviceIndex device_index) {
  TORCH_CHECK(
      GetGeneratorPrivate().has_value(),
      "Please register a generator to the PrivateUse1 dispatch key, \
      using the REGISTER_GENERATOR_PRIVATEUSE1 macro.");

  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return GetGeneratorPrivate().value()(device_index);
}

} // namespace at
