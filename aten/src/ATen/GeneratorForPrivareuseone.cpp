#include <ATen/GeneratorForPrivareuseone.h>

namespace {

c10::optional<GeneratorFuncType>& GetGeneratorPrivate() {
  static c10::optional<GeneratorFuncType> generator_privateuse1 = c10::nullopt;
  return generator_privateuse1;
}


_GeneratorRegister::_GeneratorRegister(GeneratorFuncType func) {
  TORCH_CHECK(!GetGeneratorPrivate().has_value(),
    "Only can register the Generator for `privateuseone` once!");
  auto m_generator = GetGeneratorPrivate();
  m_generator = func;
}


at::Generator GetGeneratorForPrivateuse1(c10::DeviceIndex device_index) {
  TORCH_CHECK(GetGeneratorPrivate().has_value(),
    "Please register the Generator for `privateuseone`!");
  return GetGeneratorPrivate().value()(device_index);
}

}