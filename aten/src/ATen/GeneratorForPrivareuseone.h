#pragma once

#include <ATen/core/Generator.h>
#include <c10/util/intrusive_ptr.h>

namespace at {

using GeneratorFuncType = std::function<at::Generator(c10::DeviceIndex)>;

static std::map<std::string, GeneratorFuncType>& GetGeneratorPrivateClassMap() {
  static std::map<std::string, GeneratorFuncType> generator_privateuse1_map;
  return generator_privateuse1_map;
}

class Register{
public:
  Register(std::string str, GeneratorFuncType func) {
    TORCH_CHECK(GetGeneratorPrivateClassMap().size() == 0,
      "Only can register the Generator for `privateuseone` once!");
    GetGeneratorPrivateClassMap().insert(std::make_pair(str, func));
  }
};

at::Generator GetGeneratorClassForPrivateuse1(c10::DeviceIndex device_index,
    std::string str="privateuseone") {
  TORCH_CHECK(str == "privateuseone", "Excepted got str with `privateuseone`, but got `", str, "`!");
  TORCH_CHECK(GetGeneratorPrivateClassMap().find(str) != GetGeneratorPrivateClassMap().end(),
    "Please register the Generator for `privateuseone`!");

  return GetGeneratorPrivateClassMap()[str](device_index);
}

/**
 * This is used to register Generator to PyTorch for `privateuse1` key.
 * Usage: REGISTERGENERATORFORPRIVATEUSE1(GeneratorForPrivateuse1)
 * GeneratorForPrivateuse1 func must return a argument with type of at::Generator.
 * 
 * class CustomGeneratorImpl : public c10::GeneratorImpl {
 * CustomGeneratorImpl(DeviceIndex device_index = -1);
 * ~CustomGeneratorImpl() override = default;
 * ...
 * }
 *
 * at::Generator MakeGeneratorForPrivateuse1(c10::DeviceIndex id) {
 * return at::make_generator<CustomGeneratorImpl>(id);
 * }
 * 
 * REGISTERGENERATORFORPRIVATEUSE1(MakeGeneratorForPrivateuse1)
 */
#define REGISTERGENERATORFORPRIVATEUSE1(GeneratorClassName)                                  \
  auto temp##GeneratorClassName = Register(std::string("privateuseone"), GeneratorClassName);            

} // namespace at
