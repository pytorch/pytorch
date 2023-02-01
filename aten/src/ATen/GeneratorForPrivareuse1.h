#pragma once

#include <ATen/core/Generator.h>
#include <c10/util/intrusive_ptr.h>

namespace at {

using ClassType = std::function<std::shared_ptr<void>(int8_t)>;

static std::map<std::string, ClassType>& GetGeneratorPrivateClassMap() {
  static std::map<std::string, ClassType> generator_privateuse1_map;
  return generator_privateuse1_map;
}

class Register{
public:
  Register(std::string str, ClassType func)
  {
    TORCH_CHECK(GetGeneratorPrivateClassMap().size() == 0,
      "Only can register the GeneratorImpl for `privateuse1` once!");
    GetGeneratorPrivateClassMap().insert(std::make_pair(str, func));
  }
};

at::Generator GetGeneratorClassForPrivateuse1(std::string str, int8_t device_index)
{
  TORCH_CHECK(GetGeneratorPrivateClassMap().find(str) == GetGeneratorPrivateClassMap().end(),
    "Please register the GeneratorImpl for `privateuse1`!");
  return at::Generator(GetGeneratorPrivateClassMap()[str](device_index));
}

/**
 * This is used to register GeneratorImpl to PyTorch for `privateuse1` key.
 * Usage: REGISTERGENERATORFORPRIVATEUSE1(GeneratorImplForPrivateuse1)
 * GeneratorImplForPrivateuse1 is the implement for the `privateuse1` device,
 * and it's constructed function must have one argument which means the device index.
 */
#define REGISTERGENERATORFORPRIVATEUSE1(GeneratorClassName)                                              \
  auto temp##GeneratorClassName = Register(std::string("privateuseone"), GeneratorClassName);            

} // namespace at