#pragma once

#include <ATen/core/Generator.h>
#include <c10/util/intrusive_ptr.h>

namespace at {

using GeneratorFuncType = std::function<at::Generator(c10::DeviceIndex)>;

c10::optional<GeneratorFuncType>& GetGeneratorPrivate();

class TORCH_API _GeneratorRegister {
 public:
  explicit _GeneratorRegister(GeneratorFuncType func);
};

TORCH_API at::Generator GetGeneratorForPrivateuse1(
    c10::DeviceIndex device_index);

/**
 * This is used to register Generator to PyTorch for `privateuse1` key.
 *
 * Usage: REGISTER_GENERATOR_PRIVATEUSE1(MakeGeneratorForPrivateuse1)
 *
 * class CustomGeneratorImpl : public c10::GeneratorImpl {
 *   CustomGeneratorImpl(DeviceIndex device_index = -1);
 *   explicit ~CustomGeneratorImpl() override = default;
 *   ...
 * };
 *
 * at::Generator MakeGeneratorForPrivateuse1(c10::DeviceIndex id) {
 *   return at::make_generator<CustomGeneratorImpl>(id);
 * }
 */

#define REGISTER_GENERATOR_PRIVATEUSE1(GeneratorPrivate) \
  static auto temp##GeneratorPrivate = at::_GeneratorRegister(GeneratorPrivate);

} // namespace at
