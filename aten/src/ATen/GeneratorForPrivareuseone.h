#pragma once

#include <ATen/core/Generator.h>
#include <c10/util/intrusive_ptr.h>

namespace {

using GeneratorFuncType = std::function<at::Generator(c10::DeviceIndex)>;

static c10::optional<GeneratorFuncType>& GetGeneratorPrivate();
class _GeneratorRegister{
public:
  _GeneratorRegister(GeneratorFuncType func);
};

at::Generator GetGeneratorForPrivateuse1(c10::DeviceIndex device_index);

/**
 * This is used to register Generator to PyTorch for `privateuse1` key.
 * Usage: REGISTER_GENERATOR_PRIVATEUSE1(GeneratorForPrivateuse1)
 * GeneratorForPrivateuse1 func must return a argument with type of at::Generator.
 * class CustomGeneratorImpl : public c10::GeneratorImpl {
 * CustomGeneratorImpl(DeviceIndex device_index = -1);
 * ~CustomGeneratorImpl() override = default;
 * ...
 * }
 * at::Generator MakeGeneratorForPrivateuse1(c10::DeviceIndex id) {
 * return at::make_generator<CustomGeneratorImpl>(id);
 * }
 * REGISTER_GENERATOR_PRIVATEUSE1(MakeGeneratorForPrivateuse1)
 */
#define REGISTER_GENERATOR_PRIVATEUSE1(GeneratorPrivate)                      \
  auto temp##GeneratorPrivate = _GeneratorRegister(GeneratorPrivate);            
}
