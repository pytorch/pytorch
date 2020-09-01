#include <torch/csrc/jit/runtime/captured_global_values_registry.h>

#include <aten/src/ATen/core/ivalue.h>
#include <iostream>
#include <stdexcept>

namespace torch {
namespace jit {

CapturedGlobalValuesRegistry& CapturedGlobalValuesRegistry::get() {
  static CapturedGlobalValuesRegistry registry;
  return registry;
}

c10::IValue CapturedGlobalValuesRegistry::getValueOrThrow(
    const std::string& name) const {
  auto it = registry_.find(name);
  if (it == registry_.end()) {
    throw(std::runtime_error(
        name + " is not found in CapturedGlobalValuesRegistry, " +
        "did you capture its value with " +
        "torch.jit.capture_global_constant_value()?"));
  }
  return it->second;
}

void CapturedGlobalValuesRegistry::registerValue(
    const std::string& name,
    c10::IValue value) {
  // Permit overwrite to support value update.
  registry_[name] = std::move(value);
}

void CapturedGlobalValuesRegistry::clear() {
  registry_.clear();
}

} // namespace jit
} // namespace torch
