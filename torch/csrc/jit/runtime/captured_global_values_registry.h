#pragma once

#include <aten/src/ATen/core/ivalue.h>

#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

// Class of a global singleton registry that maintains a mapping from the fully
// qualified name of a global variable to its captured value (in form of IValue)
class CapturedGlobalValuesRegistry {
 public:
  // Get global singleton registry.
  TORCH_API static CapturedGlobalValuesRegistry& get();

  // Look up captured global value by name. If not found, throw runtime_error.
  c10::IValue getValueOrThrow(const std::string& name) const;

  // Add or update a value by name.
  TORCH_API void registerValue(const std::string& name, c10::IValue value);

  // Clear registry;
  TORCH_API void clear();

  CapturedGlobalValuesRegistry(const CapturedGlobalValuesRegistry&) = delete;
  CapturedGlobalValuesRegistry(CapturedGlobalValuesRegistry&&) = delete;
  CapturedGlobalValuesRegistry& operator=(const CapturedGlobalValuesRegistry&) =
      delete;
  CapturedGlobalValuesRegistry& operator=(CapturedGlobalValuesRegistry&&) =
      delete;

 private:
  CapturedGlobalValuesRegistry() = default;

  std::unordered_map<std::string, c10::IValue> registry_;
};

} // namespace jit
} // namespace torch
