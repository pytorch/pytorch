#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <cstddef>

namespace torch {
namespace jit {

inline size_t CalculateNecessaryArgs(
    const std::vector<Argument>& schema_args,
    at::ArrayRef<Value*> actual_inputs) {
  if (schema_args.size() < actual_inputs.size()) {
    return actual_inputs.size();
  }
  // keeps track of trailing unnecessary args
  int schema_size = schema_args.size();
  for (int schema_idx = schema_size - 1; schema_idx > -1; schema_idx--) {
    // this means it is not default argument, so it is necessary
    if (!schema_args.at(schema_idx).default_value().has_value()) {
      return schema_idx + 1;
    } else {
      auto schema_value =
          schema_args.at(schema_idx).default_value().value().toIValue();
      // non-const value will become nullptr here, so will be marked necessary
      // non-const would include prim::ListConstruct, prim::DictConstruct as
      // well.
      auto actual_value = toIValue(actual_inputs[schema_idx]);
      if (!actual_value.has_value()) {
        return schema_idx + 1;
      }
      // if the IR has same value as default value of the schema,
      // it is not neccessary argument.
      if (schema_value != actual_value.value()) {
        return schema_idx + 1;
      }
    }
  }
  return 0;
}

} // namespace jit
} // namespace torch
