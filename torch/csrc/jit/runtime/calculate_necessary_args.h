#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <cstddef>

namespace torch {
namespace jit {

inline size_t CalculateNecessaryArgs(
    const std::vector<Argument>& schema_args,
    at::ArrayRef<Value*> actual_inputs,
    bool add_out_args) {
  if (schema_args.size() == 0) {
    return 0;
  }
  if (schema_args.size() < actual_inputs.size()) {
    return actual_inputs.size();
  }

  auto schema_idx = schema_args.size() - 1;
  if (add_out_args) {
    // skip over out arguments in the end.
    while (schema_idx >= 0) {
      auto current_arg = schema_args.at(schema_idx);
      if (!current_arg.is_out()) {
        break;
      }
      schema_idx--;
    }
  }

  auto num_out = schema_args.size() - schema_idx - 1;
  // keeps track of trailing unnecessary args
  while (schema_idx >= 0) {
    // this means it is not default argument, so it is necessary
    if (!schema_args.at(schema_idx).default_value().has_value()) {
      return schema_idx + 1 + num_out;
    } else {
      auto schema_value =
          schema_args.at(schema_idx).default_value().value().toIValue();
      // non-const value will become nullptr here, so will be marked necessary
      // non-const would include prim::ListConstruct, prim::DictConstruct as
      // well.
      auto actual_value = toIValue(actual_inputs[schema_idx]);
      if (!actual_value.has_value()) {
        return schema_idx + 1 + num_out;
      }
      // if the IR has same value as default value of the schema,
      // it is not neccessary argument.
      if (schema_value != actual_value.value()) {
        return schema_idx + 1 + num_out;
      }
    }
    schema_idx--;
  }
  return num_out;
}

} // namespace jit
} // namespace torch
