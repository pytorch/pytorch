#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/schema_matching.h>
#include <cstddef>

namespace torch {
namespace jit {

// Calculates the number of args that need to be passed in.
// Less args may be needed if defaults are provided.
// Returns: {number args needed, number of out args}
inline std::pair<int64_t, int64_t> CalculateNecessaryArgs(
    const std::vector<Argument>& schema_args,
    at::ArrayRef<Value*> actual_inputs,
    bool allow_trailing_out_args) {
  if (schema_args.empty()) {
    return std::make_pair(0, 0);
  }

  // count number of out arguments
  int64_t schema_idx = static_cast<int64_t>(schema_args.size()) - 1;
  if (allow_trailing_out_args) {
    // skip over out arguments in the end.
    while (schema_idx >= 0) {
      const auto& current_arg = schema_args.at(schema_idx);
      if (!current_arg.is_out()) {
        break;
      }
      schema_idx--;
    }
  }

  int64_t num_out = static_cast<int64_t>(schema_args.size()) - schema_idx - 1;

  if (schema_args.size() < actual_inputs.size()) {
    return std::make_pair(actual_inputs.size(), num_out);
  }

  // if it is the default args, we reset the index to the last element
  if (!allow_trailing_out_args) {
    schema_idx = schema_args.size() - 1;
  }
  // keeps track of trailing unnecessary args
  while (schema_idx >= 0) {
    // this means it is not default argument, so it is necessary
    if (!schema_args.at(schema_idx).default_value().has_value()) {
      return std::make_pair(schema_idx + 1, num_out);
    } else {
      auto schema_value =
          schema_args.at(schema_idx).default_value().value().toIValue();
      // non-const value will become nullptr here, so will be marked necessary
      // non-const would include prim::ListConstruct, prim::DictConstruct as
      // well.
      auto actual_value = toIValue(actual_inputs[schema_idx]);
      if (!actual_value.has_value()) {
        return std::make_pair(schema_idx + 1, num_out);
      }
      // if the IR has same value as default value of the schema,
      // it is not necessary argument.
      if (schema_value != actual_value.value()) {
        return std::make_pair(schema_idx + 1, num_out);
      }
    }
    schema_idx--;
  }
  return std::make_pair(0, num_out);
}

} // namespace jit
} // namespace torch
