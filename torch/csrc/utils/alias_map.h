#pragma once

#include <ATen/core/function_schema.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch {
namespace utils {

/**
 * class AliasMap
 *
 * Holds various aliasing and mutation information
 */

class TORCH_API AliasMap {
 public:
  AliasMap() = default;
  explicit AliasMap(const c10::FunctionSchema& schema);
  const std::vector<std::unordered_set<size_t>>& alias_map() const {
    return alias_map_;
  };
  const std::unordered_set<size_t>& const_aliasing_inputs() const {
    return const_aliasing_inputs_;
  };
  const std::unordered_set<size_t>& const_aliasing_outputs() const {
    return const_aliasing_outputs_;
  };
  const std::unordered_set<size_t>& aliasing_inputs() const {
    return aliasing_inputs_;
  };
  const std::unordered_set<size_t>& aliasing_outputs() const {
    return aliasing_outputs_;
  };
  const std::unordered_set<size_t>& mutating_inputs() const {
    return mutating_inputs_;
  };
  const std::unordered_set<size_t>& mutating_outputs() const {
    return mutating_outputs_;
  };

 private:
  void generateMaps(const c10::FunctionSchema& schema);
  // Map of aliasing information from output arguments to input arguments
  std::vector<std::unordered_set<size_t>> alias_map_;
  std::unordered_set<size_t> const_aliasing_inputs_;
  std::unordered_set<size_t> const_aliasing_outputs_;
  std::unordered_set<size_t> aliasing_inputs_;
  std::unordered_set<size_t> aliasing_outputs_;
  std::unordered_set<size_t> mutating_inputs_;
  std::unordered_set<size_t> mutating_outputs_;
};
} // namespace utils
} // namespace torch
