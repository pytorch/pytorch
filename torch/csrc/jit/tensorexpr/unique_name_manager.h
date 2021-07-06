#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace tensorexpr {

class VarHandle;
class Var;

using VarNameMap = std::unordered_map<const Var*, std::string>;

// A manager to get unique names from vars.
// It starts with the name hints of the var and append "_" + $counter until it
// hits a unique name.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class TORCH_API UniqueNameManager {
 public:
  const std::string& get_unique_name(const VarHandle& v);

  const std::string& get_unique_name(const Var* v);

 private:
  friend class ScopedVarName;
  VarNameMap unique_name_mapping_;
  std::unordered_map<std::string, int> unique_name_count_;
  std::unordered_set<std::string> all_unique_names_;
};

} // namespace tensorexpr
} // namespace jit
} // namespace torch
