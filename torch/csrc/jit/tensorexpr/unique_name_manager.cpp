#include <torch/csrc/jit/tensorexpr/unique_name_manager.h>

#include <torch/csrc/jit/tensorexpr/ir.h>
#include <cctype>

namespace torch::jit::tensorexpr {

const std::string& UniqueNameManager::get_unique_name(VarPtr v) {
  // Find if we have already encountered this variable.
  auto iter = unique_name_mapping_.find(v);
  if (iter != unique_name_mapping_.end()) {
    return iter->second;
  }

  // First use the name_hint as a prefix to check if there is another name
  // with the same prefix.
  std::string name_hint = v->name_hint();
  if (name_hint.empty()) {
    name_hint = "v";
  } else if (std::isdigit(name_hint[0])) {
    name_hint = "v" + name_hint;
  }
  int& count = unique_name_count_[name_hint];
  while (true) {
    // Even if with a new count, this name might already be used. For example
    // ("x", 1) could collidewith ("x_1", 0)
    int count_v = count++;
    std::string unique_name = name_hint;
    if (count_v > 0) {
      unique_name += "_" + std::to_string(count_v);
    }
    if (all_unique_names_.count(unique_name) == 0) {
      all_unique_names_.insert(unique_name);
      auto result = unique_name_mapping_.insert(std::make_pair(v, unique_name));
      return result.first->second;
    }
  }
}

const std::string& UniqueNameManager::get_unique_name(const VarHandle& v) {
  return get_unique_name(v.node());
}

} // namespace torch::jit::tensorexpr
