#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/script/module.h>

namespace c10 {

// This file exists because we need to reference module.h, which we can't from
// c10. Sigh...
Function* ClassType::getMethod(const std::string& name) const {
  return compilation_unit_->find_function(name);
}

CompilationUnit& ClassType::compilation_unit() {
  return *compilation_unit_;
}
const CompilationUnit& ClassType::compilation_unit() const {
  return *compilation_unit_;
}

std::vector<Function*> ClassType::methods() const {
  std::vector<Function*> ret;
  for (const auto& pr : compilation_unit().get_functions()) {
    ret.push_back(pr.get());
  }
  return ret;
}
} // namespace c10
