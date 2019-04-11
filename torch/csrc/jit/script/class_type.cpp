#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/script/module.h>

namespace c10 {

// This file exists because we need to reference module.h, which we can't from
// c10. Sigh...
Method* ClassType::getMethod(const std::string& name) const {
  return module_? module_->find_method(name) : nullptr;
}

std::vector<Method*> ClassType::methods() const {
  std::vector<Method*> ret;
  for (const auto& pr : module_->get_methods()) {
    ret.push_back(pr.get());
  }
  return ret;
}
} // namespace c10
