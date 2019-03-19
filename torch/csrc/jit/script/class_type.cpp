#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/script/module.h>

namespace c10 {

// This file exists because we need to reference module.h, which we can't from
// c10. Sigh...
Method* ClassType::getMethod(const std::string& name) const {
  return module_->find_method(name);
}

std::vector<Method*> ClassType::methods() const {
  const auto& methods = module_->get_methods();
  std::vector<Method*> ret;
  for (const auto& pr : methods.items()) {
    ret.push_back(pr.value().get());
  }
  return ret;
}
} // namespace c10
