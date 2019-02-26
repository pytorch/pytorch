#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/script/module.h>

namespace c10 {

// This file exists because we need to reference module.h, which we can't from
// c10. Sigh...
Method* UserType::getMethod(const std::string& name) const {
  return module_->find_method(name);
}

} // namespace c10
