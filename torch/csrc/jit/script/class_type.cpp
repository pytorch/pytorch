#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/script/module.h>

namespace c10 {

// This file exists because we need to reference module.h, which we can't from
// c10. Sigh...
std::shared_ptr<Function> ClassType::getMethod(const std::string& name) const {
  return compilation_unit_->find_function(name);
}

std::shared_ptr<CompilationUnit> ClassType::compilation_unit() {
  return compilation_unit_;
}
std::shared_ptr<const CompilationUnit> ClassType::compilation_unit() const {
  return compilation_unit_;
}

std::vector<Function*> ClassType::methods() const {
  std::vector<Function*> ret;
  for (const auto& pr : compilation_unit()->get_functions()) {
    ret.push_back(pr.get());
  }
  return ret;
}

namespace ivalue {
Object::~Object() {
  if (type_->is_module()) {
    type_->compilation_unit()->drop_all_functions();
  }
}
} // namespace ivalue

} // namespace c10
