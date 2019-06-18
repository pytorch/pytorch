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

ClassTypePtr ClassType::create(
    c10::optional<QualifiedName> qualifiedName,
    std::shared_ptr<CompilationUnit> cu,
    bool is_module) {
  return ClassTypePtr(new ClassType(std::move(qualifiedName), std::move(cu), is_module));
}

ClassTypePtr ClassType::refine(at::ArrayRef<TypePtr> refined_slots) const {
  auto ptr = ClassType::create(name_, compilation_unit_);
  AT_ASSERT(numAttributes() == refined_slots.size());
  for(size_t i = 0; i < attributeNames_.size(); ++i) {
    AT_ASSERT(refined_slots[i]->isSubtypeOf(attributeTypes_[i]));
    ptr->addAttribute(attributeNames_[i], refined_slots[i]);
  }
  return ptr;
}

std::vector<Function*> ClassType::methods() const {
  std::vector<Function*> ret;
  for (const auto& pr : compilation_unit()->get_functions()) {
    ret.push_back(pr.get());
  }
  return ret;
}

ClassType::ClassType(
    c10::optional<QualifiedName> name,
    std::shared_ptr<CompilationUnit> cu,
    bool is_module)
    : SerializableType(TypeKind::ClassType, std::move(name)),
      compilation_unit_(std::move(cu)),
      is_module_(is_module) {}

c10::optional<std::string> ClassType::base_class_name() const {
  return c10::nullopt;
}

std::vector<std::tuple<std::string, TypePtr>> ClassType::attrs() const {
  return {};
}

} // namespace c10
