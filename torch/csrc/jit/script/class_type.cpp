#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/script/module.h>

namespace c10 {

// This file exists because we need to reference module.h, which we can't from
// c10. Sigh...

Function* ClassType::getMethod(const std::string& name) const {
  const auto qualname = QualifiedName(*qualified_name_obj(), name);
  return compilation_unit_->find_function(qualname);
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

size_t ClassType::addAttribute(
    const std::string& name,
    TypePtr type,
    bool is_parameter) {
  for (size_t i = 0; i < attributeNames_.size(); ++i) {
    TORCH_CHECK(
        name != attributeNames_[i],
        "attempting to add ",
        is_parameter ? "parameter"
                     : "attribute"
                       " '",
        name,
        "' but a field of the same name already exists with type ",
        attributeTypes_[i]->python_str());
  }
  size_t slot = attributeNames_.size();
  attributeNames_.push_back(name);
  attributeTypes_.push_back(type);
  if (is_parameter) {
    TORCH_INTERNAL_ASSERT(is_module(), "adding a parameter to a non module");
  }
  if (is_module()) {
    parameterSlots_->push_back(is_parameter);
  }
  return slot;
}

std::vector<Function*> ClassType::methods() const {
  // TODO: this needs to be made more efficient!
  // This grabs all the functions in the CU and filters them by qualified name.
  auto cuFunctions = compilation_unit()->get_functions();
  const auto& classname = *qualified_name_obj();
  cuFunctions.erase(
      std::remove_if(
          cuFunctions.begin(),
          cuFunctions.end(),
          [&](Function* fn) { return !classname.isPrefixOf(fn->qualname()); }),
      cuFunctions.end());
  return cuFunctions;
}

ClassType::ClassType(
    c10::optional<QualifiedName> name,
    std::shared_ptr<CompilationUnit> cu,
    bool is_module)
    : NamedType(TypeKind::ClassType, name), compilation_unit_(std::move(cu)) {
  if (is_module) {
    parameterSlots_ = std::make_shared<std::vector<bool>>();
  }
}

} // namespace c10
