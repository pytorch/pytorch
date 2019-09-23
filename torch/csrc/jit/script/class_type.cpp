#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/script/module.h>

namespace c10 {

// This file exists because we need to reference module.h, which we can't from
// c10. Sigh...
FunctionType::FunctionType(Function* function)
    : NamedType(TypeKind::FunctionType, function->qualname()),
      function_(function) {}

Function* ClassType::getMethod(const std::string& name) const {
  for (auto method : methods_) {
    if (name == method->name()) {
      return method;
    }
  }
  return nullptr;
}

std::shared_ptr<CompilationUnit> ClassType::compilation_unit() {
  auto cu = compilation_unit_.lock();
  TORCH_INTERNAL_ASSERT(cu);
  return cu;
}
std::shared_ptr<const CompilationUnit> ClassType::compilation_unit() const {
  auto cu = compilation_unit_.lock();
  TORCH_INTERNAL_ASSERT(cu);
  return cu;
}

ClassTypePtr ClassType::create(
    c10::optional<QualifiedName> qualifiedName,
    std::weak_ptr<CompilationUnit> cu,
    bool is_module) {
  return ClassTypePtr(new ClassType(std::move(qualifiedName), std::move(cu), is_module));
}

ClassTypePtr ClassType::refine(at::ArrayRef<TypePtr> refined_slots) const {
  auto ptr = ClassType::create(name(), compilation_unit_);
  AT_ASSERT(numAttributes() == refined_slots.size());
  for(size_t i = 0; i < attributeNames_.size(); ++i) {
    AT_ASSERT(refined_slots[i]->isSubtypeOf(attributeTypes_[i]));
    ptr->addAttribute(attributeNames_[i], refined_slots[i]);
  }
  // Copy methods over
  for (const auto& method : methods()) {
    ptr->addMethod(method);
  }
  return ptr;
}

void ClassType::addMethod(Function* method) {
  TORCH_CHECK(
      getMethod(method->name()) == nullptr,
      "Can't redefine method: ",
      method->name());
  methods_.push_back(method);
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

const std::vector<Function*>& ClassType::methods() const {
  return methods_;
}

ClassType::ClassType(
    c10::optional<QualifiedName> name,
    std::weak_ptr<CompilationUnit> cu,
    bool is_module)
    : NamedType(TypeKind::ClassType, std::move(name)),
      compilation_unit_(std::move(cu)) {
  if (is_module) {
    parameterSlots_ = std::make_shared<std::vector<bool>>();
  }
}

bool ClassType::isSubtypeOfExt(const TypePtr rhs, std::ostream* why_not) const {
  // to improve performance, this check can be cached
  if (auto iface = rhs->cast<InterfaceType>()) {
    for (const FunctionSchema& schema : iface->methods()) {
      auto self_method = getMethod(schema.name());
      if (!self_method) {
        if (why_not) {
          *why_not << "Class '" << python_str() << "' does not have method '"
                   << schema.name() << "' but '" << rhs->python_str()
                   << "' does.\n";
        }
        return false;
      }
      if (!self_method->getSchema().isSubtypeOf(
              schema, /*is_method=*/true, why_not)) {
        if (why_not) {
          *why_not << "Method on class '" << python_str()
                   << "' (1) is not compatible with interface '"
                   << rhs->python_str() << "' (2)\n"
                   << "  (1) " << self_method->getSchema() << "\n"
                   << "  (2) " << schema << "\n";
          return false;
        }
      }
    }
    return true;
  }
  return Type::isSubtypeOfExt(rhs, why_not);
}

} // namespace c10
