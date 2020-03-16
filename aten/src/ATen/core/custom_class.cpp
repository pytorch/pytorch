#include <torch/custom_class.h>

#include <ATen/core/jit_type.h>

#include <atomic>
#include <unordered_map>

namespace torch {

std::unordered_map<std::string, at::ClassTypePtr>& customClasses() {
  static std::unordered_map<std::string, at::ClassTypePtr> customClasses;
  return customClasses;
}

void registerCustomClass(at::ClassTypePtr class_type) {
  TORCH_INTERNAL_ASSERT(class_type->name());
  auto name = class_type->name()->qualifiedName();
  TORCH_CHECK(!customClasses().count(name))
  customClasses()[name] = std::move(class_type);
}

at::ClassTypePtr getCustomClass(const std::string& name) {
  return customClasses().count(name) ? customClasses()[name] : nullptr;
}

bool isCustomClass(const c10::IValue& v) {
  return v.isObject() && v.toObject()->type()->name() &&
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

std::vector<std::shared_ptr<jit::Function>>& customClassMethods() {
  static std::vector<std::shared_ptr<jit::Function>> customClassMethods;
  return customClassMethods;
}

void registerCustomClassMethod(std::shared_ptr<jit::Function> fn) {
  customClassMethods().emplace_back(std::move(fn));
}

} // namespace torch
