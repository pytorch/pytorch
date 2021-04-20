#include <torch/custom_class.h>

#include <ATen/core/jit_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/functional.h>

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
  TORCH_CHECK(
      !customClasses().count(name),
      "Custom class with name ",
      name,
      " is already registered. Ensure that registration with torch::class_ is only called once.");
  customClasses()[name] = std::move(class_type);
}

at::ClassTypePtr getCustomClass(const std::string& name) {
  return customClasses().count(name) ? customClasses()[name] : nullptr;
}

bool isCustomClass(const c10::IValue& v) {
  return v.isObject() && v.toObject()->type()->name() &&
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

using MethodOverloadsList = std::vector<std::unique_ptr<jit::Function>>;
using MethodMap = std::unordered_map<std::string, MethodOverloadsList>;

MethodMap& customClassMethods() {
  static MethodMap customClassMethods;
  return customClassMethods;
}

void registerCustomClassMethod(std::unique_ptr<jit::Function> fn) {
  auto& custom_class_methods = customClassMethods();

  // check if the method is already registered
  for (auto& methods : custom_class_methods) {
    for (auto& method_it : methods.second) {
      if (method_it == fn) {
        return;
      }
    }
  }

  auto it =
      custom_class_methods.insert(std::pair<std::string, MethodOverloadsList>(
          fn->name(), MethodOverloadsList()));
  it.first->second.push_back(std::move(fn));
}

std::vector<c10::FunctionSchema> customClassSchemasForBCCheck() {
  auto& MethodMap = customClassMethods();
  std::vector<c10::FunctionSchema> schemas;
  for (auto& methods : MethodMap) {
    for (auto& method_it : methods.second) {
      schemas.push_back(method_it.get()->getSchema());
    }
  }
  return schemas;
}

} // namespace torch
