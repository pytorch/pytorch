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

using method_list = std::vector<std::unique_ptr<jit::Function>>;
using method_map = std::unordered_map<std::string, method_list>;

method_map& customClassMethods() {
  static method_map customClassMethods;
  return customClassMethods;
}

void registerCustomClassMethod(std::unique_ptr<jit::Function> fn) {

  auto& custom_class_methods = customClassMethods();
  if (custom_class_methods.find(fn->name()) == custom_class_methods.end()){
    custom_class_methods[fn->name()] = method_list();
  }
  custom_class_methods[fn->name()].push_back(std::move(fn));
}

std::vector<c10::FunctionSchema> customClassSchemasForBCCheck() {
    auto& custom_class_methods = customClassMethods();
    std::vector<c10::FunctionSchema> schemas;

    // for (auto & custom_class_method : custom_class_methods) {
    //   auto methods = custom_class_method.second;
    //   for (auto & method : methods) {
    //     schemas.push_back(method->getSchema());
    //   }
    // }

    return schemas;
}



} // namespace torch
