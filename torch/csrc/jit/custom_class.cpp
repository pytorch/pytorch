#include <torch/custom_class.h>

#include <atomic>
#include <unordered_map>

namespace torch {
namespace jit {

bool isCustomClass(const c10::IValue& v) {
  return v.isObject() && v.toObject()->type()->name() &&
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

std::vector<c10::RegisterOperators>& registeredOps() {
  static std::vector<c10::RegisterOperators> ops;
  return ops;
}

#ifndef C10_MOBILE

std::shared_ptr<script::CompilationUnit>& classCU() {
  static std::shared_ptr<script::CompilationUnit> cu =
      std::make_shared<script::CompilationUnit>();
  return cu;
}

namespace {

at::TypePtr realCustomClassHandler(const std::string& name) {
  return classCU()->get_type(name);
}

} // namespace

int register_custom_class_handler() {
  setGetCustomClassFn(realCustomClassHandler);
  return 0;
};

static int ensure_custom_class_handler_registered = register_custom_class_handler();

#else // C10_MOBILE

std::unordered_map<std::string, at::ClassTypePtr> mobileCustomClassRegistry;

void registerCustomClassForMobile(at::ClassTypePtr classTypePtr) {
  TORCH_INTERNAL_ASSERT(classTypePtr->name());
  mobileCustomClassRegistry[classTypePtr->name()->qualifiedName()] = classTypePtr;
}

TORCH_API at::TypePtr getCustomClass(const std::string& name) {
  return mobileCustomClassRegistry.at(name);
}

#endif // C10_MOBILE

} // namespace jit
} // namespace torch
