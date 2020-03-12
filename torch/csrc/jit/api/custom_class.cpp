#include <torch/custom_class.h>

#include <atomic>

namespace torch {
namespace jit {

std::vector<c10::RegisterOperators>& registeredOps() {
  static std::vector<c10::RegisterOperators> ops;
  return ops;
}

std::shared_ptr<script::CompilationUnit>& classCU() {
  static std::shared_ptr<script::CompilationUnit> cu =
      std::make_shared<script::CompilationUnit>();
  return cu;
}

bool isCustomClass(const c10::IValue& v) {
  return v.isObject() && v.toObject()->type()->name() &&
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

namespace {

TypePtr realCustomClassHandler(const std::string& name) {
  return classCU()->get_type(name);
}

} // namespace

int register_custom_class_handler() {
  setGetCustomClassFn(realCustomClassHandler);
  return 0;
};

static int ensure_custom_class_handler_registered = register_custom_class_handler();

} // namespace jit
} // namespace torch
