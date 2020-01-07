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

namespace {

TypePtr realCustomClassHandler(const std::string& name) {
  return classCU()->get_type(name);
}

} // namespace

int register_custom_class_handler = []() {
  setGetCustomClassFn(realCustomClassHandler);
  return 0;
}();

} // namespace jit
} // namespace torch
