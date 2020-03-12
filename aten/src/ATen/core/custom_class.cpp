#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/api/custom_class.h>

#include <atomic>

namespace torch {
namespace jit {

namespace {

at::TypePtr noOpGetter(const std::string& /*unused*/) {
  return nullptr;
}

std::atomic<GetCustomClassFnType> custom_class_fn{noOpGetter};

}  // namespace

void setGetCustomClassFn(GetCustomClassFnType fn) {
  custom_class_fn.store(fn);
}

at::TypePtr getCustomClass(const std::string& name) {
  return custom_class_fn.load()(name);
}

} // namespace jit
} // namespace torch
