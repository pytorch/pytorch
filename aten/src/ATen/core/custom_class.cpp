#include <torch/custom_class.h>

#include <atomic>

namespace torch {
namespace jit {

namespace {

TypePtr noOpGetter(const std::string& /*unused*/) {
  return nullptr;
}

std::atomic<GetCustomClassFnType> custom_class_fn{noOpGetter};

}  // namespace

void setGetCustomClassFn(GetCustomClassFnType fn) {
  custom_class_fn.store(fn);
}

TypePtr getCustomClass(const std::string& name) {
  return custom_class_fn.load()(name);
}

} // namespace jit
} // namespace torch
