#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/custom_class.h>

#include <atomic>

namespace torch {
namespace jit {

std::unordered_map<std::string, detail::RegisteredClassRecord>& registeredClasses() {
  static std::unordered_map<std::string, detail::RegisteredClassRecord> registry;
  return registry;
}

namespace {
  std::vector<ClassRegistrationCallback> &class_callbacks() {
    static std::vector<ClassRegistrationCallback> cbs;
    return cbs;
  }

  std::vector<MethodRegistrationCallback> &method_callbacks() {
    static std::vector<MethodRegistrationCallback> cbs;
    return cbs;
  }
}  // namespace

void registerClassRegistrationCallback(ClassRegistrationCallback cb) {
  class_callbacks().emplace_back(std::move(cb));
}

void registerMethodRegistrationCallback(MethodRegistrationCallback cb) {
  method_callbacks().emplace_back(std::move(cb));
}

void invokeClassRegistrationCallbacks(const detail::RegisteredClassRecord& class_record) {
  for (auto & cb : class_callbacks()) {
    cb(class_record);
  }
}

void invokeMethodRegistrationCallbacks(const detail::RegisteredClassRecord& class_record, const std::string& method_name) {
  for (auto & cb : method_callbacks()) {
    cb(class_record, method_name);
  }
}


} // namespace jit
} // namespace torch
