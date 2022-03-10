#include <torch/csrc/jit/mobile/model_tracer/CustomClassTracer.h>
#include <mutex>

namespace torch {
namespace jit {
namespace mobile {
CustomClassTracer::CustomClassTracer() {
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    std::string name = fn.name();
    std::lock_guard<std::mutex> guard(getMutex());
    getLoadedClasses().insert(name);
    return nullptr;
  };

  handle_ = at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                      .scopes({at::RecordScope::CUSTOM_CLASS}));
}

CustomClassTracer::custom_classes_type& CustomClassTracer::getLoadedClasses() {
  static custom_classes_type loaded_classes;
  return loaded_classes;
}

std::mutex& CustomClassTracer::getMutex() {
  static std::mutex m;
  return m;
}

} // namespace mobile
} // namespace jit
} // namespace torch
