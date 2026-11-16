#include <torch/csrc/jit/mobile/model_tracer/CustomClassTracer.h>
#include <mutex>

namespace torch::jit::mobile {
CustomClassTracer::CustomClassTracer() {
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    std::string name = fn.name();
    getLoadedClasses().withLock(
        [&name](CustomClassTracer::custom_classes_type& custom_classes) {
          custom_classes.insert(name);
        });
    return nullptr;
  };

  handle_ = at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                      .scopes({at::RecordScope::CUSTOM_CLASS}));
}

c10::Synchronized<CustomClassTracer::custom_classes_type>& CustomClassTracer::
    getLoadedClasses() {
  static c10::Synchronized<custom_classes_type> loaded_classes;
  return loaded_classes;
}

} // namespace torch::jit::mobile
