#include <torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.h>
#include <mutex>

namespace torch {
namespace jit {
namespace mobile {
BuildFeatureTracer::BuildFeatureTracer() {
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    std::string name = fn.name();
    std::lock_guard<std::mutex> guard(getMutex());
    getBuildFeatures().insert(name);
    return nullptr;
  };

  handle_ =
      at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                .scopes({at::RecordScope::BUILD_FEATURE}));
}

BuildFeatureTracer::build_feature_type& BuildFeatureTracer::getBuildFeatures() {
  static build_feature_type build_features;
  return build_features;
}

std::mutex& BuildFeatureTracer::getMutex() {
  static std::mutex m;
  return m;
}

} // namespace mobile
} // namespace jit
} // namespace torch
