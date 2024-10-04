#include <torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.h>
#include <mutex>

namespace torch::jit::mobile {
BuildFeatureTracer::BuildFeatureTracer() {
  auto recorder_cb =
      [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
    std::string name = fn.name();
    getBuildFeatures().withLock(
        [&](BuildFeatureTracer::build_feature_type& build_features) {
          build_features.insert(name);
        });
    return nullptr;
  };

  handle_ =
      at::addGlobalCallback(at::RecordFunctionCallback(recorder_cb)
                                .scopes({at::RecordScope::BUILD_FEATURE}));
}

c10::Synchronized<BuildFeatureTracer::build_feature_type>& BuildFeatureTracer::
    getBuildFeatures() {
  static c10::Synchronized<build_feature_type> build_features;
  return build_features;
}

} // namespace torch::jit::mobile
