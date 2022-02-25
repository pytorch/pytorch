#include <torch/csrc/jit/mobile/model_tracer/BuildFeatureTracer.h>

namespace torch {
namespace jit {
namespace mobile {
BuildFeatureTracer::build_feature_type& BuildFeatureTracer::getBuildFeatures() {
  static build_feature_type build_features;
  return build_features;
}

} // namespace mobile
} // namespace jit
} // namespace torch
