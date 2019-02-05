#include <ATen/core/opschema/layer_norm.h>
#include "caffe2/core/operator_c10wrapper.h"

namespace {

struct AxisParameter final {
  using type = int;
  static constexpr const char* name() {
    return "axis";
  }
  static constexpr int default_value() {
    return 1;
  }
};
struct EpsilonParameter final {
  using type = float;
  static constexpr const char* name() {
    return "epsilon";
  }
  static constexpr float default_value() {
    return 0.001f;
  }
};
} // namespace


namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    c10::core::opschema::LayerNorm,
    C10LayerNorm_DontUseThisOpYet,
    3,
    ParameterHelper<AxisParameter>,
    ParameterHelper<EpsilonParameter>)
}
