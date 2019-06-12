#include "caffe2/operators/experimental/c10/schemas/fc.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::ops::FullyConnected);

namespace {
struct AxisParameter final {
  using type = int32_t;
  static constexpr const char* name() {
    return "axis";
  }
  static constexpr int32_t default_value() {
    return 1;
  }
};
struct AxisWParameter final {
  using type = int32_t;
  static constexpr const char* name() {
    return "axis_w";
  }
  static constexpr int32_t default_value() {
    return 1;
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::FullyConnected,
    ops::FullyConnected::Cache,
    C10FC_DontUseThisOpYet,
    ParameterHelper<AxisParameter>,
    ParameterHelper<AxisWParameter>)
}
