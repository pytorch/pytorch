#include "caffe2/operators/experimental/c10/schemas/batch_matmul.h"
#include "caffe2/core/dispatch/OpSchemaRegistration.h"
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;
using caffe2::Tensor;

C10_DEFINE_OP_SCHEMA(caffe2::ops::BatchMatmul);

namespace {
struct TransAParameter final {
  using type = int;
  static constexpr const char* name() {
    return "trans_a";
  }
  static constexpr int default_value() {
    return 0;
  }
};
struct TransBParameter final {
  using type = int;
  static constexpr const char* name() {
    return "trans_b";
  }
  static constexpr int default_value() {
    return 0;
  }
};
struct BroadcastParameter final {
  using type = int;
  static constexpr const char* name() {
    return "broadcast";
  }
  static constexpr int default_value() {
    return 0;
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::BatchMatmul,
    ops::BatchMatmul::State,
    C10BatchMatMul_DontUseThisOpYet,
    ParameterHelper<TransAParameter>,
    ParameterHelper<TransBParameter>,
    ParameterHelper<BroadcastParameter>)
}
