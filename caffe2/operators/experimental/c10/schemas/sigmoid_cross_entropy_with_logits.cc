#include "caffe2/operators/experimental/c10/schemas/sigmoid_cross_entropy_with_logits.h"
#include <ATen/core/dispatch/OpSchemaRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"

using caffe2::CPUContext;

namespace caffe2 {
namespace ops {
// TODO Parse schema string instead of creating FunctionSchema manually
C10_DEFINE_OP_SCHEMA(SigmoidCrossEntropyWithLogits, FunctionSchema(
    "_c10_experimental::SigmoidCrossEntropyWithLogits",
    (std::vector<c10::Argument>{
      c10::Argument("input1"),
      c10::Argument("input2"),
      c10::Argument("output"),
      c10::Argument("log_D_trick", BoolType::get()),
      c10::Argument("unjoined_lr_loss", BoolType::get())
    }), (std::vector<c10::Argument>{
    })
));
}
}

namespace {
struct LogDTrickParameter final {
  using type = bool;
  static constexpr const char* name() {
    return "log_D_trick";
  }
  static constexpr bool default_value() {
    return false;
  }
};
struct UnjoinedLRLossParameter final {
  using type = bool;
  static constexpr const char* name() {
    return "unjoined_lr_loss";
  }
  static constexpr bool default_value() {
    return false;
  }
};
} // namespace

namespace caffe2 {
REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_WITH_PARAMETERS(
    ops::SigmoidCrossEntropyWithLogits,
    C10SigmoidCrossEntropyWithLogits_DontUseThisOpYet,
    1,
    ParameterHelper<LogDTrickParameter>,
    ParameterHelper<UnjoinedLRLossParameter>)
}
