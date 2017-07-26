#include "ctc_op.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace detail {
template <>
ctcComputeInfo workspaceInfo<CPUContext>(const CPUContext& /*context*/) {
  ctcComputeInfo result;
  result.loc = CTC_CPU;
  result.num_threads = 1;
  return result;
}
}

REGISTER_CPU_OPERATOR(CTC, CTCOp<float, CPUContext>);
OPERATOR_SCHEMA(CTC)
    .NumInputs(4)
    .NumOutputs(3);
//    .EnforceInputOutputGradient({{0, 0}});

namespace {
class GetCTCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Copy", "", vector<string>{O(0)}, vector<string>{GI(0)});
  }
};
}
REGISTER_GRADIENT(CTC, GetCTCGradient);
}
