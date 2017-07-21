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

namespace {
REGISTER_CPU_OPERATOR(CTC, CTCOp<float, CPUContext>);
OPERATOR_SCHEMA(CTC)
    .NumInputs(4)
    .NumOutputs(3);
//    .EnforceInputOutputGradient({{0, 0}});
NO_GRADIENT(CTC);

}
}
