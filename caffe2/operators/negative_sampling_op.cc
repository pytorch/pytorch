#include "caffe2/operators/negative_sampling_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    UniformNegativeSamplingOp,
    UniformNegativeSamplingOp<CPUContext>);

OPERATOR_SCHEMA(UniformNegativeSamplingOp)
    .NumInputs(2)
    .NumOutputs(3)
    .SetDoc(R"DOC(
For each input positive example, generate several negative samples uniformly that are different from it.
Return indices containing both positive and negative samples.
)DOC")
    .Input(0, "INDICES", "input indices")
    .Input(1, "LENGTHS", "input lengths")
    .Output(0, "OUPUT_INDICES", "indices containing negative examples")
    .Output(1, "OUTPUT_LENGTHS", "lengths of output sparse segment")
    .Output(2, "OUTPUT_LABELS", "label corresponding to each indice");

NO_GRADIENT(UniformNegativeSamplingOp);
} // namespace caffe2
