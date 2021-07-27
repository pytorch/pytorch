#include "caffe2/operators/ngram_ops.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    NGramFromCategorical,
    NGramFromCategoricalOp<float, int64_t, CPUContext>);
NO_GRADIENT(NGramFromCategorical);
OPERATOR_SCHEMA(NGramFromCategorical).NumInputs(1).NumOutputs(1);
} // namespace caffe2
