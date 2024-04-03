#include "caffe2/operators/key_split_ops.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(KeySplit, KeySplitOp<int64_t, CPUContext>);
NO_GRADIENT(KeySplitOp);
OPERATOR_SCHEMA(KeySplit).NumInputs(1).NumOutputs(1, INT_MAX);
} // namespace caffe2
