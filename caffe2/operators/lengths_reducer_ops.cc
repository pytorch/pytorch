#include "caffe2/operators/lengths_reducer_ops.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(
    SparseLengthsSum,
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 0>);
REGISTER_CPU_OPERATOR(
    SparseLengthsWeightedSum,
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 1, 0>);
REGISTER_CPU_OPERATOR(
    SparseLengthsMean,
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 1>);

} // namespace caffe2
