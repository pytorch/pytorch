#include "caffe2/operators/lengths_reducer_ops.h"
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// Use _STR option because the schema is declared using _STR version too in
// generic fashion. Otherwise it'd break schema declaration check.
// TODO(dzhulgakov): remove _STR when all lengths ops are off generic version.

REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsSum",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 0>);
REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsWeightedSum",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 1, 0>);
REGISTER_CPU_OPERATOR_STR(
    "SparseLengthsMean",
    CPUSparseLengthsReductionOp<float, TensorTypes<float, float16>, 0, 1>);

} // namespace caffe2
