#include "caffe2/operators/utility_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(Free, FreeOp);
REGISTER_CPU_OPERATOR(Print, PrintOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(PrintInt, PrintOp<int, CPUContext>);
REGISTER_CPU_OPERATOR(Flatten, FlattenOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Alias, AliasOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ReshapeLike, ReshapeLikeOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Split, SplitOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Sum, SumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(WeightedSum, WeightedSumOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(Copy, CopyOp<float, CPUContext, CPUContext, CPUContext>);


}  // namespace
}  // namespace caffe2


