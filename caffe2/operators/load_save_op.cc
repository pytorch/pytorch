#include "caffe2/operators/load_save_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(LoadFloatTensor, LoadFloatTensorOp<CPUContext>);
REGISTER_CPU_OPERATOR(SaveFloatTensor, SaveFloatTensorOp<CPUContext>);
}  // namespace
}  // namespace caffe2
