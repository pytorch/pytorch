#include "caffe2/operators/load_save_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(LoadTensor, LoadTensorOp<CPUContext>);
REGISTER_CPU_OPERATOR(Save, SaveOp<CPUContext>);
REGISTER_CPU_OPERATOR(Snapshot, SnapshotOp<CPUContext>);

OPERATOR_SCHEMA(LoadTensor).NumInputs(0).NumOutputs(1, INT_MAX);
OPERATOR_SCHEMA(Save).NumInputs(1, INT_MAX).NumOutputs(0);
OPERATOR_SCHEMA(Snapshot).NumInputs(2, INT_MAX).NumOutputs(0);


SHOULD_NOT_DO_GRADIENT(LoadTensor);
SHOULD_NOT_DO_GRADIENT(Save);
SHOULD_NOT_DO_GRADIENT(Snapshot);
}  // namespace
}  // namespace caffe2
