#include "caffe2/operators/load_save_op.h"

namespace caffe2 {
namespace {
REGISTER_CPU_OPERATOR(LoadTensor, LoadTensorOp<CPUContext>);
REGISTER_CPU_OPERATOR(Save, SaveOp<CPUContext>);
REGISTER_CPU_OPERATOR(Snapshot, SnapshotOp<CPUContext>);
}  // namespace
}  // namespace caffe2
