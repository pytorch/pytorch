#include "torch/csrc/jit/aten_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ATen2, ATenOp2<CPUContext>);
template <>
at::Backend ATenOp2<CPUContext>::backend() const {
  return at::Backend::CPU;
}

OPERATOR_SCHEMA(ATen2);

} // namespace caffe2
