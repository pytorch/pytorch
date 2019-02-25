#include "caffe2/contrib/aten/aten_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ATen, ATenOp<CPUContext>);
template<>
at::Backend ATenOp<CPUContext>::backend() const {
  return at::Backend::CPU;
}

OPERATOR_SCHEMA(ATen);

namespace math {
template <>
void Set<at::Half, CPUContext>(
    const int /*N*/,
    const at::Half h,
    at::Half* v,
    CPUContext* c) {
  Set(0, h.x, (uint16_t*) v, c);
}
}

}
