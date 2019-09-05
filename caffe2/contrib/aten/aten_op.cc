#include "caffe2/contrib/aten/aten_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ATen, ATenOp<CPUContext>);
template <>
at::Backend ATenOp<CPUContext>::backend() const {
  return at::Backend::CPU;
}

OPERATOR_SCHEMA(ATen);

namespace math {

template <>
void Set<at::Half, CPUContext>(
    const std::int64_t /* N */,
    const at::Half h,
    at::Half* v,
    CPUContext* c) {
  Set(0, h.x, (uint16_t*)v, c);
}

template <>
void Set<at::BFloat16, CPUContext>(
    const std::int64_t /* N */,
    const at::BFloat16 b,
    at::BFloat16* v,
    CPUContext* c) {
  Set(0, b.x, (uint16_t*)v, c);
}


} // namespace math

} // namespace caffe2
