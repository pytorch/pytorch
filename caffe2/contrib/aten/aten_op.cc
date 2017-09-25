#include "aten_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(ATen, ATenOp<CPUContext>);
template<>
at::Backend ATenOp<CPUContext>::backend() const {
  return at::kCPU;
}

OPERATOR_SCHEMA(ATen);
CAFFE_KNOWN_TYPE(at::Half);

namespace math {
template<>
void Set<at::Half,CPUContext>(TIndex N, at::Half h, at::Half* v, CPUContext * c) {
  Set(0, h.x, (uint16_t*) v, c);
}
}

}
