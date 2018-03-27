#include "caffe2/contrib/aten/aten_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(ATen, ATenOp<CUDAContext>);
template<>
at::Backend ATenOp<CUDAContext>::backend() const {
  return at::kCUDA;
}

namespace math {
template<>
void Set<at::Half,CUDAContext>(const size_t N, const at::Half h, at::Half* v, CUDAContext * c) {
  Set(0, h.x, (uint16_t*) v, c);
}
}

}
