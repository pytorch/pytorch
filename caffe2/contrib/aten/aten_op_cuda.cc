#include "caffe2/contrib/aten/aten_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(ATen, ATenOp<CUDAContext>);
template<>
at::Backend ATenOp<CUDAContext>::backend() const {
  return at::Backend::CUDA;
}

}
