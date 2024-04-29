#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/im2col_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(Im2Col, Im2ColOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(Col2Im, Col2ImOp<float, CUDAContext>);

} // namespace caffe2
