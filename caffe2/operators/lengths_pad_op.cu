#include "caffe2/operators/lengths_pad_op.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(LengthsPad, LengthsPadOp<CUDAContext>);
} // namespace caffe2
