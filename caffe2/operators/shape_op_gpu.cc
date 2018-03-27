#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/shape_op.h"

namespace caffe2 {
REGISTER_CUDA_OPERATOR(Shape, ShapeOp<CUDAContext>);
}
