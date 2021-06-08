#include "caffe2/operators/unsafe_coalesce.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(UnsafeCoalesce, UnsafeCoalesceOp<CUDAContext>);

}
