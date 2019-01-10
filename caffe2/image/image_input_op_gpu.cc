#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/image/image_input_op.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(ImageInput, ImageInputOp<CUDAContext>);

}  // namespace caffe2
