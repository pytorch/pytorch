#include <caffe2/core/common_gpu.h>
#include <caffe2/core/context_gpu.h>
#include <caffe2/video/video_input_op.h>

namespace caffe2 {

REGISTER_CUDA_OPERATOR(VideoInput, VideoInputOp<CUDAContext>);

} // namespace caffe2
