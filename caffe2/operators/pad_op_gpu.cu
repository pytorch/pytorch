#include <algorithm>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/pad_op.h"

namespace caffe2 {

template <>
bool PadImageOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  CAFFE_NOT_IMPLEMENTED;
}

template<>
bool PadImageOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  CAFFE_NOT_IMPLEMENTED;
}

template<>
bool PadImageGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  CAFFE_NOT_IMPLEMENTED;
}

template<>
bool PadImageGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  CAFFE_NOT_IMPLEMENTED;
}


REGISTER_CUDA_OPERATOR(PadImage, PadImageOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(PadImageGradient,
                       PadImageGradientOp<float, CUDAContext>);
}  // namespace caffe2
