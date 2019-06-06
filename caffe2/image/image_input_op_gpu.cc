#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/image/image_input_op.h"

namespace caffe2 {

template <>
bool ImageInputOp<CUDAContext>::ApplyTransformOnGPU(
    const std::vector<std::int64_t>& dims,
    const c10::Device& type) {
  // GPU transform kernel allows explicitly setting output type
  if (output_type_ == TensorProto_DataType_FLOAT) {
    auto* image_output =
        OperatorBase::OutputTensor(0, dims, at::dtype<float>().device(type));
    TransformOnGPU<uint8_t, float, CUDAContext>(
        prefetched_image_on_device_,
        image_output,
        mean_gpu_,
        std_gpu_,
        &context_);
  } else if (output_type_ == TensorProto_DataType_FLOAT16) {
    auto* image_output =
        OperatorBase::OutputTensor(0, dims, at::dtype<at::Half>().device(type));
    TransformOnGPU<uint8_t, at::Half, CUDAContext>(
        prefetched_image_on_device_,
        image_output,
        mean_gpu_,
        std_gpu_,
        &context_);
  } else {
    return false;
  }
  return true;
}

REGISTER_CUDA_OPERATOR(ImageInput, ImageInputOp<CUDAContext>);

} // namespace caffe2
