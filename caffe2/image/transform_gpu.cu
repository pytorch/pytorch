#include "caffe2/core/context_gpu.h"
#include "caffe2/image/transform_gpu.h"
#include "caffe2/utils/conversions.h"

/**
 *
 * Copyright (c) 2016, NVIDIA CORPORATION, All rights reserved
 * Distributed under 2-clause BSD license; see accompanying LICENSE file
 *
 **/

namespace caffe2 {

namespace {

// input in (int8, NHWC), output in (fp32, NCHW)
template <typename In, typename Out>
__global__ void transform_kernel(
    const int N,
    const int C,
    const int H,
    const int W,
    const float* mean,
    const float* std,
    const In* in,
    Out* out) {
  const int n = blockIdx.x;

  const int nStride = C*H*W;

  // pointers to data for this image
  const In* input_ptr = &in[n*nStride];
  Out* output_ptr = &out[n*nStride];

  // either read or write uncoalesced - try reading
  for (int c=0; c < C; ++c) {
    for (int h=threadIdx.y; h < H; h += blockDim.y) {
      for (int w=threadIdx.x; w < W; w += blockDim.x) {
        int in_idx = c + C*w + C*W*h;  // HWC
        int out_idx = c*H*W + h*W + w;  // CHW

        output_ptr[out_idx] = convert::To<float,Out>(
          (convert::To<In,float>(input_ptr[in_idx])-mean[c]) * std[c]);
      }
    }
  }
}

}

template <typename T_IN, typename T_OUT, class Context>

bool TransformOnGPU(
    Tensor& X,
    Tensor* Y,
    Tensor& mean,
    Tensor& std,
    Context* context) {
  const int N = X.dim32(0), C = X.dim32(3), H = X.dim32(1), W = X.dim32(2);
  auto* input_data = X.template data<T_IN>();
  auto* output_data = Y->template mutable_data<T_OUT>();

  transform_kernel<
    T_IN, T_OUT><<<N, dim3(16, 16), 0, context->cuda_stream()>>>(
      N, C, H, W, mean.template data<float>(), std.template data<float>(),
      input_data, output_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
};

template bool TransformOnGPU<uint8_t, float, CUDAContext>(
    Tensor& X,
    Tensor* Y,
    Tensor& mean,
    Tensor& std,
    CUDAContext* context);

template bool TransformOnGPU<uint8_t, at::Half, CUDAContext>(
    Tensor& X,
    Tensor* Y,
    Tensor& mean,
    Tensor& std,
    CUDAContext* context);

}  // namespace caffe2
