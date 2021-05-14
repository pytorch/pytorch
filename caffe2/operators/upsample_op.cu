// copied and pasted from pytorch to test if this passes the build.

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/upsample_op.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

inline __device__ int idx(
    const int n,
    const int num_channels,
    const int c,
    const int height,
    const int width,
    const int y,
    const int x) {
  return ((n * num_channels + c) * height + y) * width + x;
}

// input is X, output is Y
__global__ void UpsampleBilinearKernel(
    const int num_batch,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* __restrict__ X,
    float* __restrict__ Y) {


    const int size = output_height * output_width;
    CUDA_1D_KERNEL_LOOP(index, size) {
    int indexTemp = index;
    const int out_x = indexTemp % output_width;
    indexTemp /= output_width;
    const int out_y = indexTemp % output_height;
    indexTemp /= output_height;
    indexTemp /= num_channels;

    const float rheight =
         output_height > 1 ? (input_height - 1.f) / (output_height - 1.f) : 0.f;
    const float rwidth =
         output_width > 1 ? (input_width - 1.f) / (output_width - 1.f) : 0.f;

    // Compute Y axis lambdas
    const float h1r = rheight * out_y;
    const int h1 = (int)h1r;
    const int h1p = (h1 < input_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.f - h1lambda;

    // Compute X axis lambdas
    const float w1r = rwidth * out_x;
    const int w1 = (int)w1r;
    const int w1p = (w1 < input_width - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.f - w1lambda;

    for (int n = 0; n < num_batch; n++){
      for (int c = 0; c < num_channels; c++) {

        float X0 = X[idx(n, num_channels, c, input_height, input_width, h1, w1)];
        float X1 = X[idx(n, num_channels, c, input_height, input_width, h1, w1 + w1p)];
        float X2 = X[idx(n, num_channels, c, input_height, input_width, h1 + h1p, w1)];
        float X3 = X[idx(n, num_channels, c, input_height, input_width, h1 + h1p, w1 + w1p)];

        Y[idx(n, num_channels, c, output_height, output_width, out_y, out_x)] =
                    h0lambda * (w0lambda * X0 + w1lambda * X1) +
                    h1lambda * (w0lambda * X2 + w1lambda * X3);
      }
    }
  }
}
// input is dY, output is dX
__global__ void UpsampleBilinearGradientKernel(
    const int input_size,
    const int num_channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const float height_scale,
    const float width_scale,
    const float* dY,
    float* dX) {
  CUDA_1D_KERNEL_LOOP(index, input_size) {
    int indexTemp = index;
    const int in_x = indexTemp % input_width;
    indexTemp /= input_width;
    const int in_y = indexTemp % input_height;
    indexTemp /= input_height;
    const int c = indexTemp % num_channels;
    indexTemp /= num_channels;
    const int n = indexTemp;

    const int out_y = fminf(in_y / height_scale, output_height - 1);
    const int out_x = fminf(in_x / width_scale, output_width - 1);

    const float rheight =
        output_height > 1 ? (output_height - 1.f) / (input_height - 1.f) : 0.f;
    const float rwidth =
        output_width > 1 ? (output_width - 1.f) / (input_width - 1.f) : 0.f;

    // Compute Y axis lambdas
    const float h1r = rheight * in_y;
    const int h1 = (int)h1r;
    const int h1p = (h1 < output_height - 1) ? 1 : 0;
    const float h1lambda = h1r - h1;
    const float h0lambda = 1.f - h1lambda;

    // Compute X axis lambdas
    const float w1r = rwidth * in_x;
    const int w1 = (int)w1r;
    const int w1p = (w1 < output_width - 1) ? 1 : 0;
    const float w1lambda = w1r - w1;
    const float w0lambda = 1.f - w1lambda;

#if __CUDA_ARCH__ >= 350
    const float dYi = __ldg(&dY[index]);
#else
    const float dYi = dY[index];
#endif

    atomicAdd(
        &dX[idx(n, num_channels, c, output_height, output_width, h1, w1)],
        h0lambda * w0lambda * dYi);
    atomicAdd(
        &dX[idx(n, num_channels, c, output_height, output_width, h1, w1 + w1p)],
        h0lambda * w1lambda * dYi);
    atomicAdd(
        &dX[idx(n, num_channels, c, output_height, output_width, h1 + h1p, w1)],
        h1lambda * w0lambda * dYi);
    atomicAdd(
        &dX[idx(
            n,
            num_channels,
            c,
            output_height,
            output_width,
            h1 + h1p,
            w1 + w1p)],
        h1lambda * w1lambda * dYi);
  }
}

} // namespace

template <>
bool UpsampleBilinearOp<float, CUDAContext>::RunOnDevice() {
  const auto& X = Input(0);

  const auto inputDims = X.sizes();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = X.dim32(0), num_channels = X.dim32(1),
            input_height = X.dim32(2), input_width = X.dim32(3);
  if (InputSize() == 2) {
    const auto& scales = Input(1);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    float scales_data[2];
    context_.CopyToCPU<float>(2, scales.data<float>(), scales_data);
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  int output_width = input_width * width_scale_;
  int output_height = input_height * height_scale_;
  auto* Y = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());

  const auto size = output_height * output_width;
  UpsampleBilinearKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      batch_size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      X.data<float>(),
      Y->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}
template <>
bool UpsampleBilinearGradientOp<float, CUDAContext>::RunOnDevice() {
  const auto& dY = Input(0);
  const auto& X = Input(1);
  const auto inputDims = dY.sizes();
  CAFFE_ENFORCE_EQ(4, inputDims.size());
  const int batch_size = dY.dim32(0);
  const int num_channels = dY.dim32(1);
  const int input_height = dY.dim32(2);
  const int input_width = dY.dim32(3);
  const int output_height = X.dim32(2);
  const int output_width = X.dim32(3);
  if (InputSize() == 3) {
    const auto& scales = Input(2);
    CAFFE_ENFORCE_EQ(scales.dim(), 1);
    CAFFE_ENFORCE_EQ(scales.numel(), 2);
    float scales_data[2];
    context_.CopyToCPU<float>(2, scales.data<float>(), scales_data);
    height_scale_ = scales_data[0];
    width_scale_ = scales_data[1];
  }
  auto* dX = Output(
      0,
      {batch_size, num_channels, output_height, output_width},
      at::dtype<float>());
  math::Set<float, CUDAContext>(
      dX->numel(), 0.0f, dX->mutable_data<float>(), &context_);
  const auto size = dY.numel();
  UpsampleBilinearGradientKernel<<<
      CAFFE_GET_BLOCKS(size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      size,
      num_channels,
      input_height,
      input_width,
      output_height,
      output_width,
      height_scale_,
      width_scale_,
      dY.data<float>(),
      dX->template mutable_data<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}
REGISTER_CUDA_OPERATOR(
    UpsampleBilinear,
    UpsampleBilinearOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    UpsampleBilinearGradient,
    UpsampleBilinearGradientOp<float, CUDAContext>);
} // namespace caffe2
