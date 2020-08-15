#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/dropout_op.h"

namespace caffe2 {

namespace {
__global__ void DropoutKernel(
    const int N,
    const float ratio,
    const float* Xdata,
    float* Ydata,
    bool* maskdata) {
  const float scale = 1. / (1. - ratio);
  CUDA_1D_KERNEL_LOOP(i, N) {
    maskdata[i] = (Ydata[i] > ratio);
    Ydata[i] = Xdata[i] * scale * maskdata[i];
  }
}
} // namespace

template <>
bool DropoutOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());
  if (is_test_) {
    if (Y != &X) {
      context_.CopySameDevice<float>(
          X.numel(), X.data<float>(), Y->template mutable_data<float>());
    }
    return true;
  } else {
    // We do a simple trick here: since curand cannot generate random
    // boolean numbers, we will generate into dY and write the result to
    // mask.
    float* Ydata = Y->template mutable_data<float>();
    auto* mask = Output(1, X.sizes(), at::dtype<bool>());
    CAFFE_ENFORCE(X.data<float>() != Ydata, "In-place GPU dropout is broken");
    CURAND_ENFORCE(
        curandGenerateUniform(context_.curand_generator(), Ydata, X.numel()));
    DropoutKernel<<<
        CAFFE_GET_BLOCKS(X.numel()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        X.numel(),
        ratio_,
        X.data<float>(),
        Ydata,
        mask->template mutable_data<bool>());
    return true;
  }
}

namespace {
__global__ void DropoutGradientKernel(
    const int N,
    const float* dYdata,
    const bool* maskdata,
    const float scale,
    float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dXdata[i] = dYdata[i] * maskdata[i] * scale;
  }
}
} // namespace

template <>
bool DropoutGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto* dX = Output(0, dY.sizes(), at::dtype<float>());
  if (is_test_) {
    if (dX != &dY) {
      context_.CopySameDevice<float>(
          dY.numel(), dY.data<float>(), dX->template mutable_data<float>());
    }
    return true;
  } else {
    auto& mask = Input(1);
    CAFFE_ENFORCE_EQ(dY.numel(), mask.numel());
    const float scale = 1. / (1. - ratio_);
    DropoutGradientKernel<<<
        CAFFE_GET_BLOCKS(dY.numel()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        dY.numel(),
        dY.data<float>(),
        mask.data<bool>(),
        scale,
        dX->template mutable_data<float>());
    return true;
  }
}

REGISTER_CUDA_OPERATOR(Dropout, DropoutOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(DropoutGrad, DropoutGradientOp<float, CUDAContext>);
} // namespace caffe2
