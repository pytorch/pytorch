#include <cfloat>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/instance_norm_op.h"

namespace caffe2 {

namespace {
__global__ void InstanceNormMeanKernel(
    int N,
    int C,
    int dim,
    int N_stride,
    int C_stride,
    int dim_stride,
    const float* input_data,
    float* mean_data) {
  CUDA_1D_KERNEL_LOOP(i, N * C) {
    const auto n = i / C;
    const auto c = i % C;
    mean_data[i] = 0;
    auto input_offset = input_data + n * N_stride + c * C_stride;
    for (int j = 0; j < dim; ++j) {
      mean_data[i] += *input_offset;
      input_offset += dim_stride;
    }
    mean_data[i] /= dim;
  }
}

__global__ void InstanceNormInvStdevKernel(
    int N,
    int C,
    int dim,
    int N_stride,
    int C_stride,
    int dim_stride,
    float epsilon,
    const float* input_data,
    const float* mean_data,
    float* inv_stdev_data) {
  CUDA_1D_KERNEL_LOOP(i, N * C) {
    const auto n = i / C;
    const auto c = i % C;
    inv_stdev_data[i] = 0;
    auto input_offset = input_data + n * N_stride + c * C_stride;
    for (int j = 0; j < dim; ++j) {
      float diff = *input_offset - mean_data[i];
      inv_stdev_data[i] += diff * diff;
      input_offset += dim_stride;
    }
    inv_stdev_data[i] /= dim;
    inv_stdev_data[i] += epsilon;
    inv_stdev_data[i] = 1.0 / std::sqrt(inv_stdev_data[i]);
  }
}

__global__ void InstanceNormKernel(
    int N,
    int C,
    int dim,
    int N_stride,
    int C_stride,
    int dim_stride,
    const float* input_data,
    const float* scale_data,
    const float* bias_data,
    const float* mean_data,
    const float* inv_stdev_data,
    float* output_data) {
  CUDA_1D_KERNEL_LOOP(i, N * C * dim) {
    auto index = i;
    const auto j = index % dim;
    index /= dim;
    const auto c = index % C;
    index /= C;
    const auto n = index;

    index = n * N_stride + c * C_stride + j * dim_stride;

    const auto stat_idx = n * C + c;

    output_data[index] = (input_data[index] - mean_data[stat_idx]) *
            inv_stdev_data[stat_idx] * scale_data[c] +
        bias_data[c];
  }
}

__global__ void InstanceNormGradientKernel(
    int N,
    int C,
    int dim,
    int N_stride,
    int C_stride,
    int dim_stride,
    const float* input_data,
    const float* scale_data,
    const float* bias_data,
    const float* output_grad_data,
    const float* mean_data,
    const float* inv_stdev_data,
    float* input_grad_data) {
  CUDA_1D_KERNEL_LOOP(i, N * C) {
    const auto n = i / C;
    const auto c = i % C;

    auto input_grad_offset = input_grad_data + n * N_stride + c * C_stride;
    auto input_offset = input_data + n * N_stride + c * C_stride;
    for (int j = 0; j < dim; ++j) {
      *input_grad_offset = *input_offset - mean_data[i];
      input_grad_offset += dim_stride;
      input_offset += dim_stride;
    }

    auto temp = 0.0;
    input_grad_offset = input_grad_data + n * N_stride + c * C_stride;
    auto output_grad_offset = output_grad_data + n * N_stride + c * C_stride;
    for (int j = 0; j < dim; ++j) {
      temp += *input_grad_offset * *output_grad_offset;
      input_grad_offset += dim_stride;
      output_grad_offset += dim_stride;
    }

    temp *= -powf(inv_stdev_data[i], 3.0) / dim;

    input_grad_offset = input_grad_data + n * N_stride + c * C_stride;
    output_grad_offset = output_grad_data + n * N_stride + c * C_stride;
    auto mean = 0.0;
    for (int j = 0; j < dim; ++j) {
      *input_grad_offset *= temp;
      *input_grad_offset += *output_grad_offset * inv_stdev_data[i];
      mean += *input_grad_offset;
      input_grad_offset += dim_stride;
      output_grad_offset += dim_stride;
    }
    mean /= dim;

    input_grad_offset = input_grad_data + n * N_stride + c * C_stride;
    for (int j = 0; j < dim; ++j) {
      *input_grad_offset -= mean;
      *input_grad_offset *= scale_data[c];
      input_grad_offset += dim_stride;
    }
  }
}

__global__ void InstanceNormScaleBiasGradientKernel(
    int N,
    int C,
    int dim,
    int N_stride,
    int C_stride,
    int dim_stride,
    const float* input_data,
    const float* mean_data,
    const float* output_grad_data,
    const float* inv_stdev_data,
    float* scale_grad_data,
    float* bias_grad_data) {
  CUDA_1D_KERNEL_LOOP(c, C) {
    scale_grad_data[c] = 0;
    bias_grad_data[c] = 0;
    auto input_offset = input_data + c * C_stride;
    auto output_grad_offset = output_grad_data + c * C_stride;
    auto mean_offset = mean_data + c;
    auto inv_stdev_offset = inv_stdev_data + c;
    for (int n = 0; n < N; ++n) {
      auto input_offset_inner = input_offset + n * N_stride;
      auto output_grad_offset_inner = output_grad_offset + n * N_stride;
      for (int i = 0; i < dim; ++i) {
        scale_grad_data[c] += (*input_offset_inner - *mean_offset) *
            *inv_stdev_offset * *output_grad_offset_inner;
        bias_grad_data[c] += *output_grad_offset_inner;
        input_offset_inner += dim_stride;
        output_grad_offset_inner += dim_stride;
      }
      mean_offset += C;
      inv_stdev_offset += C;
    }
  }
}

} // namespace

template <>
bool InstanceNormOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  const auto& input = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);
  auto output = Output(OUTPUT);
  auto mean = OutputSize() >= 2 ? Output(MEAN) : &mean_;
  auto inv_stdev = OutputSize() >= 3 ? Output(INV_STDEV) : &inv_stdev_;
  CAFFE_ENFORCE_EQ(4, input.ndim());
  const int N = input.dim32(0);
  const int H = input.dim32(1);
  const int W = input.dim32(2);
  const int C = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.ndim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.ndim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  output->ResizeLike(input);
  mean->Resize(N, C);
  inv_stdev->Resize(N, C);

  const auto input_data = input.data<float>();
  const auto scale_data = scale.data<float>();
  const auto bias_data = bias.data<float>();
  auto output_data = output->mutable_data<float>();
  auto mean_data = mean->mutable_data<float>();
  auto inv_stdev_data = inv_stdev->mutable_data<float>();

  const auto dim = H * W;
  const auto N_stride = C * H * W;
  const auto C_stride = 1;
  const auto dim_stride = C;

  InstanceNormMeanKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, C, dim, N_stride, C_stride, dim_stride, input_data, mean_data);

  InstanceNormInvStdevKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      epsilon_,
      input_data,
      mean_data,
      inv_stdev_data);

  InstanceNormKernel<<<
      CAFFE_GET_BLOCKS(N * C * H * W),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      input_data,
      scale_data,
      bias_data,
      mean_data,
      inv_stdev_data,
      output_data);

  return true;
}

template <>
bool InstanceNormOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& input = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);
  auto output = Output(OUTPUT);
  auto mean = OutputSize() >= 2 ? Output(MEAN) : &mean_;
  auto inv_stdev = OutputSize() >= 3 ? Output(INV_STDEV) : &inv_stdev_;
  CAFFE_ENFORCE_EQ(4, input.ndim());
  const int N = input.dim32(0);
  const int C = input.dim32(1);
  const int H = input.dim32(2);
  const int W = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.ndim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.ndim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  output->ResizeLike(input);
  mean->Resize(N, C);
  inv_stdev->Resize(N, C);

  const auto input_data = input.data<float>();
  const auto scale_data = scale.data<float>();
  const auto bias_data = bias.data<float>();
  auto output_data = output->mutable_data<float>();
  auto mean_data = mean->mutable_data<float>();
  auto inv_stdev_data = inv_stdev->mutable_data<float>();

  const auto dim = H * W;
  const auto N_stride = C * H * W;
  const auto C_stride = H * W;
  const auto dim_stride = 1;

  InstanceNormMeanKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, C, dim, N_stride, C_stride, dim_stride, input_data, mean_data);

  InstanceNormInvStdevKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      epsilon_,
      input_data,
      mean_data,
      inv_stdev_data);

  InstanceNormKernel<<<
      CAFFE_GET_BLOCKS(N * C * H * W),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      input_data,
      scale_data,
      bias_data,
      mean_data,
      inv_stdev_data,
      output_data);

  return true;
}

template <>
bool InstanceNormGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC() {
  const auto& input = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);
  const auto& output_grad = Input(OUTPUT_GRAD);
  const auto& mean = InputSize() >= 5 ? Input(MEAN) : mean_;
  const auto& inv_stdev = InputSize() >= 6 ? Input(INV_STDEV) : inv_stdev_;
  auto input_grad = Output(INPUT_GRAD);
  auto scale_grad = Output(SCALE_GRAD);
  auto bias_grad = Output(BIAS_GRAD);
  CAFFE_ENFORCE_EQ(4, input.ndim());
  const int N = input.dim32(0);
  const int H = input.dim32(1);
  const int W = input.dim32(2);
  const int C = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.ndim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.ndim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.ndim());
  CAFFE_ENFORCE_EQ(N, output_grad.dim32(0));
  CAFFE_ENFORCE_EQ(H, output_grad.dim32(1));
  CAFFE_ENFORCE_EQ(W, output_grad.dim32(2));
  CAFFE_ENFORCE_EQ(C, output_grad.dim32(3));
  input_grad->ResizeLike(input);
  scale_grad->ResizeLike(scale);
  bias_grad->ResizeLike(bias);

  const auto input_data = input.data<float>();
  const auto scale_data = scale.data<float>();
  const auto bias_data = bias.data<float>();
  const auto output_grad_data = output_grad.data<float>();

  auto input_grad_data = input_grad->mutable_data<float>();
  auto scale_grad_data = scale_grad->mutable_data<float>();
  auto bias_grad_data = bias_grad->mutable_data<float>();

  const auto dim = H * W;
  const auto N_stride = C * H * W;
  const auto C_stride = 1;
  const auto dim_stride = C;

  if (InputSize() < 5) {
    mean_.Resize(N, C);
    auto mean_mutable_data = mean_.mutable_data<float>();
    InstanceNormMeanKernel<<<
        CAFFE_GET_BLOCKS(N * C),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        C,
        dim,
        N_stride,
        C_stride,
        dim_stride,
        input_data,
        mean_mutable_data);
  }
  CAFFE_ENFORCE_EQ(2, mean.ndim());
  CAFFE_ENFORCE_EQ(N, mean.dim32(0));
  CAFFE_ENFORCE_EQ(C, mean.dim32(1));

  const auto mean_data = mean.data<float>();

  if (InputSize() < 6) {
    inv_stdev_.Resize(N, C);
    auto inv_stdev_mutable_data = inv_stdev_.mutable_data<float>();
    InstanceNormInvStdevKernel<<<
        CAFFE_GET_BLOCKS(N * C),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        C,
        dim,
        N_stride,
        C_stride,
        dim_stride,
        epsilon_,
        input_data,
        mean_data,
        inv_stdev_mutable_data);
  }
  CAFFE_ENFORCE_EQ(2, inv_stdev.ndim());
  CAFFE_ENFORCE_EQ(N, inv_stdev.dim32(0));
  CAFFE_ENFORCE_EQ(C, inv_stdev.dim32(1));

  const auto inv_stdev_data = inv_stdev.data<float>();

  InstanceNormScaleBiasGradientKernel<<<
      CAFFE_GET_BLOCKS(C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      input_data,
      mean_data,
      output_grad_data,
      inv_stdev_data,
      scale_grad_data,
      bias_grad_data);

  InstanceNormGradientKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      input_data,
      scale_data,
      bias_data,
      output_grad_data,
      mean_data,
      inv_stdev_data,
      input_grad_data);

  return true;
}

template <>
bool InstanceNormGradientOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW() {
  const auto& input = Input(INPUT);
  const auto& scale = Input(SCALE);
  const auto& bias = Input(BIAS);
  const auto& output_grad = Input(OUTPUT_GRAD);
  const auto& mean = InputSize() >= 5 ? Input(MEAN) : mean_;
  const auto& inv_stdev = InputSize() >= 6 ? Input(INV_STDEV) : inv_stdev_;
  auto input_grad = Output(INPUT_GRAD);
  auto scale_grad = Output(SCALE_GRAD);
  auto bias_grad = Output(BIAS_GRAD);
  CAFFE_ENFORCE_EQ(4, input.ndim());
  const int N = input.dim32(0);
  const int C = input.dim32(1);
  const int H = input.dim32(2);
  const int W = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.ndim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.ndim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.ndim());
  CAFFE_ENFORCE_EQ(N, output_grad.dim32(0));
  CAFFE_ENFORCE_EQ(C, output_grad.dim32(1));
  CAFFE_ENFORCE_EQ(H, output_grad.dim32(2));
  CAFFE_ENFORCE_EQ(W, output_grad.dim32(3));
  input_grad->ResizeLike(input);
  scale_grad->ResizeLike(scale);
  bias_grad->ResizeLike(bias);

  const auto input_data = input.data<float>();
  const auto scale_data = scale.data<float>();
  const auto bias_data = bias.data<float>();
  const auto output_grad_data = output_grad.data<float>();

  auto input_grad_data = input_grad->mutable_data<float>();
  auto scale_grad_data = scale_grad->mutable_data<float>();
  auto bias_grad_data = bias_grad->mutable_data<float>();

  const auto dim = H * W;
  const auto N_stride = C * H * W;
  const auto C_stride = H * W;
  const auto dim_stride = 1;

  if (InputSize() < 5) {
    mean_.Resize(N, C);
    auto mean_mutable_data = mean_.mutable_data<float>();
    InstanceNormMeanKernel<<<
        CAFFE_GET_BLOCKS(N * C),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        C,
        dim,
        N_stride,
        C_stride,
        dim_stride,
        input_data,
        mean_mutable_data);
  }
  CAFFE_ENFORCE_EQ(2, mean.ndim());
  CAFFE_ENFORCE_EQ(N, mean.dim32(0));
  CAFFE_ENFORCE_EQ(C, mean.dim32(1));

  const auto mean_data = mean.data<float>();

  if (InputSize() < 6) {
    inv_stdev_.Resize(N, C);
    auto inv_stdev_mutable_data = inv_stdev_.mutable_data<float>();
    InstanceNormInvStdevKernel<<<
        CAFFE_GET_BLOCKS(N * C),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        C,
        dim,
        N_stride,
        C_stride,
        dim_stride,
        epsilon_,
        input_data,
        mean_data,
        inv_stdev_mutable_data);
  }
  CAFFE_ENFORCE_EQ(2, inv_stdev.ndim());
  CAFFE_ENFORCE_EQ(N, inv_stdev.dim32(0));
  CAFFE_ENFORCE_EQ(C, inv_stdev.dim32(1));

  const auto inv_stdev_data = inv_stdev.data<float>();

  InstanceNormScaleBiasGradientKernel<<<
      CAFFE_GET_BLOCKS(C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      input_data,
      mean_data,
      output_grad_data,
      inv_stdev_data,
      scale_grad_data,
      bias_grad_data);

  InstanceNormGradientKernel<<<
      CAFFE_GET_BLOCKS(N * C),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      C,
      dim,
      N_stride,
      C_stride,
      dim_stride,
      input_data,
      scale_data,
      bias_data,
      output_grad_data,
      mean_data,
      inv_stdev_data,
      input_grad_data);
  return true;
}

REGISTER_CUDA_OPERATOR(InstanceNorm, InstanceNormOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    InstanceNormGradient,
    InstanceNormGradientOp<float, CUDAContext>);
} // namespace caffe2
