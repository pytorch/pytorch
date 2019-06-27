#include "caffe2/operators/instance_norm_op.h"

#include "caffe2/core/context_gpu.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

template <typename T>
__global__ void ComputeFusedParamsCUDAKernel(
    const int64_t N,
    const int64_t C,
    const T* mean,
    const T* rstd,
    const T* gamma,
    const T* beta,
    T* scale,
    T* bias) {
  const int64_t index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C) {
    const int64_t c = index % C;
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    const T scale_val = __ldg(gamma + c) * __ldg(rstd + index);
    scale[index] = scale_val;
    bias[index] = __ldg(beta + c) - scale_val * __ldg(mean + index);
#else
    const T scale_val = gamma[c] * rstd[index];
    scale[index] = scale_val;
    bias[index] = beta[c] - scale_val * mean[index];
#endif
  }
}

template <typename T, StorageOrder kOrder>
__global__ void InstanceNormForwardCUDAKernel(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const T* X,
    const T* scale,
    const T* bias,
    T* Y) {
  const int64_t index = blockIdx.x * CAFFE_CUDA_NUM_THREADS + threadIdx.x;
  if (index < N * C * HxW) {
    const int nc = kOrder == StorageOrder::NCHW
        ? (index / HxW)
        : (index / (HxW * C) * C + index % C);
#if __CUDA_ARCH__ >= 350 || defined(__HIP_PLATFORM_HCC__)
    Y[index] = __ldg(scale + nc) * __ldg(X + index) + __ldg(bias + nc);
#else
    Y[index] = scale[nc] * X[index] + bias[nc];
#endif
  }
}

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
    inv_stdev_data[i] = 1.0 / sqrtf(inv_stdev_data[i]);
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
bool InstanceNormOp<float, CUDAContext>::RunOnDeviceWithOrderNCHW(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    float* mean,
    float* rstd) {
  ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CUDA));
  float* scale_data = scale_.template mutable_data<float>();
  float* bias_data = bias_.template mutable_data<float>();
  const std::array<int, 2> X_dims = {static_cast<int>(N * C),
                                     static_cast<int>(HxW)};
  const std::array<int, 2> Y_dims = {static_cast<int>(N * C), 1};
  math::Moments<float, CUDAContext>(
      2, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
  math::InvStd<float, CUDAContext>(
      static_cast<int>(N * C),
      static_cast<float>(epsilon_),
      rstd,
      rstd,
      &context_);
  int64_t B = math::DivUp<int64_t>(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamsCUDAKernel<float>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, mean, rstd, gamma, beta, scale_data, bias_data);
  B = math::DivUp<int64_t>(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  InstanceNormForwardCUDAKernel<float, StorageOrder::NCHW>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, X, scale_data, bias_data, Y);
  return true;
}

template <>
bool InstanceNormOp<float, CUDAContext>::RunOnDeviceWithOrderNHWC(
    const int64_t N,
    const int64_t C,
    const int64_t HxW,
    const float* X,
    const float* gamma,
    const float* beta,
    float* Y,
    float* mean,
    float* rstd) {
  ReinitializeTensor(&scale_, {N, C}, at::dtype<float>().device(CUDA));
  ReinitializeTensor(&bias_, {N, C}, at::dtype<float>().device(CUDA));
  float* scale_data = scale_.template mutable_data<float>();
  float* bias_data = bias_.template mutable_data<float>();
  const std::array<int, 3> X_dims = {
      static_cast<int>(N), static_cast<int>(HxW), static_cast<int>(C)};
  const std::array<int, 3> Y_dims = {
      static_cast<int>(N), 1, static_cast<int>(C)};
  math::Moments<float, CUDAContext>(
      3, X_dims.data(), Y_dims.data(), X, mean, rstd, &context_);
  math::InvStd<float, CUDAContext>(
      static_cast<int>(N * C),
      static_cast<float>(epsilon_),
      rstd,
      rstd,
      &context_);
  int64_t B = math::DivUp<int64_t>(N * C, CAFFE_CUDA_NUM_THREADS);
  ComputeFusedParamsCUDAKernel<float>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, mean, rstd, gamma, beta, scale_data, bias_data);
  B = math::DivUp<int64_t>(N * C * HxW, CAFFE_CUDA_NUM_THREADS);
  InstanceNormForwardCUDAKernel<float, StorageOrder::NHWC>
      <<<B, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
          N, C, HxW, X, scale_data, bias_data, Y);
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

  CAFFE_ENFORCE_EQ(4, input.dim());
  const int N = input.dim32(0);
  const int H = input.dim32(1);
  const int W = input.dim32(2);
  const int C = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.dim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.dim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.dim());
  CAFFE_ENFORCE_EQ(N, output_grad.dim32(0));
  CAFFE_ENFORCE_EQ(H, output_grad.dim32(1));
  CAFFE_ENFORCE_EQ(W, output_grad.dim32(2));
  CAFFE_ENFORCE_EQ(C, output_grad.dim32(3));
  auto input_grad = Output(INPUT_GRAD, input.sizes(), at::dtype<float>());
  auto scale_grad = Output(SCALE_GRAD, scale.sizes(), at::dtype<float>());
  auto bias_grad = Output(BIAS_GRAD, bias.sizes(), at::dtype<float>());

  const auto input_data = input.data<float>();
  const auto scale_data = scale.data<float>();
  const auto bias_data = bias.data<float>();
  const auto output_grad_data = output_grad.data<float>();

  auto input_grad_data = input_grad->template mutable_data<float>();
  auto scale_grad_data = scale_grad->template mutable_data<float>();
  auto bias_grad_data = bias_grad->template mutable_data<float>();

  const auto dim = H * W;
  const auto N_stride = C * H * W;
  const auto C_stride = 1;
  const auto dim_stride = C;

  if (InputSize() < 5) {
    ReinitializeTensor(&mean_, {N, C}, at::dtype<float>().device(CUDA));
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
  CAFFE_ENFORCE_EQ(2, mean.dim());
  CAFFE_ENFORCE_EQ(N, mean.dim32(0));
  CAFFE_ENFORCE_EQ(C, mean.dim32(1));

  const auto mean_data = mean.data<float>();

  if (InputSize() < 6) {
    ReinitializeTensor(&inv_stdev_, {N, C}, at::dtype<float>().device(CUDA));
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
  CAFFE_ENFORCE_EQ(2, inv_stdev.dim());
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
  CAFFE_ENFORCE_EQ(4, input.dim());
  const int N = input.dim32(0);
  const int C = input.dim32(1);
  const int H = input.dim32(2);
  const int W = input.dim32(3);
  CAFFE_ENFORCE_EQ(1, scale.dim());
  CAFFE_ENFORCE_EQ(C, scale.dim32(0));
  CAFFE_ENFORCE_EQ(1, bias.dim());
  CAFFE_ENFORCE_EQ(C, bias.dim32(0));
  CAFFE_ENFORCE_EQ(4, output_grad.dim());
  CAFFE_ENFORCE_EQ(N, output_grad.dim32(0));
  CAFFE_ENFORCE_EQ(C, output_grad.dim32(1));
  CAFFE_ENFORCE_EQ(H, output_grad.dim32(2));
  CAFFE_ENFORCE_EQ(W, output_grad.dim32(3));
  auto input_grad = Output(INPUT_GRAD, input.sizes(), at::dtype<float>());
  auto scale_grad = Output(SCALE_GRAD, scale.sizes(), at::dtype<float>());
  auto bias_grad = Output(BIAS_GRAD, bias.sizes(), at::dtype<float>());

  const auto input_data = input.data<float>();
  const auto scale_data = scale.data<float>();
  const auto bias_data = bias.data<float>();
  const auto output_grad_data = output_grad.data<float>();

  auto input_grad_data = input_grad->template mutable_data<float>();
  auto scale_grad_data = scale_grad->template mutable_data<float>();
  auto bias_grad_data = bias_grad->template mutable_data<float>();

  const auto dim = H * W;
  const auto N_stride = C * H * W;
  const auto C_stride = H * W;
  const auto dim_stride = 1;

  if (InputSize() < 5) {
    ReinitializeTensor(&mean_, {N, C}, at::dtype<float>().device(CUDA));
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
  CAFFE_ENFORCE_EQ(2, mean.dim());
  CAFFE_ENFORCE_EQ(N, mean.dim32(0));
  CAFFE_ENFORCE_EQ(C, mean.dim32(1));

  const auto mean_data = mean.data<float>();

  if (InputSize() < 6) {
    ReinitializeTensor(&inv_stdev_, {N, C}, at::dtype<float>().device(CUDA));
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
  CAFFE_ENFORCE_EQ(2, inv_stdev.dim());
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
