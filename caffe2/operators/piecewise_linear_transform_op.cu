#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/piecewise_linear_transform_op.h"

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

namespace caffe2 {

namespace {
__global__ void PieceWiseLinearTransformGeneralKernel(
    const int N,
    const int M,
    const int num_grp,
    const int num_fnc_per_grp,
    const float* bounds,
    const float* slopes,
    const float* intercepts,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N * M) {
    int col = i % M;
    const float* bounds_group = bounds + (col * (num_fnc_per_grp + 1));
    const float* slopes_group = slopes + (col * num_fnc_per_grp);
    const float* intercepts_group = intercepts + (col * num_fnc_per_grp);

    if (X[i] <= bounds_group[0]) {
      Y[i] = slopes_group[0] * bounds_group[0] + intercepts_group[0];
    } else if (X[i] >= bounds_group[num_fnc_per_grp]) {
      Y[i] = slopes_group[num_fnc_per_grp - 1] * bounds_group[num_fnc_per_grp] +
          intercepts_group[num_fnc_per_grp - 1];
    } else {
      auto low_bound = thrust::lower_bound(
          thrust::device,
          bounds_group,
          bounds_group + num_fnc_per_grp + 1,
          X[i]);
      int bounds_idx = low_bound - bounds_group - 1;
      Y[i] = slopes_group[bounds_idx] * X[i] + intercepts_group[bounds_idx];
    }
  }
}

} // namespace

namespace {
__global__ void PieceWiseLinearTransformBinaryKernel1(
    const int N,
    const int M,
    const int num_grp,
    const int num_fnc_per_grp,
    const float* bounds,
    const float* slopes,
    const float* intercepts,
    const float* X,
    float* Y) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    if (X[i] <= bounds[0]) {
      Y[i] = slopes[0] * bounds[0] + intercepts[0];
    } else if (X[i] >= bounds[num_fnc_per_grp]) {
      Y[i] = slopes[num_fnc_per_grp - 1] * bounds[num_fnc_per_grp] +
          intercepts[num_fnc_per_grp - 1];
    } else {
      auto low_bound = thrust::lower_bound(
          thrust::device, bounds, bounds + num_fnc_per_grp + 1, X[i]);
      int bounds_idx = low_bound - bounds - 1;
      Y[i] = slopes[bounds_idx] * X[i] + intercepts[bounds_idx];
    }
  }
}
} // namespace

namespace {
__global__ void PieceWiseLinearTransformBinaryKernel2(
    const int N,
    const int M,
    const int num_grp,
    const int num_fnc_per_grp,
    const float* bounds,
    const float* slopes,
    const float* intercepts,
    const float* X,
    float* Y) {
  // N*M/2 = N as M=2
  CUDA_1D_KERNEL_LOOP(i, N) {
    int index = i * M;
    if (X[index + 1] <= bounds[0]) {
      Y[index + 1] = slopes[0] * bounds[0] + intercepts[0];
    } else if (X[index + 1] >= bounds[num_fnc_per_grp]) {
      Y[index + 1] = slopes[num_fnc_per_grp - 1] * bounds[num_fnc_per_grp] +
          intercepts[num_fnc_per_grp - 1];
    } else {
      auto low_bound = thrust::lower_bound(
          thrust::device, bounds, bounds + num_fnc_per_grp + 1, X[index + 1]);
      int bounds_idx = low_bound - bounds - 1;
      Y[index + 1] = slopes[bounds_idx] * X[index + 1] + intercepts[bounds_idx];
    }
    Y[index] = 1.0f - Y[index + 1];
  }
}
} // namespace

template <>
void PiecewiseLinearTransformOp<float, CUDAContext>::setUpTensors(
    int64_t& num_func_per_group,
    int64_t& num_group,
    int64_t M) {
  if (transform_param_from_arg_) {
    if (!gpu_copied_) {
      int64_t num_bounds;
      int64_t num_slopes;
      int64_t num_intercepts;

      CAFFE_ENFORCE_EQ(InputSize(), 1);

      const float* bounds;
      const float* slopes;
      const float* intercepts;
      bounds = bounds_from_arg_.data();
      slopes = slopes_from_arg_.data();
      intercepts = intercepts_from_arg_.data();
      num_bounds = bounds_from_arg_.size();
      num_slopes = slopes_from_arg_.size();
      num_intercepts = intercepts_from_arg_.size();
      InferNumFunctionsPerGroup(
          num_bounds,
          num_slopes,
          num_intercepts,
          &num_func_per_group,
          &num_group);

      if (binary_) {
        CAFFE_ENFORCE_EQ(num_group, 1);
      } else {
        CAFFE_ENFORCE_EQ(num_group, M);
      }

      int length = num_group * num_func_per_group;
      Tensor bounds_host{CPU};
      bounds_host.Resize(length + num_group);
      memcpy(
          bounds_host.mutable_data<float>(),
          bounds,
          (length + num_group) * sizeof(float));

      Tensor intercepts_host{CPU};
      intercepts_host.Resize(length);
      memcpy(
          intercepts_host.mutable_data<float>(),
          intercepts,
          (length) * sizeof(float));
      Tensor slopes_host{CPU};
      slopes_host.Resize(length);
      memcpy(
          slopes_host.mutable_data<float>(), slopes, (length) * sizeof(float));

      bounds_device_.CopyFrom(bounds_host);
      intercepts_device_.CopyFrom(intercepts_host);
      slopes_device_.CopyFrom(slopes_host);

      gpu_copied_ = true;
    }
  } else {
    int64_t num_bounds;
    int64_t num_slopes;
    int64_t num_intercepts;
    CAFFE_ENFORCE_EQ(InputSize(), 4);
    auto& bounds_input = Input(BOUNDS);
    auto& slopes_input = Input(SLOPES);
    auto& intercepts_input = Input(INTERCEPTS);
    num_bounds = bounds_input.numel();
    num_slopes = slopes_input.numel();
    num_intercepts = intercepts_input.numel();
    InferNumFunctionsPerGroup(
        num_bounds,
        num_slopes,
        num_intercepts,
        &num_func_per_group,
        &num_group);

    if (binary_) {
      CAFFE_ENFORCE_EQ(num_group, 1);
    } else {
      CAFFE_ENFORCE_EQ(num_group, M);
    }

    bounds_device_.CopyFrom(bounds_input);
    slopes_device_.CopyFrom(slopes_input);
    intercepts_device_.CopyFrom(intercepts_input);
  }
}

template <>
bool PiecewiseLinearTransformOp<float, CUDAContext>::TransformGeneral() {
  auto& X = Input(0);

  CAFFE_ENFORCE_EQ(X.dim(), 2);
  int64_t N = X.dim32(0);
  int64_t M = X.dim32(1);
  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  int64_t num_func_per_group;
  int64_t num_group;

  setUpTensors(num_func_per_group, num_group, M);

  PieceWiseLinearTransformGeneralKernel<<<
      CAFFE_GET_BLOCKS(X.numel()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      M,
      num_group,
      num_func_per_group,
      bounds_device_.data<float>(),
      slopes_device_.data<float>(),
      intercepts_device_.data<float>(),
      X.data<float>(),
      Y->template mutable_data<float>());

  return true;
}

template <>
bool PiecewiseLinearTransformOp<float, CUDAContext>::TransformBinary() {
  auto& X = Input(0);

  CAFFE_ENFORCE(X.dim() == 1 || X.dim() == 2);
  int64_t N = X.dim32(0);
  int64_t M = X.dim() == 2 ? X.dim32(1) : 1;
  CAFFE_ENFORCE(
      M == 1 || M == 2,
      "If binary is set to true, the input must be Nx2 or Nx1 tensor");
  auto* Y = Output(0, X.sizes(), at::dtype<float>());

  int64_t num_func_per_group;
  int64_t num_group;

  setUpTensors(num_func_per_group, num_group, M);

  if (M == 1) {
    PieceWiseLinearTransformBinaryKernel1<<<
        CAFFE_GET_BLOCKS(X.numel()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        M,
        num_group,
        num_func_per_group,
        bounds_device_.data<float>(),
        slopes_device_.data<float>(),
        intercepts_device_.data<float>(),
        X.data<float>(),
        Y->template mutable_data<float>());
  } else {
    // don't want N*M threads, only N*M/2
    PieceWiseLinearTransformBinaryKernel2<<<
        CAFFE_GET_BLOCKS(X.numel() / 2),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        N,
        M,
        num_group,
        num_func_per_group,
        bounds_device_.data<float>(),
        slopes_device_.data<float>(),
        intercepts_device_.data<float>(),
        X.data<float>(),
        Y->template mutable_data<float>());
  }

  return true;
}

REGISTER_CUDA_OPERATOR(
    PiecewiseLinearTransform,
    PiecewiseLinearTransformOp<float, CUDAContext>);

} // namespace caffe2
