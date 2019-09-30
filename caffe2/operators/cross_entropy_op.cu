#include <assert.h>
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/cross_entropy_op.h"
#include "caffe2/operators/operator_fallback_gpu.h"

namespace caffe2 {

namespace {
__global__ void LabelCrossEntropyKernel(
    const int N, const int D, const float* Xdata, const int* labeldata,
    const float log_threshold, float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    CUDA_KERNEL_ASSERT(labeldata[i] >= 0 && labeldata[i] < D);
    Ydata[i] = -logf(fmaxf(Xdata[i * D + labeldata[i]], log_threshold));
  }
}
__global__ void LabelCrossEntropyGradientKernel(
    const int N, const int D, const float* Xdata, const int* labeldata,
    const float* dYdata, const float log_threshold, float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    int idx = i * D + labeldata[i];
    dXdata[idx] = - dYdata[i] / fmaxf(Xdata[idx], log_threshold);
  }
}
}  // namespace

template <>
bool LabelCrossEntropyOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);

  int N, D;
  if (X.dim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  auto* Y = Output(0, vector<int64_t>(size_t(1), N), at::dtype<float>());
  LabelCrossEntropyKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      label.data<int>(),
      kLOG_THRESHOLD(),
      Y->template mutable_data<float>());
  return true;
}

template <>
bool LabelCrossEntropyGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& label = Input(1);
  auto& dY = Input(2);

  int N, D;
  if (X.dim() > 1) {
    N = X.dim32(0);
    D = X.size_from_dim(1);
  } else {
    N = 1;
    D = X.dim32(0);
  }
  CAFFE_ENFORCE(
      (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
  CAFFE_ENFORCE_EQ(label.dim32(0), N);
  CAFFE_ENFORCE_EQ(dY.dim(), 1);
  CAFFE_ENFORCE_EQ(dY.dim32(0), N);
  auto* dX = Output(0, X.sizes(), at::dtype<float>());
  math::Set<float, CUDAContext>(
      dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);
  LabelCrossEntropyGradientKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N,
      D,
      X.data<float>(),
      label.data<int>(),
      dY.data<float>(),
      kLOG_THRESHOLD(),
      dX->template mutable_data<float>());
  return true;
}

namespace {
__global__ void MakeTwoClassKernel(
    const int N, const float* Xdata, float* Ydata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    Ydata[i * 2] = 1.0 - Xdata[i];
    Ydata[i * 2 + 1] = Xdata[i];
  }
}
__global__ void MakeTwoClassGradientKernel(
    const int N, const float* dYdata, float* dXdata) {
  CUDA_1D_KERNEL_LOOP(i, N) {
    dXdata[i] = dYdata[i * 2 + 1] - dYdata[i * 2];
  }
}
}  // namespace

template <>
bool MakeTwoClassOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto shape = X.sizes().vec();
  shape.push_back(2);
  CAFFE_ENFORCE_LT(X.numel(), std::numeric_limits<int>::max() / 2);
  auto* Y = Output(0, shape, at::dtype<float>());
  int N = X.numel();
  MakeTwoClassKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, X.data<float>(), Y->template mutable_data<float>());
  return true;
}

template <>
bool MakeTwoClassGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& dY = Input(0);
  auto shape = dY.sizes().vec();
  CAFFE_ENFORCE_GE(shape.size(), 1);
  CAFFE_ENFORCE_EQ(shape.back(), 2);
  shape.pop_back();
  CAFFE_ENFORCE_LT(dY.numel(), std::numeric_limits<int>::max());
  auto* dX = Output(0, shape, at::dtype<float>());
  int N = dX->numel();
  MakeTwoClassGradientKernel<<<
      CAFFE_GET_BLOCKS(N),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      N, dY.data<float>(), dX->template mutable_data<float>());
  return true;
}

namespace {

__device__ float sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * (tgt - (lgt >= 0)) - log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

__device__ float sigmoid_xent_backward(float lgt, float tgt) {
  return tgt - 1. / (1. + exp(-lgt));
}

__device__ float sigmoid_partition(float lgt) {
  // computes log(1 + exp(lgt)) with only exp(x) function when x >= 0
  return lgt * (lgt >= 0) + log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

__device__ float sigmoid_xent_forward_with_log_d_trick(float lgt, float tgt) {
  return (2 * tgt - 1.) * (lgt - sigmoid_partition(lgt));
}

__device__ float sigmoid_xent_backward_with_log_d_trick(float lgt, float tgt) {
  return (2 * tgt - 1.) / (1. + exp(lgt));
}

__device__ float unjoined_sigmoid_xent_forward(float lgt, float tgt) {
  return lgt * tgt + (tgt - 1) * lgt * (lgt >= 0) -
      (1 - tgt) * log(1 + exp(lgt - 2 * lgt * (lgt >= 0)));
}

__device__ float unjoined_sigmoid_xent_backward(float lgt, float tgt) {
  return tgt - (1. - tgt) / (1. + exp(-lgt));
}

__global__ void SigmoidCrossEntropyWithLogitsKernel(
    const int outer_size,
    const int inner_size,
    const bool log_D_trick,
    const bool unjoined_lr_loss,
    const float* logits_ptr,
    const float* targets_ptr,
    float* out_ptr) {
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float value = 0;
  for (int in_idx = i * inner_size + threadIdx.x; in_idx < last_idx;
       in_idx += blockDim.x) {
    if (unjoined_lr_loss) {
      value += unjoined_sigmoid_xent_forward(
          logits_ptr[in_idx], targets_ptr[in_idx]);
    } else {
      value +=
          (log_D_trick
               ? sigmoid_xent_forward_with_log_d_trick(
                     logits_ptr[in_idx], targets_ptr[in_idx])
               : sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]));
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sum = BlockReduce(temp_storage).Sum(value);
  if (threadIdx.x == 0) {
    out_ptr[i] = -sum / inner_size;
  }
}

__global__ void SigmoidCrossEntropyGradientWithLogitsKernel(
    const int outer_size,
    const int inner_size,
    const bool log_D_trick,
    const bool unjoined_lr_loss,
    const float* g_ptr,
    const float* logits_ptr,
    const float* targets_ptr,
    float* out_ptr) {
  CUDA_1D_KERNEL_LOOP(in_idx, outer_size * inner_size) {
    int i = in_idx / inner_size;
    auto g_factor = -g_ptr[i] / inner_size;
    if (unjoined_lr_loss) {
      out_ptr[in_idx] = g_factor *
          unjoined_sigmoid_xent_backward(
                            logits_ptr[in_idx], targets_ptr[in_idx]);
    } else {
      out_ptr[in_idx] = g_factor *
          (log_D_trick ? sigmoid_xent_backward_with_log_d_trick(
                             logits_ptr[in_idx], targets_ptr[in_idx])
                       : sigmoid_xent_backward(
                             logits_ptr[in_idx], targets_ptr[in_idx]));
    }
  }
}
} // namespace

template <>
bool SigmoidCrossEntropyWithLogitsOp<float, CUDAContext>::RunOnDevice() {
  auto& logits = Input(0);
  auto& targets = Input(1);
  CAFFE_ENFORCE_EQ(logits.sizes(), targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;

  std::vector<int64_t> dims;
  if (logits.dim() != 0) {
    dims =
        std::vector<int64_t>(logits.sizes().begin(), logits.sizes().end() - 1);
  }
  auto* out = Output(0, dims, at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();

  if (logits.numel() <= 0) {
    // nothing to do, not even launching kernel
    return true;
  }

  SigmoidCrossEntropyWithLogitsKernel<<<
      outer_size,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      outer_size,
      inner_size,
      log_D_trick_,
      unjoined_lr_loss_,
      logits_ptr,
      targets_ptr,
      out_ptr);
  return true;
}

template <>
bool SigmoidCrossEntropyWithLogitsGradientOp<float, CUDAContext>::
    RunOnDevice() {
  auto& g = Input(0);
  auto& logits = Input(1);
  auto& targets = Input(2);
  CAFFE_ENFORCE_EQ(logits.sizes(), targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;
  CAFFE_ENFORCE_EQ(g.numel(), outer_size);

  auto* out = Output(0, logits.sizes(), at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* g_ptr = g.data<float>();

  SigmoidCrossEntropyGradientWithLogitsKernel<<<
      CAFFE_GET_BLOCKS(outer_size * inner_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      outer_size,
      inner_size,
      log_D_trick_,
      unjoined_lr_loss_,
      g_ptr,
      logits_ptr,
      targets_ptr,
      out_ptr);
  return true;
}

namespace {

__global__ void WeightedSigmoidCrossEntropyWithLogitsKernel(
    const int outer_size,
    const int inner_size,
    const float* logits_ptr,
    const float* targets_ptr,
    const float* weights_ptr,
    float* out_ptr) {
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float value = 0;
  for (int in_idx = i * inner_size + threadIdx.x; in_idx < last_idx;
       in_idx += blockDim.x) {
    value += sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) *
        weights_ptr[in_idx];
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sum = BlockReduce(temp_storage).Sum(value);
  if (threadIdx.x == 0) {
    out_ptr[i] = -sum / inner_size;
  }
}

__global__ void WeightedSigmoidCrossEntropyGradientWithLogitsKernel(
    const int outer_size,
    const int inner_size,
    const float* g_ptr,
    const float* logits_ptr,
    const float* targets_ptr,
    const float* weights_ptr,
    float* out_ptr) {
  CUDA_1D_KERNEL_LOOP(in_idx, outer_size * inner_size) {
    int i = in_idx / inner_size;
    auto g_factor = -g_ptr[i] / inner_size;
    out_ptr[in_idx] = g_factor *
        sigmoid_xent_backward(logits_ptr[in_idx], targets_ptr[in_idx]) *
        weights_ptr[in_idx];
  }
}
} // namespace

template <>
bool WeightedSigmoidCrossEntropyWithLogitsOp<float, CUDAContext>::
    RunOnDevice() {
  auto& logits = Input(0);
  auto& targets = Input(1);
  auto& weights = Input(2);
  CAFFE_ENFORCE_EQ(logits.sizes(), targets.sizes());
  CAFFE_ENFORCE_EQ(weights.sizes(), targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;

  std::vector<int64_t> dims;
  if (logits.dim() != 0) {
    dims =
        std::vector<int64_t>(logits.sizes().begin(), logits.sizes().end() - 1);
  }
  auto* out = Output(0, dims, at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* weights_ptr = weights.data<float>();

  WeightedSigmoidCrossEntropyWithLogitsKernel<<<
      outer_size,
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      outer_size, inner_size, logits_ptr, targets_ptr, weights_ptr, out_ptr);
  return true;
}

template <>
bool WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, CUDAContext>::
    RunOnDevice() {
  auto& g = Input(0);
  auto& logits = Input(1);
  auto& targets = Input(2);
  auto& weights = Input(3);
  CAFFE_ENFORCE_EQ(logits.sizes(), targets.sizes());
  CAFFE_ENFORCE_EQ(weights.sizes(), targets.sizes());
  const auto inner_size = logits.dim() > 0 ? logits.sizes().back() : 1;
  const auto outer_size = logits.numel() / inner_size;
  CAFFE_ENFORCE_EQ(g.numel(), outer_size);

  auto* out = Output(0, logits.sizes(), at::dtype<float>());
  auto* out_ptr = out->template mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* weights_ptr = weights.data<float>();
  auto* g_ptr = g.data<float>();

  WeightedSigmoidCrossEntropyGradientWithLogitsKernel<<<
      CAFFE_GET_BLOCKS(outer_size * inner_size),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      outer_size,
      inner_size,
      g_ptr,
      logits_ptr,
      targets_ptr,
      weights_ptr,
      out_ptr);
  return true;
}

REGISTER_CUDA_OPERATOR(LabelCrossEntropy,
                       LabelCrossEntropyOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(LabelCrossEntropyGradient,
                       LabelCrossEntropyGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    SigmoidCrossEntropyWithLogits,
    SigmoidCrossEntropyWithLogitsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    SigmoidCrossEntropyWithLogitsGradient,
    SigmoidCrossEntropyWithLogitsGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(
    WeightedSigmoidCrossEntropyWithLogits,
    WeightedSigmoidCrossEntropyWithLogitsOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(
    WeightedSigmoidCrossEntropyWithLogitsGradient,
    WeightedSigmoidCrossEntropyWithLogitsGradientOp<float, CUDAContext>);

REGISTER_CUDA_OPERATOR(MakeTwoClass,
                       MakeTwoClassOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(MakeTwoClassGradient,
                       MakeTwoClassGradientOp<float, CUDAContext>);

//TODO(surya) Add full GPU/CUDA support for the CrossEntropyOp
REGISTER_CUDA_OPERATOR(CrossEntropy, GPUFallbackOp);
REGISTER_CUDA_OPERATOR(CrossEntropyGradient, GPUFallbackOp);

}  // namespace caffe2
