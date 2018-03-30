#include "caffe2/operators/top_k.h"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/top_k_heap_selection.cuh"
#include "caffe2/operators/top_k_radix_selection.cuh"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {

void MakeTransposeParams(
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    TensorCUDA* x_dims,
    TensorCUDA* y_dims,
    TensorCUDA* axes,
    CUDAContext* context) {
  const std::array<int, 3> x_dims_vec = {static_cast<int>(prev_size),
                                         static_cast<int>(n),
                                         static_cast<int>(next_size)};
  const std::array<int, 3> y_dims_vec = {static_cast<int>(prev_size),
                                         static_cast<int>(next_size),
                                         static_cast<int>(n)};
  const std::array<int, 3> axes_vec = {0, 2, 1};
  x_dims->Resize(3);
  context->Copy<int, CPUContext, CUDAContext>(
      3, x_dims_vec.data(), x_dims->mutable_data<int>());
  y_dims->Resize(3);
  context->Copy<int, CPUContext, CUDAContext>(
      3, y_dims_vec.data(), y_dims->mutable_data<int>());
  axes->Resize(3);
  context->Copy<int, CPUContext, CUDAContext>(
      3, axes_vec.data(), axes->mutable_data<int>());
}

template <typename T, int kHeapSize, bool kSelectMax = true>
void RunHeapSelectionImpl(
    const T* input,
    const TIndex outer_size,
    const TIndex inner_size,
    const int k,
    T* values,
    TIndex* indices,
    CUDAContext* context) {
  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int smem = kNumWarps * kHeapSize * (sizeof(T) + sizeof(TIndex));
  constexpr T kInitVal = kSelectMax ? std::numeric_limits<T>::lowest()
                                    : std::numeric_limits<T>::max();
  selectRowsViaHeap<T, TIndex, TIndex, kBlockSize, kHeapSize, kSelectMax>
      <<<outer_size, kBlockSize, smem, context->cuda_stream()>>>(
          input,
          values,
          indices,
          kInitVal,
          std::numeric_limits<TIndex>::max(),
          outer_size,
          inner_size,
          k);
}

template <typename T, bool kSelectMax = true>
void RunRadixSelectionImpl(
    const T* input,
    const TIndex outer_size,
    const TIndex inner_size,
    const int k,
    T* values,
    TIndex* indices,
    CUDAContext* context) {
  const int block = std::min(
      math::roundUp(static_cast<int>(inner_size), kWarpSize),
      CAFFE_CUDA_NUM_THREADS);
  gatherTopK<T, kSelectMax, TIndex>
      <<<outer_size, block, 0, context->cuda_stream()>>>(
          input, inner_size, k, outer_size, values, indices);
  // Unfortunately the output is not currently sorted, and there is no batch
  // sorting utility available. Iterate over all of the slices and sort them
  // in-place using Thrust.
  for (int i = 0; i < outer_size; ++i) {
    thrust::sort_by_key(
        thrust::cuda::par.on(context->cuda_stream()),
        values + i * k,
        values + i * k + k,
        indices + i * k,
        thrust::greater<T>());
  }
}

template <typename T>
void RunTopKOnLastDimCUDAImpl(
    const T* input,
    const TIndex outer_size,
    const TIndex inner_size,
    const int k,
    T* values,
    TIndex* indices,
    CUDAContext* context) {
  // If k is small, uses heap selection, otherwise uses radix selection.
  if (k < 32) {
    RunHeapSelectionImpl<T, 32>(
        input, outer_size, inner_size, k, values, indices, context);
  } else if (k < 128) {
    RunHeapSelectionImpl<T, 128>(
        input, outer_size, inner_size, k, values, indices, context);
  } else if (k < 512) {
    RunHeapSelectionImpl<T, 512>(
        input, outer_size, inner_size, k, values, indices, context);
  } else {
    RunRadixSelectionImpl<T>(
        input, outer_size, inner_size, k, values, indices, context);
  }
}

__global__ void FlattenIndicesCUDA(
    const TIndex* src,
    const TIndex size,
    const TIndex stride,
    const TIndex n,
    const int k,
    TIndex* dst) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    const TIndex x = i / stride / k;
    const TIndex y = i % stride;
#if __CUDA_ARCH__ >= 350
    dst[i] = __ldg(src + i) * stride + x * n * stride + y;
#else
    dst[i] = src[i] * stride + x * n * stride + y;
#endif
  }
}

template <typename T>
__global__ void SetTopKGradientCUDA(
    const T* values,
    const TIndex* indices,
    const TIndex size,
    const TIndex stride,
    const TIndex n,
    const int k,
    T* dst) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    const TIndex x = i / stride / k;
    const TIndex y = i % stride;
#if __CUDA_ARCH__ >= 350
    dst[__ldg(indices + i) * stride + x * n * stride + y] = __ldg(values + i);
#else
    dst[indices[i] * stride + x * n * stride + y] = values[i];
#endif
  }
}

} // namespace

template <typename T>
class TopKOp<T, CUDAContext> : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  TopKOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1),
        OP_SINGLE_ARG(int, "axis", axis_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKOp(){};

  bool RunOnDevice() override;

 private:
  const int k_;
  int axis_;

  // Buffers for CUDAContext.
  TensorCUDA input_transposed_buffer_;
  TensorCUDA values_transposed_buffer_;
  TensorCUDA indices_transposed_buffer_;

  // Shape tensors on device for CUDAContext.
  TensorCUDA input_dims_device_;
  TensorCUDA input_transposed_dims_device_;
  TensorCUDA input_axes_device_;

  TensorCUDA output_dims_device_;
  TensorCUDA output_transposed_dims_device_;
  TensorCUDA output_transposed_axes_device_;
};

template <typename T>
bool TopKOp<T, CUDAContext>::RunOnDevice() {
  const auto& input = Input(0);
  auto* values = Output(0);
  auto* indices = Output(1);
  auto* flatten_indices = OutputSize() > 2 ? Output(2) : nullptr;

  const std::vector<TIndex>& input_dims = input.dims();
  if (axis_ == -1) {
    axis_ = input_dims.size() - 1;
  }
  CAFFE_ENFORCE_GE(axis_, 0);
  CAFFE_ENFORCE_LT(axis_, input_dims.size());
  CAFFE_ENFORCE_LE(
      k_,
      input_dims[axis_],
      "k argument should not be greater than the axis dim.");

  const bool need_transpose = axis_ < input_dims.size() - 1;
  std::vector<TIndex> output_dims = input_dims;
  output_dims[axis_] = k_;
  const TIndex prev_size = std::accumulate(
      input_dims.cbegin(),
      input_dims.cbegin() + axis_,
      TIndex(1),
      std::multiplies<TIndex>());
  const TIndex next_size = std::accumulate(
      input_dims.cbegin() + axis_ + 1,
      input_dims.cend(),
      TIndex(1),
      std::multiplies<TIndex>());
  const TIndex outer_size = input.size() / input_dims[axis_];
  const TIndex inner_size = input_dims[axis_];

  values->Resize(output_dims);
  indices->Resize(output_dims);
  if (flatten_indices != nullptr) {
    flatten_indices->Resize(indices->size());
  }
  const T* input_data = input.template data<T>();
  T* values_data = values->template mutable_data<T>();
  TIndex* indices_data = indices->template mutable_data<TIndex>();
  TIndex* flatten_indices_data = flatten_indices == nullptr
      ? nullptr
      : flatten_indices->template mutable_data<TIndex>();

  if (need_transpose) {
    MakeTransposeParams(
        prev_size,
        next_size,
        inner_size,
        &input_dims_device_,
        &input_transposed_dims_device_,
        &input_axes_device_,
        &context_);
    input_transposed_buffer_.Resize(
        std::vector<TIndex>{outer_size, inner_size});
    values_transposed_buffer_.Resize(std::vector<TIndex>{outer_size, k_});
    indices_transposed_buffer_.Resize(std::vector<TIndex>{outer_size, k_});
    math::Transpose(
        3,
        input_dims_device_.data<int>(),
        input_transposed_dims_device_.data<int>(),
        input_axes_device_.data<int>(),
        input.size(),
        input.template data<T>(),
        input_transposed_buffer_.mutable_data<T>(),
        &context_);
    input_data = input_transposed_buffer_.data<T>();
    values_data = values_transposed_buffer_.mutable_data<T>();
    indices_data = indices_transposed_buffer_.mutable_data<TIndex>();
  }
  RunTopKOnLastDimCUDAImpl<T>(
      input_data,
      outer_size,
      inner_size,
      k_,
      values_data,
      indices_data,
      &context_);
  if (need_transpose) {
    MakeTransposeParams(
        prev_size,
        next_size,
        k_,
        &output_dims_device_,
        &output_transposed_dims_device_,
        &output_transposed_axes_device_,
        &context_);
    math::Transpose(
        3,
        output_transposed_dims_device_.data<int>(),
        output_dims_device_.data<int>(),
        output_transposed_axes_device_.data<int>(),
        values_transposed_buffer_.size(),
        values_transposed_buffer_.data<T>(),
        values->template mutable_data<T>(),
        &context_);
    math::Transpose(
        3,
        output_transposed_dims_device_.data<int>(),
        output_dims_device_.data<int>(),
        output_transposed_axes_device_.data<int>(),
        indices_transposed_buffer_.size(),
        indices_transposed_buffer_.data<TIndex>(),
        indices->template mutable_data<TIndex>(),
        &context_);
  }

  // Flatten the indices if needed.
  if (flatten_indices != nullptr) {
    FlattenIndicesCUDA<<<
        CAFFE_GET_BLOCKS(indices->size()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        indices->template data<TIndex>(),
        indices->size(),
        next_size,
        inner_size,
        k_,
        flatten_indices->template mutable_data<TIndex>());
  }
  return true;
}

REGISTER_CUDA_OPERATOR(TopK, TopKOp<float, CUDAContext>);

template <typename T>
class TopKGradientOp<T, CUDAContext> : public Operator<CUDAContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CUDAContext);

  TopKGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CUDAContext>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, -1) {}

  ~TopKGradientOp(){};

  bool RunOnDevice() override;

 private:
  int axis_;
};

template <typename T>
bool TopKGradientOp<T, CUDAContext>::RunOnDevice() {
  const auto& values = Input(0);
  const auto& indices = Input(1);
  const auto& original_input = Input(2);
  auto* output = Output(0);
  const std::vector<TIndex>& values_dims = values.dims();
  const std::vector<TIndex>& origin_dims = original_input.dims();
  CAFFE_ENFORCE_EQ(values_dims.size(), origin_dims.size());
  output->Resize(origin_dims);
  T* output_data = output->template mutable_data<T>();
  if (axis_ == -1) {
    axis_ = values_dims.size() - 1;
  }
  const int k = values_dims[axis_];
  math::Set<T, CUDAContext>(output->size(), T(0), output_data, &context_);
  const TIndex stride = std::accumulate(
      values_dims.cbegin() + axis_ + 1,
      values_dims.cend(),
      TIndex(1),
      std::multiplies<TIndex>());
  SetTopKGradientCUDA<<<
      CAFFE_GET_BLOCKS(indices.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      values.template data<T>(),
      indices.template data<TIndex>(),
      values.size(),
      stride,
      origin_dims[axis_],
      k,
      output_data);
  return true;
}

REGISTER_CUDA_OPERATOR(TopKGradient, TopKGradientOp<float, CUDAContext>);

} // namespace caffe2
