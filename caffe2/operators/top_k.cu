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

template <typename T, int kHeapSize, bool kSelectMax = true>
void RunHeapSelectionImpl(
    const T* input,
    const int64_t outer_size,
    const int64_t inner_size,
    const int k,
    T* values,
    int64_t* indices,
    CUDAContext* context) {
  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / kWarpSize;
  constexpr int smem = kNumWarps * kHeapSize * (sizeof(T) + sizeof(int64_t));
  constexpr T kInitVal = kSelectMax ? std::numeric_limits<T>::lowest()
                                    : std::numeric_limits<T>::max();
  selectRowsViaHeap<T, int64_t, int64_t, kBlockSize, kHeapSize, kSelectMax>
      <<<outer_size, kBlockSize, smem, context->cuda_stream()>>>(
          input,
          values,
          indices,
          kInitVal,
          std::numeric_limits<int64_t>::max(),
          outer_size,
          inner_size,
          k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, bool kSelectMax = true>
void RunRadixSelectionImpl(
    const T* input,
    const int64_t outer_size,
    const int64_t inner_size,
    const int k,
    T* values,
    int64_t* indices,
    CUDAContext* context) {
  const int block = std::min(
      math::RoundUp(static_cast<int>(inner_size), kWarpSize),
      CAFFE_CUDA_NUM_THREADS);
  gatherTopK<T, kSelectMax, int64_t>
      <<<outer_size, block, 0, context->cuda_stream()>>>(
          input, inner_size, k, outer_size, values, indices);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // Unfortunately the output is not currently sorted, and there is no batch
  // sorting utility available. Iterate over all of the slices and sort them
  // in-place using Thrust.
  for (int i = 0; i < outer_size; ++i) {
    thrust::sort_by_key(
        thrust::cuda::par.on(context->cuda_stream()),
        values + i * k,
        values + i * k + (k <= inner_size ? k : inner_size),
        indices + i * k,
        thrust::greater<T>());
  }
}

template <typename T>
void RunTopKOnLastDimCUDAImpl(
    const T* input,
    const int64_t outer_size,
    const int64_t inner_size,
    const int k,
    T* values,
    int64_t* indices,
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

__global__ void FlattenIndicesCUDAKernel(
    const int64_t* src,
    const int64_t size,
    const int64_t stride,
    const int64_t n,
    const int k,
    int64_t* dst) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    if (src[i] < 0) {
      continue;
    }
    const int64_t x = i / stride / k;
    const int64_t y = i % stride;
#if __CUDA_ARCH__ >= 350
    dst[i] = __ldg(src + i) * stride + x * n * stride + y;
#else
    dst[i] = src[i] * stride + x * n * stride + y;
#endif
  }
}

template <typename T>
__global__ void SetTopKGradientCUDAKernel(
    const T* values,
    const int64_t* indices,
    const int64_t size,
    const int64_t stride,
    const int64_t n,
    const int k,
    T* dst) {
  CUDA_1D_KERNEL_LOOP(i, size) {
    if (indices[i] < 0) {
      continue;
    }
    const int64_t x = i / stride / k;
    const int64_t y = i % stride;
#if __CUDA_ARCH__ >= 350
    dst[__ldg(indices + i) * stride + x * n * stride + y] = __ldg(values + i);
#else
    dst[indices[i] * stride + x * n * stride + y] = values[i];
#endif
  }
}

} // namespace

template <typename T, typename Context>
class TopKCudaOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  TopKCudaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "k", k_, -1),
        OP_SINGLE_ARG(int, "axis", axis_, -1) {
    CAFFE_ENFORCE(k_ >= 1, "k argument must be >= 1");
  }

  ~TopKCudaOp(){};

  bool RunOnDevice() override;

 private:
  const int k_;
  int axis_;

  // Buffers for CUDAContext.
  Tensor input_transposed_buffer_;
  Tensor values_transposed_buffer_;
  Tensor indices_transposed_buffer_;

  // Shape tensors on device for CUDAContext.
  Tensor input_dims_device_{CUDA};
  Tensor input_transposed_dims_device_{CUDA};
  Tensor input_axes_device_{CUDA};

  Tensor output_dims_device_{CUDA};
  Tensor output_transposed_dims_device_{CUDA};
  Tensor output_transposed_axes_device_{CUDA};
};

template <typename T, typename Context>
bool TopKCudaOp<T, Context>::RunOnDevice() {
  const auto& input = Input(0);
  auto* values = Output(0);
  auto* indices = Output(1);
  auto* flatten_indices = OutputSize() > 2 ? Output(2) : nullptr;

  at::IntArrayRef input_dims = input.sizes();
  if (axis_ == -1) {
    axis_ = input_dims.size() - 1;
  }
  CAFFE_ENFORCE_GE(axis_, 0);
  CAFFE_ENFORCE_LT(axis_, input_dims.size());

  const bool need_transpose = axis_ < input_dims.size() - 1;
  std::vector<int64_t> output_dims = input_dims.vec();
  output_dims[axis_] = k_;
  const int64_t prev_size = std::accumulate(
      input_dims.cbegin(),
      input_dims.cbegin() + axis_,
      int64_t(1),
      std::multiplies<int64_t>());
  const int64_t next_size = std::accumulate(
      input_dims.cbegin() + axis_ + 1,
      input_dims.cend(),
      int64_t(1),
      std::multiplies<int64_t>());
  const int64_t outer_size = input.numel() / input_dims[axis_];
  const int64_t inner_size = input_dims[axis_];

  values->Resize(output_dims);
  indices->Resize(output_dims);
  if (flatten_indices != nullptr) {
    flatten_indices->Resize(indices->numel());
  }
  const T* input_data = input.template data<T>();
  T* values_data = values->template mutable_data<T>();
  int64_t* indices_data = indices->template mutable_data<int64_t>();
  int64_t* flatten_indices_data = flatten_indices == nullptr
      ? nullptr
      : flatten_indices->template mutable_data<int64_t>();

  if (need_transpose) {
    const std::array<int, 3> dims = {static_cast<int>(prev_size),
                                     static_cast<int>(inner_size),
                                     static_cast<int>(next_size)};
    const std::array<int, 3> axes = {0, 2, 1};
    ReinitializeTensor(&input_transposed_buffer_,  std::vector<int64_t>{outer_size, inner_size}, at::dtype<T>().device(CUDA));
    ReinitializeTensor(&values_transposed_buffer_, std::vector<int64_t>{outer_size, k_}, at::dtype<T>().device(CUDA));
    ReinitializeTensor(&indices_transposed_buffer_, std::vector<int64_t>{outer_size, k_}, at::dtype<int64_t>().device(CUDA));
    math::Transpose(
        3,
        dims.data(),
        axes.data(),
        input.template data<T>(),
        input_transposed_buffer_.template mutable_data<T>(),
        &context_);
    input_data = input_transposed_buffer_.template data<T>();
    values_data = values_transposed_buffer_.template mutable_data<T>();
    indices_data = indices_transposed_buffer_.template mutable_data<int64_t>();
  }

  // init values as the default value
  math::Set<T, CUDAContext>(values->numel(), T(0), values_data, &context_);
  math::Set<int64_t, CUDAContext>(
      indices->numel(), int64_t(-1), indices_data, &context_);
  if (flatten_indices_data != nullptr) {
    math::Set<int64_t, CUDAContext>(
        flatten_indices->numel(), int64_t(-1), flatten_indices_data, &context_);
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
    const std::array<int, 3> dims = {
        static_cast<int>(prev_size), static_cast<int>(next_size), k_};
    const std::array<int, 3> axes = {0, 2, 1};
    math::Transpose(
        3,
        dims.data(),
        axes.data(),
        values_transposed_buffer_.template data<T>(),
        values->template mutable_data<T>(),
        &context_);
    math::Transpose(
        3,
        dims.data(),
        axes.data(),
        indices_transposed_buffer_.template data<int64_t>(),
        indices->template mutable_data<int64_t>(),
        &context_);
  }

  // Flatten the indices if needed.
  if (flatten_indices != nullptr) {
    FlattenIndicesCUDAKernel<<<
        CAFFE_GET_BLOCKS(indices->numel()),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        indices->template data<int64_t>(),
        indices->numel(),
        next_size,
        inner_size,
        k_,
        flatten_indices->template mutable_data<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return true;
}

REGISTER_CUDA_OPERATOR(TopK, TopKCudaOp<float, CUDAContext>);

template <typename T, typename Context>
class TopKGradientCudaOp : public Operator<Context> {
 public:
  USE_OPERATOR_FUNCTIONS(Context);

  TopKGradientCudaOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        OP_SINGLE_ARG(int, "axis", axis_, -1) {}

  ~TopKGradientCudaOp(){};

  bool RunOnDevice() override;

 private:
  int axis_;
};

template <typename T, typename Context>
bool TopKGradientCudaOp<T, Context>::RunOnDevice() {
  const auto& values = Input(0);
  const auto& indices = Input(1);
  const auto& original_input = Input(2);
  auto* output = Output(0);
  at::IntArrayRef values_dims = values.sizes();
  at::IntArrayRef origin_dims = original_input.sizes();
  CAFFE_ENFORCE_EQ(values_dims.size(), origin_dims.size());
  output->Resize(origin_dims);
  T* output_data = output->template mutable_data<T>();
  if (axis_ == -1) {
    axis_ = values_dims.size() - 1;
  }
  const int k = values_dims[axis_];
  math::Set<T, Context>(output->size(), T(0), output_data, &context_);
  const int64_t stride = std::accumulate(
      values_dims.cbegin() + axis_ + 1,
      values_dims.cend(),
      int64_t(1),
      std::multiplies<int64_t>());
  SetTopKGradientCUDAKernel<<<
      CAFFE_GET_BLOCKS(indices.size()),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      values.template data<T>(),
      indices.template data<int64_t>(),
      values.size(),
      stride,
      origin_dims[axis_],
      k,
      output_data);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return true;
}

REGISTER_CUDA_OPERATOR(TopKGradient, TopKGradientCudaOp<float, CUDAContext>);

} // namespace caffe2
