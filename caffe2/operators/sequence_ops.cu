#include <algorithm>

#include <cub/cub.cuh>
#include "caffe2/utils/cub_namespace.cuh"
#include <c10/cuda/CUDADeviceAssertion.h>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/sequence_ops.h"

#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void AddPaddingKernel(
    const T* in,
    int block_size,
    int lengths_size,
    int outer_size,
    const int32_t* lengths_prefix_sum,
    const T* padding_start_ptr,
    int start_padding_width_blocks,
    const T* padding_end_ptr,
    int end_padding_width_blocks,
    T* out,
    int32_t* lengths_out,
    TORCH_DSA_KERNEL_ARGS) {
  int element_idx = blockIdx.x;
  int prior_padding =
      element_idx * (start_padding_width_blocks + end_padding_width_blocks);
  int out_start_idx = element_idx == 0
      ? 0
      : lengths_prefix_sum[element_idx - 1] + prior_padding;
  int len_blocks;
  int in_start_idx;
  if (lengths_prefix_sum) {
    len_blocks = lengths_prefix_sum[element_idx] -
        (element_idx == 0 ? 0 : lengths_prefix_sum[element_idx - 1]);
    in_start_idx = lengths_prefix_sum[element_idx] - len_blocks;
  } else {
    // Only one element, use the outer size
    CUDA_KERNEL_ASSERT2(lengths_size == 1);
    len_blocks = outer_size;
    in_start_idx = 0;
  }

  out_start_idx *= block_size;
  in_start_idx *= block_size;

  int len = len_blocks * block_size;
  int start_padding_width = start_padding_width_blocks * block_size;
  int end_padding_width = end_padding_width_blocks * block_size;

  // start pad
  T* out_ptr = out + out_start_idx;
  for (int i = threadIdx.x; i < start_padding_width; i += blockDim.x) {
    T fill = padding_start_ptr ? padding_start_ptr[i % block_size] : T(0);
    out_ptr[i] = fill;
  }

  // payload
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    out_ptr[i + start_padding_width] = in[in_start_idx + i];
  }

  // end pad
  for (int i = threadIdx.x; i < end_padding_width; i += blockDim.x) {
    T fill = padding_end_ptr ? padding_end_ptr[i % block_size] : T(0);
    out_ptr[i + start_padding_width + len] = fill;
  }

  // update the lengths
  if (threadIdx.x == 0 && lengths_out != nullptr) {
    lengths_out[element_idx] =
        len_blocks + start_padding_width_blocks + end_padding_width_blocks;
  }
}

template <typename T>
__global__ void RemovePaddingKernel(
    const T* in,
    int block_size,
    int lengths_size,
    int outer_size,
    const int32_t* lengths_prefix_sum,
    int start_padding_width_blocks,
    int end_padding_width_blocks,
    T* out,
    int32_t* lengths_out,
    TORCH_DSA_KERNEL_ARGS) {
  int element_idx = blockIdx.x;
  int prior_padding =
      element_idx * (start_padding_width_blocks + end_padding_width_blocks);
  int out_start_idx = element_idx == 0
      ? 0
      : lengths_prefix_sum[element_idx - 1] - prior_padding;
  int len_blocks;
  int in_start_idx;
  if (lengths_prefix_sum) {
    len_blocks = lengths_prefix_sum[element_idx] -
        (element_idx == 0 ? 0 : lengths_prefix_sum[element_idx - 1]);
    in_start_idx = lengths_prefix_sum[element_idx] - len_blocks;
  } else {
    // Only one element, use the outer size
    CUDA_KERNEL_ASSERT2(lengths_size == 1);
    len_blocks = outer_size;
    in_start_idx = 0;
  }

  out_start_idx *= block_size;
  in_start_idx *= block_size;

  int len = len_blocks * block_size;
  int start_padding_width = start_padding_width_blocks * block_size;

  // payload
  T* out_ptr = out + out_start_idx;
  for (int i = threadIdx.x; i < len; i += blockDim.x) {
    out_ptr[in_start_idx + i] = in[i + start_padding_width];
  }

  // update the lengths
  if (threadIdx.x == 0 && lengths_out != nullptr) {
    lengths_out[element_idx] =
        len_blocks - (start_padding_width_blocks + end_padding_width_blocks);
  }
}

template <bool Inclusive = true>
void lengths_prefix_sum(
    const int32_t* lengths,
    int32_t num_items,
    Tensor* prefix_buffer,
    Tensor* prefix_sum,
    CUDAContext* context) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  prefix_sum->Resize(num_items);
  if (Inclusive) {
    cub::DeviceScan::InclusiveSum(
        NULL,
        temp_storage_bytes,
        lengths,
        prefix_sum->template mutable_data<int32_t>(),
        num_items,
        context->cuda_stream());
  } else {
    cub::DeviceScan::ExclusiveSum(
        NULL,
        temp_storage_bytes,
        lengths,
        prefix_sum->template mutable_data<int32_t>(),
        num_items,
        context->cuda_stream());
  }

  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(int32_t)) / sizeof(int32_t);
  prefix_buffer->Resize(buffer_size);
  void* d_temp_storage =
      static_cast<void*>(prefix_buffer->template mutable_data<int32_t>());

  if (Inclusive) {
    cub::DeviceScan::InclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        lengths,
        prefix_sum->template mutable_data<int32_t>(),
        num_items,
        context->cuda_stream());
  } else {
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        lengths,
        prefix_sum->template mutable_data<int32_t>(),
        num_items,
        context->cuda_stream());
  }
}
} // namespace

template <>
template <typename T>
bool AddPaddingOp<CUDAContext>::MakePadding(
    const T* in_ptr,
    T* out_ptr,
    const int32_t* lengths_ptr,
    int32_t lengths_size,
    int32_t outer_size,
    const T* padding_start_ptr,
    const T* padding_end_ptr,
    int64_t block_size) {
  // Step 1: compute prefix sum over the lengths -- unless
  // there were no lengths given, i.e there is only one segment
  const int32_t* lengths_prefix_sum_ptr = nullptr;
  if (lengths_ptr != nullptr) {
    lengths_prefix_sum(
        lengths_ptr,
        lengths_size,
        &lengths_prefix_sum_buffer_,
        &lengths_prefix_sum_,
        &context_);
    lengths_prefix_sum_ptr = lengths_prefix_sum_.data<int32_t>();
  }

  int32_t* lengths_out_ptr = nullptr;
  if (OutputSize() > 1) {
    auto* lengths_out = Output(1, {lengths_size}, at::dtype<int32_t>());
    lengths_out_ptr = lengths_out->template mutable_data<int32_t>();
  }

  if (lengths_size == 0) {
    return true;
  }

  // Compute the padding using the accumulated lengths
  TORCH_DSA_KERNEL_LAUNCH(
          AddPaddingKernel<T>,
          lengths_size, CAFFE_CUDA_NUM_THREADS, 0, context_.stream(),
          in_ptr,
          block_size,
          lengths_size,
          outer_size,
          lengths_prefix_sum_ptr,
          padding_start_ptr,
          startPaddingWidth_,
          padding_end_ptr,
          endPaddingWidth_,
          out_ptr,
          lengths_out_ptr);

  return true;
}

REGISTER_CUDA_OPERATOR(AddPadding, AddPaddingOp<CUDAContext>);

template <>
template <typename T>
bool RemovePaddingOp<CUDAContext>::DoRunWithType() {
  const auto& in = Input(0);
  CAFFE_ENFORCE_GE(in.dim(), 1);
  const int32_t outer_size = in.sizes()[0];
  const auto block_size = std::accumulate(
      in.sizes().begin() + 1, in.sizes().end(), 1, std::multiplies<int64_t>());

  // if no lengths is provided, assume it is a single full-span entry
  const int32_t* lengths_ptr = nullptr;
  int32_t lengths_size = 1;
  if (InputSize() > 1) {
    const auto& lengths = Input(1);
    lengths_ptr = lengths.data<int32_t>();
    lengths_size = lengths.numel();
  }

  auto out_dims = in.sizes().vec();
  out_dims[0] -= (startPaddingWidth_ + endPaddingWidth_) * lengths_size;
  auto* out = Output(0, out_dims, at::dtype<T>());
  const auto* in_ptr = in.template data<T>();
  auto* out_ptr = out->template mutable_data<T>();

  // Step 1: compute prefix sum over the (padded) lengths -- unless
  // there were no lengths given, i.e there is only one segment
  const int32_t* lengths_prefix_sum_ptr = nullptr;
  if (lengths_ptr != nullptr) {
    lengths_prefix_sum(
        lengths_ptr,
        lengths_size,
        &lengths_prefix_sum_buffer_,
        &lengths_prefix_sum_,
        &context_);
    lengths_prefix_sum_ptr = lengths_prefix_sum_.data<int32_t>();
  }

  int32_t* lengths_out_ptr = nullptr;
  if (OutputSize() > 1) {
    auto* lengths_out = Output(1, {lengths_size}, at::dtype<int32_t>());
    lengths_out_ptr = lengths_out->template mutable_data<int32_t>();
  }

  if (lengths_size == 0) {
    return true;
  }

  // Compute the padding using the accumulated lengths
  TORCH_DSA_KERNEL_LAUNCH(
          RemovePaddingKernel<T>,
          lengths_size, CAFFE_CUDA_NUM_THREADS, 0, context_.stream(),
          in_ptr,
          block_size,
          lengths_size,
          outer_size,
          lengths_prefix_sum_ptr,
          startPaddingWidth_,
          endPaddingWidth_,
          out_ptr,
          lengths_out_ptr);

  return true;
}

template <typename T>
__global__ void gather_padding_kernel(
    const int K,
    const int N,
    const int Y0Width,
    const int Y1Width,
    const T* X,
    const int* I,
    const int* L,
    T* Y0,
    T* Y1) {
  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage y0_tmp;
  __shared__ typename BlockReduce::TempStorage y1_tmp;
  for (int i = blockIdx.x; i < N; i += gridDim.x) {
    T sum_1 = T(0);
    T sum_2 = T(0);
    for (int j = threadIdx.x; j < K * Y0Width; j += blockDim.x) {
      const int j1 = j / Y0Width;
      const int j2 = j % Y0Width;
      const int idx1 = N * (L[j1] + j2);
      sum_1 += X[idx1 + i];
    }
    for (int j = threadIdx.x; j < K * Y1Width; j += blockDim.x) {
      const int j1 = j / Y1Width;
      const int j2 = j % Y1Width;
      const int idx1 = N * L[j1];
      const int idx2 = idx1 + N * (I[j1] - Y1Width + j2);
      sum_2 += X[idx2 + i];
    }
    sum_1 = BlockReduce(y0_tmp).Reduce(sum_1, cub::Sum());
    sum_2 = BlockReduce(y1_tmp).Reduce(sum_2, cub::Sum());
    if (threadIdx.x == 0) {
      Y0[i] = sum_1;
      Y0 != Y1 ? Y1[i] = sum_2 : Y0[i] = sum_1 + sum_2;
    }
    __syncthreads();
  }
}

template <>
template <typename T>
void GatherPaddingOp<CUDAContext>::GatherPadding(
    const int outer_size,
    const int lengths_size,
    const int block_size,
    const int pad_width,
    const T* in_ptr,
    const int* lengths_ptr,
    T* padding_start_ptr,
    T* padding_end_ptr) {
  if (lengths_size > 0) {
    lengths_prefix_sum<false>(
        lengths_ptr,
        lengths_size,
        &lengths_prefix_sum_buffer_,
        &lengths_prefix_sum_,
        &context_);
    gather_padding_kernel<T>
        <<<std::min(block_size, CAFFE_MAXIMUM_NUM_BLOCKS),
           CAFFE_CUDA_NUM_THREADS,
           0,
           context_.cuda_stream()>>>(
            lengths_size,
            block_size,
            startPaddingWidth_,
            endPaddingWidth_,
            in_ptr,
            lengths_ptr,
            lengths_prefix_sum_.template data<int>(),
            padding_start_ptr,
            padding_end_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
REGISTER_CUDA_OPERATOR(RemovePadding, RemovePaddingOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(GatherPadding, GatherPaddingOp<CUDAContext>);
} // namespace caffe2
