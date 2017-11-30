/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cub/cub.cuh>

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
    int32_t* lengths_out) {
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
    CUDA_KERNEL_ASSERT(lengths_size == 1);
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
    int32_t* lengths_out) {
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
    CUDA_KERNEL_ASSERT(lengths_size == 1);
    len_blocks = outer_size;
    in_start_idx = 0;
  }

  out_start_idx *= block_size;
  in_start_idx *= block_size;

  int len = len_blocks * block_size;
  int start_padding_width = start_padding_width_blocks * block_size;
  int end_padding_width = end_padding_width_blocks * block_size;

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

void lengths_prefix_sum(
    const int32_t* lengths,
    int32_t num_items,
    Tensor<CUDAContext>* prefix_buffer,
    Tensor<CUDAContext>* prefix_sum,
    CUDAContext* context) {
  // Retrieve buffer size
  size_t temp_storage_bytes = 0;
  prefix_sum->Resize(num_items);
  cub::DeviceScan::InclusiveSum(
      NULL,
      temp_storage_bytes,
      lengths,
      prefix_sum->mutable_data<int32_t>(),
      num_items,
      context->cuda_stream());

  // Allocate temporary storage
  auto buffer_size = (temp_storage_bytes + sizeof(int32_t)) / sizeof(int32_t);
  prefix_buffer->Resize(buffer_size);
  void* d_temp_storage =
      static_cast<void*>(prefix_buffer->mutable_data<int32_t>());

  cub::DeviceScan::InclusiveSum(
      d_temp_storage,
      temp_storage_bytes,
      lengths,
      prefix_sum->mutable_data<int32_t>(),
      num_items,
      context->cuda_stream());
}
} // namespace

template <>
template <typename T>
bool AddPaddingOp<CUDAContext>::DoRunWithType() {
  const auto& in = Input(0);
  CAFFE_ENFORCE_GE(in.ndim(), 1);
  const int32_t outer_size = in.dims()[0];
  const auto block_size = std::accumulate(
      in.dims().begin() + 1, in.dims().end(), 1, std::multiplies<TIndex>());

  // if no lengths is provided, assume it is a single full-span entry
  const int32_t* lengths_ptr = nullptr;
  int32_t lengths_size = 1;
  if (InputSize() > 1) {
    const auto& lengths = Input(1);
    lengths_ptr = lengths.data<int32_t>();
    lengths_size = lengths.size();
  }

  // fetch paddings
  // input_size == 2 : pad with zeros
  // input_size == 3 : start and end paddings are the same
  // input_size == 4 : different start and end paddings
  const T* padding_start_ptr = nullptr;
  const T* padding_end_ptr = nullptr;
  if (InputSize() >= 3) {
    auto& padding_start = Input(2);
    CAFFE_ENFORCE_EQ(block_size, padding_start.size());
    padding_start_ptr = padding_start.template data<T>();
  }
  if (InputSize() == 4) {
    auto& padding_end = Input(3);
    CAFFE_ENFORCE_EQ(block_size, padding_end.size());
    padding_end_ptr = padding_end.template data<T>();
  } else {
    padding_end_ptr = padding_start_ptr;
  }

  auto* out = Output(0);
  {
    auto out_dims = in.dims();
    out_dims[0] += (startPaddingWidth_ + endPaddingWidth_) * lengths_size;
    out->Resize(std::move(out_dims));
  }
  const auto* in_ptr = in.template data<T>();
  auto* out_ptr = out->template mutable_data<T>();

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
    auto* lengths_out = Output(1);
    lengths_out->Resize(lengths_size);
    lengths_out_ptr = lengths_out->mutable_data<int32_t>();
  }

  if (lengths_size == 0) {
    return true;
  }

  // Compute the padding using the accumulated lengths
  AddPaddingKernel<T>
      <<<lengths_size, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
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
  CAFFE_ENFORCE_GE(in.ndim(), 1);
  const int32_t outer_size = in.dims()[0];
  const auto block_size = std::accumulate(
      in.dims().begin() + 1, in.dims().end(), 1, std::multiplies<TIndex>());

  // if no lengths is provided, assume it is a single full-span entry
  const int32_t* lengths_ptr = nullptr;
  int32_t lengths_size = 1;
  if (InputSize() > 1) {
    const auto& lengths = Input(1);
    lengths_ptr = lengths.data<int32_t>();
    lengths_size = lengths.size();
  }

  auto* out = Output(0);
  {
    auto out_dims = in.dims();
    out_dims[0] -= (startPaddingWidth_ + endPaddingWidth_) * lengths_size;
    out->Resize(std::move(out_dims));
  }
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
    auto* lengths_out = Output(1);
    lengths_out->Resize(lengths_size);
    lengths_out_ptr = lengths_out->mutable_data<int32_t>();
  }

  if (lengths_size == 0) {
    return true;
  }

  // Compute the padding using the accumulated lengths
  RemovePaddingKernel<T>
      <<<lengths_size, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
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

REGISTER_CUDA_OPERATOR(RemovePadding, RemovePaddingOp<CUDAContext>);
} // namespace caffe2
